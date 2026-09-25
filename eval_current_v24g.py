#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Current standalone evaluator for the user's GSF / V24-G Occ3D codebase.

Key properties:
1. Uses the same model inputs and metric path as the current train.py.
2. Supports single-GPU execution and torchrun DDP.
3. Evaluates each validation sample exactly once.
4. Preserves the Occ3D camera-visibility mask without popping it per sample.
5. Recursively moves nested tensors to CUDA.
6. Can override the validation YOLO2D cache and proposal source from CLI.
7. Saves JSON/TXT summaries with per-class IoU, precision and recall.
"""

import argparse
import datetime
import json
import os
import os.path as osp
import signal
import sys
import time
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine import Config
from mmengine.logging import MMLogger
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor

warnings.filterwarnings("ignore")


LEGACY_OCC16_LABELS = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]


def setup_runtime_env() -> None:
    """Set safe defaults for CUDA/DDP evaluation."""
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "0")


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return (not dist_ready()) or dist.get_rank() == 0


def cleanup_distributed() -> None:
    """Best-effort process-group cleanup."""
    try:
        if dist_ready():
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass
    except Exception:
        pass


def move_to_cuda(obj: Any, non_blocking: bool = True) -> Any:
    """Recursively move tensors inside nested dictionaries/lists/tuples to CUDA."""
    if isinstance(obj, torch.Tensor):
        return obj.cuda(non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {key: move_to_cuda(value, non_blocking=non_blocking) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_to_cuda(value, non_blocking=non_blocking) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_cuda(value, non_blocking=non_blocking) for value in obj)
    return obj


def _set_enabled(container: Any, key: str, value: bool) -> bool:
    """Set container[key].enabled when the field exists."""
    try:
        node = container[key] if isinstance(container, dict) else getattr(container, key)
    except Exception:
        return False

    try:
        if isinstance(node, dict):
            node["enabled"] = bool(value)
        else:
            setattr(node, "enabled", bool(value))
        return True
    except Exception:
        return False


def _get_enabled(container: Any, key: str) -> Optional[bool]:
    try:
        node = container[key] if isinstance(container, dict) else getattr(container, key)
        if isinstance(node, dict):
            return bool(node.get("enabled", False))
        return bool(getattr(node, "enabled"))
    except Exception:
        return None


def apply_eval_overrides(cfg: Config, args: argparse.Namespace) -> None:
    """Apply CLI-only evaluation overrides without editing the source config."""
    if args.val_cache:
        val_cache = osp.abspath(osp.expanduser(args.val_cache))
        if not osp.isfile(val_cache):
            raise FileNotFoundError(f"Validation YOLO2D cache does not exist: {val_cache}")
        cfg.val_dataset_config["yolo2d_cache"] = val_cache

    if args.val_batch_size > 0:
        cfg.val_loader["batch_size"] = int(args.val_batch_size)

    if args.val_workers >= 0:
        cfg.val_loader["num_workers"] = int(args.val_workers)
        if int(args.val_workers) == 0:
            cfg.val_loader.pop("prefetch_factor", None)
            cfg.val_loader.pop("persistent_workers", None)

    if args.source != "config":
        if not hasattr(cfg, "model"):
            raise KeyError("Config has no cfg.model; cannot override proposal source.")

        source_state = {
            "offline": (False, True, False),
            "gt2d": (False, False, True),
            "online": (True, False, False),
        }
        online_enabled, offline_enabled, gt2d_enabled = source_state[args.source]

        ok_online = _set_enabled(cfg.model, "online_yolo", online_enabled)
        ok_offline = _set_enabled(cfg.model, "offline_yolo2d", offline_enabled)
        ok_gt2d = _set_enabled(cfg.model, "gt_projected_2d", gt2d_enabled)

        if not (ok_online and ok_offline and ok_gt2d):
            raise KeyError(
                "Could not override one or more model source fields: "
                "model.online_yolo, model.offline_yolo2d, model.gt_projected_2d"
            )

        if args.source == "offline" and not args.val_cache:
            cache_path = cfg.val_dataset_config.get("yolo2d_cache", None)
            if not cache_path:
                raise ValueError(
                    "--source offline requires --val-cache or a valid "
                    "cfg.val_dataset_config.yolo2d_cache."
                )


def resolve_checkpoint(cfg: Config, args: argparse.Namespace) -> str:
    """Resolve checkpoint from CLI, work_dir, or cfg.load_from."""
    candidates = []

    explicit = args.checkpoint or args.resume_from
    if explicit:
        candidates.append(explicit)

    candidates.extend(
        [
            osp.join(args.work_dir, "best.pth"),
            osp.join(args.work_dir, "latest.pth"),
        ]
    )

    cfg_load_from = cfg.get("load_from", "")
    if cfg_load_from:
        candidates.append(cfg_load_from)

    checked = []
    for candidate in candidates:
        if not candidate:
            continue
        path = osp.abspath(osp.expanduser(str(candidate)))
        checked.append(path)
        if osp.isfile(path):
            return path

    raise FileNotFoundError(
        "No evaluation checkpoint was found. Checked:\n  - " + "\n  - ".join(checked)
    )


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(str(key).startswith("module.") for key in state_dict.keys()):
        return {str(key)[7:]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    strict: bool,
    logger: MMLogger,
) -> Tuple[Dict[str, Any], int]:
    """Load checkpoint and report missing/unexpected keys."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint state_dict type: {type(state_dict)}")

    state_dict = _strip_module_prefix(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=bool(strict))

    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))

    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    logger.info(
        f"Checkpoint load strict={bool(strict)}, "
        f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
    )
    if missing_keys:
        logger.warning(f"Missing keys preview: {missing_keys[:30]}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys preview: {unexpected_keys[:30]}")

    global_iter = 0
    if isinstance(checkpoint, dict):
        global_iter = int(checkpoint.get("global_iter", 0))

    return checkpoint if isinstance(checkpoint, dict) else {}, global_iter


def init_runtime(args: argparse.Namespace) -> Tuple[bool, int, int]:
    """
    Initialize single-GPU or torchrun DDP.

    Returns:
        distributed, local_rank, world_size
    """
    setup_runtime_env()

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if env_world_size > 1:
        distributed = True
        local_rank = env_local_rank
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=int(args.ddp_timeout_minutes)),
        )
        world_size = dist.get_world_size()
    else:
        distributed = False
        local_rank = int(args.device)
        world_size = 1
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by this GSF evaluator.")
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid --device {local_rank}; visible CUDA device count={torch.cuda.device_count()}."
            )
        torch.cuda.set_device(local_rank)

    return distributed, local_rank, world_size


def build_logger(work_dir: str) -> MMLogger:
    os.makedirs(work_dir, exist_ok=True)
    log_file = None
    if is_main_process():
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        log_file = osp.join(work_dir, f"eval_{timestamp}.log")

    logger = MMLogger("gf3d", log_file=log_file)
    MMLogger._instance_dict["gf3d"] = logger
    return logger


def save_summary(
    work_dir: str,
    checkpoint_path: str,
    config_path: str,
    val_cache: Optional[str],
    source: str,
    miou: float,
    occ_iou: float,
    metric: Any,
    num_batches: int,
    num_samples: int,
    elapsed_seconds: float,
    peak_memory_mb: float,
) -> Tuple[str, str]:
    """Save machine-readable and human-readable evaluation summaries."""
    os.makedirs(work_dir, exist_ok=True)

    labels = list(getattr(metric, "label_str", []))
    ious = list(getattr(metric, "last_ious", []))
    precisions = list(getattr(metric, "last_precs", []))
    recalls = list(getattr(metric, "last_recas", []))

    classes = []
    for idx, name in enumerate(labels):
        item = {
            "index": int(idx),
            "name": str(name),
            "iou_percent": float(ious[idx] * 100.0) if idx < len(ious) else None,
            "precision": float(precisions[idx]) if idx < len(precisions) else None,
            "recall": float(recalls[idx]) if idx < len(recalls) else None,
        }
        classes.append(item)

    summary = {
        "checkpoint": osp.abspath(checkpoint_path),
        "config": osp.abspath(config_path),
        "val_cache": osp.abspath(val_cache) if val_cache else None,
        "proposal_source": source,
        "mIoU_percent": float(miou),
        "occupancy_iou_percent": float(occ_iou),
        "num_batches_local_rank0": int(num_batches),
        "num_samples_local_rank0": int(num_samples),
        "elapsed_seconds": float(elapsed_seconds),
        "peak_cuda_memory_mb": float(peak_memory_mb),
        "classes": classes,
    }

    json_path = osp.join(work_dir, "eval_summary.json")
    txt_path = osp.join(work_dir, "eval_summary.txt")

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(f"checkpoint: {summary['checkpoint']}\n")
        file.write(f"config: {summary['config']}\n")
        file.write(f"val_cache: {summary['val_cache']}\n")
        file.write(f"proposal_source: {summary['proposal_source']}\n")
        file.write(f"mIoU: {summary['mIoU_percent']:.6f}\n")
        file.write(f"occupancy_iou: {summary['occupancy_iou_percent']:.6f}\n")
        file.write(f"elapsed_seconds: {summary['elapsed_seconds']:.3f}\n")
        file.write(f"peak_cuda_memory_mb: {summary['peak_cuda_memory_mb']:.3f}\n")
        for item in classes:
            file.write(
                f"{item['name']}: "
                f"IoU={item['iou_percent'] if item['iou_percent'] is not None else 0.0:.4f}, "
                f"Precision={item['precision'] if item['precision'] is not None else 0.0:.6f}, "
                f"Recall={item['recall'] if item['recall'] is not None else 0.0:.6f}\n"
            )

    return json_path, txt_path


def main(args: argparse.Namespace) -> None:
    distributed, local_rank, world_size = init_runtime(args)
    main_process = is_main_process()

    try:
        set_random_seed(int(args.seed))
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        args.work_dir = osp.abspath(osp.expanduser(args.work_dir))
        logger = build_logger(args.work_dir)

        cfg = Config.fromfile(args.py_config)
        cfg.work_dir = args.work_dir
        os.environ["GSF_WORK_DIR"] = args.work_dir
        os.environ["eval"] = "true"

        apply_eval_overrides(cfg, args)

        if distributed:
            cfg.gpu_ids = range(world_size)

        if main_process:
            cfg.dump(osp.join(args.work_dir, "eval_effective_config.py"))
            logger.info(f"Config file: {osp.abspath(args.py_config)}")
            logger.info(f"Work directory: {args.work_dir}")
            logger.info(f"Distributed: {distributed}, world_size={world_size}")
            logger.info(
                "Proposal source flags: "
                f"online={_get_enabled(cfg.model, 'online_yolo')}, "
                f"offline={_get_enabled(cfg.model, 'offline_yolo2d')}, "
                f"gt2d={_get_enabled(cfg.model, 'gt_projected_2d')}"
            )
            logger.info(
                f"Validation cache: {cfg.val_dataset_config.get('yolo2d_cache', None)}"
            )
            logger.info(
                f"Validation loader: batch_size={cfg.val_loader.get('batch_size', None)}, "
                f"num_workers={cfg.val_loader.get('num_workers', None)}"
            )

        # Register the project modules before building model/dataset objects.
        import model  # noqa: F401
        from dataset import get_dataloader
        from misc.metric_util import MeanIoU

        model_instance = build_segmentor(cfg.model)
        model_instance.init_weights()

        checkpoint_path = resolve_checkpoint(cfg, args)
        _, checkpoint_global_iter = load_checkpoint(
            model_instance,
            checkpoint_path=checkpoint_path,
            strict=args.strict_load,
            logger=logger,
        )

        model_instance = model_instance.cuda()
        if distributed:
            if bool(cfg.get("syncBN", True)):
                model_instance = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_instance)
            model_instance = torch.nn.parallel.DistributedDataParallel(
                model_instance,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=bool(cfg.get("find_unused_parameters", False)),
            )

        loader_result = get_dataloader(
            cfg.train_dataset_config,
            cfg.val_dataset_config,
            cfg.train_loader,
            cfg.val_loader,
            dist=distributed,
            val_only=True,
        )
        if isinstance(loader_result, (tuple, list)) and len(loader_result) == 2:
            _, val_loader = loader_result
        else:
            val_loader = loader_result

        metric = MeanIoU(
            list(range(1, 17)),
            17,
            LEGACY_OCC16_LABELS,
            True,
            17,
            filter_minmax=False,
            training=False,
        )
        metric.reset()

        amp_from_cfg = bool(cfg.get("amp", False))
        if args.amp == "on":
            use_amp = True
        elif args.amp == "off":
            use_amp = False
        else:
            use_amp = amp_from_cfg

        model_instance.eval()
        torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        local_sample_count = 0
        local_batch_count = 0
        print_freq = int(args.print_freq or cfg.get("print_freq", 50))
        print_freq = max(print_freq, 1)

        with torch.inference_mode():
            for batch_index, data in enumerate(val_loader):
                if args.max_batches > 0 and batch_index >= int(args.max_batches):
                    break

                data = move_to_cuda(data)
                input_imgs = data.pop("img")
                input_points = data.pop("points", None)
                input_lidar_features = data.pop("lidar_feature_maps", None)
                input_dpt = data.pop("dpt", None)
                input_anchor_points = data.pop("anchor_points", None)

                # Keep the mask in metas. Do not pop it inside the per-sample loop.
                mask_all = data.get("occ3d_mask_camera", None)
                if isinstance(mask_all, torch.Tensor) and mask_all.ndim > 2:
                    mask_all = mask_all.flatten(1)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    result_dict = model_instance(
                        imgs=input_imgs,
                        points=input_points,
                        lidar_feature_maps=input_lidar_features,
                        dpt=input_dpt,
                        anchor_points=input_anchor_points,
                        metas=data,
                        global_iter=int(checkpoint_global_iter + batch_index),
                    )

                if "pred_occ" not in result_dict:
                    raise KeyError("Model output has no 'pred_occ'.")
                if "sampled_label" not in result_dict:
                    raise KeyError("Model output has no 'sampled_label'.")

                pred_container = result_dict["pred_occ"]
                pred_batch = pred_container[-1] if isinstance(pred_container, (list, tuple)) else pred_container
                target_batch = result_dict["sampled_label"]

                batch_size = int(pred_batch.shape[0])
                if int(target_batch.shape[0]) != batch_size:
                    raise ValueError(
                        f"Prediction/target batch mismatch: pred={tuple(pred_batch.shape)}, "
                        f"target={tuple(target_batch.shape)}"
                    )

                for sample_index in range(batch_size):
                    pred_occ = pred_batch[sample_index].argmax(0)
                    gt_occ = target_batch[sample_index]
                    sample_mask = mask_all[sample_index] if mask_all is not None else None

                    # Exactly one metric update per sample.
                    metric._after_step(pred_occ, gt_occ, sample_mask)
                    local_sample_count += 1

                local_batch_count += 1

                if main_process and (
                    batch_index == 0
                    or (batch_index + 1) % print_freq == 0
                    or (batch_index + 1) == len(val_loader)
                ):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[EVAL] batch={batch_index + 1}/{len(val_loader)}, "
                        f"local_samples={local_sample_count}, elapsed={elapsed:.1f}s"
                    )

        miou, occ_iou = metric._after_epoch()
        elapsed_seconds = time.time() - start_time
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

        if main_process:
            logger.info(f"mIoU: {miou}")
            logger.info(f"iou2: {occ_iou}")
            logger.info(f"Elapsed seconds: {elapsed_seconds:.3f}")
            logger.info(f"Peak CUDA memory: {peak_memory_mb:.3f} MB")

            val_cache = cfg.val_dataset_config.get("yolo2d_cache", None)
            json_path, txt_path = save_summary(
                work_dir=args.work_dir,
                checkpoint_path=checkpoint_path,
                config_path=args.py_config,
                val_cache=val_cache,
                source=args.source,
                miou=miou,
                occ_iou=occ_iou,
                metric=metric,
                num_batches=local_batch_count,
                num_samples=local_sample_count,
                elapsed_seconds=elapsed_seconds,
                peak_memory_mb=peak_memory_mb,
            )
            logger.info(f"Evaluation JSON saved to: {json_path}")
            logger.info(f"Evaluation TXT saved to: {txt_path}")

        metric.reset()

    finally:
        cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone evaluator for current GSF / V24-G Occ3D code."
    )
    parser.add_argument(
        "--py-config",
        required=True,
        help="Current GSF config file.",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Directory for evaluation logs and summaries.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint to evaluate. Preferred explicit argument.",
    )
    parser.add_argument(
        "--resume-from",
        default="",
        help="Backward-compatible alias of --checkpoint.",
    )
    parser.add_argument(
        "--val-cache",
        default="",
        help="Override cfg.val_dataset_config.yolo2d_cache.",
    )
    parser.add_argument(
        "--source",
        choices=["config", "offline", "gt2d", "online"],
        default="config",
        help=(
            "Proposal source. 'config' keeps the config unchanged; "
            "'offline' forces offline YOLO cache; 'gt2d' forces projected GT2D; "
            "'online' forces online YOLO."
        ),
    )
    parser.add_argument("--device", type=int, default=0, help="Single-GPU CUDA index.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp",
        choices=["config", "on", "off"],
        default="config",
        help="Use config AMP setting or override it.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Require an exact checkpoint/model key match.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=0,
        help="Override validation batch size when > 0.",
    )
    parser.add_argument(
        "--val-workers",
        type=int,
        default=-1,
        help="Override validation worker count when >= 0.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Debug only: stop after this many local validation batches; 0 evaluates all.",
    )
    parser.add_argument(
        "--print-freq",
        type=int,
        default=0,
        help="Progress logging interval; 0 uses cfg.print_freq.",
    )
    parser.add_argument("--ddp-timeout-minutes", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    def _handle_signal(signum, frame):
        cleanup_distributed()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        main(args)
    except KeyboardInterrupt:
        cleanup_distributed()
        if is_main_process():
            print("[EVAL] Interrupted; process group cleaned.")
        sys.exit(130)
    except Exception:
        cleanup_distributed()
        raise
