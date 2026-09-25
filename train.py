import time, argparse, os.path as osp, os, sys, signal, datetime
import torch, numpy as np
import torch.distributed as dist
from copy import deepcopy

import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim import build_optim_wrapper
from mmengine.logging import MMLogger
from mmengine.utils import symlink
from mmseg.models import build_segmentor
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler
import json

# 中文注释：训练脚本只使用本地日志、TensorBoard 和 eval 文件，不依赖任何外部实验平台。

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_dist_avail_and_initialized()) or dist.get_rank() == 0


def scalar_value(value):
    # 多卡安全日志：把 Tensor/NumPy 标量转成 Python float，只在 rank0 写日志/TensorBoard。
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    return float(value)


def tb_add_scalar(writer, tag, value, step):
    # 中文注释：用 TensorBoard 记录训练/验证指标；写入失败不影响训练主流程。
    if writer is None:
        return
    try:
        writer.add_scalar(tag, scalar_value(value), int(step))
    except Exception:
        pass




def tensor_debug_value(value):
    """Return a compact debug value for scalar tensors/values."""
    if isinstance(value, torch.Tensor):
        try:
            return float(value.detach().float().cpu().reshape(-1)[0].item())
        except Exception:
            return None
    if isinstance(value, np.ndarray):
        try:
            return float(value.reshape(-1)[0])
        except Exception:
            return None
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def format_aux_values(prefix, mapping, keys):
    parts = []
    for key in keys:
        value = mapping.get(key, None) if isinstance(mapping, dict) else None
        scalar = tensor_debug_value(value)
        if scalar is None:
            parts.append(f"{key}=None")
        else:
            parts.append(f"{key}={scalar:.6f}")
    return prefix + "; ".join(parts)


def build_loss_input_from_result(result_dict, metas, loss_input_convertion):
    """Build MultiLoss input and keep auxiliary scalar losses connected.

    Some experimental modules, such as Box-Hypothesis Mamba, produce scalar
    losses inside the model forward.  This helper makes sure those tensors are
    present in the loss input even when an old config copy misses the key.
    """
    loss_input = {'metas': metas}
    for loss_input_key, loss_input_val in loss_input_convertion.items():
        if loss_input_val in result_dict:
            loss_input[loss_input_key] = result_dict[loss_input_val]
    for aux_key in ('loss_boxhyp', 'loss_boxhyp_obj', 'loss_boxhyp_center'):
        if aux_key in result_dict and aux_key not in loss_input:
            loss_input[aux_key] = result_dict[aux_key]
    return loss_input


def maybe_add_aux_loss_fallback(loss, loss_dict, result_dict, enabled=True):
    """Fallback-add BoxHyp loss if MultiLoss did not consume it.

    Returns:
        loss, loss_dict, manual_added
    """
    manual_added = False
    if not bool(enabled):
        return loss, loss_dict, manual_added
    boxhyp = result_dict.get('loss_boxhyp', None) if isinstance(result_dict, dict) else None
    if not isinstance(boxhyp, torch.Tensor):
        return loss, loss_dict, manual_added
    boxhyp_val = tensor_debug_value(boxhyp)
    if boxhyp_val is None or abs(boxhyp_val) <= 1.0e-12:
        return loss, loss_dict, manual_added
    scalar_val = None
    if isinstance(loss_dict, dict):
        # MultiLoss names the wrapper by module class name in this codebase.
        scalar_val = tensor_debug_value(loss_dict.get('ScalarTensorLoss', None))
    if scalar_val is None or abs(scalar_val) <= 1.0e-12:
        loss = loss + boxhyp
        if isinstance(loss_dict, dict):
            loss_dict['ScalarTensorLoss'] = boxhyp.detach()
        manual_added = True
    return loss, loss_dict, manual_added



def _safe_shape(value):
    try:
        return tuple(value.shape)
    except Exception:
        return None


def _debug_count_item(value):
    """Best-effort count for GT-like objects without materializing large data."""
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return 1
        return int(value.shape[0])
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return 1
        return int(value.shape[0])
    if hasattr(value, 'tensor'):
        try:
            t = value.tensor
            if isinstance(t, torch.Tensor):
                return int(t.shape[0]) if t.ndim > 0 else 1
            if isinstance(t, np.ndarray):
                return int(t.shape[0]) if t.ndim > 0 else 1
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, dict):
        for k in ('gt_bboxes_3d', 'gt_boxes', 'gt_boxes_3d', 'boxes', 'gt_labels_3d', 'gt_labels', 'gt_names'):
            if k in value:
                return _debug_count_item(value[k])
        return len(value)
    try:
        return len(value)
    except Exception:
        return -1


def _debug_summarize_one(value, max_items=6):
    """Compact summary for one batch field."""
    if value is None:
        return 'None'
    if isinstance(value, torch.Tensor):
        return f'tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})'
    if isinstance(value, np.ndarray):
        return f'ndarray(shape={value.shape}, dtype={value.dtype})'
    if hasattr(value, 'tensor'):
        try:
            t = value.tensor
            return f'{type(value).__name__}(tensor_shape={_safe_shape(t)}, count={_debug_count_item(value)})'
        except Exception:
            return f'{type(value).__name__}(count={_debug_count_item(value)})'
    if isinstance(value, (list, tuple)):
        counts = []
        types = []
        for item in list(value)[:max_items]:
            counts.append(_debug_count_item(item))
            types.append(type(item).__name__)
        suffix = '' if len(value) <= max_items else f', ...(+{len(value)-max_items})'
        return f'{type(value).__name__}(len={len(value)}, item_types={types}{suffix}, item_counts={counts}{suffix})'
    if isinstance(value, dict):
        keys = list(value.keys())
        shown = keys[:max_items]
        suffix = '' if len(keys) <= max_items else f', ...(+{len(keys)-max_items})'
        return f'dict(keys={shown}{suffix})'
    return f'{type(value).__name__}(count={_debug_count_item(value)})'


def _format_gt_data_debug(data, forward_step, show_all_keys=False):
    """Format GT/debug keys currently present in the dataloader batch."""
    if not isinstance(data, dict):
        return f'[GT DATA DEBUG][train] forward_step={forward_step}, data_type={type(data).__name__}'
    candidate_keys = [
        'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes_3d_has_ann',
        'gt_boxes', 'gt_boxes_3d', 'gt_bboxes', 'gt_labels', 'gt_names',
        'ann_info', 'ann_infos', 'annos', 'annotations',
        'sample_idx', 'sample_token', 'token', 'scene_token', 'lidar_token',
    ]
    present_candidates = [k for k in candidate_keys if k in data]
    if show_all_keys:
        key_text = ','.join(str(k) for k in data.keys())
    else:
        key_text = ','.join(str(k) for k in present_candidates)
    parts = [f'[GT DATA DEBUG][train] forward_step={forward_step}, keys={key_text}']
    if len(present_candidates) == 0:
        # Still show a compact all-key preview so we can see whether the dataset
        # uses a different name for detection annotations.
        all_keys = list(data.keys())
        preview = all_keys[:40]
        suffix = '' if len(all_keys) <= 40 else f', ...(+{len(all_keys)-40})'
        parts.append(f'all_key_preview={preview}{suffix}')
    for key in present_candidates:
        parts.append(f'{key}={_debug_summarize_one(data.get(key, None))}')
    return '; '.join(parts)


def setup_runtime_env():
    """Runtime defaults that make DDP interruption and CUDA errors easier to handle."""
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "0")


def cleanup_distributed():
    """Best-effort cleanup. Safe to call for single GPU and DDP."""
    try:
        if dist.is_available() and dist.is_initialized():
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


def move_to_cuda(obj, non_blocking=True):
    """Move nested tensors/lists/dicts to current CUDA device.

    This keeps variable-length point clouds as lists while moving their tensor
    elements to CUDA, which is required for per-GPU batch_size > 1.
    """
    if isinstance(obj, torch.Tensor):
        return obj.cuda(non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: move_to_cuda(v, non_blocking=non_blocking) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_cuda(v, non_blocking=non_blocking) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_cuda(v, non_blocking=non_blocking) for v in obj)
    return obj


def reduce_sum_count(local_sum, local_count, device=None):
    # DDP 下聚合各 rank 的验证 loss，避免只记录 rank0 子集的均值。
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = torch.tensor([float(local_sum), float(local_count)], device=device, dtype=torch.float64)
    if is_dist_avail_and_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_count = max(float(stats[1].item()), 1.0)
    return float(stats[0].item() / total_count)


def save_eval_summary(work_dir, epoch, miou, iou2, val_loss, metric):
    # 中文注释：自动保存每轮 eval 结果，方便离线追踪训练。
    eval_dir = osp.join(work_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    class_iou = getattr(metric, "last_ious", None)
    class_prec = getattr(metric, "last_precs", None)
    class_recall = getattr(metric, "last_recas", None)
    labels = getattr(metric, "label_str", None)
    data = {
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "mIoU": float(miou),
        "occ_iou": float(iou2),
        "classes": [],
    }
    if class_iou is not None and labels is not None:
        for idx, name in enumerate(labels):
            item = {"name": str(name), "iou": float(class_iou[idx]) * 100.0}
            if class_prec is not None:
                item["precision"] = float(class_prec[idx])
            if class_recall is not None:
                item["recall"] = float(class_recall[idx])
            data["classes"].append(item)
    json_path = osp.join(eval_dir, f"epoch_{int(epoch):04d}.json")
    txt_path = osp.join(eval_dir, f"epoch_{int(epoch):04d}.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"epoch: {data['epoch']}\n")
        f.write(f"val_loss: {data['val_loss']:.6f}\n")
        f.write(f"mIoU: {data['mIoU']:.6f}\n")
        f.write(f"occ_iou: {data['occ_iou']:.6f}\n")
        for item in data["classes"]:
            f.write(
                f"{item['name']}: IoU={item['iou']:.4f}, "
                f"Precision={item.get('precision', 0.0):.4f}, "
                f"Recall={item.get('recall', 0.0):.4f}\n"
            )
    return json_path


def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    # Make relative visualization output directories land under the active work_dir.
    os.environ["GSF_WORK_DIR"] = str(args.work_dir)

    # Evaluation cadence interface:
    #   config: eval_every_epochs = N
    #   CLI override: --eval-every-epochs N
    # N=1 means evaluate every epoch; N=2 means every 2 epochs; N<=0 disables eval.
    if args.eval_every_epochs is not None:
        cfg.eval_every_epochs = int(args.eval_every_epochs)
    cfg.eval_every_epochs = int(cfg.get('eval_every_epochs', 1))

    # init DDP
    # 中文注释：
    # 推荐双卡使用 torchrun 启动，避免 torch.multiprocessing.spawn 在 Ctrl+C 时卡在 join。
    # 普通 python train.py 即使看到多张 GPU，也只跑单进程 cuda:0。
    setup_runtime_env()
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", str(local_rank)))

    if env_world_size > 1:
        distributed = True
        local_rank = env_local_rank
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=args.ddp_timeout_minutes),
        )
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
    else:
        distributed = False
        world_size = 1
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    main_process = is_main_process()  # 中文注释：统一判断当前进程是否为主进程，只有主进程写日志/TensorBoard/保存评估文件。
    
    if main_process:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
        from misc.tb_wrapper import WrappedTBWriter
        writer = WrappedTBWriter('gf3d', log_dir=osp.join(args.work_dir, 'tf'))
        WrappedTBWriter._instance_dict['gf3d'] = writer
    else:
        writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log') if main_process else None
    logger = MMLogger('gf3d', log_file=log_file)
    MMLogger._instance_dict['gf3d'] = logger
    if main_process:
        logger.info(f'Config:\n{cfg.pretty_text}')
        logger.info('External experiment logging disabled permanently; TensorBoard + local eval files are used instead.')

    # build model
    import model
    from dataset import get_dataloader
    from loss import OPENOCC_LOSS

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    if main_process:
        logger.info(f'Number of params: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            if main_process:
                logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    if main_process:
        logger.info('done ddp model')

    # Use get_dataloader to get train and val dataloader respectively from the config
    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        iter_resume=args.iter_resume)

    # get optimizer, loss, scheduler
    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    max_num_epochs = cfg.max_epochs
    if cfg.get('multisteplr', False):
        decay_t_lst = cfg.get('decay_t', [16])
        scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=[len(train_dataset_loader) * t for t in decay_t_lst],
            decay_rate=cfg.get('decay_rate', 0.1),
            warmup_t=cfg.get('warmup_iters', 500),
            warmup_lr_init=1e-6,
            t_in_epochs=False
        )
    else:
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=len(train_dataset_loader) * cfg.get('cycle_limit_epochs', max_num_epochs),
            lr_min=cfg.optimizer["optimizer"]["lr"] * cfg.get('lr_min_factor', 0.1),
            cycle_limit=1,
            warmup_t=cfg.get('warmup_iters', 500),
            warmup_lr_init=1e-6,
            t_in_epochs=False)
    amp = cfg.get('amp', False)
    if amp:
        scaler = torch.cuda.amp.GradScaler()
        os.environ['amp'] = 'true'
    else:
        os.environ['amp'] = 'false'
    
    # resume and load
    epoch = 0
    global_iter = 0
    last_iter = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    if main_process:
        logger.info('resume from: ' + cfg.resume_from)
        logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        optimizer.load_state_dict(ckpt['optimizer']) # TODO: Attention!
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        global_iter = ckpt['global_iter']
        best_miou = float(ckpt.get('best_miou', -1.0))
        last_iter = ckpt['last_iter'] if 'last_iter' in ckpt else 0
        if hasattr(train_dataset_loader.sampler, 'set_last_iter'):
            train_dataset_loader.sampler.set_last_iter(last_iter)
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        try:
            print(raw_model.load_state_dict(state_dict, strict=False))
        except:
            from misc.checkpoint_util import refine_load_from_sd
            print(raw_model.load_state_dict(
                refine_load_from_sd(state_dict), strict=False))
        
    # training
    print_freq = cfg.print_freq
    first_run = True
    grad_accumulation = args.gradient_accumulation
    grad_norm = 0
    best_miou = -1.0  # 中文注释：在本地保存最佳 mIoU 对应的 checkpoint。
    from misc.metric_util import MeanIoU
    miou_metric = MeanIoU(
        list(range(1, 17)),
        17, #17,
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
         True, 17, filter_minmax=False)
    miou_metric.reset()

    aux_loss_debug_cfg = cfg.get('aux_loss_debug', dict(enabled=False))
    if isinstance(aux_loss_debug_cfg, dict):
        aux_loss_debug_enabled = bool(aux_loss_debug_cfg.get('enabled', False))
        aux_loss_debug_interval = int(aux_loss_debug_cfg.get('interval', print_freq))
        aux_loss_debug_max_print = int(aux_loss_debug_cfg.get('max_print', 0))
    else:
        aux_loss_debug_enabled = bool(aux_loss_debug_cfg)
        aux_loss_debug_interval = int(cfg.get('aux_loss_debug_interval', print_freq))
        aux_loss_debug_max_print = int(cfg.get('aux_loss_debug_max_print', 0))
    aux_loss_debug_interval = max(aux_loss_debug_interval, 1)
    aux_loss_debug_max_print = max(aux_loss_debug_max_print, 0)
    aux_loss_debug_print_count = 0
    aux_loss_manual_fallback = bool(cfg.get('aux_loss_manual_fallback', True))

    gt_data_debug_cfg = cfg.get('gt_data_debug', dict(enabled=False))
    if isinstance(gt_data_debug_cfg, dict):
        gt_data_debug_enabled = bool(gt_data_debug_cfg.get('enabled', False))
        gt_data_debug_interval = int(gt_data_debug_cfg.get('interval', print_freq))
        gt_data_debug_max_print = int(gt_data_debug_cfg.get('max_print', 0))
        gt_data_debug_show_all_keys = bool(gt_data_debug_cfg.get('show_all_keys', False))
    else:
        gt_data_debug_enabled = bool(gt_data_debug_cfg)
        gt_data_debug_interval = int(cfg.get('gt_data_debug_interval', print_freq))
        gt_data_debug_max_print = int(cfg.get('gt_data_debug_max_print', 0))
        gt_data_debug_show_all_keys = bool(cfg.get('gt_data_debug_show_all_keys', False))
    gt_data_debug_interval = max(gt_data_debug_interval, 1)
    gt_data_debug_max_print = max(gt_data_debug_max_print, 0)
    gt_data_debug_print_count = 0

    while epoch < max_num_epochs:
        my_model.train()
        os.environ['eval'] = 'false'
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_list = []
        # time.sleep(10)  # removed: unnecessary delay and makes Ctrl+C feel unresponsive.
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, data in enumerate(train_dataset_loader):
            if first_run:
                i_iter = i_iter + last_iter

            data = move_to_cuda(data)
            input_imgs = data.pop('img')            
            data_time_e = time.time()
            if 'points' in data:
                input_points = data.pop('points')
            else:
                input_points = None
            if 'lidar_feature_maps' in data:
                input_lidar_features = data.pop('lidar_feature_maps') # dict
            else:
                input_lidar_features = None
            if 'dpt' in data:
                input_dpt = data.pop('dpt')
            else:
                input_dpt = None

            # DPT debug is now controlled inside the model config (dpt_debug).
            if 'anchor_points' in data:
                input_anchor_points = data.pop('anchor_points')
            else:
                input_anchor_points = None

            forward_step = global_iter + 1
            if (
                gt_data_debug_enabled
                and main_process
                and gt_data_debug_print_count < gt_data_debug_max_print
                and forward_step % gt_data_debug_interval == 0
            ):
                logger.info(_format_gt_data_debug(data, forward_step, show_all_keys=gt_data_debug_show_all_keys))
                gt_data_debug_print_count += 1

            with torch.cuda.amp.autocast(amp):
                # forward + backward + optimize
                result_dict = my_model(
                    imgs=input_imgs,
                    points=input_points,
                    lidar_feature_maps=input_lidar_features,
                    dpt=input_dpt,
                    anchor_points=input_anchor_points,
                    metas=data,
                    global_iter=int(global_iter),
                )

                loss_input = build_loss_input_from_result(
                    result_dict=result_dict,
                    metas=data,
                    loss_input_convertion=cfg.loss_input_convertion,
                )
                loss, loss_dict = loss_func(loss_input)
                loss, loss_dict, manual_aux_added = maybe_add_aux_loss_fallback(
                    loss, loss_dict, result_dict, enabled=aux_loss_manual_fallback
                )

                if (
                    aux_loss_debug_enabled
                    and main_process
                    and aux_loss_debug_print_count < aux_loss_debug_max_print
                    and forward_step % aux_loss_debug_interval == 0
                ):
                    aux_keys = [
                        'loss_boxhyp', 'loss_boxhyp_obj', 'loss_boxhyp_center',
                        'boxhyp_num_ann', 'boxhyp_num_gt', 'boxhyp_num_pos',
                        'boxhyp_bev_iou_mean', 'boxhyp_bev_iou_pos_mean', 'boxhyp_bev_iou_max',
                        'boxhyp_center_dist_mean', 'boxhyp_center_dist_pos_mean',
                    ]
                    logger.info(
                        f"[AUX LOSS DEBUG][train] forward_step={forward_step}, "
                        f"manual_added={manual_aux_added}, "
                        + format_aux_values('result: ', result_dict, aux_keys)
                        + ', '
                        + format_aux_values('loss_input: ', loss_input, aux_keys[:3])
                        + ', '
                        + format_aux_values('loss_dict: ', loss_dict, ['ScalarTensorLoss'])
                    )
                    aux_loss_debug_print_count += 1

                loss = loss / grad_accumulation
            if not amp:
                loss.backward()
                if (global_iter + 1) % grad_accumulation == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                scaler.scale(loss).backward()
                if (global_iter + 1) % grad_accumulation == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            loss_list.append(loss.detach().cpu().item())
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            should_log_train = (forward_step == 1) or (print_freq > 0 and forward_step % print_freq == 0)
            if should_log_train and main_process:
                lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.3f, lr: %.7f, time: %.3f (%.3f)'%(
                    epoch, i_iter + 1, len(train_dataset_loader), 
                    loss.item(), np.mean(loss_list), grad_norm, lr,
                    time_e - time_s, data_time_e - data_time_s))
                detailed_loss = []
                for loss_name, loss_value in loss_dict.items():
                    detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                detailed_loss = ', '.join(detailed_loss)
                logger.info(detailed_loss)
                tb_add_scalar(writer, 'train/loss', np.mean(loss_list), global_iter)
                tb_add_scalar(writer, 'train/lr', lr, global_iter)
                tb_add_scalar(writer, 'train/grad_norm', grad_norm, global_iter)
                for loss_name, loss_value in loss_dict.items():
                    tb_add_scalar(writer, f'train/{loss_name}', loss_value, global_iter)
                loss_list = []
            data_time_s = time.time()
            time_s = time.time()

            if args.iter_resume:
                if (i_iter + 1) % 50 == 0 and main_process:
                    dict_to_save = {
                        'state_dict': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_iter': global_iter,
                        'last_iter': i_iter + 1,
                        'best_miou': best_miou,
                    }
                    save_file_name = os.path.join(os.path.abspath(args.work_dir), 'iter.pth')
                    torch.save(dict_to_save, save_file_name)
                    dst_file = osp.join(args.work_dir, 'latest.pth')
                    symlink(save_file_name, dst_file)
                    logger.info(f'iter ckpt {i_iter + 1} saved!')
        
        # save checkpoint
        if main_process:
            dict_to_save = {
                'state_dict': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
                'best_miou': best_miou,
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            symlink(save_file_name, dst_file)

        epoch += 1
        first_run = False
        
        # eval
        eval_every_epochs = int(cfg.get('eval_every_epochs', 1))
        if eval_every_epochs <= 0:
            if main_process:
                logger.info(f'[EVAL] skipped at epoch {epoch}: eval_every_epochs={eval_every_epochs}')
            continue
        if epoch % eval_every_epochs != 0:
            if main_process:
                logger.info(f'[EVAL] skipped at epoch {epoch}: eval_every_epochs={eval_every_epochs}')
            continue
        my_model.eval()
        os.environ['eval'] = 'true'
        val_loss_list = []

        with torch.no_grad():
            for i_iter_val, data in enumerate(val_dataset_loader):
                data = move_to_cuda(data)
                input_imgs = data.pop('img')
                if 'points' in data:
                    input_points = data.pop('points')
                else:
                    input_points = None
                if 'lidar_feature_maps' in data:
                    input_lidar_features = data.pop('lidar_feature_maps') # dict
                else:
                    input_lidar_features = None
                if 'dpt' in data:
                    input_dpt = data.pop('dpt')
                else:
                    input_dpt = None
                if 'anchor_points' in data:
                    input_anchor_points = data.pop('anchor_points')
                else:
                    input_anchor_points = None
                
                with torch.cuda.amp.autocast(amp):
                    # Pass global_iter during eval as well so config-driven debug/visualizers
                    # do not silently skip when validation runs immediately after training.
                    result_dict = my_model(
                        imgs=input_imgs,
                        points=input_points,
                        lidar_feature_maps=input_lidar_features,
                        dpt=input_dpt,
                        anchor_points=input_anchor_points,
                        metas=data,
                        global_iter=global_iter + i_iter_val,
                    )

                    # Keep eval loss-input construction consistent with training.
                    # Some auxiliary losses (for example loss_boxhyp) are only produced
                    # during training or only when the corresponding proposal module is enabled.
                    # Direct indexing result_dict[result_key] makes eval crash when the aux key
                    # is absent, even though occupancy prediction / mIoU are valid.
                    loss_input = build_loss_input_from_result(
                        result_dict=result_dict,
                        metas=data,
                        loss_input_convertion=cfg.loss_input_convertion,
                    )
                    # If the config still contains ScalarTensorLoss(input_key='loss_boxhyp')
                    # but eval forward does not return loss_boxhyp, feed an explicit zero scalar
                    # so validation loss can be computed without changing occupancy metrics.
                    if 'loss_boxhyp' in cfg.loss_input_convertion and 'loss_boxhyp' not in loss_input:
                        ref = result_dict.get('pred_occ', None)
                        if isinstance(ref, (list, tuple)) and len(ref) > 0 and isinstance(ref[-1], torch.Tensor):
                            device = ref[-1].device
                        elif isinstance(ref, torch.Tensor):
                            device = ref.device
                        else:
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        loss_input['loss_boxhyp'] = torch.zeros((), device=device, dtype=torch.float32)
                    loss, loss_dict = loss_func(loss_input)

                # During training, the joint mask is applied in the loss function (ignored class + invisible class)
                # And also, the empty label 17 is also ignored in loss computation.
                # However, during evaluation, the joint mask is not applied!
                # Since during evaluation, we want to keep the ignored class but not the invisible class, so we have to apply the invisible mask here.
                occ3d_mask_camera_all = data.get('occ3d_mask_camera', None)
                if occ3d_mask_camera_all is not None:
                    occ3d_mask_camera_all = occ3d_mask_camera_all.flatten(1)

                for idx, pred in enumerate(result_dict['pred_occ'][-1]):
                    pred_occ = pred.argmax(0)
                    gt_occ = result_dict['sampled_label'][idx] # idx should be batch idx
                    if occ3d_mask_camera_all is not None:
                        occ3d_mask_camera = occ3d_mask_camera_all[idx]
                        miou_metric._after_step(pred_occ, gt_occ, occ3d_mask_camera)
                    else:
                        miou_metric._after_step(pred_occ, gt_occ)
                
                val_loss_list.append(loss.detach().cpu().numpy())
                if i_iter_val % print_freq == 0 and main_process:
                    logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)'%(
                        epoch, i_iter_val, loss.item(), np.mean(val_loss_list)))
                    detailed_loss = []
                    for loss_name, loss_value in loss_dict.items():
                        detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                    detailed_loss = ', '.join(detailed_loss)
                    logger.info(detailed_loss)
                        
        local_val_loss_sum = float(np.sum(val_loss_list)) if len(val_loss_list) > 0 else 0.0
        local_val_loss_count = len(val_loss_list)
        val_loss_mean = reduce_sum_count(local_val_loss_sum, local_val_loss_count)

        miou, iou2 = miou_metric._after_epoch()
        if main_process:
            logger.info(f'mIoU: {miou}, iou2: {iou2}')
            logger.info('Current val loss is %.3f' % val_loss_mean)
            tb_add_scalar(writer, 'val/loss', val_loss_mean, epoch)
            tb_add_scalar(writer, 'val/mIoU', miou, epoch)
            tb_add_scalar(writer, 'val/occ_iou', iou2, epoch)
            for label_name, class_iou in zip(miou_metric.label_str, getattr(miou_metric, 'last_ious', [])):
                tb_add_scalar(writer, f'val_class_iou/{label_name}', float(class_iou) * 100.0, epoch)

            eval_path = save_eval_summary(args.work_dir, epoch, miou, iou2, val_loss_mean, miou_metric)
            logger.info(f'Eval summary saved to {eval_path}')

            if float(miou) > best_miou:
                best_miou = float(miou)
                best_file_name = os.path.join(os.path.abspath(args.work_dir), 'best.pth')
                torch.save({
                    'state_dict': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_iter': global_iter,
                    'best_miou': best_miou,
                    'best_iou2': float(iou2),
                    'best_val_loss': float(val_loss_mean),
                }, best_file_name)
                logger.info(f'best ckpt saved: {best_file_name}, best_mIoU={best_miou:.4f}')
        miou_metric.reset()

    if writer is not None:
        writer.close()

    if distributed and is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./work_dirs/nuscenes_occ3d_gs25600')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--iter-resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='nuscenes-occ3d')
    parser.add_argument('--ddp-timeout-minutes', type=int, default=30)
    parser.add_argument(
        '--eval-every-epochs',
        type=int,
        default=None,
        help='Override config eval_every_epochs. 1=evaluate every epoch, 2=every two epochs, <=0=disable evaluation.',
    )
    # 中文注释：不再提供外部实验平台参数，日志全部写入本地文件和 TensorBoard。
    args = parser.parse_args()

    def _handle_signal(signum, frame):
        cleanup_distributed()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    ngpus = torch.cuda.device_count()
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.gpus = env_world_size if env_world_size > 1 else 1

    if env_world_size > 1:
        if env_local_rank == 0:
            print(args)
            print("[DDP] torchrun mode detected. Ctrl+C/SIGTERM will be handled by torchrun.")
        try:
            main(env_local_rank, args)
        except KeyboardInterrupt:
            cleanup_distributed()
            if env_local_rank == 0:
                print("[DDP] Interrupted. Process group cleaned.")
            sys.exit(130)
        except Exception:
            cleanup_distributed()
            raise
    else:
        # 中文注释：
        # 不再因为可见 GPU 数量大于 1 就自动 torch.multiprocessing.spawn。
        # 双卡请用 torchrun；普通 python train.py 始终单进程运行，避免 Ctrl+C 卡死。
        print(args)
        if ngpus > 1:
            print(
                "[INFO] Multiple GPUs are visible, but train.py will run single-process on cuda:0.\n"
                "[INFO] For two GPUs, use:\n"
                "       CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py "
                "--py-config <config.py> --work-dir <work_dir>"
            )
        try:
            main(0, args)
        except KeyboardInterrupt:
            cleanup_distributed()
            print("[INFO] Interrupted.")
            sys.exit(130)
        except Exception:
            cleanup_distributed()
            raise
