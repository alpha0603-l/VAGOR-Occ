#!/usr/bin/env python3
"""Generate offline YOLO2D proposal cache for GSF OQG route.

The cache coordinates are deliberately generated AFTER the same resize/crop used
by the GSF/DAOcc image pipeline: raw nuScenes 1600x900 -> resize 0.44 ->
704x396 -> bottom crop to 704x256.  Therefore cached boxes are already aligned
with the network tensor [3,256,704] and OQG can sample ROI features directly.

Output format:
    {
      'meta': {...},
      'samples': {
        sample_token: {
          'boxes':  np.ndarray [6,max_det,4] float32 xyxy in 704x256 coords,
          'scores': np.ndarray [6,max_det] float32,
          'labels': np.ndarray [6,max_det] int64,  nuScenes10 det label ids,
          'valid':  np.ndarray [6,max_det] bool,
          'counts': np.ndarray [6] int64,
        }, ...
      }
    }
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

try:
    import mmengine
except Exception:  # pragma: no cover
    mmengine = None


CAM_ORDER = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
    'CAM_BACK_LEFT',
]

# COCO -> DAOcc/nuScenes10 detection order:
# 0 car, 1 truck, 2 construction_vehicle, 3 bus, 4 trailer,
# 5 barrier, 6 motorcycle, 7 bicycle, 8 pedestrian, 9 traffic_cone.
COCO_TO_NUSCENES10 = {
    0: 8,  # person -> pedestrian
    1: 7,  # bicycle -> bicycle
    2: 0,  # car -> car
    3: 6,  # motorcycle -> motorcycle
    5: 3,  # bus -> bus
    7: 1,  # truck -> truck
}

SMALL_CLASS_IDS = (5, 6, 7, 8, 9)
VEHICLE_CLASS_IDS = (0, 1, 2, 3, 4)


def load_pickle(path: str):
    if mmengine is not None:
        return mmengine.load(path)
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(obj, path: str):
    """Dump pickle without relying on filename extension.

    mmengine.dump infers format from the last suffix and will fail for names like
    ``xxx.pkl.partial``.  Use stdlib pickle so both final and partial cache
    files are valid regardless of suffix.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def join_data_root(data_root: str, path: str) -> str:
    if path is None:
        return path
    path = str(path)
    if os.path.isabs(path):
        return path
    norm = path[2:] if path.startswith('./') else path
    root_norm = (data_root[2:] if data_root.startswith('./') else data_root).rstrip('/')
    if norm.startswith(root_norm + '/') or norm == root_norm:
        return norm
    return os.path.join(data_root, path)


def sample_token(info: dict, fallback: int) -> str:
    return str(info.get('token', info.get('sample_token', info.get('sample_idx', fallback))))


def flatten_infos(data_obj) -> List[dict]:
    infos = data_obj['infos'] if isinstance(data_obj, dict) and 'infos' in data_obj else data_obj
    if isinstance(infos, list):
        return list(infos)
    if isinstance(infos, dict):
        # GF3D nested: {scene_token: [frame0, frame1, ...]}
        out = []
        for _, v in infos.items():
            if isinstance(v, list):
                out.extend(v)
            elif isinstance(v, dict):
                out.append(v)
        return out
    raise TypeError(f'Unsupported infos type: {type(infos)}')


def get_cam_filename(info: dict, cam_name: str) -> str:
    if 'data' in info:
        return info['data'][cam_name]['filename']
    if 'cams' in info:
        cam = info['cams'][cam_name]
        return cam.get('data_path', cam.get('filename'))
    raise KeyError('Unsupported info format: expected key data or cams')


def build_jobs(infos: List[dict], data_root: str, sample_interval: int = 1, limit: int = 0) -> Tuple[List[Tuple[str, int, str]], List[str]]:
    jobs = []
    tokens = []
    selected = infos[::max(int(sample_interval), 1)]
    if limit and limit > 0:
        selected = selected[:int(limit)]
    for i, info in enumerate(selected):
        tok = sample_token(info, i)
        tokens.append(tok)
        for cam_idx, cam_name in enumerate(CAM_ORDER):
            img_path = join_data_root(data_root, get_cam_filename(info, cam_name))
            jobs.append((tok, cam_idx, img_path))
    return jobs, tokens


def preprocess_image(path: str, final_h: int, final_w: int, raw_h: int, raw_w: int, resize: float, bot_pct: float) -> np.ndarray:
    """Apply the same deterministic resize/crop as ResizeCropFlipImage.

    This matches data_aug_conf in config/_base_/occ3d_pcd_dfa3d.py:
        H=900, W=1600, final_dim=(256,704), resize_lim=(0.44,0.44), bot_pct_lim=(0,0).
    """
    img = Image.open(path).convert('RGB')
    resize_dims = (int(raw_w * resize), int(raw_h * resize))  # PIL uses W,H
    img = img.resize(resize_dims)
    new_w, new_h = resize_dims
    crop_h = int((1.0 - float(bot_pct)) * new_h) - int(final_h)
    crop_w = int(max(0, new_w - int(final_w)) / 2)
    crop = (crop_w, crop_h, crop_w + int(final_w), crop_h + int(final_h))
    img = img.crop(crop)
    arr = np.asarray(img, dtype=np.uint8)
    arr = arr[..., ::-1].copy()
    return arr


def map_labels(cls: np.ndarray, class_mode: str) -> np.ndarray:
    cls = cls.astype(np.int64, copy=False)
    if class_mode == 'nuscenes10':
        return cls
    if class_mode == 'coco':
        out = np.full_like(cls, -1, dtype=np.int64)
        for src, dst in COCO_TO_NUSCENES10.items():
            out[cls == int(src)] = int(dst)
        return out
    raise ValueError(f'Unsupported class_mode={class_mode!r}')


def threshold_by_class(labels: np.ndarray, scores: np.ndarray, conf_small: float, conf_vehicle: float) -> np.ndarray:
    keep = np.zeros(scores.shape, dtype=bool)
    for cid in SMALL_CLASS_IDS:
        keep |= (labels == cid) & (scores >= float(conf_small))
    for cid in VEHICLE_CLASS_IDS:
        keep |= (labels == cid) & (scores >= float(conf_vehicle))
    keep &= labels >= 0
    return keep


def empty_sample(max_det: int) -> Dict[str, np.ndarray]:
    return {
        'boxes': np.zeros((len(CAM_ORDER), max_det, 4), dtype=np.float32),
        'scores': np.zeros((len(CAM_ORDER), max_det), dtype=np.float32),
        'labels': np.full((len(CAM_ORDER), max_det), -1, dtype=np.int64),
        'valid': np.zeros((len(CAM_ORDER), max_det), dtype=np.bool_),
        'counts': np.zeros((len(CAM_ORDER),), dtype=np.int64),
    }



def shard_jobs_and_tokens(jobs: List[Tuple[str, int, str]], tokens: List[str], shard_id: int, num_shards: int) -> Tuple[List[Tuple[str, int, str]], List[str]]:
    """Split by sample token, not by camera image.

    Each shard processes all 6 cameras of its assigned samples.  This avoids
    multiple shards writing different cameras of the same sample and makes merge
    deterministic and cheap.
    """
    num_shards = int(num_shards)
    shard_id = int(shard_id)
    if num_shards <= 1:
        return jobs, tokens
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f'shard_id must be in [0, {num_shards - 1}], got {shard_id}')
    shard_tokens = tokens[shard_id::num_shards]
    shard_set = set(shard_tokens)
    shard_jobs = [job for job in jobs if job[0] in shard_set]
    return shard_jobs, shard_tokens


def preprocess_job(job: Tuple[str, int, str], args) -> Tuple[str, int, str, np.ndarray | None, str | None]:
    tok, cam_idx, img_path = job
    try:
        arr = preprocess_image(
            img_path,
            final_h=int(args.final_dim[0]), final_w=int(args.final_dim[1]),
            raw_h=int(args.raw_shape[0]), raw_w=int(args.raw_shape[1]),
            resize=float(args.resize), bot_pct=float(args.bot_pct),
        )
        return tok, cam_idx, img_path, arr, None
    except Exception as exc:
        return tok, cam_idx, img_path, None, str(exc)


def load_batch_images(batch: List[Tuple[str, int, str]], args) -> Tuple[List[np.ndarray], List[Tuple[str, int, str]]]:
    """Preprocess images for one YOLO batch, optionally with CPU threads.

    YOLO inference itself is usually light here.  Low GPU util is often caused by
    sequential image read/resize/crop.  Threaded preprocessing helps keep the GPU
    fed without increasing GPU memory much.
    """
    workers = max(int(getattr(args, 'preprocess_workers', 0)), 0)
    outputs = []
    if workers > 1 and len(batch) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            outputs = list(ex.map(lambda j: preprocess_job(j, args), batch))
    else:
        outputs = [preprocess_job(j, args) for j in batch]

    imgs: List[np.ndarray] = []
    kept_meta: List[Tuple[str, int, str]] = []
    for tok, cam_idx, img_path, arr, err in outputs:
        if arr is None:
            print(f'[YOLO2DCache][warn] failed to read {img_path}: {err}')
            continue
        imgs.append(arr)
        kept_meta.append((tok, cam_idx, img_path))
    return imgs, kept_meta


def merge_cache_files(input_paths: List[str], out_path: str):
    merged_samples = {}
    base_meta = None
    total_in_samples = 0
    for path in input_paths:
        obj = load_pickle(path)
        if not isinstance(obj, dict) or 'samples' not in obj:
            raise ValueError(f'Bad cache file: {path}')
        meta = obj.get('meta', {})
        samples = obj['samples']
        if base_meta is None:
            base_meta = dict(meta)
        for tok, item in samples.items():
            if tok in merged_samples:
                old = merged_samples[tok]
                if 'valid' in old and 'valid' in item:
                    # If duplicate sample exists, prefer non-empty cameras from the later shard.
                    cam_valid = item['valid'].sum(axis=1) > 0
                    old['boxes'][cam_valid] = item['boxes'][cam_valid]
                    old['scores'][cam_valid] = item['scores'][cam_valid]
                    old['labels'][cam_valid] = item['labels'][cam_valid]
                    old['valid'][cam_valid] = item['valid'][cam_valid]
                    old['counts'][cam_valid] = item['counts'][cam_valid]
                else:
                    merged_samples[tok] = item
            else:
                merged_samples[tok] = item
        total_in_samples += len(samples)
        print(f'[YOLO2DCacheMerge] loaded {path}: samples={len(samples)}')

    if base_meta is None:
        base_meta = {}
    base_meta = dict(base_meta)
    base_meta['merged_from'] = list(input_paths)
    base_meta['num_samples'] = len(merged_samples)
    base_meta['format'] = base_meta.get('format', 'gsf_yolo2d_cache_v1')

    out_obj = {'meta': base_meta, 'samples': merged_samples}
    dump_pickle(out_obj, out_path)
    total_valid = sum(int(v['valid'].sum()) for v in merged_samples.values())
    per_class = {}
    for v in merged_samples.values():
        labs = v['labels'][v['valid']]
        if labs.size:
            uniq, cnt = np.unique(labs, return_counts=True)
            for k, c in zip(uniq.tolist(), cnt.tolist()):
                per_class[int(k)] = per_class.get(int(k), 0) + int(c)
    print(f'[YOLO2DCacheMerge] saved: {out_path}')
    print(f'[YOLO2DCacheMerge] input_sample_sum={total_in_samples}, merged_samples={len(merged_samples)}, total_valid_boxes={total_valid}, per_class={per_class}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default=None, help='Path to nuscenes info pkl')
    parser.add_argument('--data-root', default='data/nuscenes/', help='nuScenes data root')
    parser.add_argument('--out', required=True, help='Output cache pkl')
    parser.add_argument('--merge-inputs', nargs='+', default=None, help='Merge shard cache pkls and exit')
    parser.add_argument('--weights', default='ckpts/yolo26m.pt', help='Ultralytics YOLO weights')
    parser.add_argument('--class-mode', default='nuscenes10', choices=['nuscenes10', 'coco'], help='YOLO label space')
    parser.add_argument('--device', default='0', help='YOLO device, e.g. 0 or cuda:0 or cpu')
    parser.add_argument('--batch-size', type=int, default=24, help='Number of camera images per YOLO batch')
    parser.add_argument('--raw-max-det', type=int, default=300, help='max_det passed to YOLO before class-specific filtering')
    parser.add_argument('--max-det', type=int, default=100, help='Padded max boxes per camera saved to cache')
    parser.add_argument('--conf-small', type=float, default=0.05, help='score threshold for barrier/ped/moto/bicycle/cone')
    parser.add_argument('--conf-vehicle', type=float, default=0.10, help='score threshold for vehicle classes')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[256, 704], metavar=('H', 'W'), help='YOLO imgsz')
    parser.add_argument('--final-dim', type=int, nargs=2, default=[256, 704], metavar=('H', 'W'), help='network final image size')
    parser.add_argument('--raw-shape', type=int, nargs=2, default=[900, 1600], metavar=('H', 'W'), help='raw nuScenes image shape used by data_aug_conf')
    parser.add_argument('--resize', type=float, default=0.44, help='fixed resize from data_aug_conf')
    parser.add_argument('--bot-pct', type=float, default=0.0, help='deterministic bottom crop pct from data_aug_conf')
    parser.add_argument('--sample-interval', type=int, default=1, help='Optional sample stride for quick debug')
    parser.add_argument('--limit', type=int, default=0, help='Optional max samples for quick debug')
    parser.add_argument('--save-every', type=int, default=0, help='Save partial cache every N samples; 0 disables; recommended for speed')
    parser.add_argument('--num-shards', type=int, default=1, help='Total number of sample shards for multi-process/multi-GPU cache generation')
    parser.add_argument('--shard-id', type=int, default=0, help='Current shard id in [0, num_shards-1]')
    parser.add_argument('--preprocess-workers', type=int, default=8, help='CPU threads per process for image read/resize/crop. Increase if GPU util is low')
    args = parser.parse_args()

    if args.merge_inputs:
        merge_cache_files(args.merge_inputs, args.out)
        return
    if not args.ann:
        parser.error('--ann is required unless --merge-inputs is used')

    from ultralytics import YOLO

    data_obj = load_pickle(args.ann)
    infos = flatten_infos(data_obj)
    jobs, tokens = build_jobs(infos, args.data_root, sample_interval=args.sample_interval, limit=args.limit)
    full_tokens, full_jobs = len(tokens), len(jobs)
    jobs, tokens = shard_jobs_and_tokens(jobs, tokens, args.shard_id, args.num_shards)
    print(f'[YOLO2DCache] infos={len(infos)}, selected_samples={len(tokens)}/{full_tokens}, camera_jobs={len(jobs)}/{full_jobs}, shard={args.shard_id}/{args.num_shards}')
    print(f'[YOLO2DCache] weights={args.weights}, class_mode={args.class_mode}, device={args.device}')
    print(f'[YOLO2DCache] output coords: post-resize/crop W={args.final_dim[1]}, H={args.final_dim[0]}')

    model = YOLO(args.weights)
    max_det = int(args.max_det)
    samples = {tok: empty_sample(max_det) for tok in tokens}

    processed = 0
    sample_seen = set()
    last_partial_seen = 0
    for start in range(0, len(jobs), int(args.batch_size)):
        batch = jobs[start:start + int(args.batch_size)]
        imgs, kept_meta = load_batch_images(batch, args)
        if not imgs:
            continue

        results = model.predict(
            source=imgs,
            imgsz=tuple(int(x) for x in args.imgsz),
            conf=min(float(args.conf_small), float(args.conf_vehicle)),
            max_det=int(args.raw_max_det),
            device=args.device,
            verbose=False,
            batch=int(args.batch_size),
        )

        for res, (tok, cam_idx, _) in zip(results, kept_meta):
            sample_seen.add(tok)
            if not hasattr(res, 'boxes') or res.boxes is None or len(res.boxes) == 0:
                continue
            boxes = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            scores = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
            labels_raw = res.boxes.cls.detach().cpu().numpy().astype(np.int64)
            labels = map_labels(labels_raw, args.class_mode)
            keep = threshold_by_class(labels, scores, args.conf_small, args.conf_vehicle)
            if not keep.any():
                continue
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            # Clip to network image coords.
            final_h, final_w = int(args.final_dim[0]), int(args.final_dim[1])
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, final_w - 1)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, final_h - 1)
            wh_ok = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, scores, labels = boxes[wh_ok], scores[wh_ok], labels[wh_ok]
            if boxes.shape[0] == 0:
                continue
            order = np.argsort(-scores)[:max_det]
            boxes, scores, labels = boxes[order], scores[order], labels[order]
            n = int(boxes.shape[0])
            item = samples[tok]
            item['boxes'][cam_idx, :n] = boxes
            item['scores'][cam_idx, :n] = scores
            item['labels'][cam_idx, :n] = labels
            item['valid'][cam_idx, :n] = True
            item['counts'][cam_idx] = n

        processed += len(kept_meta)
        if processed % max(int(args.batch_size) * 20, 1) == 0 or processed == len(jobs):
            valid_boxes = sum(int(v['valid'].sum()) for v in samples.values())
            print(f'[YOLO2DCache] processed_images={processed}/{len(jobs)}, samples_seen={len(sample_seen)}, valid_boxes={valid_boxes}', flush=True)
        if args.save_every and len(sample_seen) > 0 and (len(sample_seen) - last_partial_seen) >= int(args.save_every):
            partial_out = args.out + '.partial.pkl'
            dump_pickle({'meta': {'partial': True, 'samples_seen': len(sample_seen)}, 'samples': samples}, partial_out)
            last_partial_seen = len(sample_seen)
            print(f'[YOLO2DCache] partial saved: {partial_out} samples_seen={last_partial_seen}', flush=True)

    meta = {
        'format': 'gsf_yolo2d_cache_v1',
        'ann': args.ann,
        'data_root': args.data_root,
        'weights': args.weights,
        'class_mode': args.class_mode,
        'cam_order': CAM_ORDER,
        'coord': 'post_resize_crop_network_image_xyxy',
        'final_dim_hw': [int(args.final_dim[0]), int(args.final_dim[1])],
        'raw_shape_hw': [int(args.raw_shape[0]), int(args.raw_shape[1])],
        'resize': float(args.resize),
        'bot_pct': float(args.bot_pct),
        'max_det': max_det,
        'conf_small': float(args.conf_small),
        'conf_vehicle': float(args.conf_vehicle),
        'num_samples': len(samples),
        'shard_id': int(args.shard_id),
        'num_shards': int(args.num_shards),
        'preprocess_workers': int(args.preprocess_workers),
    }
    out_obj = {'meta': meta, 'samples': samples}
    dump_pickle(out_obj, args.out)
    total_valid = sum(int(v['valid'].sum()) for v in samples.values())
    per_class = {}
    for v in samples.values():
        labs = v['labels'][v['valid']]
        if labs.size:
            uniq, cnt = np.unique(labs, return_counts=True)
            for k, c in zip(uniq.tolist(), cnt.tolist()):
                per_class[int(k)] = per_class.get(int(k), 0) + int(c)
    print(f'[YOLO2DCache] saved: {args.out}')
    print(f'[YOLO2DCache] total_valid_boxes={total_valid}, per_class={per_class}')


if __name__ == '__main__':
    main()
