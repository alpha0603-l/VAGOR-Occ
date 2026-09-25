import os
from copy import deepcopy

import mmengine
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset, get_worker_info

from . import OPENOCC_DATASET, OPENOCC_TRANSFORMS
from .utils import get_img2global, get_lidar2global


@OPENOCC_DATASET.register_module()
class NuScenesOcc3DDataset(Dataset):
    """nuScenes Occ3D dataset wrapper for GaussianFormer3D/GSF.

    Supports both GaussianFormer3D's original nested info structure:
        data['infos'][scene][frame]
    and DAOcc/MMDet3D-style list info structure:
        data['infos'][idx]

    For DAOcc pkl, Occ3D labels are read from:
        info['occ3d']['occ_path'] + '/labels.npz'

    Detection GT is recovered directly from the pkl because this lightweight
    occupancy pipeline does not run DAOcc's LoadAnnotations3D transform.
    """

    def __init__(
        self,
        data_root=None,
        imageset=None,
        data_aug_conf=None,
        pipeline=None,
        vis_indices=None,
        num_samples=0,
        phase='train',
        sample_interval=1,
        sample_offset=0,
        gt_raw_debug=None,
        coord_debug=None,
        respect_valid_flag=False,
        numeric_label_format='auto',
        yolo2d_cache=None,
        yolo2d_cache_max_det=80,
        yolo2d_cache_score_thr=0.0,
        yolo2d_cache_debug=False,
        yolo2d_cache_debug_max_print=5,
    ):
        self.data_path = data_root or ''
        data = mmengine.load(imageset)
        self.scene_infos = data['infos']
        self.metadata = data.get('metadata', None)
        self.data_aug_conf = data_aug_conf
        self.test_mode = phase != 'train'
        self.pipeline = [OPENOCC_TRANSFORMS.build(t) for t in pipeline]
        self.sensor_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_RIGHT',
            'CAM_BACK_LEFT',
        ]

        # Optional offline YOLO2D cache.  The cache is generated in the same
        # post-resize/crop image coordinate system used by the network
        # (default 704x256), so OQG can sample ROI features without any further
        # coordinate conversion.  The cache stores nuScenes-10 detection labels.
        self.yolo2d_cache_path = yolo2d_cache
        self.yolo2d_cache_max_det = int(yolo2d_cache_max_det)
        self.yolo2d_cache_score_thr = float(yolo2d_cache_score_thr)
        self.yolo2d_cache_debug = bool(yolo2d_cache_debug)
        self.yolo2d_cache_debug_max_print = int(yolo2d_cache_debug_max_print)
        self._yolo2d_cache_debug_print_count = 0
        self.yolo2d_cache = None
        self.yolo2d_cache_meta = {}
        if self.yolo2d_cache_path not in (None, '', False):
            cache_obj = mmengine.load(self.yolo2d_cache_path)
            if isinstance(cache_obj, dict) and 'samples' in cache_obj:
                self.yolo2d_cache = cache_obj.get('samples', {})
                self.yolo2d_cache_meta = cache_obj.get('meta', {}) or {}
            elif isinstance(cache_obj, dict):
                # Backward compatible: allow {sample_token: det_dict}.
                self.yolo2d_cache = cache_obj
                self.yolo2d_cache_meta = {}
            else:
                raise TypeError(f'Unsupported YOLO2D cache object type: {type(cache_obj)}')
            print(
                f'[NuScenesOcc3DDataset] loaded YOLO2D cache: {self.yolo2d_cache_path}, '
                f'samples={len(self.yolo2d_cache)}, max_det={self.yolo2d_cache_max_det}, '
                f'score_thr={self.yolo2d_cache_score_thr}',
                flush=True,
            )

        # DAOcc/default.yaml detection label order.  GSF uses the same pkl, so
        # Box-Hypothesis Mamba supervision must keep this exact order.
        self.object_classes = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
        ]
        self.name_to_label = {name: idx for idx, name in enumerate(self.object_classes)}
        # nuScenes/MMDet3D aliases that may appear in DAOcc pkl.
        self.name_to_label.update({
            'vehicle.car': 0,
            'vehicle.truck': 1,
            'vehicle.construction': 2,
            'vehicle.construction_vehicle': 2,
            'construction_vehicle': 2,
            'vehicle.bus.rigid': 3,
            'vehicle.bus.bendy': 3,
            'vehicle.bus': 3,
            'vehicle.trailer': 4,
            'movable_object.barrier': 5,
            'vehicle.motorcycle': 6,
            'vehicle.bicycle': 7,
            'human.pedestrian.adult': 8,
            'human.pedestrian.child': 8,
            'human.pedestrian.construction_worker': 8,
            'human.pedestrian.police_officer': 8,
            'human.pedestrian.personal_mobility': 8,
            'pedestrian': 8,
            'movable_object.trafficcone': 9,
            'movable_object.traffic_cone': 9,
            'trafficcone': 9,
            'traffic_cone': 9,
            'traffic-cone': 9,
        })
        # Make lookup robust to case/space/hyphen variants.
        self.name_to_label = {self._normalize_class_name(k): v for k, v in self.name_to_label.items()}
        self.label_to_name = {idx: name for idx, name in enumerate(self.object_classes)}
        # DAOcc 10-class detection labels -> official Occ3D semantic indices.
        # Occ3D keeps an extra "others" class at index 0, so object classes are
        # shifted compared with the old SurroundOcc/GSF-16 order:
        #   0 others, 1 barrier, 2 bicycle, 3 bus, 4 car,
        #   5 construction_vehicle, 6 motorcycle, 7 pedestrian, 8 traffic_cone,
        #   9 trailer, 10 truck, 11 driveable_surface, ... 16 vegetation, 17 free.
        # Keep gt_labels_3d in DAOcc order for YOLO/BoxHyp compatibility; use
        # this map only for debug and semantic-prior conversion downstream.
        self.dao_to_gsf_occ_sem = np.asarray([4, 10, 5, 3, 9, 1, 6, 2, 7, 8], dtype=np.int64)
        self.gsf_occ_to_dao = {
            0: 5, 1: 7, 2: 3, 3: 0, 4: 2,
            5: 6, 6: 8, 7: 9, 8: 4, 9: 1,
        }
        self.official_occ3d_to_dao = {
            1: 5, 2: 7, 3: 3, 4: 0, 5: 2,
            6: 6, 7: 8, 8: 9, 9: 4, 10: 1,
        }
        self.respect_valid_flag = bool(respect_valid_flag)
        self.numeric_label_format = str(numeric_label_format).lower().strip()

        coord_debug = coord_debug or {}
        self.coord_debug_enabled = bool(coord_debug.get('enabled', False))
        self.coord_debug_interval = max(int(coord_debug.get('interval', 50)), 1)
        self.coord_debug_max_print = max(int(coord_debug.get('max_print', 8)), 0)
        self.coord_debug_show_matrices = bool(coord_debug.get('show_matrices', False))
        self.coord_debug_show_label_hist = bool(coord_debug.get('show_label_hist', True))
        self._coord_debug_call_count = 0
        self._coord_debug_print_count = 0

        gt_raw_debug = gt_raw_debug or {}
        self.gt_raw_debug_enabled = bool(gt_raw_debug.get('enabled', False))
        self.gt_raw_debug_interval = max(int(gt_raw_debug.get('interval', 50)), 1)
        self.gt_raw_debug_max_print = max(int(gt_raw_debug.get('max_print', 4)), 0)
        self.gt_raw_debug_only_when_problem = bool(gt_raw_debug.get('only_when_problem', True))
        self.gt_raw_debug_show_info_keys = bool(gt_raw_debug.get('show_info_keys', True))
        self.gt_raw_debug_show_value_summaries = bool(gt_raw_debug.get('show_value_summaries', False))
        self._gt_raw_debug_print_count = 0

        # GF3D pkl: infos is dict and metadata contains (scene, frame_idx).
        # DAOcc pkl: infos is list; use list index directly.
        if isinstance(self.scene_infos, dict):
            self.keyframes = data['metadata']
            self.keyframes = sorted(self.keyframes, key=lambda x: x[0] + '{:0>3}'.format(str(x[1])))
            self.info_style = 'gf3d_nested'
        elif isinstance(self.scene_infos, list):
            self.keyframes = list(range(len(self.scene_infos)))
            self.info_style = 'list'
        else:
            raise TypeError(f'Unsupported infos type: {type(self.scene_infos)}')

        # Optional dataset thinning for quick experiments.  This changes which
        # keyframes are used as samples, but each selected sample still loads
        # its own configured multi-sweep point cloud.
        sample_interval = max(int(sample_interval), 1)
        sample_offset = max(int(sample_offset), 0)
        if sample_interval > 1:
            self.keyframes = self.keyframes[sample_offset::sample_interval]

        if vis_indices is not None:
            if len(vis_indices) > 0:
                vis_indices = [i % len(self.keyframes) for i in vis_indices]
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
            elif num_samples > 0:
                vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
        elif num_samples > 0:
            vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
            self.keyframes = [self.keyframes[idx] for idx in vis_indices]

        self._print_pipeline_range_debug()

    @staticmethod
    def _debug_print_allowed():
        """Return True only for rank0 + dataloader worker0.

        In DDP with multiple dataloader workers, every worker owns an independent
        Dataset instance.  Without this gate, a 50-call debug interval prints one
        line per worker/rank, which looks like log spam.
        """
        try:
            rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
        except Exception:
            rank = 0
        if rank != 0:
            return False
        try:
            worker = get_worker_info()
        except Exception:
            worker = None
        if worker is not None and int(worker.id) != 0:
            return False
        return True

    def _print_pipeline_range_debug(self):
        if not self.coord_debug_enabled or not self._debug_print_allowed():
            return
        items = []
        for i, transform in enumerate(self.pipeline):
            attrs = []
            for attr in ['point_cloud_range', 'pc_range', 'visibility_pc_range', 'occ_size', 'use_ego']:
                if hasattr(transform, attr):
                    try:
                        attrs.append(f'{attr}={getattr(transform, attr)}')
                    except Exception:
                        attrs.append(f'{attr}=<unprintable>')
            if attrs:
                items.append(f'{i}:{transform.__class__.__name__}(' + ', '.join(attrs) + ')')
        if items:
            print(
                '[NuScenesOcc3DDataset CONFIG DEBUG] '
                f'style={self.info_style}; samples={len(self.keyframes)}; '
                f'respect_valid_flag={self.respect_valid_flag}; '
                f'numeric_label_format={self.numeric_label_format}; '
                + ' | '.join(items),
                flush=True,
            )

    def __len__(self):
        return len(self.keyframes)

    @staticmethod
    def _normalize_class_name(name):
        return str(name).strip().lower().replace(' ', '_').replace('-', '_')

    def _label_from_name(self, name):
        key = self._normalize_class_name(name)
        return self.name_to_label.get(key, -1)

    def _numeric_labels_to_daocc(self, labels):
        """Convert numeric labels to DAOcc 10-class order when possible.

        Most DAOcc/MMDet3D pkls also contain gt_names; when names are present
        they are preferred and this function is not used.  This fallback keeps
        compatibility with three common numeric conventions:
          * daocc/default.yaml: car=0, truck=1, ..., traffic_cone=9;
          * GSF/metric Occ3D-16: barrier=0, bicycle=1, ..., truck=9;
          * official Occ3D-17(non-empty): others=0, barrier=1, ..., truck=10.
        """
        arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            return arr
        fmt = self.numeric_label_format
        out = np.full_like(arr, -1)
        if fmt == 'daocc' or (fmt == 'auto' and int(np.nanmax(arr)) <= 9):
            # Ambiguous 0..9 case: keep DAOcc because YOLO/BoxHyp and the pkl
            # detection head use this order.  If your pkl only has Occ3D-16
            # numeric labels and no gt_names, set numeric_label_format='gsf_occ16'.
            return arr
        if fmt in {'gsf_occ16', 'occ16'}:
            mapper = self.gsf_occ_to_dao
        elif fmt in {'official_occ3d', 'occ3d17', 'occ17'}:
            mapper = self.official_occ3d_to_dao
        elif fmt == 'auto':
            # Values above 10 usually mean official Occ3D non-object classes are
            # present; object boxes should still map through official indices.
            mapper = self.official_occ3d_to_dao if int(np.nanmax(arr)) >= 10 else self.gsf_occ_to_dao
        else:
            mapper = {}
        for src, dst in mapper.items():
            out[arr == int(src)] = int(dst)
        return out

    @staticmethod
    def _xyz_stats(value):
        if value is None:
            return 'none'
        arr = np.asarray(value)
        if arr.size == 0:
            return f'shape={arr.shape}, empty'
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        xyz = arr.reshape(-1, arr.shape[-1])[:, :3]
        return (
            f'shape={arr.shape}, '
            f'x=[{float(np.nanmin(xyz[:,0])):.3f},{float(np.nanmax(xyz[:,0])):.3f}], '
            f'y=[{float(np.nanmin(xyz[:,1])):.3f},{float(np.nanmax(xyz[:,1])):.3f}], '
            f'z=[{float(np.nanmin(xyz[:,2])):.3f},{float(np.nanmax(xyz[:,2])):.3f}]'
        )

    @staticmethod
    def _hist_str(value, max_items=12):
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return '{}'
        uniq, cnt = np.unique(arr, return_counts=True)
        pairs = list(zip(uniq.tolist(), cnt.tolist()))[:max_items]
        suffix = '...' if len(uniq) > max_items else ''
        return '{' + ', '.join([f'{int(k)}:{int(v)}' for k, v in pairs]) + suffix + '}'

    def _box_summary(self, boxes, labels=None, names=None):
        boxes_np = np.asarray(boxes) if boxes is not None else np.zeros((0, 7), dtype=np.float32)
        if boxes_np.size == 0:
            return 'boxes=empty'
        boxes_np = boxes_np.reshape(-1, boxes_np.shape[-1])
        n = boxes_np.shape[0]
        centers = boxes_np[:, :3]
        dims = boxes_np[:, 3:6] if boxes_np.shape[1] >= 6 else np.zeros((n, 3), dtype=np.float32)
        # Print both center-origin and bottom-origin interpretation so z_origin
        # mistakes can be spotted immediately in logs.
        z_center = centers[:, 2]
        h = dims[:, 2]
        bottom_if_center = z_center - 0.5 * h
        top_if_center = z_center + 0.5 * h
        top_if_bottom = z_center + h
        label_part = ''
        if labels is not None:
            label_part += f', labels_daocc={self._hist_str(labels)}'
            try:
                labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
                valid = (labels_np >= 0) & (labels_np < len(self.dao_to_gsf_occ_sem))
                occ_labels = np.full_like(labels_np, -1)
                occ_labels[valid] = self.dao_to_gsf_occ_sem[labels_np[valid]]
                label_part += f', labels_occ3d_sem={self._hist_str(occ_labels)}'
            except Exception:
                pass
        if names is not None:
            names_arr = np.asarray(names).reshape(-1)
            label_part += f', names_preview={names_arr[:8].tolist()}'
        return (
            f'boxes_shape={boxes_np.shape}, center_z=[{float(np.nanmin(z_center)):.3f},{float(np.nanmax(z_center)):.3f}], '
            f'bottom_if_center=[{float(np.nanmin(bottom_if_center)):.3f},{float(np.nanmax(bottom_if_center)):.3f}], '
            f'top_if_center=[{float(np.nanmin(top_if_center)):.3f},{float(np.nanmax(top_if_center)):.3f}], '
            f'top_if_bottom=[{float(np.nanmin(top_if_bottom)):.3f},{float(np.nanmax(top_if_bottom)):.3f}]'
            f'{label_part}'
        )

    def _maybe_print_coord_debug(self, input_dict, index=None, key=None):
        if not self.coord_debug_enabled or self._coord_debug_print_count >= self.coord_debug_max_print:
            return
        if not self._debug_print_allowed():
            return
        if self._coord_debug_call_count % self.coord_debug_interval != 0:
            return
        pts = input_dict.get('points', None)
        vpts = input_dict.get('visibility_points', None)
        occ_xyz = input_dict.get('occ_xyz', None)
        boxes = input_dict.get('gt_bboxes_3d', None)
        labels = input_dict.get('gt_labels_3d', None)
        names = input_dict.get('gt_names', None)
        occ_label = input_dict.get('occ_label', None)
        msg = [
            f"[NuScenesOcc3DDataset COORD DEBUG] call={self._coord_debug_call_count}; idx={index}; key={key}; ",
            f"sample={input_dict.get('sample_idx', None)}; scene={input_dict.get('scene_name', input_dict.get('scene_token', None))}; ",
            f"respect_valid_flag={self.respect_valid_flag}; ann={bool(input_dict.get('gt_bboxes_3d_has_ann', False))}; ",
            f"points={self._xyz_stats(pts)}; visibility_points={self._xyz_stats(vpts)}; occ_xyz={self._xyz_stats(occ_xyz)}; ",
            self._box_summary(boxes, labels=labels, names=names),
        ]
        if self.coord_debug_show_label_hist and occ_label is not None:
            msg.append(f"occ_label_hist={self._hist_str(occ_label)}; ")
        if self.coord_debug_show_matrices and 'ego2lidar' in input_dict:
            e2l = np.asarray(input_dict['ego2lidar'])
            msg.append(f"ego2lidar_t={e2l[:3, 3].round(4).tolist()}; ")
        print(' '.join(msg), flush=True)
        self._coord_debug_print_count += 1

    @staticmethod
    def _is_numeric_array_like(value):
        try:
            arr = np.asarray(value)
        except Exception:
            return False
        return np.issubdtype(arr.dtype, np.number)

    @staticmethod
    def _value_summary(value):
        if isinstance(value, np.ndarray):
            flat = value.reshape(-1)
            preview = flat[: min(5, flat.size)].tolist()
            return f"ndarray(shape={value.shape}, dtype={value.dtype}, preview={preview})"
        if isinstance(value, (list, tuple)):
            item_types = [type(x).__name__ for x in value[: min(3, len(value))]]
            return f"{type(value).__name__}(len={len(value)}, item_types={item_types})"
        return f"{type(value).__name__}({value})"

    def _maybe_print_raw_gt_debug(self, info, ann_out, index=None, key=None, reason=None, source=None, boxes=None, labels=None, names=None):
        if not self.gt_raw_debug_enabled or self._gt_raw_debug_print_count >= self.gt_raw_debug_max_print:
            return
        if not self._debug_print_allowed():
            return
        if index is not None and int(index) % self.gt_raw_debug_interval != 0:
            return

        has_ann = bool(ann_out.get('gt_bboxes_3d_has_ann', False))
        gt_count = int(np.asarray(ann_out.get('gt_bboxes_3d', np.zeros((0, 7)))).shape[0])
        problem = (not has_ann) or (gt_count == 0)
        if self.gt_raw_debug_only_when_problem and not problem:
            return

        top_keys = list(info.keys()) if isinstance(info, dict) else []
        gt_like_keys = [k for k in top_keys if ('gt' in k.lower() or 'ann' in k.lower() or 'label' in k.lower() or 'name' in k.lower() or 'valid' in k.lower())]
        msg = [
            f"[DATASET RAW GT DEBUG] idx={index}; key={key}; sample={self._get_sample_token(info)}; style={self.info_style};",
            f"source={source}; reason={reason}; has_ann={has_ann}; gt_count={gt_count}; respect_valid_flag={self.respect_valid_flag};",
        ]
        if self.gt_raw_debug_show_info_keys:
            msg.append(f"top_keys={top_keys}; gt_like_keys={gt_like_keys};")
        if self.gt_raw_debug_show_value_summaries:
            values = {}
            for k in gt_like_keys[:12]:
                try:
                    values[k] = self._value_summary(info[k])
                except Exception as exc:
                    values[k] = f"summary_failed({exc})"
            if boxes is not None:
                values['parsed_boxes'] = self._value_summary(np.asarray(boxes))
            if labels is not None:
                values['parsed_labels'] = self._value_summary(np.asarray(labels))
            if names is not None:
                values['parsed_names'] = self._value_summary(np.asarray(names))
            msg.append("values={" + '; '.join([f"{k}={v}" for k, v in values.items()]) + "}")
        print(' '.join(msg), flush=True)
        self._gt_raw_debug_print_count += 1

    def _join_data_root(self, path):
        if path is None:
            return None
        path = str(path)
        if os.path.isabs(path):
            return path
        # DAOcc often stores paths like ./data/nuscenes/xxx or data/nuscenes/xxx.
        norm = path[2:] if path.startswith('./') else path
        data_root_norm = self.data_path[2:] if self.data_path.startswith('./') else self.data_path
        data_root_norm = data_root_norm.rstrip('/')
        if norm.startswith(data_root_norm + '/') or norm == data_root_norm:
            return norm
        return os.path.join(self.data_path, path)

    def _sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = bool(self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]))
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def _empty_yolo2d_cache_entry(self):
        max_det = max(int(self.yolo2d_cache_max_det), 1)
        cam = len(self.sensor_types)
        return {
            'yolo_boxes_2d': np.zeros((cam, max_det, 4), dtype=np.float32),
            'yolo_scores_2d': np.zeros((cam, max_det), dtype=np.float32),
            'yolo_labels_2d': np.full((cam, max_det), -1, dtype=np.int64),
            'yolo_valid_2d': np.zeros((cam, max_det), dtype=np.bool_),
            'yolo_source_is_offline_cache': np.asarray(True, dtype=np.bool_),
        }

    def _get_yolo2d_cache_entry(self, sample_token):
        """Return fixed-shape YOLO2D proposals for one sample.

        Cache format produced by tools/generate_yolo2d_cache.py:
            samples[token]['boxes']  [6,M,4] xyxy in resized/cropped 704x256 coords
            samples[token]['scores'] [6,M]
            samples[token]['labels'] [6,M] nuScenes-10 detection labels
            samples[token]['valid']  [6,M]
        """
        out = self._empty_yolo2d_cache_entry()
        if self.yolo2d_cache is None:
            out['yolo_source_is_offline_cache'] = np.asarray(False, dtype=np.bool_)
            return out
        key = str(sample_token)
        entry = self.yolo2d_cache.get(key, None)
        if entry is None:
            # Some info files use sample_idx/token inconsistently.  Keep a
            # clean empty tensor rather than crashing mid-training.
            if self.yolo2d_cache_debug and self._yolo2d_cache_debug_print_count < self.yolo2d_cache_debug_max_print:
                print(f'[YOLO2DCache][miss] sample_token={key}', flush=True)
                self._yolo2d_cache_debug_print_count += 1
            return out

        boxes = np.asarray(entry.get('boxes', entry.get('boxes_2d', [])), dtype=np.float32)
        scores = np.asarray(entry.get('scores', entry.get('scores_2d', [])), dtype=np.float32)
        labels = np.asarray(entry.get('labels', entry.get('labels_2d', [])), dtype=np.int64)
        valid = np.asarray(entry.get('valid', entry.get('valid_2d', [])), dtype=np.bool_)
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            return out
        cam = min(boxes.shape[0], len(self.sensor_types))
        src_m = boxes.shape[1]
        max_det = out['yolo_boxes_2d'].shape[1]
        m = min(src_m, max_det)
        if m <= 0 or cam <= 0:
            return out
        boxes = boxes[:cam, :m]
        scores = scores[:cam, :m] if scores.ndim == 2 else np.zeros((cam, m), dtype=np.float32)
        labels = labels[:cam, :m] if labels.ndim == 2 else np.full((cam, m), -1, dtype=np.int64)
        valid = valid[:cam, :m] if valid.ndim == 2 else np.ones((cam, m), dtype=np.bool_)

        finite = np.isfinite(boxes).all(axis=-1) & np.isfinite(scores)
        size_ok = (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])
        label_ok = (labels >= 0) & (labels < 10)
        score_ok = scores >= self.yolo2d_cache_score_thr
        keep = valid & finite & size_ok & label_ok & score_ok

        out['yolo_boxes_2d'][:cam, :m] = boxes
        out['yolo_scores_2d'][:cam, :m] = scores
        out['yolo_labels_2d'][:cam, :m] = labels
        out['yolo_valid_2d'][:cam, :m] = keep
        if self.yolo2d_cache_debug and self._yolo2d_cache_debug_print_count < self.yolo2d_cache_debug_max_print:
            hist = {}
            if keep.any():
                labs, cnt = np.unique(labels[keep], return_counts=True)
                hist = {int(k): int(v) for k, v in zip(labs.tolist(), cnt.tolist())}
            print(f'[YOLO2DCache] sample={key}, valid={int(keep.sum())}, hist={hist}', flush=True)
            self._yolo2d_cache_debug_print_count += 1
        return out

    def __getitem__(self, index):
        key = self.keyframes[index]
        if self.info_style == 'gf3d_nested':
            scene_token, frame_index = key
            info = deepcopy(self.scene_infos[scene_token][frame_index])
            input_dict = self.get_data_info(info, scene_token=scene_token, index=index, key=key)
        else:
            info = deepcopy(self.scene_infos[key])
            input_dict = self.get_data_info(info, scene_token=info.get('scene_name', info.get('scene_token', None)), index=index, key=key)

        # Guarantee fixed GT keys before and after pipeline, so batch collation
        # stays aligned even when one sample has no valid detection GT.
        for k, v in self._empty_gt_ann(has_ann=False, reason='missing_before_pipeline').items():
            input_dict.setdefault(k, v)

        if self.data_aug_conf is not None:
            input_dict['aug_configs'] = self._sample_augmentation()
        for transform in self.pipeline:
            input_dict = transform(input_dict)

        self._coord_debug_call_count += 1
        self._maybe_print_coord_debug(input_dict, index=index, key=key)

        return_dict = {
            'img': input_dict['img'],
            'projection_mat': input_dict['projection_mat'],
            'image_wh': input_dict['image_wh'],
            'occ_label': input_dict['occ_label'],
            'occ_xyz': input_dict['occ_xyz'],
            'occ_cam_mask': input_dict['occ_cam_mask'],
        }
        # Force GT keys to exist in every returned sample.  This prevents
        # custom_collate from producing list(len<batch_size) for variable GT.
        for k, v in self._empty_gt_ann(has_ann=False, reason='missing_after_pipeline').items():
            input_dict.setdefault(k, v)

        for k in [
            'lidar_feature_maps', 'points', 'visibility_points', 'ego2lidar', 'img_filename', 'dpt', 'anchor_points',
            'occ3d_mask_camera', 'mask_camera', 'mask_lidar', 'pts_filename', 'occ_gt_path',
            'sample_idx', 'scene_name', 'scene_token', 'lidar_path',
            'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes_3d_has_ann', 'gt_names', 'gt_ann_source', 'gt_ann_reason',
        ]:
            if k in input_dict:
                return_dict[k] = input_dict[k]

        if self.yolo2d_cache is not None:
            yolo_cache_out = self._get_yolo2d_cache_entry(input_dict.get('sample_idx', self._get_sample_token(info)))
            return_dict.update(yolo_cache_out)
        return return_dict

    @staticmethod
    def _quat_to_mat(q):
        return Quaternion(q).rotation_matrix

    @staticmethod
    def _make_rt(translation, rotation):
        rt = np.eye(4, dtype=np.float64)
        rt[:3, :3] = Quaternion(rotation).rotation_matrix
        rt[:3, 3] = np.asarray(translation, dtype=np.float64)
        return rt

    def _get_occ_path(self, info):
        if 'occ3d' in info and isinstance(info['occ3d'], dict) and 'occ_path' in info['occ3d']:
            return info['occ3d']['occ_path']
        for key in ['occ_path', 'gt_path']:
            if key in info:
                return info[key]
        return None

    def _get_sample_token(self, info):
        return info.get('token', info.get('sample_token', info.get('sample_idx', None)))

    def _get_lidar_info_gf3d(self, info):
        return info['data']['LIDAR_TOP']

    @staticmethod
    def _empty_gt_ann(has_ann=False, reason='empty', source='none'):
        return {
            'gt_bboxes_3d': np.zeros((0, 7), dtype=np.float32),
            'gt_labels_3d': np.zeros((0,), dtype=np.int64),
            'gt_names': np.asarray([], dtype=str),
            'gt_bboxes_3d_has_ann': np.asarray(bool(has_ann), dtype=np.bool_),
            'gt_ann_reason': str(reason),
            'gt_ann_source': str(source),
        }

    def _extract_gt_ann(self, info, index=None, key=None):
        """Extract DAOcc/MMDet3D-style 3D detection GT from the pkl if present.

        Boxes are expected in LiDAR/ego frame as [x, y, z, l, w, h, yaw].
        This parser deliberately prefers gt_names when present because some
        DAOcc ann_infos tuples contain non-class bookkeeping in the second item.
        The previous parser let ann_infos[1] override gt_names, causing labels
        to become all -1 even when gt_names=['car', ...] was valid.
        """
        boxes = None
        labels = None
        names = None
        source = 'none'
        has_annotation_field = False

        # Common DAOcc/MMDet3D info format.
        for box_key in ['gt_boxes', 'gt_bboxes_3d', 'gt_boxes_3d', 'gt_bboxes']:
            if box_key in info:
                boxes = info[box_key]
                source = f'top.{box_key}'
                has_annotation_field = True
                break
        for label_key in ['gt_labels_3d', 'gt_labels', 'labels']:
            if label_key in info:
                labels = info[label_key]
                has_annotation_field = True
                break
        for name_key in ['gt_names', 'gt_name', 'names']:
            if name_key in info:
                names = info[name_key]
                has_annotation_field = True
                break

        # Some pkl files store annotations as ann_infos=(boxes, labels/names).
        ann_infos = info.get('ann_infos', info.get('ann_info', None))
        if ann_infos is not None:
            has_annotation_field = True
            if isinstance(ann_infos, (list, tuple)) and len(ann_infos) >= 2:
                if boxes is None:
                    boxes = ann_infos[0]
                    source = 'ann_infos[0]'
                # Only use ann_infos[1] if neither a top-level numeric label nor
                # gt_names is available.  In the DAOcc pkl you showed, gt_names
                # is correct and ann_infos[1] should not override it.
                if labels is None and names is None:
                    labels = ann_infos[1]
            elif isinstance(ann_infos, dict):
                if boxes is None:
                    for box_key in ['gt_boxes', 'gt_bboxes_3d', 'boxes_3d', 'bboxes_3d']:
                        if box_key in ann_infos:
                            boxes = ann_infos[box_key]
                            source = f'ann_infos.{box_key}'
                            break
                if labels is None:
                    for label_key in ['gt_labels_3d', 'gt_labels', 'labels_3d', 'labels']:
                        if label_key in ann_infos:
                            labels = ann_infos[label_key]
                            break
                if names is None:
                    for name_key in ['gt_names', 'names']:
                        if name_key in ann_infos:
                            names = ann_infos[name_key]
                            break

        if boxes is None:
            out = self._empty_gt_ann(has_ann=has_annotation_field, reason='no_boxes', source=source)
            self._maybe_print_raw_gt_debug(info, out, index=index, key=key, reason='no_boxes', source=source)
            return out

        try:
            boxes = np.asarray(boxes, dtype=np.float32)
        except Exception:
            out = self._empty_gt_ann(has_ann=has_annotation_field, reason='boxes_parse_failed', source=source)
            self._maybe_print_raw_gt_debug(info, out, index=index, key=key, reason='boxes_parse_failed', source=source)
            return out
        if boxes.ndim != 2 or boxes.shape[0] == 0 or boxes.shape[1] < 7:
            out = self._empty_gt_ann(has_ann=has_annotation_field, reason='empty_or_bad_boxes', source=source)
            self._maybe_print_raw_gt_debug(info, out, index=index, key=key, reason='empty_or_bad_boxes', source=source, boxes=boxes)
            return out
        boxes = boxes[:, :7].astype(np.float32, copy=False)

        names_arr = None
        if names is not None:
            try:
                names_arr = np.asarray(names).reshape(-1)
            except Exception:
                names_arr = None

        # Prefer names if they align with boxes.  This is the robust path for
        # your DAOcc pkl where gt_names contains simple class names.
        if names_arr is not None and names_arr.shape[0] == boxes.shape[0]:
            labels = np.asarray([self._label_from_name(n) for n in names_arr], dtype=np.int64)
        elif labels is not None:
            # Numeric labels are accepted directly.  String labels are mapped.
            if self._is_numeric_array_like(labels):
                try:
                    labels = self._numeric_labels_to_daocc(labels)
                except Exception:
                    labels = np.full((boxes.shape[0],), -1, dtype=np.int64)
            else:
                try:
                    label_names = np.asarray(labels).reshape(-1)
                    labels = np.asarray([self._label_from_name(n) for n in label_names], dtype=np.int64)
                    if names_arr is None:
                        names_arr = label_names
                except Exception:
                    labels = np.full((boxes.shape[0],), -1, dtype=np.int64)
        else:
            labels = np.full((boxes.shape[0],), -1, dtype=np.int64)

        if labels.shape[0] != boxes.shape[0]:
            labels = np.full((boxes.shape[0],), -1, dtype=np.int64)

        valid = labels >= 0
        # DAOcc/MMDet3D valid_flag usually means "has enough LiDAR points".
        # For oracle occupancy completion we keep boxes even when valid_flag is
        # false, because missing/weak-LiDAR small objects are exactly where box
        # completion should help.  Set respect_valid_flag=True to restore the
        # detection-style filter.
        if self.respect_valid_flag and 'valid_flag' in info:
            vf = np.asarray(info['valid_flag']).astype(bool).reshape(-1)
            if vf.shape[0] == boxes.shape[0]:
                valid &= vf

        if not bool(valid.any()):
            out = self._empty_gt_ann(has_ann=has_annotation_field, reason='no_valid_label_or_valid_flag', source=source)
            self._maybe_print_raw_gt_debug(
                info, out, index=index, key=key, reason='no_valid_label_or_valid_flag', source=source,
                boxes=boxes, labels=labels, names=names_arr,
            )
            return out

        if names_arr is None or names_arr.shape[0] != boxes.shape[0]:
            names_arr = np.asarray([self.label_to_name.get(int(x), 'unknown') for x in labels], dtype=str)

        out = {
            'gt_bboxes_3d': boxes[valid].astype(np.float32, copy=False),
            'gt_labels_3d': labels[valid].astype(np.int64, copy=False),
            'gt_names': names_arr[valid].astype(str),
            'gt_bboxes_3d_has_ann': np.asarray(True, dtype=np.bool_),
            'gt_ann_reason': 'ok',
            'gt_ann_source': source,
        }
        self._maybe_print_raw_gt_debug(info, out, index=index, key=key, reason='ok', source=source, boxes=boxes, labels=labels, names=names_arr)
        return out

    def _get_data_info_gf3d(self, info, scene_token=None, index=None, key=None):
        image_paths = []
        lidar2img_rts = []
        img2lidar_rts = []
        cam_intrinsics = []
        cam2ego_rts = []
        ego2image_rts = []
        lidar2cam_rts = []

        lidar_info = info['data']['LIDAR_TOP']
        lidar2ego = self._make_rt(lidar_info['calib']['translation'], lidar_info['calib']['rotation'])
        ego2lidar = np.linalg.inv(lidar2ego)
        lidar2global = get_lidar2global(lidar_info['calib'], lidar_info['pose'])
        ego2global = self._make_rt(lidar_info['pose']['translation'], lidar_info['pose']['rotation'])

        for cam_type in self.sensor_types:
            cam_info = info['data'][cam_type]
            image_paths.append(self._join_data_root(cam_info['filename']))
            img2global = get_img2global(cam_info['calib'], cam_info['pose'])
            lidar2img = np.linalg.inv(img2global) @ lidar2global
            img2lidar = np.linalg.inv(lidar2global) @ img2global
            cam2ego = self._make_rt(cam_info['calib']['translation'], cam_info['calib']['rotation'])
            lidar2cam = np.linalg.inv(cam2ego) @ lidar2ego
            intrinsic = np.asarray(cam_info['calib']['camera_intrinsic'], dtype=np.float64)
            viewpad = np.eye(4, dtype=np.float64)
            viewpad[:3, :3] = intrinsic
            lidar2img_rts.append(lidar2img)
            img2lidar_rts.append(img2lidar)
            cam_intrinsics.append(viewpad)
            cam2ego_rts.append(cam2ego)
            ego2image_rts.append(np.linalg.inv(img2global) @ ego2global)
            lidar2cam_rts.append(lidar2cam)

        pts_filename = self._join_data_root(lidar_info['filename'])
        lidar_path = self._join_data_root(info.get('lidar_path', lidar_info.get('filename')))
        ann_info = self._extract_gt_ann(info, index=index, key=key)
        return dict(
            sample_idx=self._get_sample_token(info),
            scene_token=scene_token,
            scene_name=info.get('scene_name', scene_token),
            pts_filename=pts_filename,
            lidar_path=lidar_path,
            occ_path=self._get_occ_path(info),
            timestamp=info['timestamp'] / 1e6,
            ego2global=ego2global,
            lidar2global=lidar2global,
            img_filename=image_paths,
            lidar2img=np.asarray(lidar2img_rts),
            img2lidar=np.asarray(img2lidar_rts),
            cam_intrinsic=np.asarray(cam_intrinsics),
            ori_intrinsic=np.asarray(cam_intrinsics).copy(),
            ego2lidar=ego2lidar,
            cam2ego=np.asarray(cam2ego_rts),
            ego2img=np.asarray(ego2image_rts),
            lidar2cam=np.asarray(lidar2cam_rts),
            sweeps=info.get('sweeps', []),
            **ann_info,
        )

    def _get_data_info_mmdet3d(self, info, scene_token=None, index=None, key=None):
        image_paths = []
        lidar2img_rts = []
        img2lidar_rts = []
        cam_intrinsics = []
        cam2ego_rts = []
        ego2image_rts = []
        lidar2cam_rts = []

        lidar2ego = self._make_rt(info['lidar2ego_translation'], info['lidar2ego_rotation'])
        ego2lidar = np.linalg.inv(lidar2ego)
        ego2global = self._make_rt(info['ego2global_translation'], info['ego2global_rotation'])
        lidar2global = ego2global @ lidar2ego

        cams = info['cams']
        for cam_type in self.sensor_types:
            cam_info = cams[cam_type]
            image_paths.append(self._join_data_root(cam_info.get('data_path', cam_info.get('filename'))))

            cam2ego = self._make_rt(cam_info['sensor2ego_translation'], cam_info['sensor2ego_rotation'])
            ego_cam2global = self._make_rt(cam_info['ego2global_translation'], cam_info['ego2global_rotation'])
            img2global = ego_cam2global @ cam2ego
            lidar2cam = np.linalg.inv(img2global) @ lidar2global

            intrinsic = None
            for intrinsic_key in ['cam_intrinsic', 'camera_intrinsic', 'camera_intrinsics', 'intrinsic', 'cam2img']:
                if intrinsic_key in cam_info:
                    intrinsic = cam_info[intrinsic_key]
                    break
            if intrinsic is None:
                raise KeyError(
                    'Cannot find camera intrinsic in cam_info. '
                    f'cam_type={cam_type}, available keys={list(cam_info.keys())}'
                )
            intrinsic = np.asarray(intrinsic, dtype=np.float64)
            if intrinsic.shape == (4, 4):
                intrinsic = intrinsic[:3, :3]
            elif intrinsic.shape == (3, 4):
                intrinsic = intrinsic[:3, :3]
            if intrinsic.shape != (3, 3):
                raise ValueError(
                    f'Camera intrinsic should be 3x3/3x4/4x4, got {intrinsic.shape} for {cam_type}'
                )
            viewpad = np.eye(4, dtype=np.float64)
            viewpad[:3, :3] = intrinsic
            lidar2img = viewpad @ lidar2cam
            img2lidar = np.linalg.inv(lidar2global) @ img2global

            lidar2img_rts.append(lidar2img)
            img2lidar_rts.append(img2lidar)
            cam_intrinsics.append(viewpad)
            cam2ego_rts.append(cam2ego)
            # MMDet3D/DAOcc list-style infos build img2global as cam2global
            # without the camera intrinsic.  NuScenesAdaptor(use_ego=True)
            # later exposes ego2img as projection_mat, so this must be the
            # full pixel projection K @ ego2cam, not only ego2cam.  Otherwise
            # projected boxes collapse near normalized camera coordinates and
            # GT-projected-2D / YOLO2DTo3D diagnostics produce boxes2d=0.
            ego2cam = np.linalg.inv(img2global) @ ego2global
            ego2image_rts.append(viewpad @ ego2cam)
            lidar2cam_rts.append(lidar2cam)

        lidar_path = self._join_data_root(info['lidar_path'])
        ann_info = self._extract_gt_ann(info, index=index, key=key)
        return dict(
            sample_idx=self._get_sample_token(info),
            scene_token=info.get('scene_token', scene_token),
            scene_name=info.get('scene_name', scene_token),
            pts_filename=lidar_path,
            lidar_path=lidar_path,
            occ_path=self._get_occ_path(info),
            timestamp=info['timestamp'] / 1e6 if info.get('timestamp', 0) > 1e10 else info.get('timestamp', 0),
            ego2global=ego2global,
            lidar2global=lidar2global,
            img_filename=image_paths,
            lidar2img=np.asarray(lidar2img_rts),
            img2lidar=np.asarray(img2lidar_rts),
            cam_intrinsic=np.asarray(cam_intrinsics),
            ori_intrinsic=np.asarray(cam_intrinsics).copy(),
            ego2lidar=ego2lidar,
            cam2ego=np.asarray(cam2ego_rts),
            ego2img=np.asarray(ego2image_rts),
            lidar2cam=np.asarray(lidar2cam_rts),
            sweeps=info.get('sweeps', []),
            **ann_info,
        )

    def get_data_info(self, info, scene_token=None, index=None, key=None):
        if 'data' in info:
            return self._get_data_info_gf3d(info, scene_token=scene_token, index=index, key=key)
        if 'cams' in info:
            return self._get_data_info_mmdet3d(info, scene_token=scene_token, index=index, key=key)
        raise KeyError('Unsupported info format: expected key "data" or "cams".')
