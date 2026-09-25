import torch, torch.nn as nn
from torch.nn import functional as F
from mmseg.registry import MODELS
from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid, point_cloud_map
try:
    from .visibility_grid import UNKNOWN, WEAK_FREE, STRONG_FREE, SURFACE_HIT, MIXED
except Exception:
    UNKNOWN, WEAK_FREE, STRONG_FREE, SURFACE_HIT, MIXED = 0, 1, 2, 3, 4


@MODELS.register_module()
class GaussianLifter(BaseLifter):
    def __init__(
        self,
        num_anchor, # number of gaussians
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        phi_activation='sigmoid',
        semantics=False,
        semantic_dim=None,
        include_opa=True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        
        xyz = torch.rand(num_anchor, 3, dtype=torch.float) # randomly set x y z in range [0, 1]
        if phi_activation == 'sigmoid':
            xyz = safe_inverse_sigmoid(xyz) # map x y z from [0, 1] to [-inf, inf] and safely apply inverse sigmoid
        elif phi_activation == 'loop':
            xyz[:, :2] = safe_inverse_sigmoid(xyz[:, :2])
        else:
            raise NotImplementedError
            
        scale = torch.rand_like(xyz) # ranging between [0, 1)
        scale = safe_inverse_sigmoid(scale) # ranging between (-9.21, 9.21)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if include_opa:
            opacity = safe_inverse_sigmoid(0.1 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float) # (N, 17)

        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1) # gaussians initialization, (N, 28)

        self.num_anchor = num_anchor
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]), # (N, 128)
            requires_grad=feat_grad,
        )

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def forward(self, ms_img_feats, **kwargs):
        # ms_img_feats: list of multi-scale image features, each element is a tensor of shape (B, BN/B, C, H, W)
        batch_size = ms_img_feats[0].shape[0]
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1) # (B, N, 128)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1)) # (B, N, 28)

        return {
            'rep_features': instance_feature,
            'representation': anchor,
        }


@MODELS.register_module()
class GaussianLifterLiDARPoint(BaseLifter):
    def __init__(
        self,
        num_anchor, # number of gaussians
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        phi_activation='sigmoid',
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        use_intensity=False,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        
        xyz = torch.rand(num_anchor, 3, dtype=torch.float) # randomly set x y z in range [0, 1]
        if phi_activation == 'sigmoid':
            xyz = safe_inverse_sigmoid(xyz) # map x y z from [0, 1] to [-inf, inf] and safely apply inverse sigmoid
        elif phi_activation == 'loop':
            xyz[:, :2] = safe_inverse_sigmoid(xyz[:, :2])
        else:
            raise NotImplementedError
            
        scale = torch.rand_like(xyz) # ranging between [0, 1)
        scale = safe_inverse_sigmoid(scale) # ranging between (-9.21, 9.21)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if include_opa:
            opacity = safe_inverse_sigmoid(0.1 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float) # (N, 17)

        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1) # gaussians initialization, (N, 28)

        self.num_anchor = num_anchor
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]), # (N, 128)
            requires_grad=feat_grad,
        )
        self.use_intensity = use_intensity

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def forward(self, ms_img_feats, anchor_points, **kwargs):
        # ms_img_feats: list of multi-scale image features, each element is a tensor of shape (B, BN/B, C, H, W)
        batch_size = ms_img_feats[0].shape[0]
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1) # (B, N, 128)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1)) # (B, N, 28)

        for batch_idx in range(batch_size):
            anchor_points_single = anchor_points[batch_idx] # (N, 4) or (N, 3)
            assert anchor_points_single.shape[0] == self.num_anchor
            anchor_points_coords = anchor_points_single[:, :3] # (N, 3)
            anchor_points_coords_logits = safe_inverse_sigmoid(anchor_points_coords) # (N, 3) and each element is in range [-9.21024, 9.21024]
            anchor[batch_idx][:, :3] = anchor_points_coords_logits
            if self.use_intensity:
                anchor_points_intensity = anchor_points_single[:, 3]
                anchor_points_intensity_logits = safe_inverse_sigmoid(anchor_points_intensity)
                anchor[batch_idx][:, 10] = anchor_points_intensity_logits

        return {
            'rep_features': instance_feature,
            'representation': anchor,
        }


@MODELS.register_module()
class GaussianLifterLiDAR(BaseLifter):
    def __init__(
        self,
        num_anchor, # number of gaussians
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        phi_activation='sigmoid',
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        use_intensity=True,
        occ_annotation="surroundocc",
    ):
        super().__init__()
        self.embed_dims = embed_dims
        
        xyz = torch.rand(num_anchor, 3, dtype=torch.float) # randomly set x y z in range [0, 1]
        if phi_activation == 'sigmoid':
            xyz = safe_inverse_sigmoid(xyz) # map x y z from [0, 1] to [-9.21024, 9.21024]
        elif phi_activation == 'loop':
            xyz[:, :2] = safe_inverse_sigmoid(xyz[:, :2])
        else:
            raise NotImplementedError
            
        scale = torch.rand_like(xyz)
        scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if include_opa:
            opacity = safe_inverse_sigmoid(0.1 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float) # (N, 17)

        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1) # gaussians initialization, (N, 28)

        self.num_anchor = num_anchor
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]), # (N, 128)
            requires_grad=feat_grad,
        )
        self.use_intensity = use_intensity
        self.occ_annotation = occ_annotation

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def forward(self, ms_img_feats, voxel_lidar_feats, coors_batch, **kwargs):
        batch_size = ms_img_feats[0].shape[0]
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1) # (B, N, 128)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1)) # (B, N, 28)

        for batch_idx in range(batch_size):
            batch_mask = coors_batch[:, 0] == batch_idx
            voxel_lidar_feats_single = voxel_lidar_feats[batch_mask] # (Ni, 5)
            voxel_map_coords = point_cloud_map(voxel_lidar_feats_single, self.occ_annotation)
            voxel_map_coords_logits = safe_inverse_sigmoid(voxel_map_coords) # (Ni, 3)
            if self.use_intensity:
                voxel_map_intensity = voxel_lidar_feats_single[:, 3] / 255.0 # [0, 1]
                voxel_map_intensity_logits = safe_inverse_sigmoid(voxel_map_intensity) # (Ni, )

            N = anchor.shape[1]
            Ni = voxel_map_coords_logits.shape[0]
            if N > Ni:
                # randomly sample Ni points from N points
                idx = torch.randperm(N)[:Ni]
                anchor[batch_idx][idx][:, :3] = voxel_map_coords_logits
                if self.use_intensity:
                    anchor[batch_idx][idx][:, 10] = voxel_map_intensity_logits
            else:
                # randomly sample N points from Ni points
                idx = torch.randperm(Ni)[:N]
                anchor[batch_idx][:, :3] = voxel_map_coords_logits[idx]
                if self.use_intensity:
                    anchor[batch_idx][:, 10] = voxel_map_intensity_logits[idx]

        return {
            'rep_features': instance_feature,
            'representation': anchor,
        }





# ===================== Octree / visibility measured Gaussian lifter =====================

def _rotation_matrix_to_quaternion_wxyz(rot: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices [N,3,3] to normalized quaternions [N,4] in wxyz order.

    The measured Gaussian probe is no-grad, but the converted quaternion is used
    by the normal GSF3D encoder/refinement path afterwards.
    """
    if rot.numel() == 0:
        return rot.new_empty((0, 4))
    r = rot.float()
    m00, m01, m02 = r[:, 0, 0], r[:, 0, 1], r[:, 0, 2]
    m10, m11, m12 = r[:, 1, 0], r[:, 1, 1], r[:, 1, 2]
    m20, m21, m22 = r[:, 2, 0], r[:, 2, 1], r[:, 2, 2]
    qw = torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=1e-8)) * 0.5
    qx = torch.sign(m21 - m12) * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-8)) * 0.5
    qy = torch.sign(m02 - m20) * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-8)) * 0.5
    qz = torch.sign(m10 - m01) * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-8)) * 0.5
    q = torch.stack([qw, qx, qy, qz], dim=-1)
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    q[:, 0] = torch.where(q.norm(dim=-1) < 1e-6, torch.ones_like(q[:, 0]), q[:, 0])
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return q.to(dtype=rot.dtype)


@MODELS.register_module()
class OctreeGaussianLifter(BaseLifter):
    """Initialize GSF3D Gaussians with the DAOcc visibility + octree measured path.

    This module deliberately keeps GSF3D's output contract unchanged:
        {'rep_features': [B,N,C], 'representation': [B,N,D]}

    The visibility / octree probe runs under torch.no_grad() as a physical
    initializer. Gradients still flow through the normal GSF3D encoder,
    refinement module, GaussianHead, localagg and occupancy loss after the
    initialized representation is returned.
    """

    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        phi_activation='sigmoid',
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        use_intensity=True,
        occ_annotation='occ3d',
        pc_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        scale_range=(0.01, 1.44),
        gaussian_probe=None,
        intensity_opacity_weight=0.35,
        suppress_fallback_opacity=True,
        fallback_opacity=0.03,
        detach_initialized_anchor=False,
        zero_base_semantics=True,
        background_gaussian=None,
        build_visibility_neighborhood_maps=True,
        visibility_neighborhood_kernel=3,
    ):
        super().__init__()
        self.num_anchor = int(num_anchor)
        self.embed_dims = int(embed_dims)
        self.phi_activation = phi_activation
        self.semantics = bool(semantics)
        self.semantic_dim = int(semantic_dim or 0) if semantics else 0
        self.include_opa = bool(include_opa)
        self.use_intensity = bool(use_intensity)
        self.occ_annotation = occ_annotation
        self.pc_range = tuple(float(v) for v in pc_range)
        self.scale_range = tuple(float(v) for v in scale_range)
        self.intensity_opacity_weight = float(intensity_opacity_weight)
        self.suppress_fallback_opacity = bool(suppress_fallback_opacity)
        self.fallback_opacity = float(fallback_opacity)
        self.detach_initialized_anchor = bool(detach_initialized_anchor)
        self.zero_base_semantics = bool(zero_base_semantics)
        self.build_visibility_neighborhood_maps = bool(build_visibility_neighborhood_maps)
        kernel = max(int(visibility_neighborhood_kernel), 1)
        if kernel % 2 == 0:
            kernel += 1
        self.visibility_neighborhood_kernel = kernel

        # v9.0 background completion budget.  These anchors are reserved at the
        # tail of the fixed representation and are never overwritten by object/box
        # completion.  They are initialized from the 10-sweep background point
        # support with weak visibility filtering and simple stratified sampling.
        bg_cfg = dict(background_gaussian or {})
        self.bg_enabled = bool(bg_cfg.get('enabled', False))
        self.bg_num = max(int(bg_cfg.get('num_background', 0)), 0) if self.bg_enabled else 0
        self.bg_num = min(self.bg_num, self.num_anchor)
        self.normal_num_anchor = self.num_anchor - self.bg_num
        self.bg_ground_ratio = float(bg_cfg.get('ground_ratio', 0.60))
        self.bg_vertical_ratio = float(bg_cfg.get('vertical_ratio', 0.25))
        self.bg_far_ratio = float(bg_cfg.get('far_ratio', 0.15))
        self.bg_ground_max_z = float(bg_cfg.get('ground_max_z', 0.80))
        self.bg_vertical_min_z = float(bg_cfg.get('vertical_min_z', 0.35))
        self.bg_far_range = float(bg_cfg.get('far_range', 32.0))
        self.bg_sample_oversample = max(int(bg_cfg.get('oversample_factor', 4)), 1)
        self.bg_sampling_mode = str(bg_cfg.get('sampling_mode', 'jitter')).lower()
        self.bg_frontier_steps_xy = tuple(float(v) for v in bg_cfg.get('frontier_steps_xy', (0.8, 1.2, 1.6)))
        if len(self.bg_frontier_steps_xy) <= 0:
            self.bg_frontier_steps_xy = (0.8, 1.2, 1.6)
        self.bg_frontier_z_tol = float(bg_cfg.get('frontier_z_tol', 0.30))
        self.bg_frontier_allow_weak_free = bool(bg_cfg.get('frontier_allow_weak_free', True))
        self.bg_frontier_allow_mixed = bool(bg_cfg.get('frontier_allow_mixed', True))
        self.bg_frontier_fallback_to_jitter = bool(bg_cfg.get('frontier_fallback_to_jitter', True))
        self.bg_frontier_min_keep_ratio = float(bg_cfg.get('frontier_min_keep_ratio', 0.25))
        self.bg_exclude_object_boxes = bool(bg_cfg.get('exclude_object_boxes', True))
        self.bg_box_margin_xy = float(bg_cfg.get('box_margin_xy', 0.50))
        self.bg_box_margin_z = float(bg_cfg.get('box_margin_z', 0.25))
        self.bg_filter_strong_free = bool(bg_cfg.get('filter_strong_free', True))
        self.bg_opacity_ground = float(bg_cfg.get('opacity_ground', 0.12))
        self.bg_opacity_vertical = float(bg_cfg.get('opacity_vertical', 0.10))
        self.bg_opacity_far = float(bg_cfg.get('opacity_far', 0.07))
        self.bg_prior_logit = float(bg_cfg.get('background_prior_logit', 0.50))
        self.bg_object_suppress_logit = float(bg_cfg.get('object_suppress_logit', -0.75))
        self.bg_jitter_ground_xy = float(bg_cfg.get('jitter_ground_xy', 0.85))
        self.bg_jitter_ground_z = float(bg_cfg.get('jitter_ground_z', 0.06))
        self.bg_jitter_vertical_xy = float(bg_cfg.get('jitter_vertical_xy', 0.45))
        self.bg_jitter_vertical_z = float(bg_cfg.get('jitter_vertical_z', 0.75))
        self.bg_jitter_far_xy = float(bg_cfg.get('jitter_far_xy', 1.40))
        self.bg_jitter_far_z = float(bg_cfg.get('jitter_far_z', 0.25))
        self.bg_scale_ground = tuple(float(x) for x in bg_cfg.get('scale_ground', (0.85, 0.85, 0.12)))
        self.bg_scale_vertical = tuple(float(x) for x in bg_cfg.get('scale_vertical', (0.35, 0.35, 0.95)))
        self.bg_scale_far = tuple(float(x) for x in bg_cfg.get('scale_far', (1.20, 1.20, 0.30)))
        self.bg_debug = bool(bg_cfg.get('debug', False))
        self.bg_debug_interval = max(int(bg_cfg.get('debug_interval', 200)), 1)
        self.bg_debug_max_print = max(int(bg_cfg.get('debug_max_print', 5)), 0)
        self._bg_debug_print_count = 0

        # Base fallback anchors. These keep the number of Gaussians fixed even
        # when the measured octree returns fewer than num_anchor candidates.
        xyz = torch.rand(self.num_anchor, 3, dtype=torch.float)
        if phi_activation == 'sigmoid':
            xyz = safe_inverse_sigmoid(xyz)
        elif phi_activation == 'loop':
            xyz[:, :2] = safe_inverse_sigmoid(xyz[:, :2])
        else:
            raise NotImplementedError
        scale = safe_inverse_sigmoid(torch.rand_like(xyz))
        rots = torch.zeros(self.num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1.0
        if include_opa:
            opacity = safe_inverse_sigmoid(0.1 * torch.ones((self.num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((self.num_anchor, 0), dtype=torch.float)
        semantic = torch.randn(self.num_anchor, self.semantic_dim, dtype=torch.float)
        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1)

        self.anchor = nn.Parameter(anchor.clone().float(), requires_grad=anchor_grad)
        self.anchor_init = anchor.clone().float()
        self.instance_feature = nn.Parameter(
            torch.zeros([self.num_anchor, self.embed_dims], dtype=torch.float32),
            requires_grad=feat_grad,
        )

        if gaussian_probe is None:
            gaussian_probe = {}
        cfg = dict(gaussian_probe)
        cfg.setdefault('enabled', True)
        cfg.setdefault('always_compute', True)
        cfg.setdefault('retain_last_result', False)
        cfg.setdefault('debug', False)
        cfg.setdefault('visualize', False)
        cfg.setdefault('grid_range', self.pc_range)
        cfg.setdefault('coarse_voxel_size', 0.8)
        cfg.setdefault('refined_voxel_size', 0.4)
        # 中文注释：Strict GPU 版 GaussianProbeGPU 不再使用旧 CPU ray marching 参数。
        # 不再向 probe 传递 ray_step_size / max_ray_steps / ray_chunk_size，避免无意义 warning。
        cfg.setdefault('endpoint_margin', 0.3)
        cfg.setdefault('min_ray_range', 1.0)
        cfg.setdefault('ray_count_tau', 3.0)
        cfg.setdefault('strong_free_threshold', 0.75)
        cfg.setdefault('temporal_support', dict(
            xyz_dims=(0, 1, 2), intensity_dim=3, time_lag_dim=4,
            decay_dim=5, current_flag_dim=6,
            current_weight=1.0, min_point_weight=0.01,
            strong_free_gamma=4.0, weak_free_gamma=2.0,
            mixed_factor=0.7, time_bin_size=0.02,
        ))
        cfg.setdefault('octree', dict(
            octree_range=self.pc_range,
            levels=(3.2, 1.6, 0.8, 0.4),
            min_points_by_level={3.2: 20, 1.6: 12, 0.8: 7, 0.4: 3},
            min_weighted_support_by_level={3.2: 12.0, 1.6: 8.0, 0.8: 4.0, 0.4: 2.0},
            max_normal_rms_by_level={3.2: 0.12, 1.6: 0.09, 0.8: 0.07, 0.4: 0.05},
            max_normal_scale_by_level={3.2: 0.25, 1.6: 0.16, 0.8: 0.10, 0.4: 0.06},
            max_temporal_conflict_ratio=0.35,
            support_tau=10.0,
            fit_sigma=0.12,
            min_opacity=0.10,
            max_opacity=0.95,
            free_overlap_reject=0.35,
            max_gaussians=self.num_anchor,
            ground_refinement=dict(
                enabled=True,
                source_levels=(3.2,),
                min_abs_normal_z=0.85,
                min_mean_z=self.pc_range[2],
                max_mean_z=0.60,
                min_major_scale=0.80,
                min_minor_scale=0.45,
            ),
            fitter=dict(
                sigma_scale_factor=2.0,
                min_gaussian_scale=0.05,
                max_scale_ratio=0.45,
                min_normal_scale=0.04,
                max_normal_scale_ratio=0.10,
                covariance_epsilon=1e-5,
            ),
        ))
        self.debug = bool(cfg.get('debug', False))
        self.debug_interval = max(int(cfg.get('debug_interval', 50)), 1)
        self.debug_max_print = max(int(cfg.get('debug_max_print', 5)), 0)
        self._runtime_forward_count = 0
        self._runtime_debug_print_count = 0

        # Import here to avoid forcing the old probe dependency when users keep
        # using GaussianLifter / GaussianLifterLiDAR baselines.
        from .gaussian_probe_gpu import GaussianProbeGPU
        self.gaussian_probe = GaussianProbeGPU(**cfg)

    @staticmethod
    def _is_main_process():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    @staticmethod
    def _first_tensor(value):
        # 中文注释：从 list / tuple / BasePoints / Tensor 中安全取出一个 tensor，用于推断 batch 和 device。
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value
        if hasattr(value, 'tensor') and isinstance(value.tensor, torch.Tensor):
            return value.tensor
        if isinstance(value, (list, tuple)):
            for item in value:
                t = OctreeGaussianLifter._first_tensor(item)
                if t is not None:
                    return t
        return None

    def _infer_batch_size_device(self, ms_img_feats=None, metas=None, points=None, **kwargs):
        # 中文注释：Mamba 快速版可以跳过图像/深度分支，此时 ms_img_feats=None。
        # 这里改为从 points / visibility_points / coors_batch 等张量推断 batch_size 和 device。
        if isinstance(ms_img_feats, (list, tuple)) and len(ms_img_feats) > 0 and isinstance(ms_img_feats[0], torch.Tensor):
            return int(ms_img_feats[0].shape[0]), ms_img_feats[0].device, 'ms_img_feats'
        if isinstance(ms_img_feats, torch.Tensor):
            return int(ms_img_feats.shape[0]), ms_img_feats.device, 'ms_img_feats_tensor'

        visibility_points = kwargs.get('visibility_points', None)
        if visibility_points is None and isinstance(metas, dict):
            visibility_points = metas.get('visibility_points', None)
        if isinstance(visibility_points, (list, tuple)) and len(visibility_points) > 0:
            t = self._first_tensor(visibility_points)
            if t is not None:
                return int(len(visibility_points)), t.device, 'visibility_points_list'
        if isinstance(visibility_points, torch.Tensor):
            if visibility_points.ndim >= 3:
                return int(visibility_points.shape[0]), visibility_points.device, 'visibility_points_tensor3d'
            if visibility_points.ndim == 2:
                return 1, visibility_points.device, 'visibility_points_tensor2d'

        if isinstance(points, (list, tuple)) and len(points) > 0:
            t = self._first_tensor(points)
            if t is not None:
                return int(len(points)), t.device, 'points_list'
        if isinstance(points, torch.Tensor):
            if points.ndim >= 3:
                return int(points.shape[0]), points.device, 'points_tensor3d'
            if points.ndim == 2:
                return 1, points.device, 'points_tensor2d'

        coors_batch = kwargs.get('coors_batch', None)
        if isinstance(coors_batch, torch.Tensor):
            if coors_batch.numel() > 0 and coors_batch.ndim >= 2:
                bs = int(coors_batch[:, 0].max().detach().item()) + 1
                return max(bs, 1), coors_batch.device, 'coors_batch'
            return 1, coors_batch.device, 'coors_batch_empty'

        # 最后兜底，避免启动阶段因为纯 LiDAR/Mamba 路径没有图像特征而崩溃。
        return 1, self.anchor.device, 'fallback_parameter'

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def _as_point_list(self, points, batch_size):
        if points is None:
            return None
        if isinstance(points, torch.Tensor):
            if points.ndim == 2:
                return [points]
            if points.ndim == 3:
                return [points[i] for i in range(points.shape[0])]
        if isinstance(points, (list, tuple)):
            out = []
            for p in points:
                out.append(p.tensor if hasattr(p, 'tensor') else p)
            return out
        if hasattr(points, 'tensor'):
            return [points.tensor]
        return None

    def _get_visibility_points(self, points=None, metas=None, **kwargs):
        v = kwargs.get('visibility_points', None)
        if v is None and isinstance(metas, dict):
            v = metas.get('visibility_points', None)
        # Fallback is allowed for debugging, but the intended path is 7D
        # visibility_points with is_current at dim 6.
        if v is None:
            v = points
        return v

    def _get_lidar_origins(self, batch_size, device, dtype, metas=None, **kwargs):
        origins = kwargs.get('lidar_origins', None)
        if origins is None and isinstance(metas, dict):
            origins = metas.get('lidar_origins', None)
        if origins is not None:
            t = origins.to(device=device, dtype=dtype) if isinstance(origins, torch.Tensor) else torch.as_tensor(origins, device=device, dtype=dtype)
            if t.ndim == 1:
                t = t.view(1, -1).expand(batch_size, -1)
            return t[:, :3].contiguous()
        ego2lidar = None
        if isinstance(metas, dict):
            ego2lidar = metas.get('ego2lidar', None)
        if ego2lidar is not None:
            t = ego2lidar.to(device=device, dtype=dtype) if isinstance(ego2lidar, torch.Tensor) else torch.as_tensor(ego2lidar, device=device, dtype=dtype)
            if t.ndim == 2:
                t = t.view(1, 4, 4).expand(batch_size, -1, -1)
            lidar2ego = torch.linalg.inv(t.float()).to(dtype=dtype)
            return lidar2ego[:, :3, 3].contiguous()
        return torch.zeros((batch_size, 3), device=device, dtype=dtype)

    def _encode_xyz(self, xyz):
        xyz01 = point_cloud_map(xyz, self.occ_annotation).clamp(1e-4, 1.0 - 1e-4)
        return safe_inverse_sigmoid(xyz01)

    def _encode_scale(self, scales):
        smin, smax = self.scale_range
        s = scales.clamp(min=smin, max=smax)
        sn = (s - smin) / max(smax - smin, 1e-6)
        return safe_inverse_sigmoid(sn.clamp(1e-4, 1.0 - 1e-4))

    def _normalize_intensity(self, intensity):
        if intensity is None or (isinstance(intensity, torch.Tensor) and intensity.numel() == 0):
            return None
        x = intensity.float()
        # NuScenes .bin intensity is often stored as 0~255 in this codebase;
        # keep already-normalized 0~1 values unchanged.
        max_val = torch.nan_to_num(x.detach().max(), nan=0.0)
        if bool(max_val > 1.5):
            x = x / 255.0
        return x.clamp(1e-4, 1.0 - 1e-4)

    def _fill_from_gaussians(self, anchor_b, gaussians, max_slots=None):
        if gaussians is None or len(gaussians) == 0:
            return 0, None
        device = anchor_b.device
        dtype = anchor_b.dtype
        g = gaussians.to(device) if hasattr(gaussians, 'to') else gaussians
        limit = int(anchor_b.shape[0]) if max_slots is None else min(int(max_slots), int(anchor_b.shape[0]))
        n = min(int(len(g)), limit)
        if n <= 0:
            return 0, None

        # Prefer higher-confidence measured Gaussians if the probe produced more
        # than num_anchor candidates.
        if int(len(g)) > n and hasattr(g, 'confidences') and g.confidences is not None:
            idx = torch.topk(g.confidences.float(), k=n, largest=True, sorted=False).indices
            means = g.means[idx]
            scales = g.scales[idx]
            rotations = g.rotations[idx]
            opacities = g.opacities[idx]
            intensity = getattr(g, 'intensity_mean', None)
            intensity = intensity[idx] if isinstance(intensity, torch.Tensor) else None
            levels = getattr(g, 'levels', None)
            levels = levels[idx] if isinstance(levels, torch.Tensor) else None
        else:
            means = g.means[:n]
            scales = g.scales[:n]
            rotations = g.rotations[:n]
            opacities = g.opacities[:n]
            intensity = getattr(g, 'intensity_mean', None)
            intensity = intensity[:n] if isinstance(intensity, torch.Tensor) else None
            levels = getattr(g, 'levels', None)
            levels = levels[:n] if isinstance(levels, torch.Tensor) else None

        anchor_b[:n, 0:3] = self._encode_xyz(means.to(device=device, dtype=dtype))
        anchor_b[:n, 3:6] = self._encode_scale(scales.to(device=device, dtype=dtype))
        if rotations.ndim == 3:
            anchor_b[:n, 6:10] = _rotation_matrix_to_quaternion_wxyz(rotations.to(device=device, dtype=dtype))
        else:
            q = rotations[:n].to(device=device, dtype=dtype)
            if q.shape[-1] == 4:
                anchor_b[:n, 6:10] = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.include_opa:
            opa = opacities[:n].to(device=device, dtype=dtype).float().clamp(1e-4, 1.0 - 1e-4)
            inten = self._normalize_intensity(intensity)
            if self.use_intensity and inten is not None:
                inten = inten.to(device=device, dtype=opa.dtype)
                w = min(max(float(self.intensity_opacity_weight), 0.0), 1.0)
                opa = ((1.0 - w) * opa + w * inten).clamp(1e-4, 1.0 - 1e-4)
            anchor_b[:n, 10] = safe_inverse_sigmoid(opa.to(dtype=dtype))
        if isinstance(levels, torch.Tensor):
            levels = levels.to(device=device, dtype=torch.float32, non_blocking=True).reshape(-1)[:n]
        return n, levels

    def _get_batch_boxes_labels(self, kwargs, metas, batch_size, device):
        boxes = kwargs.get('gt_bboxes_3d', None)
        labels = kwargs.get('gt_labels_3d', None)
        if boxes is None and isinstance(metas, dict):
            boxes = metas.get('gt_bboxes_3d', None)
        if labels is None and isinstance(metas, dict):
            labels = metas.get('gt_labels_3d', None)
        # Fallback to OQG selected boxes when GT is not present.  These are
        # already in [x,y,z,l,w,h,yaw] style, but for exclusion we only need a
        # conservative axis-aligned/yaw box test.
        if boxes is None:
            boxes = kwargs.get('yolo_box3d_selected_flat', None)
            labels = kwargs.get('yolo_box3d_selected_flat_labels', labels)
        return self._as_tensor_list(boxes, batch_size, device, torch.float32), self._as_tensor_list(labels, batch_size, device, None)

    @staticmethod
    def _as_tensor_list(value, batch_size, device, dtype=None):
        if value is None:
            return [None for _ in range(batch_size)]
        if isinstance(value, torch.Tensor):
            if value.ndim >= 3 and int(value.shape[0]) == batch_size:
                out = [value[b] for b in range(batch_size)]
            elif batch_size == 1:
                out = [value]
            else:
                out = [None for _ in range(batch_size)]
        elif isinstance(value, (list, tuple)):
            out = [value[i] if i < len(value) else None for i in range(batch_size)]
        else:
            out = [value] if batch_size == 1 else [None for _ in range(batch_size)]
        ret = []
        for x in out:
            if x is None:
                ret.append(None)
            else:
                if hasattr(x, 'tensor') and isinstance(x.tensor, torch.Tensor):
                    x = x.tensor
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x)
                ret.append(x.to(device=device, dtype=dtype) if dtype is not None else x.to(device=device))
        return ret

    def _points_inside_boxes_wlh(self, xyz, boxes, labels=None):
        if boxes is None or xyz is None or xyz.numel() == 0 or boxes.numel() == 0:
            return torch.zeros((0 if xyz is None else xyz.shape[0],), device=xyz.device, dtype=torch.bool)
        boxes = boxes.reshape(-1, boxes.shape[-1])[:, :7]
        valid = torch.isfinite(boxes).all(dim=-1)
        if labels is not None and labels.numel() >= boxes.shape[0]:
            lab = labels.reshape(-1)[:boxes.shape[0]].long()
            valid &= (lab >= 0) & (lab < 10)
        if not bool(valid.any()):
            return torch.zeros((xyz.shape[0],), device=xyz.device, dtype=torch.bool)
        boxes = boxes[valid]
        centers = boxes[:, :3]
        # DAOcc GT default is [x,y,z,w,l,h,yaw].  OQG selected boxes are close
        # enough for exclusion; the margin makes this conservative.
        w = boxes[:, 3].abs().clamp_min(0.05)
        l = boxes[:, 4].abs().clamp_min(0.05)
        h = boxes[:, 5].abs().clamp_min(0.05)
        yaw = boxes[:, 6]
        diff = xyz[:, None, :] - centers[None, :, :]
        c = torch.cos(yaw).view(1, -1)
        s = torch.sin(yaw).view(1, -1)
        local_x = diff[..., 0] * c + diff[..., 1] * s
        local_y = -diff[..., 0] * s + diff[..., 1] * c
        local_z = diff[..., 2]
        hx = 0.5 * l.view(1, -1) + float(self.bg_box_margin_xy)
        hy = 0.5 * w.view(1, -1) + float(self.bg_box_margin_xy)
        hz = 0.5 * h.view(1, -1) + float(self.bg_box_margin_z)
        inside = (local_x.abs() <= hx) & (local_y.abs() <= hy) & (local_z.abs() <= hz)
        return inside.any(dim=1)

    def _query_coarse_state(self, visibility, xyz):
        if visibility is None or xyz.numel() == 0 or not hasattr(visibility, 'coarse_state'):
            return None
        state = visibility.coarse_state
        if not isinstance(state, torch.Tensor) or state.numel() == 0:
            return None
        grid_min = xyz.new_tensor(visibility.grid_range[:3])
        vs = float(visibility.coarse_voxel_size)
        idx = torch.floor((xyz - grid_min.view(1, 3)) / max(vs, 1.0e-6)).long()
        shape = state.shape
        valid = (
            (idx[:, 0] >= 0) & (idx[:, 0] < int(shape[0])) &
            (idx[:, 1] >= 0) & (idx[:, 1] < int(shape[1])) &
            (idx[:, 2] >= 0) & (idx[:, 2] < int(shape[2]))
        )
        out = torch.full((xyz.shape[0],), -1, device=xyz.device, dtype=torch.long)
        if bool(valid.any()):
            ii = idx[valid]
            out[valid] = state[ii[:, 0], ii[:, 1], ii[:, 2]].long()
        return out

    def _choice_with_repeat(self, idx, num):
        num = int(max(num, 0))
        if num <= 0:
            return idx.new_empty((0,), dtype=torch.long)
        if idx.numel() == 0:
            return idx.new_empty((0,), dtype=torch.long)
        if idx.numel() >= num:
            perm = torch.randperm(idx.numel(), device=idx.device)[:num]
            return idx[perm]
        rep = int((num + idx.numel() - 1) // idx.numel())
        expanded = idx.repeat(rep)
        perm = torch.randperm(expanded.numel(), device=idx.device)[:num]
        return expanded[perm]

    def _make_bg_semantic_prior(self, kind, n, device, dtype):
        sem = torch.zeros((n, self.semantic_dim), device=device, dtype=dtype)
        if self.semantic_dim <= 0 or n <= 0:
            return sem
        # Occ3D semantic ids: 11 driveable, 12 other_flat, 13 sidewalk,
        # 14 terrain, 15 manmade, 16 vegetation.
        obj_end = min(11, self.semantic_dim)
        if obj_end > 0:
            sem[:, :obj_end] = float(self.bg_object_suppress_logit)
        def add(ids, val):
            for cid in ids:
                if 0 <= int(cid) < self.semantic_dim:
                    sem[:, int(cid)] = float(val)
        if kind == 'ground':
            add((11, 12, 13, 14), self.bg_prior_logit)
            add((15, 16), 0.10)
        elif kind == 'vertical':
            add((15,), self.bg_prior_logit + 0.15)
            add((16,), 0.20)
            add((11, 12, 13, 14), -0.15)
        else:  # far / vegetation / weak coverage
            add((11, 12, 13, 14, 15, 16), self.bg_prior_logit * 0.55)
            add((16,), self.bg_prior_logit)
        return sem

    def _build_bg_anchor_chunk(self, xyz, scales, opacity, kind, anchor_dim, dtype):
        n = int(xyz.shape[0])
        device = xyz.device
        anchor = torch.zeros((n, anchor_dim), device=device, dtype=dtype)
        anchor[:, 0:3] = self._encode_xyz(xyz).to(dtype=dtype)
        anchor[:, 3:6] = self._encode_scale(scales).to(dtype=dtype)
        anchor[:, 6] = 1.0
        if self.include_opa and anchor_dim > 10:
            opa = torch.full((n,), float(opacity), device=device, dtype=dtype).clamp(1e-4, 1.0 - 1e-4)
            anchor[:, 10] = safe_inverse_sigmoid(opa)
        sem_s = 10 + int(self.include_opa)
        sem_e = sem_s + self.semantic_dim
        if self.semantic_dim > 0 and anchor_dim >= sem_e:
            anchor[:, sem_s:sem_e] = self._make_bg_semantic_prior(kind, n, device, dtype)
        return anchor

    def _frontier_allowed_state_mask(self, state):
        if state is None:
            return None
        keep = state == int(UNKNOWN)
        if self.bg_frontier_allow_weak_free:
            keep = keep | (state == int(WEAK_FREE))
        if self.bg_frontier_allow_mixed:
            keep = keep | (state == int(MIXED))
        # Never use strong free as a background surface candidate.
        keep = keep & (state != int(STRONG_FREE))
        return keep

    def _sample_background_frontier_kind(
        self,
        kind,
        pts_valid,
        seed_mask,
        num,
        visibility,
        boxes,
        labels,
        jitter_z,
        scale_tuple,
        opacity,
        anchor_dim,
        dtype,
    ):
        """Sample background anchors by extending measured support into unknown/free frontier cells.

        This is intentionally Gaussian-level sparse sampling, not dense BEV sampling.
        It uses measured points only as support seeds and places candidates in nearby
        UNKNOWN / WEAK_FREE / MIXED coarse visibility cells.  Strong-free cells and
        dynamic object boxes are rejected.
        """
        num = int(max(num, 0))
        device = pts_valid.device
        if num <= 0:
            return torch.empty((0, anchor_dim), device=device, dtype=dtype), 0, 0
        seed_idx = torch.where(seed_mask)[0]
        if seed_idx.numel() == 0:
            return torch.empty((0, anchor_dim), device=device, dtype=dtype), 0, 0
        # More oversampling than old jitter because frontier filtering is stricter.
        need = max(num * max(int(self.bg_sample_oversample), 1) * 3, num)
        chosen = self._choice_with_repeat(seed_idx, need)
        base = pts_valid[chosen]

        dirs = base.new_tensor([
            [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
            [0.7071, 0.7071], [0.7071, -0.7071], [-0.7071, 0.7071], [-0.7071, -0.7071],
        ])
        didx = torch.randint(0, dirs.shape[0], (need,), device=device)
        dir_xy = dirs[didx]
        step_vals = base.new_tensor(self.bg_frontier_steps_xy)
        sidx = torch.randint(0, step_vals.shape[0], (need,), device=device)
        step = step_vals[sidx].view(-1, 1)
        xyz = base.clone()
        xyz[:, 0:2] = base[:, 0:2] + dir_xy * step
        # Keep ground-like anchors thin.  Vertical/far only have small budgets here;
        # they still use frontier placement, but with their own z jitter.
        z_noise = torch.randn((need,), device=device, dtype=torch.float32) * float(jitter_z)
        xyz[:, 2] = base[:, 2] + z_noise.clamp(-abs(float(self.bg_frontier_z_tol)), abs(float(self.bg_frontier_z_tol)))

        r = xyz.new_tensor(self.pc_range)
        xyz[:, 0] = xyz[:, 0].clamp(r[0] + 1e-3, r[3] - 1e-3)
        xyz[:, 1] = xyz[:, 1].clamp(r[1] + 1e-3, r[4] - 1e-3)
        xyz[:, 2] = xyz[:, 2].clamp(r[2] + 1e-3, r[5] - 1e-3)

        keep = torch.isfinite(xyz).all(dim=-1)
        state = self._query_coarse_state(visibility, xyz) if visibility is not None else None
        state_keep = self._frontier_allowed_state_mask(state)
        if state_keep is not None:
            keep &= state_keep
        elif self.bg_filter_strong_free and visibility is not None:
            # Defensive path; normally state_keep already handles strong free.
            state2 = self._query_coarse_state(visibility, xyz)
            if state2 is not None:
                keep &= state2 != int(STRONG_FREE)
        if self.bg_exclude_object_boxes and boxes is not None:
            keep &= ~self._points_inside_boxes_wlh(xyz, boxes, labels)
        xyz = xyz[keep]
        num_frontier = int(xyz.shape[0])
        if xyz.shape[0] < num:
            return torch.empty((0, anchor_dim), device=device, dtype=dtype), num_frontier, 0
        xyz = xyz[:num]
        scales = xyz.new_tensor(scale_tuple).view(1, 3).expand(xyz.shape[0], 3).clone()
        return self._build_bg_anchor_chunk(xyz, scales, opacity, kind, anchor_dim, dtype), num_frontier, int(xyz.shape[0])

    def _sample_background_jitter_kind(
        self,
        kind,
        pts_valid,
        mask,
        num,
        visibility,
        boxes,
        labels,
        jitter_xy,
        jitter_z,
        scale_tuple,
        opacity,
        anchor_dim,
        dtype,
    ):
        device = pts_valid.device
        num = int(max(num, 0))
        if num <= 0:
            return torch.empty((0, anchor_dim), device=device, dtype=dtype), 0
        pool_local = torch.where(mask)[0]
        if pool_local.numel() == 0:
            pool_local = torch.arange(pts_valid.shape[0], device=device)
        need = num * int(self.bg_sample_oversample)
        chosen = self._choice_with_repeat(pool_local, need)
        base = pts_valid[chosen]
        noise = torch.randn((need, 3), device=device, dtype=torch.float32)
        noise[:, 0:2] *= float(jitter_xy)
        noise[:, 2] *= float(jitter_z)
        xyz = base + noise
        r = xyz.new_tensor(self.pc_range)
        xyz[:, 0] = xyz[:, 0].clamp(r[0] + 1e-3, r[3] - 1e-3)
        xyz[:, 1] = xyz[:, 1].clamp(r[1] + 1e-3, r[4] - 1e-3)
        xyz[:, 2] = xyz[:, 2].clamp(r[2] + 1e-3, r[5] - 1e-3)
        keep = torch.isfinite(xyz).all(dim=-1)
        if self.bg_exclude_object_boxes and boxes is not None:
            keep &= ~self._points_inside_boxes_wlh(xyz, boxes, labels)
        if self.bg_filter_strong_free and visibility is not None:
            state = self._query_coarse_state(visibility, xyz)
            if state is not None:
                keep &= state != int(STRONG_FREE)
        xyz = xyz[keep]
        if xyz.shape[0] < num:
            extra = self._choice_with_repeat(torch.arange(max(xyz.shape[0], 1), device=device), num - xyz.shape[0])
            if xyz.shape[0] == 0:
                xyz = base[:1].repeat(num, 1)
            else:
                xyz = torch.cat([xyz, xyz[extra]], dim=0)
        else:
            xyz = xyz[:num]
        scales = xyz.new_tensor(scale_tuple).view(1, 3).expand(xyz.shape[0], 3).clone()
        return self._build_bg_anchor_chunk(xyz, scales, opacity, kind, anchor_dim, dtype), int(xyz.shape[0])

    def _sample_background_anchors(self, point_tensor, visibility, boxes, labels, anchor_dim, dtype):
        device = self.anchor.device if point_tensor is None else point_tensor.device
        if self.bg_num <= 0:
            return torch.empty((0, anchor_dim), device=device, dtype=dtype), {}
        if point_tensor is None or point_tensor.numel() == 0:
            # Absolute fallback: low-opacity random background anchors.  This
            # should rarely be used because normal training has 10-sweep points.
            r = torch.tensor(self.pc_range, device=device, dtype=torch.float32)
            xyz = torch.rand((self.bg_num, 3), device=device, dtype=torch.float32)
            xyz = torch.stack([
                xyz[:, 0] * (r[3] - r[0]) + r[0],
                xyz[:, 1] * (r[4] - r[1]) + r[1],
                xyz[:, 2] * (r[5] - r[2]) + r[2],
            ], dim=-1)
            sc = xyz.new_tensor(self.bg_scale_far).view(1, 3).expand(self.bg_num, 3)
            return self._build_bg_anchor_chunk(xyz, sc, self.bg_opacity_far, 'far', anchor_dim, dtype), {'fallback': self.bg_num}
        pts = point_tensor[:, :3].to(device=device, dtype=torch.float32)
        finite = torch.isfinite(pts).all(dim=-1)
        r = torch.tensor(self.pc_range, device=device, dtype=torch.float32)
        in_range = (pts[:, 0] >= r[0]) & (pts[:, 0] <= r[3]) & (pts[:, 1] >= r[1]) & (pts[:, 1] <= r[4]) & (pts[:, 2] >= r[2]) & (pts[:, 2] <= r[5])
        valid = finite & in_range
        if self.bg_exclude_object_boxes and boxes is not None:
            valid &= ~self._points_inside_boxes_wlh(pts, boxes, labels)
        idx_all = torch.where(valid)[0]
        if idx_all.numel() == 0:
            idx_all = torch.where(finite & in_range)[0]
        if idx_all.numel() == 0:
            return self._sample_background_anchors(None, None, None, None, anchor_dim, dtype)

        pts_valid = pts[idx_all]
        dist = torch.linalg.norm(pts_valid[:, :2], dim=-1)
        z = pts_valid[:, 2]
        ground_mask = z <= float(self.bg_ground_max_z)
        vertical_mask = (z > float(self.bg_vertical_min_z)) & (dist < float(self.bg_far_range))
        far_mask = dist >= float(self.bg_far_range)

        n_ground = int(round(self.bg_num * self.bg_ground_ratio))
        n_vertical = int(round(self.bg_num * self.bg_vertical_ratio))
        n_far = self.bg_num - n_ground - n_vertical
        counts = {'ground': max(n_ground, 0), 'vertical': max(n_vertical, 0), 'far': max(n_far, 0)}
        anchors = []
        stats = {'mode': self.bg_sampling_mode}
        for kind, mask, num, jitter_xy, jitter_z, scale_tuple, opa in [
            ('ground', ground_mask, counts['ground'], self.bg_jitter_ground_xy, self.bg_jitter_ground_z, self.bg_scale_ground, self.bg_opacity_ground),
            ('vertical', vertical_mask, counts['vertical'], self.bg_jitter_vertical_xy, self.bg_jitter_vertical_z, self.bg_scale_vertical, self.bg_opacity_vertical),
            ('far', far_mask, counts['far'], self.bg_jitter_far_xy, self.bg_jitter_far_z, self.bg_scale_far, self.bg_opacity_far),
        ]:
            if num <= 0:
                continue
            chunk = None
            if self.bg_sampling_mode in ('frontier', 'hole', 'frontier_hole') and visibility is not None:
                chunk, frontier_candidates, frontier_used = self._sample_background_frontier_kind(
                    kind, pts_valid, mask, num, visibility, boxes, labels,
                    jitter_z, scale_tuple, opa, anchor_dim, dtype,
                )
                stats[f'{kind}_frontier_candidates'] = int(frontier_candidates)
                stats[f'{kind}_frontier_used'] = int(frontier_used)
                # If frontier filtering is too strict for the current sample, fall back
                # to the old jitter sampler for the missing kind instead of silently
                # duplicating a tiny set of candidates.
                min_keep = int(max(1, round(num * float(self.bg_frontier_min_keep_ratio))))
                if chunk.shape[0] < num or frontier_used < min_keep:
                    chunk = None
            if chunk is None:
                if not self.bg_frontier_fallback_to_jitter and self.bg_sampling_mode in ('frontier', 'hole', 'frontier_hole'):
                    # Empty pad is safer than repeating arbitrary measured points when
                    # explicitly testing frontier-only sampling.
                    chunk = torch.empty((0, anchor_dim), device=device, dtype=dtype)
                else:
                    chunk, used = self._sample_background_jitter_kind(
                        kind, pts_valid, mask, num, visibility, boxes, labels,
                        jitter_xy, jitter_z, scale_tuple, opa, anchor_dim, dtype,
                    )
                    stats[f'{kind}_jitter_used'] = int(used)
            if chunk.shape[0] > 0:
                anchors.append(chunk)
            stats[kind] = int(chunk.shape[0])
        if len(anchors) == 0:
            return self._sample_background_anchors(None, None, None, None, anchor_dim, dtype)
        out = torch.cat(anchors, dim=0)
        if out.shape[0] > self.bg_num:
            out = out[:self.bg_num]
        elif out.shape[0] < self.bg_num:
            pad = out[:1].repeat(self.bg_num - out.shape[0], 1) if out.shape[0] > 0 else torch.zeros((self.bg_num - out.shape[0], anchor_dim), device=device, dtype=dtype)
            out = torch.cat([out, pad], dim=0)
        stats['total'] = int(out.shape[0])
        return out.to(dtype=dtype), stats

    def forward(self, ms_img_feats=None, metas=None, points=None, **kwargs):
        self._runtime_forward_count += 1
        batch_size, device, batch_source = self._infer_batch_size_device(
            ms_img_feats=ms_img_feats, metas=metas, points=points, **kwargs
        )
        # Keep anchors/features in their parameter dtype (normally fp32). This
        # matches the original GSF3D lifter and avoids half-precision logits for
        # xyz/scale/opacity under AMP.
        dtype = self.anchor.dtype
        instance_feature = torch.tile(self.instance_feature[None], (batch_size, 1, 1)).to(device=device)
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1)).to(device=device).clone()

        # Measured octree Gaussians do not carry semantic labels here.  Clear
        # the random learnable semantic logits before measured filling/local
        # completion so local children do not inherit random class priors.
        if self.zero_base_semantics and self.semantics and self.semantic_dim > 0:
            sem_s = 10 + int(self.include_opa)
            sem_e = sem_s + self.semantic_dim
            if anchor.shape[-1] >= sem_e:
                anchor[..., sem_s:sem_e] = 0.0

        if self.include_opa and self.suppress_fallback_opacity:
            anchor[:, :, 10] = safe_inverse_sigmoid(
                torch.full_like(anchor[:, :, 10], min(max(self.fallback_opacity, 1e-4), 1.0 - 1e-4))
            )

        visibility_points = self._get_visibility_points(points=points, metas=metas, **kwargs)
        point_list = self._as_point_list(visibility_points, batch_size)
        # V22-C1 keeps the visibility branch entirely on CUDA.  We collect only
        # the already-computed coarse state tensors, stack them once, and build
        # fixed neighborhood maps with cuDNN/PyTorch 3D pooling.  No Python
        # dictionary per sample, no CPU ray traversal, and no per-Gaussian
        # support-point loop is introduced.
        visibility_state_items = [None for _ in range(batch_size)]
        visibility_grid_min = None
        visibility_voxel_size = None
        probe_out = None
        octree_debug = []
        filled_anchor_counts = [0 for _ in range(batch_size)]
        measured_levels = torch.full((batch_size, int(anchor.shape[1])), -1.0, device=device, dtype=torch.float32)
        # 0 measured/fallback, 1 completion/object, 2 background-only.
        source_types = torch.zeros((batch_size, int(anchor.shape[1])), device=device, dtype=torch.long)
        if self.bg_num > 0:
            source_types[:, self.normal_num_anchor:] = 2
        if point_list is not None and len(point_list) > 0:
            point_list = [p.to(device=device, dtype=torch.float32, non_blocking=True) for p in point_list]
            origins = self._get_lidar_origins(batch_size, device, torch.float32, metas=metas, **kwargs)
            probe_out = self.gaussian_probe(point_list, origins, metas=metas)
            if probe_out is not None:
                by_batch = {int(x.batch_index): x for x in probe_out}
                for b in range(batch_size):
                    item = by_batch.get(b, None)
                    if item is None:
                        octree_debug.append({'batch_index': b, 'measured': 0})
                        continue
                    visibility = getattr(item, 'visibility', None)
                    if visibility is not None and isinstance(visibility.coarse_state, torch.Tensor):
                        coarse_state = visibility.coarse_state.detach().to(
                            device=device, dtype=torch.uint8, non_blocking=True
                        ).contiguous()
                        visibility_state_items[b] = coarse_state
                        if visibility_grid_min is None:
                            visibility_grid_min = torch.as_tensor(
                                visibility.grid_range[:3], device=device, dtype=torch.float32
                            )
                            visibility_voxel_size = float(visibility.coarse_voxel_size)
                    filled, levels = self._fill_from_gaussians(anchor[b, :self.normal_num_anchor], item.gaussians, max_slots=self.normal_num_anchor)
                    filled_anchor_counts[b] = int(filled)
                    if isinstance(levels, torch.Tensor) and int(filled) > 0:
                        measured_levels[b, :int(filled)] = levels[:int(filled)].to(device=device, dtype=torch.float32)
                    info = dict(getattr(item, 'debug_info', {}) or {})
                    info['filled_anchor'] = int(filled)
                    octree_debug.append(info)

        # V22-C1: build the batched visibility-neighborhood maps here, after the
        # Octree/visibility probe has produced ``visibility_state_items``.
        # The previous package accidentally placed this block in the unrelated
        # GaussianLifter.forward(), where these variables do not exist, and did
        # not execute it in OctreeGaussianLifter.forward().
        visibility_state_map = None
        visibility_strong_any_map = None
        visibility_strong_ratio_map = None
        visibility_weak_ratio_map = None
        visibility_valid_batch = torch.zeros(
            (batch_size,), device=device, dtype=torch.bool
        )
        if self.build_visibility_neighborhood_maps:
            reference = next((x for x in visibility_state_items if x is not None), None)
            if reference is not None:
                state_items = [
                    x if x is not None else torch.full_like(
                        reference, int(UNKNOWN), dtype=torch.uint8
                    )
                    for x in visibility_state_items
                ]
                visibility_state_map = torch.stack(state_items, dim=0).contiguous()
                visibility_valid_batch = torch.as_tensor(
                    [x is not None for x in visibility_state_items],
                    device=device,
                    dtype=torch.bool,
                )

                # [B,1,X,Y,Z]. These are standard CUDA pooling operations and
                # run once per batch. Local children later query the resulting
                # maps with flattened torch.gather; there is no 13-point loop.
                strong = (
                    visibility_state_map == int(STRONG_FREE)
                ).unsqueeze(1).to(dtype=torch.float32)
                weak = (
                    visibility_state_map == int(WEAK_FREE)
                ).unsqueeze(1).to(dtype=torch.float32)
                k = int(self.visibility_neighborhood_kernel)
                pad = k // 2
                visibility_strong_any_map = F.max_pool3d(
                    strong, kernel_size=k, stride=1, padding=pad
                ).squeeze(1).to(dtype=torch.uint8)
                visibility_strong_ratio_map = F.avg_pool3d(
                    strong,
                    kernel_size=k,
                    stride=1,
                    padding=pad,
                    count_include_pad=True,
                ).squeeze(1).to(dtype=torch.float16)
                visibility_weak_ratio_map = F.avg_pool3d(
                    weak,
                    kernel_size=k,
                    stride=1,
                    padding=pad,
                    count_include_pad=True,
                ).squeeze(1).to(dtype=torch.float16)

        # Initialize the reserved tail slots with background-only Gaussians.
        # This happens after measured filling and before box completion.  Box
        # completion is told to stop at gaussian_background_start so these 2000
        # anchors cannot be stolen by object completion.
        bg_stats_all = []
        if self.bg_enabled and self.bg_num > 0:
            boxes_list, labels_list = self._get_batch_boxes_labels(kwargs, metas, batch_size, device)
            by_batch_probe = {}
            try:
                if probe_out is not None:
                    by_batch_probe = {int(x.batch_index): x for x in probe_out}
            except Exception:
                by_batch_probe = {}
            for b in range(batch_size):
                vis = getattr(by_batch_probe.get(b, None), 'visibility', None)
                pt_b = point_list[b] if point_list is not None and b < len(point_list) else None
                bg_anchor, bg_stats = self._sample_background_anchors(
                    pt_b,
                    vis,
                    boxes_list[b],
                    labels_list[b],
                    anchor_dim=int(anchor.shape[-1]),
                    dtype=dtype,
                )
                if bg_anchor.shape[0] > 0:
                    anchor[b, self.normal_num_anchor:self.normal_num_anchor + self.bg_num] = bg_anchor[:self.bg_num].to(device=device, dtype=dtype)
                bg_stats_all.append(bg_stats)
            if (
                self.bg_debug and self._is_main_process()
                and self._bg_debug_print_count < self.bg_debug_max_print
                and self._runtime_forward_count % self.bg_debug_interval == 0
            ):
                print(
                    '[BackgroundGaussianSampler] '
                    f'call={self._runtime_forward_count}, total_anchor={self.num_anchor}, normal={self.normal_num_anchor}, '
                    f'bg={self.bg_num}, filled={filled_anchor_counts}, stats={bg_stats_all}',
                    flush=True,
                )
                self._bg_debug_print_count += 1

        if (
            self.debug
            and self._is_main_process()
            and self._runtime_debug_print_count < self.debug_max_print
            and self._runtime_forward_count % self.debug_interval == 0
        ):
            point_counts = []
            if point_list is not None:
                point_counts = [int(p.shape[0]) for p in point_list]
            filled_counts = [int(x.get('filled_anchor', 0)) for x in octree_debug if isinstance(x, dict)]
            print(
                '[OctreeGaussianLifter Runtime Debug] '
                f'call={self._runtime_forward_count}, batch_size={batch_size}, device={device}, '
                f'batch_source={batch_source}, ms_img_feats_none={ms_img_feats is None}, '
                f'point_counts={point_counts}, filled_anchor={filled_counts}',
                flush=True,
            )
            self._runtime_debug_print_count += 1

        if self.detach_initialized_anchor:
            anchor = anchor.detach()
        return {
            'rep_features': instance_feature,
            'representation': anchor,
            'octree_gaussian_debug': octree_debug,
            'gaussian_filled_count': torch.as_tensor(filled_anchor_counts, device=device, dtype=torch.long),
            'gaussian_measured_levels': measured_levels,
            'gaussian_source_types': source_types,
            'gaussian_background_start': torch.full((batch_size,), int(self.normal_num_anchor), device=device, dtype=torch.long),
            'gaussian_background_count': torch.full((batch_size,), int(self.bg_num), device=device, dtype=torch.long),
            'gaussian_visibility_state_map': visibility_state_map,
            'gaussian_visibility_strong_any_map': visibility_strong_any_map,
            'gaussian_visibility_strong_ratio_map': visibility_strong_ratio_map,
            'gaussian_visibility_weak_ratio_map': visibility_weak_ratio_map,
            'gaussian_visibility_valid_batch': visibility_valid_batch,
            'gaussian_visibility_grid_min': visibility_grid_min,
            'gaussian_visibility_voxel_size': visibility_voxel_size,
        }
