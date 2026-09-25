from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from model.utils.safe_ops import safe_inverse_sigmoid
from model.lifter.visibility_grid import UNKNOWN, WEAK_FREE, STRONG_FREE, SURFACE_HIT, MIXED


class ObjectGaussianSeedCompletionHead(nn.Module):
    """V22-C2: decoupled geometry/semantic child quality on top of V22-C1 visibility gating.

    The fixed GSF representation already reserves ``num_anchor`` slots.  V12
    therefore does not impose separate total limits on object, local, generic,
    or other completion branches.  Every branch may propose as many Gaussians as its local
    rule produces; only the real remaining representation capacity is enforced.

    Main steps:
      1. Project every measured Gaussian into every valid 2D object box.
      2. Predict an ownership probability for each (box, Gaussian) pair.
      3. Globally assign each measured Gaussian to at most one object.
      4. Generate local and object-level children from owned measured seeds.
      5. Generate conservative generic children from non-owned measured seeds.
      6. Reject/attenuate Local children using a fast center-state lookup in the current-frame visibility grid.

    Training still uses matched GT 3D boxes to supervise the V12 ownership
    classifier.  Generated positions, scales, and rotations remain the stable
    V12 fixed geometry and are detached before entering the Gaussian encoder.

    V22-C2 keeps the isolated child-quality network but separates its two jobs.
    The non-empty head predicts a geometry gate, while the semantic head predicts
    class correctness conditioned on the sampled location being non-empty.
    Geometry controls opacity and the persistent render gate; geometry times
    semantic confidence controls object-class features and capacity ranking.
    The quality loss never moves Gaussian geometry and the detached gates avoid
    coupling the auxiliary classifier to the main occupancy path.
    """

    # DAOcc 10-class detection label -> Occ3D semantic label.
    DAO_TO_OCC = (4, 10, 5, 3, 9, 1, 6, 2, 7, 8)
    SMALL_CLASS_IDS = (5, 6, 7, 8, 9)
    LARGE_CLASS_IDS = (1, 2, 3, 4)

    def __init__(
        self,
        enabled: bool = True,
        point_cloud_range: Sequence[float] = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        scale_range: Sequence[float] = (0.01, 1.44),
        semantic_dim: int = 17,
        feature_dim: int = 128,
        include_opa: bool = True,
        # Projection / geometric prior.
        core_shrink_x: float = 0.12,
        core_shrink_top: float = 0.10,
        core_shrink_bottom: float = 0.18,
        min_projected_gaussians: int = 1,
        depth_cluster_gap_abs: float = 0.90,
        depth_cluster_gap_rel: float = 0.030,
        max_depth_clusters: int = 6,
        support_norm: float = 5.0,
        # Learned ownership.
        owner_feature_dim: int = 16,
        owner_class_dim: int = 8,
        owner_hidden_dim: int = 128,
        owner_threshold: float = 0.55,
        owner_min_keep_score: float = 0.35,
        owner_focal_alpha: float = 0.75,
        owner_focal_gamma: float = 2.0,
        owner_prior_strength: float = 1.0,
        owner_semantic_prior_logit: float = 3.0,
        inject_seed_semantics: bool = True,
        inject_seed_features: bool = True,
        # Local completion.  Counts are per parent, not branch-total limits.
        generate_seed_children: bool = True,
        seed_children_per_parent: int = 3,
        small_seed_children_per_parent: int = 4,
        large_seed_children_per_parent: int = 4,
        child_offset_ratio: float = 0.90,
        child_scale_ratio: float = 0.62,
        min_gaussian_scale: float = 0.04,
        max_gaussian_scale: float = 0.32,
        completion_opacity: float = 0.18,
        # V22-C1 GPU center-hard / neighborhood-soft gate retained in V22-C2.
        enable_visibility_gate: bool = True,
        visibility_grid_range: Sequence[float] = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        visibility_voxel_size: float = 0.8,
        visibility_strong_ratio_hard: float = 0.25,
        visibility_strong_ratio_weight: float = 0.35,
        visibility_weak_ratio_weight: float = 0.15,
        visibility_soft_gate_min: float = 0.50,
        visibility_gate_local_only: bool = True,
        visibility_filter_hard_reject: bool = False,
        visibility_encoder_gain_power: float = 0.50,
        visibility_debug: bool = False,
        # Object-level completion.  Counts are per physical object.
        enable_object_completion: bool = True,
        class_typical_dims: Sequence[Sequence[float]] = (
            (4.5, 1.8, 1.6), (7.0, 2.5, 3.0), (6.0, 2.6, 3.0),
            (10.0, 2.6, 3.2), (10.0, 2.6, 3.0), (2.0, 0.5, 0.8),
            (2.1, 0.8, 1.4), (1.8, 0.6, 1.5), (0.8, 0.8, 1.75),
            (0.5, 0.5, 0.8),
        ),
        object_children_min: Sequence[int] = (12, 20, 24, 24, 20, 8, 6, 6, 4, 2),
        object_children_max: Sequence[int] = (40, 56, 64, 64, 56, 24, 12, 12, 8, 4),
        object_merge_radii: Sequence[float] = (1.8, 2.6, 2.6, 3.0, 3.0, 1.2, 0.9, 0.9, 0.7, 0.5),
        object_center_oqg_blend: float = 0.10,
        object_scale_min: float = 0.08,
        object_scale_max: float = 0.60,
        object_completion_opacity: float = 0.14,
        object_min_record_score: float = 0.15,
        # V14 child-quality calibration.  These settings never cap the number
        # of generated Gaussians and never move a Gaussian's geometry.
        enable_child_quality: bool = True,
        occ_empty_label: int = 17,
        child_quality_feature_dim: int = 24,
        child_quality_class_dim: int = 8,
        child_quality_kind_dim: int = 8,
        child_quality_hidden_dim: int = 96,
        child_quality_prior_strength: float = 1.0,
        child_quality_nonempty_weight: float = 0.50,
        child_quality_semantic_weight: float = 0.50,
        # V22-C2: geometry and semantic gates are independent and may reach zero.
        child_quality_decoupled: bool = True,
        child_quality_semantic_on_nonempty_only: bool = True,
        child_quality_geometry_zero_threshold: float = 0.20,
        child_quality_semantic_zero_threshold: float = 0.20,
        child_quality_persistent_geometry_gate: bool = True,
        child_quality_opacity_floor: float = 0.0,
        child_quality_feature_floor: float = 0.0,
        child_quality_rank_floor: float = 0.0,
        # Generic measured-parent completion for all non-owned measured Gaussians.
        enable_generic_completion: bool = True,
        generic_opacity_thr: float = 0.35,
        generic_max_parent_scale: float = 0.30,
        generic_children_per_parent: int = 1,
        generic_child_offset_ratio: float = 0.80,
        generic_child_scale_ratio: float = 0.72,
        generic_completion_opacity: float = 0.12,
        # Deprecated constructor fields retained only for old-config compatibility.
        enable_oqg_fallback: bool = True,
        fallback_score_thr: float = 0.02,
        fallback_gaussians_per_box: int = 1,
        fallback_scale_ratio: float = 0.08,
        fallback_min_scale: float = 0.08,
        fallback_max_scale: float = 0.28,
        fallback_opacity: float = 0.055,
        fallback_semantic_prior_logit: float = 1.10,
        fallback_nms_default_radius: float = 1.0,
        fallback_nms_class_radii: Sequence[float] = (
            1.8, 2.4, 2.4, 2.5, 2.5, 1.2, 0.8, 0.8, 0.7, 0.6,
        ),
        # Learned feature priors.
        add_completion_feature: bool = True,
        # Sparse debug.
        debug: bool = False,
        debug_interval: int = 100,
        debug_train_max_print: int = 12,
        debug_eval_max_print: int = 12,
        # Backward-compatible legacy total-cap arguments.  V12 intentionally
        # ignores them; the only total limit is the representation capacity N.
        max_projected_gaussians_per_box: int = 0,
        max_seed_parents_per_box: int = 0,
        max_seed_parents_per_sample: int = 0,
        seed_parent_caps: Sequence[int] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        max_local_gaussians: int = 0,
        max_object_gaussians: int = 0,
        max_generic_gaussians: int = 0,
        max_completed_gaussians: int = 0,
        fallback_max_per_sample: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.semantic_dim = max(int(semantic_dim), 0)
        self.feature_dim = max(int(feature_dim), 1)
        self.include_opa = bool(include_opa)

        pc = torch.tensor(point_cloud_range, dtype=torch.float32)
        sr = torch.tensor(scale_range, dtype=torch.float32)
        if pc.numel() != 6:
            raise ValueError("point_cloud_range must contain 6 values")
        if sr.numel() != 2:
            raise ValueError("scale_range must contain 2 values")
        self.register_buffer("pc_range_tensor", pc, persistent=False)
        self.register_buffer("scale_range_tensor", sr, persistent=False)
        self.register_buffer(
            "dao_to_occ_sem",
            torch.tensor(self.DAO_TO_OCC, dtype=torch.long),
            persistent=False,
        )

        self.core_shrink_x = float(min(max(core_shrink_x, 0.0), 0.45))
        self.core_shrink_top = float(min(max(core_shrink_top, 0.0), 0.45))
        self.core_shrink_bottom = float(min(max(core_shrink_bottom, 0.0), 0.45))
        self.min_projected_gaussians = max(int(min_projected_gaussians), 1)
        self.depth_cluster_gap_abs = max(float(depth_cluster_gap_abs), 1.0e-3)
        self.depth_cluster_gap_rel = max(float(depth_cluster_gap_rel), 0.0)
        self.max_depth_clusters = max(int(max_depth_clusters), 1)
        self.support_norm = max(float(support_norm), 1.0e-3)

        self.owner_threshold = float(min(max(owner_threshold, 0.01), 0.99))
        self.owner_min_keep_score = float(min(max(owner_min_keep_score, 0.0), self.owner_threshold))
        self.owner_focal_alpha = float(min(max(owner_focal_alpha, 0.0), 1.0))
        self.owner_focal_gamma = max(float(owner_focal_gamma), 0.0)
        self.owner_prior_strength = max(float(owner_prior_strength), 0.0)
        self.owner_semantic_prior_logit = float(owner_semantic_prior_logit)
        self.inject_seed_semantics = bool(inject_seed_semantics)
        self.inject_seed_features = bool(inject_seed_features)

        owner_feature_dim = max(int(owner_feature_dim), 4)
        owner_class_dim = max(int(owner_class_dim), 4)
        owner_hidden_dim = max(int(owner_hidden_dim), 16)
        # V22-B removes the two OQG depth fields; 14 geometry values remain.
        owner_geom_dim = 14
        self.owner_feature_proj = nn.Sequential(
            nn.Linear(self.feature_dim, owner_feature_dim),
            nn.LayerNorm(owner_feature_dim),
            nn.SiLU(),
        )
        self.owner_class_embedding = nn.Embedding(10, owner_class_dim)
        self.owner_mlp = nn.Sequential(
            nn.Linear(owner_geom_dim + owner_feature_dim + owner_class_dim, owner_hidden_dim),
            nn.SiLU(),
            nn.Linear(owner_hidden_dim, max(owner_hidden_dim // 2, 16)),
            nn.SiLU(),
            nn.Linear(max(owner_hidden_dim // 2, 16), 1),
        )
        nn.init.normal_(self.owner_class_embedding.weight, mean=0.0, std=0.02)
        # Start as a residual correction around the geometric prior rather than
        # producing random ownership decisions at iteration 1.
        nn.init.zeros_(self.owner_mlp[-1].weight)
        nn.init.zeros_(self.owner_mlp[-1].bias)

        self.generate_seed_children = bool(generate_seed_children)
        self.seed_children_per_parent = max(int(seed_children_per_parent), 0)
        self.small_seed_children_per_parent = max(int(small_seed_children_per_parent), 0)
        self.large_seed_children_per_parent = max(int(large_seed_children_per_parent), 0)
        self.child_offset_ratio = max(float(child_offset_ratio), 0.0)
        self.child_scale_ratio = max(float(child_scale_ratio), 1.0e-3)
        self.min_gaussian_scale = max(float(min_gaussian_scale), 1.0e-4)
        self.max_gaussian_scale = max(float(max_gaussian_scale), self.min_gaussian_scale)
        self.completion_opacity = float(min(max(completion_opacity, 1.0e-4), 1.0 - 1.0e-4))

        self.enable_visibility_gate = bool(enable_visibility_gate)
        if len(visibility_grid_range) != 6:
            raise ValueError("visibility_grid_range must contain 6 values")
        vis_range = torch.as_tensor(visibility_grid_range, dtype=torch.float32)
        self.register_buffer("visibility_grid_min", vis_range[:3].clone(), persistent=False)
        self.visibility_voxel_size = max(float(visibility_voxel_size), 1.0e-6)
        self.visibility_strong_ratio_hard = float(min(max(visibility_strong_ratio_hard, 1.0e-3), 1.0))
        self.visibility_strong_ratio_weight = float(min(max(visibility_strong_ratio_weight, 0.0), 1.0))
        self.visibility_weak_ratio_weight = float(min(max(visibility_weak_ratio_weight, 0.0), 1.0))
        self.visibility_soft_gate_min = float(min(max(visibility_soft_gate_min, 0.0), 1.0))
        self.visibility_gate_local_only = bool(visibility_gate_local_only)
        # Default false in V22-C1: a zero persistent render gate is cheaper and
        # avoids dynamic boolean compaction.  The option remains available.
        self.visibility_filter_hard_reject = bool(visibility_filter_hard_reject)
        self.visibility_encoder_gain_power = float(min(max(visibility_encoder_gain_power, 0.0), 1.0))
        self.visibility_debug = bool(visibility_debug)

        self.enable_object_completion = bool(enable_object_completion)
        dims = torch.as_tensor(class_typical_dims, dtype=torch.float32)
        if tuple(dims.shape) != (10, 3):
            raise ValueError("class_typical_dims must have shape [10,3]")
        self.register_buffer("class_typical_dims", dims.clamp_min(0.05), persistent=False)
        obj_min = tuple(max(int(x), 0) for x in object_children_min)
        obj_max = tuple(max(int(x), 0) for x in object_children_max)
        merge_r = tuple(max(float(x), 0.05) for x in object_merge_radii)
        if len(obj_min) != 10 or len(obj_max) != 10 or len(merge_r) != 10:
            raise ValueError("object child counts and merge radii must contain 10 values")
        self.object_children_min = obj_min
        self.object_children_max = tuple(max(a, b) for a, b in zip(obj_min, obj_max))
        self.object_merge_radii = merge_r
        # Backward-compatible constructor argument; V22-B never consumes OQG geometry.
        _ = object_center_oqg_blend
        self.object_scale_min = max(float(object_scale_min), 1.0e-4)
        self.object_scale_max = max(float(object_scale_max), self.object_scale_min)
        self.object_completion_opacity = float(min(max(object_completion_opacity, 1.0e-4), 1.0 - 1.0e-4))
        self.object_min_record_score = float(min(max(object_min_record_score, 0.0), 1.0))

        self.enable_child_quality = bool(enable_child_quality)
        self.occ_empty_label = int(occ_empty_label)
        self.child_quality_prior_strength = max(float(child_quality_prior_strength), 0.0)
        self.child_quality_nonempty_weight = max(float(child_quality_nonempty_weight), 0.0)
        self.child_quality_semantic_weight = max(float(child_quality_semantic_weight), 0.0)
        self.child_quality_decoupled = bool(child_quality_decoupled)
        if not self.child_quality_decoupled:
            raise ValueError("V22-C2 requires child_quality_decoupled=True")
        self.child_quality_semantic_on_nonempty_only = bool(child_quality_semantic_on_nonempty_only)
        self.child_quality_geometry_zero_threshold = float(
            min(max(child_quality_geometry_zero_threshold, 0.0), 1.0)
        )
        self.child_quality_semantic_zero_threshold = float(
            min(max(child_quality_semantic_zero_threshold, 0.0), 1.0)
        )
        self.child_quality_persistent_geometry_gate = bool(child_quality_persistent_geometry_gate)
        # Legacy floor fields are kept for old configs, but V22-C2 defaults all
        # three to zero so an unreliable candidate has no forced residual signal.
        self.child_quality_opacity_floor = float(min(max(child_quality_opacity_floor, 0.0), 1.0))
        self.child_quality_feature_floor = float(min(max(child_quality_feature_floor, 0.0), 1.0))
        self.child_quality_rank_floor = float(min(max(child_quality_rank_floor, 0.0), 1.0))

        quality_feat_dim = max(int(child_quality_feature_dim), 4)
        quality_cls_dim = max(int(child_quality_class_dim), 4)
        quality_kind_dim = max(int(child_quality_kind_dim), 4)
        quality_hidden_dim = max(int(child_quality_hidden_dim), 16)
        # 11 class IDs: 10 DAOcc object classes + one generic/unknown class.
        self.child_quality_feature_proj = nn.Sequential(
            nn.Linear(self.feature_dim, quality_feat_dim),
            nn.LayerNorm(quality_feat_dim),
            nn.SiLU(),
        )
        self.child_quality_class_embedding = nn.Embedding(11, quality_cls_dim)
        self.child_quality_kind_embedding = nn.Embedding(4, quality_kind_dim)
        quality_geom_dim = 11
        self.child_quality_mlp = nn.Sequential(
            nn.Linear(quality_geom_dim + quality_feat_dim + quality_cls_dim + quality_kind_dim, quality_hidden_dim),
            nn.SiLU(),
            nn.Linear(quality_hidden_dim, max(quality_hidden_dim // 2, 16)),
            nn.SiLU(),
        )
        quality_out_dim = max(quality_hidden_dim // 2, 16)
        self.child_nonempty_head = nn.Linear(quality_out_dim, 1)
        self.child_semantic_head = nn.Linear(quality_out_dim, 1)
        nn.init.normal_(self.child_quality_class_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.child_quality_kind_embedding.weight, mean=0.0, std=0.02)
        # Residual heads start at zero, so iteration 1 remains close to V12 and
        # uses only the conservative quality prior below.
        nn.init.zeros_(self.child_nonempty_head.weight)
        nn.init.zeros_(self.child_nonempty_head.bias)
        nn.init.zeros_(self.child_semantic_head.weight)
        nn.init.zeros_(self.child_semantic_head.bias)

        self.enable_generic_completion = bool(enable_generic_completion)
        self.generic_opacity_thr = float(min(max(generic_opacity_thr, 0.0), 1.0))
        self.generic_max_parent_scale = max(float(generic_max_parent_scale), 1.0e-4)
        self.generic_children_per_parent = max(int(generic_children_per_parent), 0)
        self.generic_child_offset_ratio = max(float(generic_child_offset_ratio), 0.0)
        self.generic_child_scale_ratio = max(float(generic_child_scale_ratio), 1.0e-3)
        self.generic_completion_opacity = float(min(max(generic_completion_opacity, 1.0e-4), 1.0 - 1.0e-4))

        self.enable_oqg_fallback = False  # V22-C2 runtime is strictly OQG-free
        self.fallback_score_thr = max(float(fallback_score_thr), 0.0)
        self.fallback_gaussians_per_box = max(int(fallback_gaussians_per_box), 1)
        self.fallback_scale_ratio = max(float(fallback_scale_ratio), 1.0e-4)
        self.fallback_min_scale = max(float(fallback_min_scale), 1.0e-4)
        self.fallback_max_scale = max(float(fallback_max_scale), self.fallback_min_scale)
        self.fallback_opacity = float(min(max(fallback_opacity, 1.0e-4), 1.0 - 1.0e-4))
        self.fallback_semantic_prior_logit = float(fallback_semantic_prior_logit)
        self.fallback_nms_default_radius = max(float(fallback_nms_default_radius), 0.0)
        radii = tuple(float(x) for x in fallback_nms_class_radii)
        if len(radii) != 10:
            raise ValueError("fallback_nms_class_radii must contain 10 values")
        self.fallback_nms_class_radii = radii

        self.add_completion_feature = bool(add_completion_feature)
        if self.add_completion_feature:
            self.completion_feature = nn.Parameter(torch.zeros(self.feature_dim, dtype=torch.float32))
            nn.init.normal_(self.completion_feature, mean=0.0, std=0.02)
        else:
            self.register_parameter("completion_feature", None)
        self.generic_completion_feature = nn.Parameter(torch.zeros(self.feature_dim, dtype=torch.float32))
        nn.init.normal_(self.generic_completion_feature, mean=0.0, std=0.02)
        self.class_feature = nn.Embedding(10, self.feature_dim)
        nn.init.normal_(self.class_feature.weight, mean=0.0, std=0.02)

        self.debug = bool(debug)
        self.debug_interval = max(int(debug_interval), 1)
        self.debug_train_max_print = max(int(debug_train_max_print), 0)
        self.debug_eval_max_print = max(int(debug_eval_max_print), 0)
        self._call_count = 0
        self._debug_train_print_count = 0
        self._debug_eval_print_count = 0

        patterns = torch.tensor(
            [
                [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                [1.0, 1.0, 0.0], [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
                [0.7, 0.0, 0.7], [-0.7, 0.0, 0.7],
            ],
            dtype=torch.float32,
        )
        patterns = patterns / patterns.norm(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("child_patterns", patterns, persistent=False)

        # Farthest-point ordering over an object-normalized grid.  Prefixes stay
        # spatially distributed for any dynamic per-object child count.
        xs = torch.linspace(-0.45, 0.45, steps=6)
        ys = torch.linspace(-0.45, 0.45, steps=5)
        zs = torch.linspace(-0.40, 0.40, steps=4)
        grid = torch.cartesian_prod(xs, ys, zs).to(torch.float32)
        chosen = [int(torch.argmin(grid.square().sum(dim=-1)).item())]
        min_dist = torch.cdist(grid, grid[chosen]).amin(dim=1)
        while len(chosen) < int(grid.shape[0]):
            nxt = int(torch.argmax(min_dist).item())
            chosen.append(nxt)
            min_dist = torch.minimum(min_dist, torch.linalg.norm(grid - grid[nxt], dim=-1))
        self.register_buffer(
            "object_patterns",
            grid[torch.tensor(chosen, dtype=torch.long)],
            persistent=False,
        )

    @staticmethod
    def _is_rank0() -> bool:
        try:
            import torch.distributed as dist
            return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        except Exception:
            return int(os.environ.get("RANK", "0")) == 0

    @staticmethod
    def _get_projection(metas: Optional[Dict[str, Any]], device: torch.device, dtype: torch.dtype) -> Tensor:
        if not isinstance(metas, dict) or "projection_mat" not in metas:
            raise RuntimeError("ObjectGaussianSeedCompletionHead requires metas['projection_mat']")
        proj = metas["projection_mat"]
        if not isinstance(proj, Tensor):
            proj = torch.as_tensor(proj)
        proj = proj.to(device=device, dtype=dtype)
        if proj.ndim == 3:
            proj = proj.unsqueeze(0)
        if proj.ndim != 4 or proj.shape[-2:] != (4, 4):
            raise RuntimeError(f"projection_mat must be [B,Cam,4,4], got {tuple(proj.shape)}")
        return proj

    @staticmethod
    def _project_points(points_xyz: Tensor, proj: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if points_xyz.numel() == 0:
            empty = points_xyz.new_empty((0,))
            return empty, empty, empty
        ones = torch.ones((points_xyz.shape[0], 1), device=points_xyz.device, dtype=points_xyz.dtype)
        hom = torch.cat([points_xyz[:, :3], ones], dim=-1)
        img_h = hom @ proj.transpose(0, 1)
        depth = img_h[:, 2]
        safe = depth.clamp_min(1.0e-4)
        return img_h[:, 0] / safe, img_h[:, 1] / safe, depth

    def _anchor_xyz_scale_opacity(self, anchor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r = self.pc_range_tensor.to(device=anchor.device, dtype=anchor.dtype)
        sr = self.scale_range_tensor.to(device=anchor.device, dtype=anchor.dtype)
        xyz01 = anchor[:, 0:3].sigmoid()
        xyz = torch.stack(
            [
                xyz01[:, 0] * (r[3] - r[0]) + r[0],
                xyz01[:, 1] * (r[4] - r[1]) + r[1],
                xyz01[:, 2] * (r[5] - r[2]) + r[2],
            ],
            dim=-1,
        )
        s01 = anchor[:, 3:6].sigmoid()
        scales = s01 * (sr[1] - sr[0]) + sr[0]
        if self.include_opa and anchor.shape[-1] > 10:
            opacity = anchor[:, 10].sigmoid()
        else:
            opacity = torch.ones((anchor.shape[0],), device=anchor.device, dtype=anchor.dtype)
        return xyz, scales.clamp_min(self.min_gaussian_scale), opacity.clamp(0.0, 1.0)

    def _xyz_to_anchor_logit(self, xyz: Tensor) -> Tensor:
        r = self.pc_range_tensor.to(device=xyz.device, dtype=xyz.dtype)
        xyz01 = torch.stack(
            [
                (xyz[:, 0] - r[0]) / (r[3] - r[0]).clamp_min(1.0e-6),
                (xyz[:, 1] - r[1]) / (r[4] - r[1]).clamp_min(1.0e-6),
                (xyz[:, 2] - r[2]) / (r[5] - r[2]).clamp_min(1.0e-6),
            ],
            dim=-1,
        ).clamp(1.0e-4, 1.0 - 1.0e-4)
        return safe_inverse_sigmoid(xyz01)

    def _scale_to_anchor_logit(self, scales: Tensor) -> Tensor:
        sr = self.scale_range_tensor.to(device=scales.device, dtype=scales.dtype)
        scales = scales.clamp(min=max(float(sr[0]), 1.0e-6), max=float(sr[1]))
        sn = (scales - sr[0]) / (sr[1] - sr[0]).clamp_min(1.0e-6)
        return safe_inverse_sigmoid(sn.clamp(1.0e-4, 1.0 - 1.0e-4))

    def _query_visibility_neighborhood_maps(
        self,
        points: Tensor,
        state_map: Optional[Tensor],
        strong_any_map: Optional[Tensor],
        strong_ratio_map: Optional[Tensor],
        weak_ratio_map: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample precomputed CUDA maps with one flattened gather per map.

        All maps are built once in OctreeGaussianLifter using max_pool3d /
        avg_pool3d.  This function performs no CPU conversion, no .item(), no
        search/sort, and no per-Gaussian Python loop.
        """
        n = int(points.shape[0])
        device = points.device
        states = torch.full((n,), int(UNKNOWN), device=device, dtype=torch.long)
        strong_any = torch.zeros((n,), device=device, dtype=torch.bool)
        strong_ratio = torch.zeros((n,), device=device, dtype=torch.float32)
        weak_ratio = torch.zeros((n,), device=device, dtype=torch.float32)
        valid = torch.zeros((n,), device=device, dtype=torch.bool)
        if n == 0 or not isinstance(state_map, Tensor) or state_map.ndim != 3:
            return states, strong_any, strong_ratio, weak_ratio, valid

        # Maps should already be on the representation device.  The fallback
        # transfers below are non-blocking and occur at most once per sample.
        state_map = state_map.to(device=device, non_blocking=True).contiguous()
        if isinstance(strong_any_map, Tensor):
            strong_any_map = strong_any_map.to(device=device, non_blocking=True).contiguous()
        if isinstance(strong_ratio_map, Tensor):
            strong_ratio_map = strong_ratio_map.to(device=device, non_blocking=True).contiguous()
        if isinstance(weak_ratio_map, Tensor):
            weak_ratio_map = weak_ratio_map.to(device=device, non_blocking=True).contiguous()

        grid_min = self.visibility_grid_min.to(device=device, dtype=torch.float32)
        idx = torch.floor(
            (points.detach().to(torch.float32) - grid_min.view(1, 3))
            / self.visibility_voxel_size
        ).to(torch.long)
        sx, sy, sz = state_map.shape
        valid = (
            (idx[:, 0] >= 0) & (idx[:, 0] < sx)
            & (idx[:, 1] >= 0) & (idx[:, 1] < sy)
            & (idx[:, 2] >= 0) & (idx[:, 2] < sz)
        )
        safe_x = idx[:, 0].clamp(0, sx - 1)
        safe_y = idx[:, 1].clamp(0, sy - 1)
        safe_z = idx[:, 2].clamp(0, sz - 1)
        flat_idx = (safe_x * sy + safe_y) * sz + safe_z

        states_all = torch.gather(state_map.reshape(-1).to(torch.long), 0, flat_idx)
        states = torch.where(valid, states_all, states)
        if isinstance(strong_any_map, Tensor) and strong_any_map.shape == state_map.shape:
            any_all = torch.gather(strong_any_map.reshape(-1).to(torch.uint8), 0, flat_idx) > 0
            strong_any = valid & any_all
        if isinstance(strong_ratio_map, Tensor) and strong_ratio_map.shape == state_map.shape:
            ratio_all = torch.gather(strong_ratio_map.reshape(-1), 0, flat_idx).to(torch.float32)
            strong_ratio = torch.where(valid, ratio_all, strong_ratio)
        if isinstance(weak_ratio_map, Tensor) and weak_ratio_map.shape == state_map.shape:
            weak_all = torch.gather(weak_ratio_map.reshape(-1), 0, flat_idx).to(torch.float32)
            weak_ratio = torch.where(valid, weak_all, weak_ratio)
        return states, strong_any, strong_ratio, weak_ratio, valid

    def _local_visibility_gate(
        self,
        child_xyz: Tensor,
        child_scales: Tensor,
        child_quat: Tensor,
        child_labels: Tensor,
        state_map: Optional[Tensor],
        strong_any_map: Optional[Tensor],
        strong_ratio_map: Optional[Tensor],
        weak_ratio_map: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """V22-C1 center-hard / neighborhood-soft Local visibility gate.

        The precomputed maps are queried entirely on CUDA.  Only a Local child
        whose CENTER lies in STRONG_FREE is assigned a zero persistent gate.
        Neighborhood strong/weak ratios are bounded soft evidence and cannot
        hard-reject a child.  ``strong_any`` is retained for diagnostics only.
        """
        del child_scales, child_quat
        total = int(child_xyz.shape[0])
        device = child_xyz.device
        gate = torch.ones((total,), device=device, dtype=torch.float32)
        keep = torch.ones((total,), device=device, dtype=torch.bool)

        zero = torch.zeros((), device=device, dtype=torch.long)
        zero_f = torch.zeros((), device=device, dtype=torch.float32)
        zero_stats = {
            'proposed': torch.full((), total, device=device, dtype=torch.long),
            'rejected': zero, 'strong_center': zero, 'strong_any': zero,
            'strong_ratio_mean': zero_f, 'weak_mean': zero_f,
            'unknown_mean': zero_f, 'car_rejected': zero, 'bus_rejected': zero,
        }
        if total == 0 or not self.enable_visibility_gate or state_map is None:
            return gate, keep, zero_stats

        with torch.no_grad():
            states, strong_any, strong_ratio, weak_ratio, valid = (
                self._query_visibility_neighborhood_maps(
                    child_xyz, state_map, strong_any_map,
                    strong_ratio_map, weak_ratio_map,
                )
            )
            center_strong = valid & (states == int(STRONG_FREE))
            center_unknown = (~valid) | (states == int(UNKNOWN))

            # V22-C1: neighborhood evidence is deliberately soft.  A real
            # vehicle-surface Gaussian normally touches free space, so neither
            # ``strong_any`` nor a high neighborhood ratio is a hard conflict.
            # Only a CENTER located in STRONG_FREE is physically contradictory.
            soft_gate = (
                (1.0 - self.visibility_strong_ratio_weight * strong_ratio).clamp(0.0, 1.0)
                * (1.0 - self.visibility_weak_ratio_weight * weak_ratio).clamp(0.0, 1.0)
            ).clamp(self.visibility_soft_gate_min, 1.0)
            gate = torch.where(
                center_strong,
                torch.zeros_like(soft_gate),
                soft_gate,
            )
            if self.visibility_filter_hard_reject:
                keep = ~center_strong

            rejected = center_strong
            denom = float(max(total, 1))
            stats = {
                'proposed': torch.full((), total, device=device, dtype=torch.long),
                'rejected': rejected.sum().to(torch.long),
                'strong_center': center_strong.sum().to(torch.long),
                'strong_any': strong_any.sum().to(torch.long),
                'strong_ratio_mean': strong_ratio.sum() / denom,
                'weak_mean': weak_ratio.sum() / denom,
                'unknown_mean': center_unknown.to(torch.float32).sum() / denom,
                'car_rejected': (rejected & (child_labels == 0)).sum().to(torch.long),
                'bus_rejected': (rejected & (child_labels == 3)).sum().to(torch.long),
            }
        return gate, keep, stats

    @staticmethod
    def _as_gt_list(
        value: Any,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Optional[Tensor]]:
        if value is None:
            return [None] * batch_size
        if isinstance(value, (list, tuple)):
            out: List[Optional[Tensor]] = []
            for b in range(batch_size):
                if b >= len(value) or value[b] is None:
                    out.append(None)
                    continue
                item = value[b].tensor if hasattr(value[b], "tensor") else value[b]
                if not isinstance(item, Tensor):
                    try:
                        item = torch.as_tensor(item)
                    except Exception:
                        out.append(None)
                        continue
                out.append(item.to(device=device, dtype=dtype))
            return out
        item = value.tensor if hasattr(value, "tensor") else value
        if isinstance(item, Tensor) and item.ndim >= 3 and int(item.shape[0]) == batch_size:
            return [item[b].to(device=device, dtype=dtype) for b in range(batch_size)]
        if isinstance(item, Tensor) and batch_size == 1:
            return [item.to(device=device, dtype=dtype)]
        return [None] * batch_size

    @staticmethod
    def _as_batched_tensor(
        value: Any,
        batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        item_ndim: int = 3,
    ) -> Optional[Tensor]:
        if value is None:
            return None
        tensor: Optional[Tensor] = None
        if isinstance(value, Tensor):
            tensor = value
        elif isinstance(value, (list, tuple)):
            items: List[Tensor] = []
            for item in value[:batch_size]:
                if item is None:
                    return None
                if not isinstance(item, Tensor):
                    try:
                        item = torch.as_tensor(item)
                    except Exception:
                        return None
                items.append(item)
            if not items:
                return None
            try:
                tensor = torch.stack(items, dim=0)
            except Exception:
                return None
        else:
            try:
                tensor = torch.as_tensor(value)
            except Exception:
                return None
        if tensor is None:
            return None
        if tensor.ndim == item_ndim:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != item_ndim + 1:
            return None
        if tensor.shape[0] < batch_size:
            return None
        return tensor[:batch_size].to(device=device, dtype=dtype, non_blocking=True)

    def _extract_occ_tensors(
        self,
        metas: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        occ_label = kwargs.get("occ_label", None)
        occ_xyz = kwargs.get("occ_xyz", None)
        if isinstance(metas, dict):
            occ_label = metas.get("occ_label", occ_label)
            occ_xyz = metas.get("occ_xyz", occ_xyz)
        labels = self._as_batched_tensor(
            occ_label,
            batch_size,
            device,
            dtype=torch.long,
            item_ndim=3,
        )
        xyz = self._as_batched_tensor(
            occ_xyz,
            batch_size,
            device,
            dtype=torch.float32,
            item_ndim=4,
        )
        if xyz is not None and xyz.shape[-1] != 3:
            xyz = None
        return labels, xyz

    @staticmethod
    def _occ_grid_parameters(occ_label: Tensor, occ_xyz: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        origin = occ_xyz[0, 0, 0]
        sx = (
            (occ_xyz[1, 0, 0, 0] - occ_xyz[0, 0, 0, 0]).abs()
            if occ_xyz.shape[0] > 1
            else occ_xyz.new_tensor(0.4)
        )
        sy = (
            (occ_xyz[0, 1, 0, 1] - occ_xyz[0, 0, 0, 1]).abs()
            if occ_xyz.shape[1] > 1
            else occ_xyz.new_tensor(0.4)
        )
        sz = (
            (occ_xyz[0, 0, 1, 2] - occ_xyz[0, 0, 0, 2]).abs()
            if occ_xyz.shape[2] > 1
            else occ_xyz.new_tensor(0.4)
        )
        step = torch.stack([sx, sy, sz]).clamp_min(1.0e-6)
        shape = torch.tensor(occ_label.shape[:3], device=occ_label.device, dtype=torch.long)
        return origin, step, shape

    def _lookup_occ_labels(
        self,
        points: Tensor,
        occ_label: Optional[Tensor],
        occ_xyz: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        labels = torch.full(
            (points.shape[0],),
            self.occ_empty_label,
            device=points.device,
            dtype=torch.long,
        )
        valid = torch.zeros((points.shape[0],), device=points.device, dtype=torch.bool)
        if points.numel() == 0 or occ_label is None or occ_xyz is None:
            return labels, valid
        origin, step, shape = self._occ_grid_parameters(occ_label, occ_xyz)
        ijk = torch.round((points.to(torch.float32) - origin) / step).long()
        valid = (
            (ijk[:, 0] >= 0)
            & (ijk[:, 0] < shape[0])
            & (ijk[:, 1] >= 0)
            & (ijk[:, 1] < shape[1])
            & (ijk[:, 2] >= 0)
            & (ijk[:, 2] < shape[2])
        )
        if bool(valid.any()):
            iv = ijk[valid]
            labels[valid] = occ_label[iv[:, 0], iv[:, 1], iv[:, 2]].long()
        return labels, valid

    @staticmethod
    def _balanced_bce(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        if logits.numel() == 0 or not bool(mask.any()):
            return logits.sum() * 0.0
        logits = logits[mask]
        targets = targets[mask].to(dtype=logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        positive = targets >= 0.5
        negative = ~positive
        pieces: List[Tensor] = []
        if bool(positive.any()):
            pieces.append(loss[positive].mean())
        if bool(negative.any()):
            pieces.append(loss[negative].mean())
        return torch.stack(pieces).mean() if pieces else logits.sum() * 0.0

    def _child_quality_logits(
        self,
        anchors: Tensor,
        features: Tensor,
        base_quality: Tensor,
        dao_labels: Tensor,
        kinds: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        xyz, scales, opacity = self._anchor_xyz_scale_opacity(anchors)
        r = self.pc_range_tensor.to(device=xyz.device, dtype=xyz.dtype)
        xyz_norm = torch.stack(
            [
                (xyz[:, 0] - r[0]) / (r[3] - r[0]).clamp_min(1.0e-6),
                (xyz[:, 1] - r[1]) / (r[4] - r[1]).clamp_min(1.0e-6),
                (xyz[:, 2] - r[2]) / (r[5] - r[2]).clamp_min(1.0e-6),
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        radial = torch.linalg.norm(xyz[:, :2], dim=-1) / 60.0
        known_class = (dao_labels >= 0).to(torch.float32)
        geom = torch.cat(
            [
                xyz_norm,
                (scales / 0.60).clamp(0.0, 4.0),
                opacity[:, None],
                base_quality.to(torch.float32).clamp(0.0, 1.5)[:, None],
                radial.clamp(0.0, 2.0)[:, None],
                xyz_norm[:, 2:3],
                known_class[:, None],
            ],
            dim=-1,
        )
        feat = self.child_quality_feature_proj(features.to(torch.float32))
        cls_index = torch.where(
            (dao_labels >= 0) & (dao_labels < 10),
            dao_labels,
            torch.full_like(dao_labels, 10),
        )
        cls = self.child_quality_class_embedding(cls_index)
        kind = self.child_quality_kind_embedding(kinds.clamp(0, 3))
        hidden = self.child_quality_mlp(torch.cat([geom, feat, cls, kind], dim=-1))

        # Conservative fixed priors preserve V12 at initialization.  The MLP
        # heads learn only residual corrections around them.
        q = base_quality.to(torch.float32).clamp(0.0, 1.0)
        nonempty_prior = (0.55 + 0.35 * q).clamp(0.55, 0.92)
        semantic_prior = (0.50 + 0.38 * q).clamp(0.50, 0.90)
        nonempty_logit = self.child_nonempty_head(hidden).squeeze(-1)
        semantic_logit = self.child_semantic_head(hidden).squeeze(-1)
        nonempty_logit = nonempty_logit + self.child_quality_prior_strength * safe_inverse_sigmoid(nonempty_prior)
        semantic_logit = semantic_logit + self.child_quality_prior_strength * safe_inverse_sigmoid(semantic_prior)
        return nonempty_logit, semantic_logit, xyz, opacity

    def _zero_child_quality_loss(self) -> Tensor:
        zero = self.child_quality_feature_proj[0].weight.sum() * 0.0
        zero = zero + self.child_quality_class_embedding.weight.sum() * 0.0
        zero = zero + self.child_quality_kind_embedding.weight.sum() * 0.0
        zero = zero + self.child_quality_mlp[0].weight.sum() * 0.0
        zero = zero + self.child_nonempty_head.weight.sum() * 0.0
        zero = zero + self.child_semantic_head.weight.sum() * 0.0
        return zero

    @staticmethod
    def _inside_oriented_box(points: Tensor, box: Tensor, expand: float = 1.10) -> Tensor:
        if points.numel() == 0 or box.numel() < 7:
            return torch.zeros((points.shape[0],), device=points.device, dtype=torch.bool)
        center = box[:3]
        dims = box[3:6].abs().clamp_min(0.05) * float(expand)
        yaw = box[6]
        diff = points[:, :3] - center.view(1, 3)
        c, s = torch.cos(yaw), torch.sin(yaw)
        local_x = diff[:, 0] * c + diff[:, 1] * s
        local_y = -diff[:, 0] * s + diff[:, 1] * c
        local_z = diff[:, 2]
        half = 0.5 * dims
        return (
            (local_x.abs() <= half[0])
            & (local_y.abs() <= half[1])
            & (local_z.abs() <= half[2])
        )

    def _depth_clusters(self, depth: Tensor) -> List[Tensor]:
        if depth.numel() == 0:
            return []
        order = torch.argsort(depth)
        sd = depth[order]
        if sd.numel() == 1:
            return [order]
        gaps = sd[1:] - sd[:-1]
        threshold = self.depth_cluster_gap_abs + self.depth_cluster_gap_rel * sd[:-1].clamp_min(0.0)
        split = torch.where(gaps > threshold)[0] + 1
        bounds = [0] + [int(x.item()) for x in split[: self.max_depth_clusters - 1]] + [int(sd.numel())]
        clusters: List[Tensor] = []
        for i in range(len(bounds) - 1):
            if bounds[i + 1] > bounds[i]:
                clusters.append(order[bounds[i] : bounds[i + 1]])
        return clusters[: self.max_depth_clusters]

    def _best_cluster_prior(
        self,
        u: Tensor,
        v: Tensor,
        depth: Tensor,
        core: Tensor,
        opacity: Tensor,
        scales: Tensor,
        box: Tensor,
    ) -> Tuple[Tensor, Tensor, int]:
        """Return the best depth cluster without any OQG signal."""
        clusters = self._depth_clusters(depth)
        if not clusters:
            mask = torch.ones_like(depth, dtype=torch.bool)
            return mask, depth.median(), 0
        x1, y1, x2, y2 = box
        bw = (x2 - x1).clamp_min(1.0)
        bh = (y2 - y1).clamp_min(1.0)
        cu, cv = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        best_score = -1.0
        best_cluster = clusters[0]
        for cluster in clusters:
            weight = opacity[cluster].clamp_min(0.05)
            support = 1.0 - math.exp(-float(weight.sum().detach().item()) / self.support_norm)
            core_ratio = float((weight * core[cluster].float()).sum().detach().item() / weight.sum().clamp_min(1.0e-6).detach().item())
            du = (u[cluster] - cu) / (0.5 * bw).clamp_min(1.0)
            dv = (v[cluster] - cv) / (0.5 * bh).clamp_min(1.0)
            centrality = float(torch.exp(-0.5 * (du.square() + dv.square())).mean().detach().item())
            opacity_mean = float(opacity[cluster].mean().detach().item())
            fine = float(torch.exp(-scales[cluster].amax(dim=-1).mean() / 0.8).detach().item())
            # +0.075 preserves the old no-OQG baseline offset (0.15 * 0.5)
            # while removing all dependence on a predicted OQG depth.
            score = support * (0.35 * core_ratio + 0.20 * centrality + 0.20 * opacity_mean + 0.10 * fine + 0.075)
            if score > best_score:
                best_score = score
                best_cluster = cluster
        mask = torch.zeros_like(depth, dtype=torch.bool)
        mask[best_cluster] = True
        return mask, depth[best_cluster].median(), len(clusters)

    def _owner_logits(
        self,
        gaussian_features: Tensor,
        labels: Tensor,
        u: Tensor,
        v: Tensor,
        depth: Tensor,
        core: Tensor,
        best_cluster: Tensor,
        reference_depth: Tensor,
        opacity: Tensor,
        scales: Tensor,
        box: Tensor,
        box_score: float,
    ) -> Tuple[Tensor, Tensor]:
        x1, y1, x2, y2 = box
        bw = (x2 - x1).clamp_min(1.0)
        bh = (y2 - y1).clamp_min(1.0)
        cu, cv = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        du = (u - cu) / (0.5 * bw).clamp_min(1.0)
        dv = (v - cv) / (0.5 * bh).clamp_min(1.0)
        left = (u - x1) / bw
        right = (x2 - u) / bw
        top = (v - y1) / bh
        bottom = (y2 - v) / bh
        edge = torch.stack([left, right, top, bottom], dim=-1).amin(dim=-1).clamp(0.0, 0.5) * 2.0
        depth_delta = (depth - reference_depth) / (1.0 + 0.10 * reference_depth.abs())
        scale_max = scales.amax(dim=-1)
        scale_mean = scales.mean(dim=-1)
        box_score_t = depth.new_full(depth.shape, float(box_score))
        geom = torch.stack(
            [
                du, dv, du.abs(), dv.abs(), edge,
                depth / 60.0,
                depth_delta.clamp(-4.0, 4.0),
                depth_delta.abs().clamp(0.0, 4.0),
                opacity,
                (scale_max / 0.8).clamp(0.0, 4.0),
                (scale_mean / 0.8).clamp(0.0, 4.0),
                core.float(),
                best_cluster.float(),
                box_score_t,
            ],
            dim=-1,
        )
        feat = self.owner_feature_proj(gaussian_features.to(torch.float32))
        cls = self.owner_class_embedding(labels.clamp(0, 9))
        residual = self.owner_mlp(torch.cat([geom, feat, cls], dim=-1)).squeeze(-1)

        centrality = torch.exp(-0.5 * (du.square() + dv.square()))
        depth_prior = torch.exp(-depth_delta.abs())
        prior = (
            0.055  # old 0.03 base + neutral no-OQG contribution 0.025
            + 0.12 * core.float()
            + 0.18 * best_cluster.float()
            + 0.12 * centrality
            + 0.10 * opacity
            + 0.10 * depth_prior
        ).clamp(0.03, 0.90)
        prior_logit = safe_inverse_sigmoid(prior)
        return residual + self.owner_prior_strength * prior_logit, prior

    def _owner_focal_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        target = target.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = logits.sigmoid()
        pt = target * prob + (1.0 - target) * (1.0 - prob)
        alpha_t = target * self.owner_focal_alpha + (1.0 - target) * (1.0 - self.owner_focal_alpha)
        loss = alpha_t * (1.0 - pt).pow(self.owner_focal_gamma) * bce
        pos = target > 0.5
        neg = ~pos
        pieces: List[Tensor] = []
        if bool(pos.any()):
            pieces.append(loss[pos].mean())
        if bool(neg.any()):
            pieces.append(loss[neg].mean())
        if not pieces:
            return logits.sum() * 0.0
        return torch.stack(pieces).mean()

    def _children_per_parent(self, labels: Tensor) -> Tensor:
        out = torch.full_like(labels, self.seed_children_per_parent, dtype=torch.long)
        for cls in self.SMALL_CLASS_IDS:
            out = torch.where(labels == int(cls), torch.full_like(out, self.small_seed_children_per_parent), out)
        for cls in self.LARGE_CLASS_IDS:
            out = torch.where(labels == int(cls), torch.full_like(out, self.large_seed_children_per_parent), out)
        return out.clamp_min(0)

    def _build_seed_children(
        self,
        parent_anchor: Tensor,
        parent_xyz: Tensor,
        parent_scales: Tensor,
        parent_labels: Tensor,
        parent_scores: Tensor,
        parent_gt_idx: Tensor,
        visibility_state_map: Optional[Tensor],
        visibility_strong_any_map: Optional[Tensor],
        visibility_strong_ratio_map: Optional[Tensor],
        visibility_weak_ratio_map: Optional[Tensor],
        anchor_dim: int,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        device = parent_anchor.device
        empty_a = torch.empty((0, anchor_dim), device=device, dtype=dtype)
        empty_i = torch.empty((0,), device=device, dtype=torch.long)
        empty_q = torch.empty((0,), device=device, dtype=torch.float32)
        zero_l = torch.zeros((), device=device, dtype=torch.long)
        zero_f = torch.zeros((), device=device, dtype=torch.float32)
        zero_stats = {
            'proposed': zero_l, 'rejected': zero_l,
            'strong_center': zero_l, 'strong_any': zero_l,
            'strong_ratio_mean': zero_f, 'weak_mean': zero_f,
            'unknown_mean': zero_f, 'car_rejected': zero_l, 'bus_rejected': zero_l,
        }
        if parent_anchor.numel() == 0 or not self.generate_seed_children:
            return empty_a, empty_i, empty_i, empty_q, empty_i, empty_q, zero_stats
        counts = self._children_per_parent(parent_labels)
        total = int(counts.sum().item())
        if total <= 0:
            return empty_a, empty_i, empty_i, empty_q, empty_i, empty_q, zero_stats
        pidx = torch.repeat_interleave(torch.arange(parent_anchor.shape[0], device=device), counts)
        starts = torch.cumsum(counts, dim=0) - counts
        kord = torch.arange(total, device=device) - torch.repeat_interleave(starts, counts)
        patterns = self.child_patterns.to(device=device, dtype=parent_xyz.dtype)
        pat = patterns[kord % patterns.shape[0]]
        base_scale = parent_scales[pidx].clamp(self.min_gaussian_scale, self.max_gaussian_scale)
        offset_scale = base_scale.clamp(min=0.08, max=0.35)
        child_xyz = parent_xyz[pidx] + pat * offset_scale * self.child_offset_ratio
        r = self.pc_range_tensor.to(device=device, dtype=child_xyz.dtype)
        child_xyz[:, 0] = child_xyz[:, 0].clamp(r[0] + 1.0e-3, r[3] - 1.0e-3)
        child_xyz[:, 1] = child_xyz[:, 1].clamp(r[1] + 1.0e-3, r[4] - 1.0e-3)
        child_xyz[:, 2] = child_xyz[:, 2].clamp(r[2] + 1.0e-3, r[5] - 1.0e-3)
        child_scale = (base_scale * self.child_scale_ratio).clamp(self.min_gaussian_scale, self.max_gaussian_scale)
        labels = parent_labels[pidx]
        scores = parent_scores[pidx].clamp(0.0, 1.0)
        child_quat = parent_anchor[pidx, 6:10] if parent_anchor.shape[1] >= 10 else torch.zeros((total, 4), device=device, dtype=dtype)
        if child_quat.shape[1] == 4 and parent_anchor.shape[1] < 10:
            child_quat[:, 0] = 1.0

        render_gate, keep, vis_stats = self._local_visibility_gate(
            child_xyz, child_scale, child_quat, labels,
            visibility_state_map, visibility_strong_any_map,
            visibility_strong_ratio_map, visibility_weak_ratio_map,
        )
        if self.visibility_filter_hard_reject:
            pidx = pidx[keep]
            child_xyz = child_xyz[keep]
            child_scale = child_scale[keep]
            child_quat = child_quat[keep]
            labels = labels[keep]
            scores = scores[keep]
            render_gate = render_gate[keep]
        total = int(child_xyz.shape[0])
        if total <= 0:
            return empty_a, empty_i, empty_i, empty_q, empty_i, empty_q, vis_stats

        encoder_gain = render_gate.clamp(0.0, 1.0).pow(self.visibility_encoder_gain_power)
        anchor = torch.zeros((total, anchor_dim), device=device, dtype=dtype)
        anchor[:, 0:3] = self._xyz_to_anchor_logit(child_xyz).to(dtype=dtype)
        anchor[:, 3:6] = self._scale_to_anchor_logit(child_scale).to(dtype=dtype)
        if anchor_dim >= 10:
            anchor[:, 6:10] = child_quat.to(dtype=dtype)
        else:
            anchor[:, 6] = 1.0
        if self.include_opa and anchor_dim > 10:
            opacity = (self.completion_opacity * (0.60 + 0.40 * scores) * encoder_gain).clamp(1.0e-4, 1.0 - 1.0e-4)
            anchor[:, 10] = safe_inverse_sigmoid(opacity.to(dtype=dtype))
        sem_start = 11
        if self.semantic_dim > 0 and anchor_dim >= sem_start + self.semantic_dim:
            occ_idx = self.dao_to_occ_sem.to(device=device)[labels.clamp(0, 9)]
            valid_sem = (occ_idx >= 0) & (occ_idx < self.semantic_dim)
            anchor[valid_sem, sem_start + occ_idx[valid_sem]] = (
                self.owner_semantic_prior_logit * encoder_gain[valid_sem]
            ).to(dtype=dtype)
        quality = ((0.45 + 0.55 * scores) * render_gate).clamp(0.0, 1.0)
        return anchor, labels, pidx, quality, parent_gt_idx[pidx], render_gate, vis_stats

    def _merge_object_records(
        self,
        records: List[Dict[str, Any]],
        measured_xyz: Tensor,
    ) -> List[Dict[str, Any]]:
        if len(records) <= 1:
            return records
        records = sorted(records, key=lambda r: float(r["score"]), reverse=True)
        merged: List[Dict[str, Any]] = []
        for rec in records:
            idx = rec["parent_idx"]
            if idx.numel() == 0:
                continue
            weights = rec["parent_scores"].clamp_min(1.0e-4)
            center = (measured_xyz[idx] * weights[:, None]).sum(dim=0) / weights.sum()
            target: Optional[Dict[str, Any]] = None
            for existing in merged:
                if int(existing["label"]) != int(rec["label"]):
                    continue
                ex_idx = existing["parent_idx"]
                shared = bool(torch.isin(idx, ex_idx).any()) if hasattr(torch, "isin") else False
                ex_w = existing["parent_scores"].clamp_min(1.0e-4)
                ex_center = (measured_xyz[ex_idx] * ex_w[:, None]).sum(dim=0) / ex_w.sum()
                radius = self.object_merge_radii[min(max(int(rec["label"]), 0), 9)]
                if shared or float(torch.linalg.norm(center[:2] - ex_center[:2]).item()) <= radius:
                    target = existing
                    break
            if target is None:
                merged.append(dict(rec))
                continue
            score_map: Dict[int, float] = {}
            for pi, ps in zip(target["parent_idx"], target["parent_scores"]):
                score_map[int(pi.item())] = float(ps.item())
            for pi, ps in zip(idx, rec["parent_scores"]):
                p = int(pi.item())
                score_map[p] = max(score_map.get(p, 0.0), float(ps.item()))
            ordered = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
            target["parent_idx"] = torch.tensor([x[0] for x in ordered], device=idx.device, dtype=torch.long)
            target["parent_scores"] = torch.tensor([x[1] for x in ordered], device=idx.device, dtype=torch.float32)
            if float(rec["score"]) > float(target["score"]):
                target["score"] = rec["score"]
                target["gt_idx"] = rec.get("gt_idx", -1)
        return merged

    def _build_object_children(
        self,
        records: List[Dict[str, Any]],
        measured_xyz: Tensor,
        anchor_dim: int,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        device = measured_xyz.device
        empty_a = torch.empty((0, anchor_dim), device=device, dtype=dtype)
        empty_i = torch.empty((0,), device=device, dtype=torch.long)
        empty_q = torch.empty((0,), device=device, dtype=torch.float32)
        if not self.enable_object_completion or not records:
            return empty_a, empty_i, empty_i, empty_q, empty_i
        anchors: List[Tensor] = []
        labels_out: List[Tensor] = []
        reps_out: List[Tensor] = []
        quality_out: List[Tensor] = []
        gt_out: List[Tensor] = []
        patterns = self.object_patterns.to(device=device, dtype=torch.float32)
        r = self.pc_range_tensor.to(device=device, dtype=torch.float32)
        for rec in records:
            rec_score = float(rec["score"])
            if rec_score < self.object_min_record_score:
                continue
            idx = rec["parent_idx"]
            if idx.numel() == 0:
                continue
            label = min(max(int(rec["label"]), 0), 9)
            ps = rec["parent_scores"].to(device=device, dtype=torch.float32).clamp_min(1.0e-4)
            seeds = measured_xyz[idx].to(torch.float32)
            seed_center = (seeds * ps[:, None]).sum(dim=0) / ps.sum()
            typical = self.class_typical_dims[label].to(device=device).clone()
            dims = typical.clone()

            # OQG-free rollback path: infer yaw only from owned measured seeds.
            yaw = seeds.new_tensor(0.0)
            if seeds.shape[0] >= 3:
                xy = seeds[:, :2] - seed_center[:2].view(1, 2)
                weighted = xy * ps[:, None].sqrt()
                cov = weighted.transpose(0, 1) @ weighted / ps.sum().clamp_min(1.0)
                eigvals, eigvecs = torch.linalg.eigh(cov)
                ratio = float((eigvals[-1] / eigvals[-2].clamp_min(1.0e-5)).item())
                if ratio >= 1.60 and float(eigvals[-1].sqrt().item()) >= 0.25:
                    vec = eigvecs[:, -1]
                    pca_yaw = torch.atan2(vec[1], vec[0])
                    yaw = pca_yaw

            c, s = torch.cos(yaw), torch.sin(yaw)
            diff = seeds - seed_center.view(1, 3)
            local = torch.stack(
                [
                    diff[:, 0] * c + diff[:, 1] * s,
                    -diff[:, 0] * s + diff[:, 1] * c,
                    diff[:, 2],
                ],
                dim=-1,
            )
            if local.shape[0] >= 4:
                low = torch.quantile(local, 0.10, dim=0)
                high = torch.quantile(local, 0.90, dim=0)
                span = (high - low).abs()
            else:
                span = local.amax(dim=0) - local.amin(dim=0)
            coverage_axis = (span / dims.clamp_min(0.05)).clamp(0.0, 1.0)
            coverage = float(coverage_axis.mean().item())
            confidence = float(ps.mean().clamp(0.0, 1.0).item())
            low_n = self.object_children_min[label]
            high_n = self.object_children_max[label]
            missing = max(0.0, 1.0 - coverage)
            count = int(round(low_n + (high_n - low_n) * missing * (0.45 + 0.55 * confidence)))
            count = max(1, min(count, int(patterns.shape[0])))

            local_grid = patterns * dims.view(1, 3)
            xy = torch.stack(
                [
                    local_grid[:, 0] * c - local_grid[:, 1] * s,
                    local_grid[:, 0] * s + local_grid[:, 1] * c,
                ],
                dim=-1,
            )
            candidate_xyz = seed_center.view(1, 3) + torch.cat([xy, local_grid[:, 2:3]], dim=-1)
            candidate_xyz[:, 0] = candidate_xyz[:, 0].clamp(r[0] + 1.0e-3, r[3] - 1.0e-3)
            candidate_xyz[:, 1] = candidate_xyz[:, 1].clamp(r[1] + 1.0e-3, r[4] - 1.0e-3)
            candidate_xyz[:, 2] = candidate_xyz[:, 2].clamp(r[2] + 1.0e-3, r[5] - 1.0e-3)
            min_seed_dist = torch.cdist(candidate_xyz, seeds).amin(dim=1)
            order_bonus = torch.linspace(0.15, 0.0, steps=patterns.shape[0], device=device)
            choose = torch.topk(min_seed_dist + order_bonus, k=count, largest=True).indices
            xyz = candidate_xyz[choose]
            scale = (dims / dims.new_tensor([10.0, 8.0, 6.0])).clamp(self.object_scale_min, self.object_scale_max)
            scale = scale.view(1, 3).expand(count, 3)
            anchor = torch.zeros((count, anchor_dim), device=device, dtype=dtype)
            anchor[:, 0:3] = self._xyz_to_anchor_logit(xyz).to(dtype=dtype)
            anchor[:, 3:6] = self._scale_to_anchor_logit(scale).to(dtype=dtype)
            anchor[:, 6] = torch.cos(0.5 * yaw).to(dtype=dtype)
            anchor[:, 9] = torch.sin(0.5 * yaw).to(dtype=dtype)
            if self.include_opa and anchor_dim > 10:
                opacity = self.object_completion_opacity * (0.50 + 0.50 * confidence)
                opacity *= 0.80 + 0.20 * missing
                anchor[:, 10] = safe_inverse_sigmoid(
                    anchor.new_full((count,), opacity).clamp(1.0e-4, 1.0 - 1.0e-4)
                )
            sem_start = 11
            if self.semantic_dim > 0 and anchor_dim >= sem_start + self.semantic_dim:
                occ_idx = int(self.dao_to_occ_sem[label].item())
                if 0 <= occ_idx < self.semantic_dim:
                    anchor[:, sem_start + occ_idx] = self.owner_semantic_prior_logit
            anchors.append(anchor)
            labels_out.append(torch.full((count,), label, device=device, dtype=torch.long))
            rep = idx[torch.argmax(ps)].expand(count)
            reps_out.append(rep)
            normalized_dist = (min_seed_dist[choose] / torch.linalg.norm(dims).clamp_min(0.1)).clamp(0.0, 1.0)
            quality = (0.40 + 0.45 * confidence + 0.15 * normalized_dist).clamp(0.0, 1.0)
            quality_out.append(quality)
            gt_out.append(torch.full((count,), int(rec.get("gt_idx", -1)), device=device, dtype=torch.long))
        if not anchors:
            return empty_a, empty_i, empty_i, empty_q, empty_i
        return (
            torch.cat(anchors, dim=0),
            torch.cat(labels_out, dim=0),
            torch.cat(reps_out, dim=0),
            torch.cat(quality_out, dim=0),
            torch.cat(gt_out, dim=0),
        )

    @staticmethod
    def _quat_rotate(q: Tensor, v: Tensor) -> Tensor:
        if q.numel() == 0:
            return v
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        qw = q[:, :1]
        qv = q[:, 1:4]
        return v + 2.0 * torch.cross(qv, torch.cross(qv, v, dim=-1) + qw * v, dim=-1)

    def _build_generic_children(
        self,
        measured_anchor: Tensor,
        measured_xyz: Tensor,
        measured_scales: Tensor,
        measured_opacity: Tensor,
        eligible_mask: Tensor,
        anchor_dim: int,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        device = measured_anchor.device
        empty_a = torch.empty((0, anchor_dim), device=device, dtype=dtype)
        empty_i = torch.empty((0,), device=device, dtype=torch.long)
        empty_q = torch.empty((0,), device=device, dtype=torch.float32)
        if not self.enable_generic_completion or self.generic_children_per_parent <= 0:
            return empty_a, empty_i, empty_q
        scale_max = measured_scales.amax(dim=-1)
        valid = eligible_mask & (measured_opacity >= self.generic_opacity_thr) & (scale_max <= self.generic_max_parent_scale)
        idx = torch.where(valid)[0]
        if idx.numel() == 0:
            return empty_a, empty_i, empty_q
        quality_parent = measured_opacity[idx] * torch.exp(-scale_max[idx] / self.generic_max_parent_scale)
        pidx = idx.repeat_interleave(self.generic_children_per_parent)
        quality = quality_parent.repeat_interleave(self.generic_children_per_parent).clamp(0.0, 1.0)
        total = int(pidx.numel())
        base_scale = measured_scales[pidx]
        axis = torch.argmax(base_scale, dim=-1)
        local_dir = torch.zeros((total, 3), device=device, dtype=torch.float32)
        local_dir[torch.arange(total, device=device), axis] = 1.0
        signs = torch.where((torch.arange(total, device=device) % 2) == 0, 1.0, -1.0).view(-1, 1)
        if anchor_dim >= 10:
            q = measured_anchor[pidx, 6:10].to(torch.float32)
        else:
            q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).view(1, 4).expand(total, 4)
        direction = self._quat_rotate(q, local_dir) * signs
        magnitude = base_scale.amax(dim=-1, keepdim=True).clamp(0.06, 0.20) * self.generic_child_offset_ratio
        xyz = measured_xyz[pidx] + direction * magnitude
        r = self.pc_range_tensor.to(device=device, dtype=torch.float32)
        xyz[:, 0] = xyz[:, 0].clamp(r[0] + 1.0e-3, r[3] - 1.0e-3)
        xyz[:, 1] = xyz[:, 1].clamp(r[1] + 1.0e-3, r[4] - 1.0e-3)
        xyz[:, 2] = xyz[:, 2].clamp(r[2] + 1.0e-3, r[5] - 1.0e-3)
        scale = (base_scale * self.generic_child_scale_ratio).clamp(self.min_gaussian_scale, self.max_gaussian_scale)
        child = measured_anchor[pidx].clone().to(dtype=dtype)
        child[:, 0:3] = self._xyz_to_anchor_logit(xyz).to(dtype=dtype)
        child[:, 3:6] = self._scale_to_anchor_logit(scale).to(dtype=dtype)
        if self.include_opa and anchor_dim > 10:
            opacity = self.generic_completion_opacity * (0.65 + 0.35 * measured_opacity[pidx].clamp(0.0, 1.0))
            child[:, 10] = safe_inverse_sigmoid(opacity.to(dtype=dtype).clamp(1.0e-4, 1.0 - 1.0e-4))
        return child, pidx, quality

    def _fallback_nms(self, centers: Tensor, labels: Tensor, scores: Tensor) -> Tensor:
        if centers.numel() == 0:
            return torch.empty((0,), device=centers.device, dtype=torch.long)
        order = torch.argsort(scores, descending=True)
        keep: List[int] = []
        for idx_t in order:
            idx = int(idx_t.item())
            label = int(labels[idx].item())
            radius = self.fallback_nms_class_radii[label] if 0 <= label < 10 else self.fallback_nms_default_radius
            reject = False
            for kept in keep:
                if int(labels[kept].item()) != label:
                    continue
                if float(torch.linalg.norm(centers[idx, :2] - centers[kept, :2]).item()) <= radius:
                    reject = True
                    break
            if not reject:
                keep.append(idx)
        return torch.tensor(keep, device=centers.device, dtype=torch.long)

    def _build_fallback_anchors(
        self,
        boxes: Tensor,
        labels: Tensor,
        scores: Tensor,
        gt_indices: Tensor,
        anchor_dim: int,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        device = boxes.device
        empty_a = torch.empty((0, anchor_dim), device=device, dtype=dtype)
        empty_i = torch.empty((0,), device=device, dtype=torch.long)
        empty_q = torch.empty((0,), device=device, dtype=torch.float32)
        if boxes.numel() == 0 or not self.enable_oqg_fallback:
            return empty_a, empty_i, empty_q, empty_i
        valid = (
            torch.isfinite(boxes[:, :7]).all(dim=-1)
            & torch.isfinite(scores)
            & (scores >= self.fallback_score_thr)
            & (labels >= 0)
            & (labels < 10)
        )
        if not bool(valid.any()):
            return empty_a, empty_i, empty_q, empty_i
        boxes = boxes[valid]
        labels = labels[valid].long()
        scores = scores[valid]
        gt_indices = gt_indices[valid]
        keep = self._fallback_nms(boxes[:, :3], labels, scores)
        if keep.numel() == 0:
            return empty_a, empty_i, empty_q, empty_i
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep].clamp(0.0, 1.0)
        gt_indices = gt_indices[keep]
        k = self.fallback_gaussians_per_box
        nbox = int(boxes.shape[0])
        # V12 default k=1.  Additional entries remain supported as a local
        # per-box rule, but there is no branch-total limit.
        base_pattern = boxes.new_tensor(
            [[0.0, 0.0, 0.0], [0.12, 0.0, 0.0], [-0.12, 0.0, 0.0], [0.0, 0.0, -0.18]]
        )
        if k > base_pattern.shape[0]:
            base_pattern = torch.cat(
                [base_pattern, torch.zeros((k - base_pattern.shape[0], 3), device=device, dtype=boxes.dtype)],
                dim=0,
            )
        pattern = base_pattern[:k]
        dims = boxes[:, 3:6].abs().clamp_min(0.05)
        yaw = boxes[:, 6]
        local = pattern.view(1, k, 3) * dims[:, None, :]
        c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        xy = torch.stack(
            [local[:, :, 0] * c - local[:, :, 1] * s, local[:, :, 0] * s + local[:, :, 1] * c],
            dim=-1,
        )
        xyz = boxes[:, None, :3] + torch.cat([xy, local[:, :, 2:3]], dim=-1)
        xyz = xyz.reshape(-1, 3)
        labels_out = labels[:, None].expand(nbox, k).reshape(-1)
        scores_out = scores[:, None].expand(nbox, k).reshape(-1)
        gt_out = gt_indices[:, None].expand(nbox, k).reshape(-1)
        yaw_out = yaw[:, None].expand(nbox, k).reshape(-1)
        scale = (dims * self.fallback_scale_ratio).clamp(self.fallback_min_scale, self.fallback_max_scale)
        scale = scale[:, None, :].expand(nbox, k, 3).reshape(-1, 3)
        r = self.pc_range_tensor.to(device=device, dtype=xyz.dtype)
        xyz[:, 0] = xyz[:, 0].clamp(r[0] + 1.0e-3, r[3] - 1.0e-3)
        xyz[:, 1] = xyz[:, 1].clamp(r[1] + 1.0e-3, r[4] - 1.0e-3)
        xyz[:, 2] = xyz[:, 2].clamp(r[2] + 1.0e-3, r[5] - 1.0e-3)
        total = int(xyz.shape[0])
        anchor = torch.zeros((total, anchor_dim), device=device, dtype=dtype)
        anchor[:, 0:3] = self._xyz_to_anchor_logit(xyz).to(dtype=dtype)
        anchor[:, 3:6] = self._scale_to_anchor_logit(scale).to(dtype=dtype)
        anchor[:, 6] = torch.cos(0.5 * yaw_out).to(dtype=dtype)
        anchor[:, 9] = torch.sin(0.5 * yaw_out).to(dtype=dtype)
        if self.include_opa and anchor_dim > 10:
            opacity = self.fallback_opacity * (0.50 + 0.50 * scores_out)
            anchor[:, 10] = safe_inverse_sigmoid(opacity.to(dtype=dtype).clamp(1.0e-4, 1.0 - 1.0e-4))
        sem_start = 11
        if self.semantic_dim > 0 and anchor_dim >= sem_start + self.semantic_dim:
            occ_idx = self.dao_to_occ_sem.to(device=device)[labels_out]
            valid_sem = (occ_idx >= 0) & (occ_idx < self.semantic_dim)
            anchor[valid_sem, sem_start + occ_idx[valid_sem]] = self.fallback_semantic_prior_logit
        quality = (0.05 + 0.25 * scores_out).clamp(0.0, 0.30)
        return anchor, labels_out, quality, gt_out

    def _zero_owner_loss(self) -> Tensor:
        zero = self.owner_mlp[-1].weight.sum() * 0.0
        zero = zero + self.owner_feature_proj[0].weight.sum() * 0.0
        zero = zero + self.owner_class_embedding.weight.sum() * 0.0
        return zero

    def forward(
        self,
        representation: Optional[Tensor] = None,
        rep_features: Optional[Tensor] = None,
        gaussian_filled_count: Optional[Tensor] = None,
        gaussian_source_types: Optional[Tensor] = None,
        gaussian_visibility_state_map: Optional[Tensor] = None,
        gaussian_visibility_strong_any_map: Optional[Tensor] = None,
        gaussian_visibility_strong_ratio_map: Optional[Tensor] = None,
        gaussian_visibility_weak_ratio_map: Optional[Tensor] = None,
        yolo_boxes_2d: Optional[Tensor] = None,
        yolo_scores_2d: Optional[Tensor] = None,
        yolo_labels_2d: Optional[Tensor] = None,
        yolo_valid_2d: Optional[Tensor] = None,
        yolo_gt_indices_2d: Optional[Tensor] = None,
        gt_bboxes_3d: Any = None,
        metas: Optional[Dict[str, Any]] = None,
        imgs: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        self._call_count += 1
        if (
            (not self.enabled)
            or representation is None
            or rep_features is None
            or gaussian_filled_count is None
            or yolo_boxes_2d is None
            or yolo_labels_2d is None
            or yolo_valid_2d is None
        ):
            early = {
                "loss_gaussian_owner": self._zero_owner_loss(),
                "loss_gaussian_child_quality": self._zero_child_quality_loss(),
            }
            if isinstance(representation, Tensor) and representation.ndim == 3:
                early["gaussian_completion_render_gate"] = torch.ones(
                    representation.shape[:2], device=representation.device, dtype=torch.float32
                )
                early["gaussian_diagnostic_geometry_gates"] = torch.full(
                    representation.shape[:2], -1.0, device=representation.device, dtype=torch.float32
                )
                early["gaussian_diagnostic_semantic_gates"] = torch.full(
                    representation.shape[:2], -1.0, device=representation.device, dtype=torch.float32
                )
            return early

        B, N, A = representation.shape
        device, dtype = representation.device, representation.dtype
        if rep_features.shape[:2] != representation.shape[:2]:
            raise RuntimeError("rep_features and representation must have matching [B,N]")
        filled = gaussian_filled_count.to(device=device, dtype=torch.long).reshape(-1)
        if filled.numel() < B:
            filled = torch.cat([filled, torch.zeros((B - filled.numel(),), device=device, dtype=torch.long)])
        filled = filled[:B].clamp(0, N)
        boxes2d = yolo_boxes_2d.to(device=device, dtype=torch.float32)
        labels2d = yolo_labels_2d.to(device=device, dtype=torch.long)
        valid2d = yolo_valid_2d.to(device=device, dtype=torch.bool)
        scores2d = (
            yolo_scores_2d.to(device=device, dtype=torch.float32)
            if isinstance(yolo_scores_2d, Tensor)
            else torch.ones_like(labels2d, dtype=torch.float32)
        )
        if boxes2d.ndim != 4 or boxes2d.shape[-1] != 4:
            raise RuntimeError(f"yolo_boxes_2d must be [B,Cam,M,4], got {tuple(boxes2d.shape)}")
        Cam = int(boxes2d.shape[1])
        proj = self._get_projection(metas, device=device, dtype=torch.float32)
        image_h, image_w = (
            (int(imgs.shape[-2]), int(imgs.shape[-1]))
            if isinstance(imgs, Tensor) and imgs.ndim == 5
            else (256, 704)
        )

        out_anchor = representation.clone()
        out_feat = rep_features.clone()
        # Persistent gate survives encoder opacity refinement and is consumed by
        # GaussianHead. V22-C2 combines the V22-C1 Local visibility gate with
        # the detached geometry-quality gate for every generated candidate.
        out_render_gate = torch.ones((B, N), device=device, dtype=torch.float32)
        if (
            isinstance(gaussian_source_types, Tensor)
            and gaussian_source_types.shape[:2] == representation.shape[:2]
        ):
            out_source = gaussian_source_types.to(device=device, dtype=torch.long).clone()
        else:
            out_source = torch.zeros((B, N), device=device, dtype=torch.long)

        # V21-D0 side-channel provenance; it never enters the model computation.
        # 0 measured, 1 object child, 2 local child, 3 generic child,
        # 4 reserved legacy fallback slot type, 5 untouched/base anchor.
        out_diag_branch = torch.full((B, N), 5, device=device, dtype=torch.long)
        out_diag_label = torch.full((B, N), -1, device=device, dtype=torch.long)
        out_diag_gt = torch.full((B, N), -1, device=device, dtype=torch.long)
        out_diag_owner = torch.full((B, N), -1.0, device=device, dtype=torch.float32)
        out_diag_gate = torch.full((B, N), -1.0, device=device, dtype=torch.float32)
        out_diag_geometry_gate = torch.full((B, N), -1.0, device=device, dtype=torch.float32)
        out_diag_semantic_gate = torch.full((B, N), -1.0, device=device, dtype=torch.float32)
        gt_list = self._as_gt_list(gt_bboxes_3d, B, device=device, dtype=torch.float32)
        occ_labels, occ_xyz = self._extract_occ_tensors(metas, kwargs, B, device)
        gt_idx = (
            yolo_gt_indices_2d.to(device=device, dtype=torch.long)
            if isinstance(yolo_gt_indices_2d, Tensor) and yolo_gt_indices_2d.shape == labels2d.shape
            else None
        )

        owner_losses: List[Tensor] = []
        child_quality_losses: List[Tensor] = []
        object_counts: List[int] = []
        local_counts: List[int] = []
        generic_counts: List[int] = []
        fallback_counts: List[int] = []
        parent_counts: List[int] = []
        total_counts: List[int] = []
        candidate_counts: List[int] = []
        capacity_drop_counts: List[int] = []

        boxes_total_stat = 0
        boxes_owned_stat = 0
        boxes_fallback_stat = 0
        owner_candidate_total = 0
        owner_selected_before_global = 0
        duplicate_removed = 0
        owner_tp = owner_fp = owner_fn = 0
        gt_box_seed_hit = gt_box_seed_total = 0
        gt_seed_inside = gt_seed_total = 0
        child_inside = child_total = 0
        child_occ_valid = child_nonempty = 0
        child_semantic_valid = child_semantic_match = 0
        quality_nonempty_tp = quality_nonempty_fp = quality_nonempty_fn = 0
        quality_semantic_tp = quality_semantic_fp = quality_semantic_fn = 0
        quality_prob_stat: List[float] = []
        quality_gate_stat: List[float] = []
        quality_geometry_stat: List[float] = []
        quality_semantic_stat: List[float] = []
        quality_geometry_zero = 0
        quality_semantic_zero = 0
        quality_known_total = 0
        cluster_count_stat: List[float] = []
        owner_prob_stat: List[float] = []
        vis_proposed_counts: List[Tensor] = []
        vis_rejected_counts: List[Tensor] = []
        vis_strong_center_counts: List[Tensor] = []
        vis_strong_any_counts: List[Tensor] = []
        vis_strong_ratio_means: List[Tensor] = []
        vis_weak_means: List[Tensor] = []
        vis_unknown_means: List[Tensor] = []
        vis_car_rejected_counts: List[Tensor] = []
        vis_bus_rejected_counts: List[Tensor] = []

        for b in range(B):
            start = int(filled[b].item())
            capacity = max(N - start, 0)
            if start > 0:
                out_diag_branch[b, :start] = 0
            measured_anchor = out_anchor[b, :start]
            measured_feat = out_feat[b, :start]
            if start > 0:
                measured_xyz, measured_scales, measured_opacity = self._anchor_xyz_scale_opacity(measured_anchor.detach())
            else:
                measured_xyz = torch.empty((0, 3), device=device, dtype=torch.float32)
                measured_scales = torch.empty((0, 3), device=device, dtype=torch.float32)
                measured_opacity = torch.empty((0,), device=device, dtype=torch.float32)

            # The lifter always emits a full batched map when visibility is
            # available.  Missing samples are filled with UNKNOWN on CUDA, so
            # there is no scalar Tensor -> Python bool synchronization here.
            visibility_state_b = (
                gaussian_visibility_state_map[b]
                if isinstance(gaussian_visibility_state_map, Tensor)
                and gaussian_visibility_state_map.ndim == 4
                and b < int(gaussian_visibility_state_map.shape[0])
                else None
            )
            visibility_strong_any_b = (
                gaussian_visibility_strong_any_map[b]
                if isinstance(gaussian_visibility_strong_any_map, Tensor)
                and gaussian_visibility_strong_any_map.ndim == 4
                and b < int(gaussian_visibility_strong_any_map.shape[0])
                else None
            )
            visibility_strong_ratio_b = (
                gaussian_visibility_strong_ratio_map[b]
                if isinstance(gaussian_visibility_strong_ratio_map, Tensor)
                and gaussian_visibility_strong_ratio_map.ndim == 4
                and b < int(gaussian_visibility_strong_ratio_map.shape[0])
                else None
            )
            visibility_weak_ratio_b = (
                gaussian_visibility_weak_ratio_map[b]
                if isinstance(gaussian_visibility_weak_ratio_map, Tensor)
                and gaussian_visibility_weak_ratio_map.ndim == 4
                and b < int(gaussian_visibility_weak_ratio_map.shape[0])
                else None
            )
            # box_records are first built independently.  The global ownership
            # stage later guarantees one measured Gaussian -> at most one object.
            box_records: List[Dict[str, Any]] = []
            projected_cache: Dict[int, Tuple[Tensor, Tensor, Tensor, Tensor]] = {}
            object_id = 0
            selected_before_global_b = 0

            for cidx in range(Cam):
                if start > 0:
                    u_all, v_all, d_all = self._project_points(measured_xyz, proj[b, cidx])
                    project_valid = (
                        torch.isfinite(u_all)
                        & torch.isfinite(v_all)
                        & torch.isfinite(d_all)
                        & (d_all > 1.0e-3)
                        & (u_all >= -2.0)
                        & (u_all <= image_w + 2.0)
                        & (v_all >= -2.0)
                        & (v_all <= image_h + 2.0)
                    )
                    projected_cache[cidx] = (u_all, v_all, d_all, project_valid)
                valid_rows = torch.where(
                    valid2d[b, cidx]
                    & (labels2d[b, cidx] >= 0)
                    & (labels2d[b, cidx] < 10)
                )[0]
                for mt in valid_rows:
                    m = int(mt.item())
                    boxes_total_stat += 1
                    object_id += 1
                    box = boxes2d[b, cidx, m].clone()
                    x1, y1, x2, y2 = box.unbind()
                    x1 = x1.clamp(0.0, image_w - 1.0)
                    x2 = x2.clamp(0.0, image_w - 1.0)
                    y1 = y1.clamp(0.0, image_h - 1.0)
                    y2 = y2.clamp(0.0, image_h - 1.0)
                    box = torch.stack([x1, y1, x2, y2])
                    bw = float((x2 - x1).item())
                    bh = float((y2 - y1).item())
                    label = int(labels2d[b, cidx, m].item())
                    box_score = float(scores2d[b, cidx, m].clamp(0.0, 1.0).item())
                    gi = int(gt_idx[b, cidx, m].item()) if gt_idx is not None else -1
                    selected_idx = torch.empty((0,), device=device, dtype=torch.long)
                    selected_scores = torch.empty((0,), device=device, dtype=torch.float32)
                    logits: Optional[Tensor] = None
                    targets: Optional[Tensor] = None
                    if start > 0 and bw >= 1.0 and bh >= 1.0:
                        u_all, v_all, d_all, project_valid = projected_cache[cidx]
                        in_box = (
                            project_valid
                            & (u_all >= x1)
                            & (u_all <= x2)
                            & (v_all >= y1)
                            & (v_all <= y2)
                        )
                        idx = torch.where(in_box)[0]
                        if idx.numel() >= self.min_projected_gaussians:
                            u = u_all[idx]
                            v = v_all[idx]
                            d = d_all[idx]
                            cx1 = x1 + self.core_shrink_x * (x2 - x1)
                            cx2 = x2 - self.core_shrink_x * (x2 - x1)
                            cy1 = y1 + self.core_shrink_top * (y2 - y1)
                            cy2 = y2 - self.core_shrink_bottom * (y2 - y1)
                            core = (u >= cx1) & (u <= cx2) & (v >= cy1) & (v <= cy2)
                            best_cluster, ref_depth, n_clusters = self._best_cluster_prior(
                                u,
                                v,
                                d,
                                core,
                                measured_opacity[idx],
                                measured_scales[idx],
                                box,
                            )
                            cluster_count_stat.append(float(n_clusters))
                            label_tensor = torch.full((idx.numel(),), label, device=device, dtype=torch.long)
                            logits, _ = self._owner_logits(
                                measured_feat[idx],
                                label_tensor,
                                u,
                                v,
                                d,
                                core,
                                best_cluster,
                                ref_depth,
                                measured_opacity[idx],
                                measured_scales[idx],
                                box,
                                box_score,
                            )
                            prob = logits.sigmoid()
                            owner_candidate_total += int(prob.numel())
                            owner_prob_stat.append(float(prob.mean().detach().item()))

                            gt_boxes_b = gt_list[b]
                            if gt_boxes_b is not None and 0 <= gi < int(gt_boxes_b.shape[0]):
                                targets = self._inside_oriented_box(measured_xyz[idx], gt_boxes_b[gi], expand=1.05)
                                owner_losses.append(self._owner_focal_loss(logits, targets.float()))
                                pred = prob.detach() >= self.owner_threshold
                                owner_tp += int((pred & targets).sum().item())
                                owner_fp += int((pred & ~targets).sum().item())
                                owner_fn += int((~pred & targets).sum().item())

                            keep = prob.detach() >= self.owner_threshold
                            if not bool(keep.any()) and prob.numel() > 0:
                                max_prob, max_local = prob.detach().max(dim=0)
                                if float(max_prob.item()) >= self.owner_min_keep_score:
                                    keep[max_local] = True
                            selected_idx = idx[keep]
                            selected_scores = prob.detach()[keep]
                            owner_selected_before_global += int(selected_idx.numel())
                            selected_before_global_b += int(selected_idx.numel())

                    if selected_idx.numel() > 0:
                        boxes_owned_stat += 1
                        box_records.append(
                            {
                                "object_id": object_id,
                                "label": label,
                                "score": float(selected_scores.mean().item()),
                                "parent_idx": selected_idx,
                                "parent_scores": selected_scores,
                                "gt_idx": gi,
                            }
                        )
                        if targets is not None:
                            selected_target = targets[(selected_idx[:, None] == idx[None, :]).float().argmax(dim=1)] if selected_idx.numel() else targets[:0]
                            gt_box_seed_total += 1
                            gt_box_seed_hit += int(bool(selected_target.any()))
                            gt_seed_total += int(selected_target.numel())
                            gt_seed_inside += int(selected_target.sum().item())

            # Global one-Gaussian/one-object assignment.
            winner: Dict[int, Tuple[int, float]] = {}
            for rid, rec in enumerate(box_records):
                for pi, ps in zip(rec["parent_idx"], rec["parent_scores"]):
                    p = int(pi.item())
                    score = float(ps.item())
                    if p not in winner or score > winner[p][1]:
                        winner[p] = (rid, score)
            duplicate_removed += selected_before_global_b - len(winner)
            filtered_records: List[Dict[str, Any]] = []
            for rid, rec in enumerate(box_records):
                mask = torch.tensor(
                    [winner.get(int(pi.item()), (-1, -1.0))[0] == rid for pi in rec["parent_idx"]],
                    device=device,
                    dtype=torch.bool,
                )
                if bool(mask.any()):
                    new_rec = dict(rec)
                    new_rec["parent_idx"] = rec["parent_idx"][mask]
                    new_rec["parent_scores"] = rec["parent_scores"][mask]
                    new_rec["score"] = float(new_rec["parent_scores"].mean().item())
                    filtered_records.append(new_rec)
            object_records = self._merge_object_records(filtered_records, measured_xyz)

            # Rebuild unique parent tensors after merging/ownership.
            parent_meta: Dict[int, Tuple[int, float, int]] = {}
            for rec in object_records:
                for pi, ps in zip(rec["parent_idx"], rec["parent_scores"]):
                    p = int(pi.item())
                    score = float(ps.item())
                    old = parent_meta.get(p)
                    if old is None or score > old[1]:
                        parent_meta[p] = (int(rec["label"]), score, int(rec.get("gt_idx", -1)))
            parent_items = sorted(parent_meta.items(), key=lambda kv: kv[1][1], reverse=True)
            if parent_items:
                parent_idx = torch.tensor([x[0] for x in parent_items], device=device, dtype=torch.long)
                parent_labels = torch.tensor([x[1][0] for x in parent_items], device=device, dtype=torch.long)
                parent_scores = torch.tensor([x[1][1] for x in parent_items], device=device, dtype=torch.float32)
                parent_gt_idx = torch.tensor([x[1][2] for x in parent_items], device=device, dtype=torch.long)
                parent_anchor = measured_anchor[parent_idx]
                parent_xyz = measured_xyz[parent_idx]
                parent_scales = measured_scales[parent_idx]
            else:
                parent_idx = torch.empty((0,), device=device, dtype=torch.long)
                parent_labels = torch.empty((0,), device=device, dtype=torch.long)
                parent_scores = torch.empty((0,), device=device, dtype=torch.float32)
                parent_gt_idx = torch.empty((0,), device=device, dtype=torch.long)
                parent_anchor = torch.empty((0, A), device=device, dtype=dtype)
                parent_xyz = torch.empty((0, 3), device=device, dtype=torch.float32)
                parent_scales = torch.empty((0, 3), device=device, dtype=torch.float32)

            if parent_idx.numel() > 0:
                out_diag_label[b, parent_idx] = parent_labels
                out_diag_gt[b, parent_idx] = parent_gt_idx
                out_diag_owner[b, parent_idx] = parent_scores
                sem_start = 11
                if self.inject_seed_semantics and self.semantic_dim > 0 and A >= sem_start + self.semantic_dim:
                    occ_idx = self.dao_to_occ_sem.to(device=device)[parent_labels.clamp(0, 9)]
                    valid_sem = (occ_idx >= 0) & (occ_idx < self.semantic_dim)
                    if bool(valid_sem.any()):
                        rows = parent_idx[valid_sem]
                        cols = occ_idx[valid_sem]
                        sem = out_anchor[b, rows, sem_start : sem_start + self.semantic_dim].clone()
                        rr = torch.arange(rows.numel(), device=device)
                        sem[rr, cols] = torch.maximum(
                            sem[rr, cols],
                            sem.new_full((rows.numel(),), self.owner_semantic_prior_logit),
                        )
                        out_anchor[b, rows, sem_start : sem_start + self.semantic_dim] = sem
                if self.inject_seed_features:
                    out_feat[b, parent_idx] = out_feat[b, parent_idx] + self.class_feature(
                        parent_labels.clamp(0, 9)
                    ).to(out_feat.dtype)

            # Build every branch without branch-total limits.
            obj_anchor, obj_label, obj_rep, obj_quality, obj_gt = self._build_object_children(
                object_records,
                measured_xyz,
                A,
                dtype,
            )
            loc_anchor, loc_label, loc_parent_local, loc_quality, loc_gt, loc_render_gate, vis_stats = self._build_seed_children(
                parent_anchor,
                parent_xyz,
                parent_scales,
                parent_labels,
                parent_scores,
                parent_gt_idx,
                visibility_state_b,
                visibility_strong_any_b,
                visibility_strong_ratio_b,
                visibility_weak_ratio_b,
                A,
                dtype,
            )
            vis_proposed_counts.append(vis_stats['proposed'])
            vis_rejected_counts.append(vis_stats['rejected'])
            vis_strong_center_counts.append(vis_stats['strong_center'])
            vis_strong_any_counts.append(vis_stats['strong_any'])
            vis_strong_ratio_means.append(vis_stats['strong_ratio_mean'])
            vis_weak_means.append(vis_stats['weak_mean'])
            vis_unknown_means.append(vis_stats['unknown_mean'])
            vis_car_rejected_counts.append(vis_stats['car_rejected'])
            vis_bus_rejected_counts.append(vis_stats['bus_rejected'])
            owned_mask = torch.zeros((start,), device=device, dtype=torch.bool)
            if parent_idx.numel() > 0:
                owned_mask[parent_idx] = True
            gen_anchor, gen_parent, gen_quality = self._build_generic_children(
                measured_anchor,
                measured_xyz,
                measured_scales,
                measured_opacity,
                ~owned_mask,
                A,
                dtype,
            )

            candidate_anchor: List[Tensor] = []
            candidate_feat: List[Tensor] = []
            candidate_quality: List[Tensor] = []
            candidate_kind: List[Tensor] = []
            candidate_source: List[Tensor] = []
            candidate_gt: List[Tensor] = []
            candidate_label: List[Tensor] = []
            candidate_render_gate: List[Tensor] = []

            if obj_anchor.numel() > 0:
                feat = self.class_feature(obj_label.clamp(0, 9)).to(out_feat.dtype)
                if self.add_completion_feature and self.completion_feature is not None:
                    feat = feat + self.completion_feature.to(out_feat.dtype)
                feat = feat + 0.25 * out_feat[b, obj_rep]
                candidate_anchor.append(obj_anchor)
                candidate_feat.append(feat)
                candidate_quality.append((obj_quality + 0.25).clamp(0.0, 1.25))
                candidate_kind.append(torch.zeros((obj_anchor.shape[0],), device=device, dtype=torch.long))
                candidate_source.append(torch.ones((obj_anchor.shape[0],), device=device, dtype=torch.long))
                candidate_gt.append(obj_gt)
                candidate_label.append(obj_label)
                candidate_render_gate.append(torch.ones((obj_anchor.shape[0],), device=device, dtype=torch.float32))

            if loc_anchor.numel() > 0:
                loc_parent_global = parent_idx[loc_parent_local]
                feat = self.class_feature(loc_label.clamp(0, 9)).to(out_feat.dtype)
                if self.add_completion_feature and self.completion_feature is not None:
                    feat = feat + self.completion_feature.to(out_feat.dtype)
                feat = feat + 0.25 * out_feat[b, loc_parent_global]
                encoder_gain = loc_render_gate.clamp(0.0, 1.0).pow(self.visibility_encoder_gain_power)
                feat = feat * encoder_gain[:, None].to(feat.dtype)
                candidate_anchor.append(loc_anchor)
                candidate_feat.append(feat)
                candidate_quality.append((loc_quality + 0.30).clamp(0.0, 1.30))
                candidate_kind.append(torch.ones((loc_anchor.shape[0],), device=device, dtype=torch.long))
                candidate_source.append(torch.ones((loc_anchor.shape[0],), device=device, dtype=torch.long))
                candidate_gt.append(loc_gt)
                candidate_label.append(loc_label)
                candidate_render_gate.append(loc_render_gate.to(torch.float32))

            if gen_anchor.numel() > 0:
                feat = 0.50 * out_feat[b, gen_parent] + self.generic_completion_feature.to(
                    device=device,
                    dtype=out_feat.dtype,
                ).view(1, -1)
                candidate_anchor.append(gen_anchor)
                candidate_feat.append(feat)
                candidate_quality.append((gen_quality + 0.10).clamp(0.0, 1.10))
                candidate_kind.append(torch.full((gen_anchor.shape[0],), 2, device=device, dtype=torch.long))
                candidate_source.append(torch.full((gen_anchor.shape[0],), 2, device=device, dtype=torch.long))
                candidate_gt.append(torch.full((gen_anchor.shape[0],), -1, device=device, dtype=torch.long))
                candidate_label.append(torch.full((gen_anchor.shape[0],), -1, device=device, dtype=torch.long))
                candidate_render_gate.append(torch.ones((gen_anchor.shape[0],), device=device, dtype=torch.float32))

            if candidate_anchor:
                cand_anchor = torch.cat(candidate_anchor, dim=0)
                cand_feat = torch.cat(candidate_feat, dim=0)
                cand_quality = torch.cat(candidate_quality, dim=0)
                cand_kind = torch.cat(candidate_kind, dim=0)
                cand_source = torch.cat(candidate_source, dim=0)
                cand_gt = torch.cat(candidate_gt, dim=0)
                cand_label = torch.cat(candidate_label, dim=0)
                cand_render_gate = torch.cat(candidate_render_gate, dim=0).clamp(0.0, 1.0)
                cand_branch = cand_kind + 1
                cand_gate = torch.ones((cand_anchor.shape[0],), device=device, dtype=torch.float32)
                cand_geometry_gate = torch.ones((cand_anchor.shape[0],), device=device, dtype=torch.float32)
                cand_semantic_gate = torch.ones((cand_anchor.shape[0],), device=device, dtype=torch.float32)
            else:
                cand_anchor = torch.empty((0, A), device=device, dtype=dtype)
                cand_feat = torch.empty((0, self.feature_dim), device=device, dtype=out_feat.dtype)
                cand_quality = torch.empty((0,), device=device, dtype=torch.float32)
                cand_kind = torch.empty((0,), device=device, dtype=torch.long)
                cand_source = torch.empty((0,), device=device, dtype=torch.long)
                cand_gt = torch.empty((0,), device=device, dtype=torch.long)
                cand_label = torch.empty((0,), device=device, dtype=torch.long)
                cand_render_gate = torch.empty((0,), device=device, dtype=torch.float32)
                cand_branch = torch.empty((0,), device=device, dtype=torch.long)
                cand_gate = torch.empty((0,), device=device, dtype=torch.float32)
                cand_geometry_gate = torch.empty((0,), device=device, dtype=torch.float32)
                cand_semantic_gate = torch.empty((0,), device=device, dtype=torch.float32)

            # V22-C2 learns reliability without moving candidate geometry.  Geometry and candidate features are
            # detached from the auxiliary quality loss, and the predicted gates
            # are detached before affecting the main occupancy path.
            if self.enable_child_quality and cand_anchor.numel() > 0:
                nonempty_logit, semantic_logit, cand_xyz, base_opacity = self._child_quality_logits(
                    cand_anchor.detach(),
                    cand_feat.detach(),
                    cand_quality.detach(),
                    cand_label,
                    cand_kind,
                )
                nonempty_prob = nonempty_logit.sigmoid()
                semantic_prob = semantic_logit.sigmoid()
                known_semantic = (cand_label >= 0) & (cand_label < 10)

                occ_label_b = occ_labels[b] if occ_labels is not None else None
                occ_xyz_b = occ_xyz[b] if occ_xyz is not None else None
                sampled_occ, sampled_valid = self._lookup_occ_labels(cand_xyz.detach(), occ_label_b, occ_xyz_b)
                nonempty_target = sampled_occ != self.occ_empty_label
                expected_occ = torch.full_like(cand_label, -1)
                if bool(known_semantic.any()):
                    expected_occ[known_semantic] = self.dao_to_occ_sem.to(device=device)[cand_label[known_semantic]]
                semantic_target = sampled_occ == expected_occ
                semantic_mask = sampled_valid & known_semantic & (expected_occ >= 0)
                if self.child_quality_semantic_on_nonempty_only:
                    # q_semantic is conditional class correctness. Empty locations
                    # are supervised only by q_geometry and are not counted again
                    # as semantic negatives.
                    semantic_mask = semantic_mask & nonempty_target

                nonempty_loss = self._balanced_bce(nonempty_logit, nonempty_target.float(), sampled_valid)
                semantic_loss = self._balanced_bce(semantic_logit, semantic_target.float(), semantic_mask)
                child_quality_losses.append(
                    self.child_quality_nonempty_weight * nonempty_loss
                    + self.child_quality_semantic_weight * semantic_loss
                )

                with torch.no_grad():
                    raw_geometry_gate = nonempty_prob.detach().clamp(0.0, 1.0)
                    raw_semantic_gate = semantic_prob.detach().clamp(0.0, 1.0)
                    geometry_gate = torch.where(
                        raw_geometry_gate >= self.child_quality_geometry_zero_threshold,
                        raw_geometry_gate,
                        torch.zeros_like(raw_geometry_gate),
                    )
                    semantic_component = torch.where(
                        raw_semantic_gate >= self.child_quality_semantic_zero_threshold,
                        raw_semantic_gate,
                        torch.zeros_like(raw_semantic_gate),
                    )
                    # Generic children have no object-class hypothesis, therefore
                    # their semantic component is neutral and only geometry gates them.
                    semantic_component = torch.where(
                        known_semantic, semantic_component, torch.ones_like(semantic_component)
                    )
                    final_gate = (geometry_gate * semantic_component).clamp(0.0, 1.0)

                    quality_prob_stat.append(float(final_gate.mean().item()))
                    quality_geometry_stat.append(float(geometry_gate.mean().item()))
                    if bool(known_semantic.any()):
                        quality_semantic_stat.append(float(raw_semantic_gate[known_semantic].mean().item()))
                    quality_geometry_zero += int((geometry_gate <= 0.0).sum().item())
                    quality_semantic_zero += int((known_semantic & (semantic_component <= 0.0)).sum().item())
                    quality_known_total += int(known_semantic.sum().item())

                    pred_nonempty = nonempty_prob >= 0.5
                    quality_nonempty_tp += int((pred_nonempty & nonempty_target & sampled_valid).sum().item())
                    quality_nonempty_fp += int((pred_nonempty & ~nonempty_target & sampled_valid).sum().item())
                    quality_nonempty_fn += int((~pred_nonempty & nonempty_target & sampled_valid).sum().item())
                    pred_semantic = semantic_prob >= 0.5
                    quality_semantic_tp += int((pred_semantic & semantic_target & semantic_mask).sum().item())
                    quality_semantic_fp += int((pred_semantic & ~semantic_target & semantic_mask).sum().item())
                    quality_semantic_fn += int((~pred_semantic & semantic_target & semantic_mask).sum().item())
                    child_occ_valid += int(sampled_valid.sum().item())
                    child_nonempty += int((nonempty_target & sampled_valid).sum().item())
                    child_semantic_valid += int(semantic_mask.sum().item())
                    child_semantic_match += int((semantic_target & semantic_mask).sum().item())

                cand_geometry_gate = geometry_gate.to(dtype=torch.float32)
                cand_semantic_gate = semantic_component.to(dtype=torch.float32)
                cand_gate = final_gate.to(dtype=torch.float32)
                quality_gate_stat.append(float(final_gate.mean().item()))

                if self.include_opa and A > 10:
                    # Opacity answers only whether geometry should exist. No
                    # semantic confidence is allowed to erase valid non-object
                    # geometry, and no positive floor is retained.
                    opacity_gain = self.child_quality_opacity_floor + (1.0 - self.child_quality_opacity_floor) * geometry_gate
                    calibrated_opacity = (base_opacity * opacity_gain).clamp(1.0e-6, 1.0 - 1.0e-4)
                    cand_anchor = cand_anchor.clone()
                    cand_anchor[:, 10] = safe_inverse_sigmoid(calibrated_opacity.to(dtype=cand_anchor.dtype))

                # Object-class evidence needs both non-empty geometry and class
                # correctness. Generic children use geometry only because they do
                # not inject a fixed car/bus/etc. class hypothesis.
                feature_gate = torch.where(known_semantic, final_gate, geometry_gate)
                feature_gain = self.child_quality_feature_floor + (1.0 - self.child_quality_feature_floor) * feature_gate
                cand_feat = cand_feat * feature_gain[:, None].to(dtype=cand_feat.dtype)

                rank_gate = torch.where(known_semantic, final_gate, geometry_gate)
                rank_gain = self.child_quality_rank_floor + (1.0 - self.child_quality_rank_floor) * rank_gate
                cand_quality = cand_quality * rank_gain

                if self.child_quality_persistent_geometry_gate:
                    # Combine with V22-C1 visibility gating. GaussianHead applies
                    # this again after encoder refinement, so empty candidates
                    # cannot be reactivated by a later opacity update.
                    cand_render_gate = cand_render_gate * geometry_gate.to(dtype=cand_render_gate.dtype)
            elif cand_anchor.numel() > 0:
                child_quality_losses.append(self._zero_child_quality_loss())

            candidate_counts.append(int(cand_anchor.shape[0]))
            if cand_anchor.shape[0] > capacity:
                # This is the only total truncation in V12: the actual 9000-slot
                # representation capacity.  There are no per-branch total caps.
                keep = torch.topk(cand_quality, k=capacity, largest=True).indices if capacity > 0 else cand_quality[:0].long()
                capacity_drop_counts.append(int(cand_anchor.shape[0] - capacity))
                cand_anchor = cand_anchor[keep]
                cand_feat = cand_feat[keep]
                cand_kind = cand_kind[keep]
                cand_source = cand_source[keep]
                cand_gt = cand_gt[keep]
                cand_label = cand_label[keep]
                cand_branch = cand_branch[keep]
                cand_gate = cand_gate[keep]
                cand_geometry_gate = cand_geometry_gate[keep]
                cand_semantic_gate = cand_semantic_gate[keep]
                cand_render_gate = cand_render_gate[keep]
            else:
                capacity_drop_counts.append(0)

            take = int(cand_anchor.shape[0])
            if take > 0:
                slots = slice(start, start + take)
                out_anchor[b, slots] = cand_anchor.detach()
                out_feat[b, slots] = out_feat[b, slots] + cand_feat
                out_source[b, slots] = cand_source
                out_diag_branch[b, slots] = cand_branch
                out_diag_label[b, slots] = cand_label
                out_diag_gt[b, slots] = cand_gt
                out_diag_gate[b, slots] = cand_gate
                out_diag_geometry_gate[b, slots] = cand_geometry_gate
                out_diag_semantic_gate[b, slots] = cand_semantic_gate
                out_render_gate[b, slots] = cand_render_gate

                # Debug-only correctness of selected generated object Gaussians.
                gt_boxes_b = gt_list[b]
                if gt_boxes_b is not None:
                    child_xyz, _, _ = self._anchor_xyz_scale_opacity(cand_anchor.detach())
                    valid_child_gt = (cand_gt >= 0) & (cand_gt < int(gt_boxes_b.shape[0])) & (cand_kind != 2)
                    for gi_value in torch.unique(cand_gt[valid_child_gt]):
                        gi_int = int(gi_value.item())
                        mask = valid_child_gt & (cand_gt == gi_int)
                        inside = self._inside_oriented_box(child_xyz[mask], gt_boxes_b[gi_int], expand=1.10)
                        child_total += int(inside.numel())
                        child_inside += int(inside.sum().item())

            object_counts.append(int((cand_kind == 0).sum().item()))
            local_counts.append(int((cand_kind == 1).sum().item()))
            generic_counts.append(int((cand_kind == 2).sum().item()))
            fallback_counts.append(int((cand_kind == 3).sum().item()))
            parent_counts.append(int(parent_idx.numel()))
            total_counts.append(start + take)

        if owner_losses:
            loss_owner = torch.stack(owner_losses).mean()
        else:
            loss_owner = self._zero_owner_loss()
        if child_quality_losses:
            loss_child_quality = torch.stack(child_quality_losses).mean()
        else:
            loss_child_quality = self._zero_child_quality_loss()

        object_t = torch.tensor(object_counts, device=device, dtype=torch.long)
        local_t = torch.tensor(local_counts, device=device, dtype=torch.long)
        generic_t = torch.tensor(generic_counts, device=device, dtype=torch.long)
        fallback_t = torch.tensor(fallback_counts, device=device, dtype=torch.long)
        parent_t = torch.tensor(parent_counts, device=device, dtype=torch.long)
        total_t = torch.tensor(total_counts, device=device, dtype=torch.long)
        completed_t = object_t + local_t + generic_t + fallback_t
        vis_proposed_t = torch.stack(vis_proposed_counts).to(dtype=torch.long)
        vis_rejected_t = torch.stack(vis_rejected_counts).to(dtype=torch.long)
        vis_strong_center_t = torch.stack(vis_strong_center_counts).to(dtype=torch.long)
        vis_strong_any_t = torch.stack(vis_strong_any_counts).to(dtype=torch.long)
        vis_strong_ratio_t = torch.stack(vis_strong_ratio_means).to(dtype=torch.float32)
        vis_weak_t = torch.stack(vis_weak_means).to(dtype=torch.float32)
        vis_unknown_t = torch.stack(vis_unknown_means).to(dtype=torch.float32)
        vis_car_rejected_t = torch.stack(vis_car_rejected_counts).to(dtype=torch.long)
        vis_bus_rejected_t = torch.stack(vis_bus_rejected_counts).to(dtype=torch.long)

        if self.debug and self._is_rank0() and self._call_count % self.debug_interval == 0:
            allow = (
                self._debug_train_print_count < self.debug_train_max_print
                if self.training
                else self._debug_eval_print_count < self.debug_eval_max_print
            )
            if allow:
                mode = "train" if self.training else "eval"
                owner_precision = owner_tp / max(owner_tp + owner_fp, 1)
                owner_recall = owner_tp / max(owner_tp + owner_fn, 1)
                owner_f1 = 2.0 * owner_precision * owner_recall / max(owner_precision + owner_recall, 1.0e-8)
                quality_nonempty_precision = quality_nonempty_tp / max(quality_nonempty_tp + quality_nonempty_fp, 1)
                quality_nonempty_recall = quality_nonempty_tp / max(quality_nonempty_tp + quality_nonempty_fn, 1)
                quality_nonempty_f1 = (
                    2.0 * quality_nonempty_precision * quality_nonempty_recall
                    / max(quality_nonempty_precision + quality_nonempty_recall, 1.0e-8)
                )
                quality_semantic_precision = quality_semantic_tp / max(quality_semantic_tp + quality_semantic_fp, 1)
                quality_semantic_recall = quality_semantic_tp / max(quality_semantic_tp + quality_semantic_fn, 1)
                quality_semantic_f1 = (
                    2.0 * quality_semantic_precision * quality_semantic_recall
                    / max(quality_semantic_precision + quality_semantic_recall, 1.0e-8)
                )
                seed_in = gt_seed_inside / max(gt_seed_total, 1)
                box_hit = gt_box_seed_hit / max(gt_box_seed_total, 1)
                child_in = child_inside / max(child_total, 1)
                child_nonempty_ratio = child_nonempty / max(child_occ_valid, 1)
                child_semantic_ratio = child_semantic_match / max(child_semantic_valid, 1)
                clusters_mean = sum(cluster_count_stat) / len(cluster_count_stat) if cluster_count_stat else 0.0
                owner_prob_mean = sum(owner_prob_stat) / len(owner_prob_stat) if owner_prob_stat else 0.0
                quality_prob_mean = sum(quality_prob_stat) / len(quality_prob_stat) if quality_prob_stat else 0.0
                quality_gate_mean = sum(quality_gate_stat) / len(quality_gate_stat) if quality_gate_stat else 0.0
                quality_geometry_mean = sum(quality_geometry_stat) / len(quality_geometry_stat) if quality_geometry_stat else 0.0
                quality_semantic_mean = sum(quality_semantic_stat) / len(quality_semantic_stat) if quality_semantic_stat else 0.0
                util = [round(100.0 * x / max(N, 1), 1) for x in total_counts]
                vis_prop_print = vis_proposed_t.detach().cpu().tolist()
                vis_reject_print = vis_rejected_t.detach().cpu().tolist()
                vis_strong_center_print = vis_strong_center_t.detach().cpu().tolist()
                vis_strong_any_print = vis_strong_any_t.detach().cpu().tolist()
                vis_strong_ratio_print = [f'{x:.3f}' for x in vis_strong_ratio_t.detach().cpu().tolist()]
                vis_weak_print = [f'{x:.3f}' for x in vis_weak_t.detach().cpu().tolist()]
                vis_unknown_print = [f'{x:.3f}' for x in vis_unknown_t.detach().cpu().tolist()]
                vis_car_print = vis_car_rejected_t.detach().cpu().tolist()
                vis_bus_print = vis_bus_rejected_t.detach().cpu().tolist()
                print(
                    "[V22C2DecoupledQuality] "
                    f"{mode} call={self._call_count}, boxes={boxes_total_stat}, owned/fallback={boxes_owned_stat}/{boxes_fallback_stat}, "
                    f"measured={filled.detach().cpu().tolist()}, ownerCand={owner_candidate_total}, owners={parent_counts}, "
                    f"ownerP/R/F1={owner_precision:.3f}/{owner_recall:.3f}/{owner_f1:.3f}, ownerProb={owner_prob_mean:.3f}, "
                    f"gtBoxSeed={box_hit:.3f}({gt_box_seed_hit}/{gt_box_seed_total}), "
                    f"seedInGT={seed_in:.3f}({gt_seed_inside}/{gt_seed_total}), childInGT={child_in:.3f}({child_inside}/{child_total}), "
                    f"childNE={child_nonempty_ratio:.3f}({child_nonempty}/{child_occ_valid}), "
                    f"childSem={child_semantic_ratio:.3f}({child_semantic_match}/{child_semantic_valid}), "
                    f"qNE_P/R/F1={quality_nonempty_precision:.3f}/{quality_nonempty_recall:.3f}/{quality_nonempty_f1:.3f}, "
                    f"qSem_P/R/F1={quality_semantic_precision:.3f}/{quality_semantic_recall:.3f}/{quality_semantic_f1:.3f}, "
                    f"qGeom/qSem/qFinal={quality_geometry_mean:.3f}/{quality_semantic_mean:.3f}/{quality_gate_mean:.3f}, "
                    f"qZeroGeom/Sem={quality_geometry_zero}/{quality_semantic_zero}({quality_known_total}), "
                    f"obj={object_counts}, local={local_counts}, generic={generic_counts}, oqg={fallback_counts}, "
                    f"visProp={vis_prop_print}, visReject={vis_reject_print}, "
                    f"visStrongCenter={vis_strong_center_print}, visStrongAny={vis_strong_any_print}, "
                    f"visStrongRatio={vis_strong_ratio_print}, visWeak={vis_weak_print}, visUnknown={vis_unknown_print}, "
                    f"visCarBusReject={list(zip(vis_car_print, vis_bus_print))}, "
                    f"proposed={candidate_counts}, capDrop={capacity_drop_counts}, total={total_counts}/{N}, util={util}%, "
                    f"dupRemoved={duplicate_removed}, clusters={clusters_mean:.2f}, ownerLoss={float(loss_owner.detach().item()):.4f}, "
                    f"qualityLoss={float(loss_child_quality.detach().item()):.4f}",
                    flush=True,
                )
                if self.training:
                    self._debug_train_print_count += 1
                else:
                    self._debug_eval_print_count += 1

        return {
            "representation": out_anchor,
            "rep_features": out_feat,
            "gaussian_source_types": out_source,
            "gaussian_diagnostic_branch_types": out_diag_branch,
            "gaussian_diagnostic_object_labels": out_diag_label,
            "gaussian_diagnostic_gt_indices": out_diag_gt,
            "gaussian_diagnostic_owner_scores": out_diag_owner,
            "gaussian_diagnostic_quality_gates": out_diag_gate,
            "gaussian_diagnostic_geometry_gates": out_diag_geometry_gate,
            "gaussian_diagnostic_semantic_gates": out_diag_semantic_gate,
            "gaussian_completion_render_gate": out_render_gate,
            "gaussian_filled_count": total_t,
            "seed_completion_count": completed_t,
            "seed_parent_count": parent_t,
            "object_completion_count": object_t,
            "seed_child_count": local_t,
            "generic_completion_count": generic_t,
            "seed_fallback_count": fallback_t,
            "total_gaussian_count": total_t,
            "loss_gaussian_owner": loss_owner,
            "loss_gaussian_child_quality": loss_child_quality,
            "visibility_local_proposed_count": vis_proposed_t,
            "visibility_local_rejected_count": vis_rejected_t,
            "visibility_local_strong_center_count": vis_strong_center_t,
            "visibility_local_strong_any_count": vis_strong_any_t,
            "visibility_local_strong_ratio_mean": vis_strong_ratio_t,
            "visibility_local_weak_ratio_mean": vis_weak_t,
            "visibility_local_unknown_ratio_mean": vis_unknown_t,
            "visibility_local_car_rejected_count": vis_car_rejected_t,
            "visibility_local_bus_rejected_count": vis_bus_rejected_t,
        }
