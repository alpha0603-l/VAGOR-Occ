from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# Source ids.
MEASURED_SOURCE = 0
COMPLETED_SOURCE = 1

# Completion type ids used by completion / resolver / fuser.
# 0 measured, 1 local child, 2 vehicle parent child, 3 box-level anchor.
COMPLETION_TYPE_MEASURED = 0
COMPLETION_TYPE_LOCAL = 1
COMPLETION_TYPE_VEHICLE = 2
COMPLETION_TYPE_BOX_ANCHOR = 3


@dataclass
class VisibilityQueryResult:
    """Per-point query result from the current-frame visibility grid.

    The fields are typed as Any to keep this structure compatible with both the
    original CPU/numpy path and the GPU/torch exact path.
    """

    valid: Any
    state: Any
    free_score: Any
    free_count: Any
    hit_count: Any
    resolution: Any
    near_current_surface: Any


@dataclass
class TemporalSupportResult:
    """Ten-sweep geometry after current-visibility reliability weighting."""

    xyz: Any
    intensity: Any
    delta_t: Any
    decay: Any
    is_current: Any
    sweep_id: Any
    weights: Any
    fit_mask: Any
    visibility_state: Any
    visibility_free_score: Any
    visibility_resolution: Any
    near_current_surface: Any
    strong_conflict_mask: Any
    debug_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GaussianFitBatchResult:
    """Batched Gaussian fit result used by the GPU/exact fitter path.

    This compatibility dataclass is needed because gaussian_fitter.py imports it
    directly.  The fields are intentionally Any so the same class works for
    torch tensors without forcing CPU/numpy conversion.
    """

    valid: Any
    mean: Any
    covariance: Any
    rotation: Any
    scales: Any
    eigenvalues: Any
    normal_rms: Any
    normal_residual_p90: Any
    normal_residual_p95: Any
    normal_residual_p99: Any
    normal_residual_robust_max: Any
    weighted_support: Any
    raw_point_count: Any
    current_point_count: Any
    current_weight: Any
    current_ratio: Any
    distinct_sweeps: Any
    temporal_conflict_ratio: Any
    connected_components: Any
    off_plane_point_count: Any
    off_plane_point_ratio: Any
    off_plane_weighted_support: Any
    off_plane_weight_ratio: Any
    off_plane_cluster_count: Any
    largest_off_plane_cluster_points: Any
    largest_off_plane_cluster_weight: Any
    off_plane_cluster_exists: Any


@dataclass
class GaussianFitResult:
    """Weighted anisotropic Gaussian fitted inside one octree node."""

    mean: Any
    covariance: Any
    rotation: Any
    scales: Any
    eigenvalues: Any

    normal_rms: Any
    normal_residual_p90: Any
    normal_residual_p95: Any
    normal_residual_p99: Any
    normal_residual_robust_max: Any

    weighted_support: Any
    raw_point_count: Any
    current_point_count: Any
    current_weight: Any
    current_ratio: Any
    distinct_sweeps: Any
    temporal_conflict_ratio: Any
    connected_components: Any

    off_plane_point_count: Any
    off_plane_point_ratio: Any
    off_plane_weighted_support: Any
    off_plane_weight_ratio: Any
    off_plane_cluster_count: Any
    largest_off_plane_cluster_points: Any
    largest_off_plane_cluster_weight: Any
    off_plane_cluster_exists: Any


@dataclass
class GaussianPrimitiveSet:
    """Flat Gaussian representation.

    This class is deliberately compatible with both paths in your current tree:
      - measured GPU/octree path may pass torch tensors plus completion_valid;
      - completion head may append torch tensors and extra completion metadata;
      - older CPU/debug code may still create numpy arrays.
    """

    means: Any
    scales: Any
    rotations: Any
    opacities: Any
    confidences: Any
    levels: Any
    support_count: Any
    weighted_support: Any
    current_ratio: Any
    distinct_sweeps: Any
    fit_residual: Any
    temporal_conflict: Any
    free_overlap: Any
    history_only: Any
    intensity_mean: Optional[Any] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    # Basic source / parent metadata.
    sources: Optional[Any] = None
    parent_indices: Optional[Any] = None
    child_indices: Optional[Any] = None

    # Required by the current GPU measured octree.
    completion_valid: Optional[Any] = None

    # Required by GT-box completion, resolver and fuser.
    completion_depth: Optional[Any] = None
    completion_opacity_gate: Optional[Any] = None
    completion_keep_prob: Optional[Any] = None
    completion_visibility_state: Optional[Any] = None
    completion_parent_valid: Optional[Any] = None
    completion_type: Optional[Any] = None
    completion_box_index: Optional[Any] = None

    @classmethod
    def empty(cls, device: Optional[Any] = None) -> "GaussianPrimitiveSet":
        """Create an empty set.

        If a torch device is supplied, return torch tensors on that device.  If
        no device is supplied, keep the original numpy empty behavior.
        """

        if device is not None and torch is not None:
            dev = torch.device(device)
            return cls(
                means=torch.empty((0, 3), device=dev, dtype=torch.float32),
                scales=torch.empty((0, 3), device=dev, dtype=torch.float32),
                rotations=torch.empty((0, 3, 3), device=dev, dtype=torch.float32),
                opacities=torch.empty((0,), device=dev, dtype=torch.float32),
                confidences=torch.empty((0,), device=dev, dtype=torch.float32),
                levels=torch.empty((0,), device=dev, dtype=torch.float32),
                support_count=torch.empty((0,), device=dev, dtype=torch.int32),
                weighted_support=torch.empty((0,), device=dev, dtype=torch.float32),
                current_ratio=torch.empty((0,), device=dev, dtype=torch.float32),
                distinct_sweeps=torch.empty((0,), device=dev, dtype=torch.int16),
                fit_residual=torch.empty((0,), device=dev, dtype=torch.float32),
                temporal_conflict=torch.empty((0,), device=dev, dtype=torch.float32),
                free_overlap=torch.empty((0,), device=dev, dtype=torch.float32),
                history_only=torch.empty((0,), device=dev, dtype=torch.bool),
                intensity_mean=torch.empty((0,), device=dev, dtype=torch.float32),
                debug_info={},
                sources=torch.empty((0,), device=dev, dtype=torch.uint8),
                parent_indices=torch.empty((0,), device=dev, dtype=torch.long),
                child_indices=torch.empty((0,), device=dev, dtype=torch.int16),
                completion_valid=torch.empty((0,), device=dev, dtype=torch.bool),
                completion_depth=torch.empty((0,), device=dev, dtype=torch.int16),
                completion_opacity_gate=torch.empty((0,), device=dev, dtype=torch.float32),
                completion_keep_prob=torch.empty((0,), device=dev, dtype=torch.float32),
                completion_visibility_state=torch.empty((0,), device=dev, dtype=torch.int32),
                completion_parent_valid=torch.empty((0,), device=dev, dtype=torch.bool),
                completion_type=torch.empty((0,), device=dev, dtype=torch.uint8),
                completion_box_index=torch.empty((0,), device=dev, dtype=torch.int32),
            )

        return cls(
            means=np.empty((0, 3), dtype=np.float32),
            scales=np.empty((0, 3), dtype=np.float32),
            rotations=np.empty((0, 3, 3), dtype=np.float32),
            opacities=np.empty((0,), dtype=np.float32),
            confidences=np.empty((0,), dtype=np.float32),
            levels=np.empty((0,), dtype=np.float32),
            support_count=np.empty((0,), dtype=np.int32),
            weighted_support=np.empty((0,), dtype=np.float32),
            current_ratio=np.empty((0,), dtype=np.float32),
            distinct_sweeps=np.empty((0,), dtype=np.int16),
            fit_residual=np.empty((0,), dtype=np.float32),
            temporal_conflict=np.empty((0,), dtype=np.float32),
            free_overlap=np.empty((0,), dtype=np.float32),
            history_only=np.empty((0,), dtype=bool),
            intensity_mean=np.empty((0,), dtype=np.float32),
            debug_info={},
            sources=np.empty((0,), dtype=np.uint8),
            parent_indices=np.empty((0,), dtype=np.int64),
            child_indices=np.empty((0,), dtype=np.int16),
            completion_valid=np.empty((0,), dtype=bool),
            completion_depth=np.empty((0,), dtype=np.int16),
            completion_opacity_gate=np.empty((0,), dtype=np.float32),
            completion_keep_prob=np.empty((0,), dtype=np.float32),
            completion_visibility_state=np.empty((0,), dtype=np.int32),
            completion_parent_valid=np.empty((0,), dtype=bool),
            completion_type=np.empty((0,), dtype=np.uint8),
            completion_box_index=np.empty((0,), dtype=np.int32),
        )

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def to(self, device: Any) -> "GaussianPrimitiveSet":
        """Move torch tensor fields to device; leave numpy/object fields unchanged."""

        if torch is None:
            return self
        kwargs: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                kwargs[key] = value.to(device)
            else:
                kwargs[key] = value
        return GaussianPrimitiveSet(**kwargs)


@dataclass
class GaussianProbeSampleResult:
    """One sampled scene processed by the measured-Gaussian probe."""

    visibility: Any
    temporal_support: TemporalSupportResult
    gaussians: GaussianPrimitiveSet
    debug_info: Dict[str, Any]
    batch_index: int = 0

    lidar_feature: Optional[Any] = None
    lidar_valid_mask: Optional[Any] = None
    lidar_debug_info: Dict[str, Any] = field(default_factory=dict)

    image_feature: Optional[Any] = None
    image_valid_mask: Optional[Any] = None
    image_camera_index: Optional[Any] = None
    image_uv: Optional[Any] = None
    image_valid_camera_count: Optional[Any] = None
    image_sample_uv: Optional[Any] = None
    image_sample_valid_mask: Optional[Any] = None
    image_sample_radius_feat: Optional[Any] = None
    image_debug_info: Dict[str, Any] = field(default_factory=dict)

    fused_feature: Optional[Any] = None
    fused_valid_mask: Optional[Any] = None
    fused_batch_index: Optional[Any] = None
    fused_feature_slices: Dict[str, Any] = field(default_factory=dict)
    fused_debug_info: Dict[str, Any] = field(default_factory=dict)
