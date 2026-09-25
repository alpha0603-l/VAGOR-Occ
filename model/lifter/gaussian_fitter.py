from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch

from .gaussian_structures import GaussianFitBatchResult, TemporalSupportResult


def _level_map(obj: Any, levels: Sequence[float]) -> Dict[float, float]:
    """Map config dictionaries with string/float keys onto the float level list."""
    if obj is None:
        return {float(level): 0.0 for level in levels}
    if not isinstance(obj, Mapping):
        return {float(level): float(obj) for level in levels}
    out: Dict[float, float] = {}
    default = obj.get("default", None)
    for level in levels:
        key_f = float(level)
        candidates = (key_f, str(key_f), f"{key_f:g}")
        value = None
        for key in candidates:
            if key in obj:
                value = obj[key]
                break
        if value is None:
            value = default if default is not None else 0.0
        out[key_f] = float(value)
    return out


class WeightedGaussianFitter:
    """Batched weighted Gaussian fitter used by the GPU top-down octree.

    All heavy operations stay on torch tensors, normally CUDA tensors.  The class
    deliberately mirrors the DAOcc measured-Gaussian statistics used by
    GaussianOctreeBuilder: weighted PCA, current/history support, temporal
    conflict, off-plane complexity and scale clamping.
    """

    def __init__(
        self,
        sigma_scale_factor: float = 2.0,
        min_gaussian_scale: float = 0.05,
        max_scale_ratio: float = 0.45,
        min_normal_scale: float = 0.04,
        max_normal_scale_ratio: float = 0.10,
        covariance_epsilon: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        self.sigma_scale_factor = float(sigma_scale_factor)
        self.min_gaussian_scale = float(min_gaussian_scale)
        self.max_scale_ratio = float(max_scale_ratio)
        self.min_normal_scale = float(min_normal_scale)
        self.max_normal_scale_ratio = float(max_normal_scale_ratio)
        self.covariance_epsilon = float(covariance_epsilon)

    @torch.no_grad()
    def fit_groups(
        self,
        support: TemporalSupportResult,
        active_indices: torch.Tensor,
        group_inverse: torch.Tensor,
        node_min: torch.Tensor,
        node_size: float,
        off_plane_distance: float,
        off_plane_cluster_resolution: float,
        off_plane_min_cluster_points: int,
        off_plane_min_cluster_weighted_support: float,
        max_normal_scale: float,
    ) -> GaussianFitBatchResult:
        device = support.xyz.device
        dtype = support.xyz.dtype
        active = active_indices.long()
        inv = group_inverse.long()
        num_groups = int(node_min.shape[0])
        if num_groups == 0 or active.numel() == 0:
            return self._empty(num_groups, device, dtype)

        xyz = support.xyz[active].to(dtype=torch.float32)
        weights = support.weights[active].to(dtype=torch.float32).clamp_min(0.0)
        is_current = support.is_current[active].bool()
        sweep_id = support.sweep_id[active].long().clamp_min(0).clamp_max(32767)
        strong_conflict = support.strong_conflict_mask[active].bool()
        intensity = support.intensity[active].to(dtype=torch.float32)

        one_i32 = torch.ones((active.numel(),), device=device, dtype=torch.int32)
        one_f = torch.ones((active.numel(),), device=device, dtype=torch.float32)
        raw_point_count = torch.zeros((num_groups,), device=device, dtype=torch.int32).index_add_(0, inv, one_i32)
        weighted_support = torch.zeros((num_groups,), device=device, dtype=torch.float32).index_add_(0, inv, weights)
        denom = weighted_support.clamp_min(1e-6)

        weighted_xyz = xyz * weights[:, None]
        mean = torch.zeros((num_groups, 3), device=device, dtype=torch.float32).index_add_(0, inv, weighted_xyz)
        mean = mean / denom[:, None]

        centered = xyz - mean[inv]
        cov_terms = torch.stack(
            [
                centered[:, 0] * centered[:, 0],
                centered[:, 1] * centered[:, 1],
                centered[:, 2] * centered[:, 2],
                centered[:, 0] * centered[:, 1],
                centered[:, 0] * centered[:, 2],
                centered[:, 1] * centered[:, 2],
            ],
            dim=-1,
        ) * weights[:, None]
        cov_terms_sum = torch.zeros((num_groups, 6), device=device, dtype=torch.float32).index_add_(0, inv, cov_terms)
        cov_terms_sum = cov_terms_sum / denom[:, None]
        covariance = torch.zeros((num_groups, 3, 3), device=device, dtype=torch.float32)
        covariance[:, 0, 0] = cov_terms_sum[:, 0]
        covariance[:, 1, 1] = cov_terms_sum[:, 1]
        covariance[:, 2, 2] = cov_terms_sum[:, 2]
        covariance[:, 0, 1] = covariance[:, 1, 0] = cov_terms_sum[:, 3]
        covariance[:, 0, 2] = covariance[:, 2, 0] = cov_terms_sum[:, 4]
        covariance[:, 1, 2] = covariance[:, 2, 1] = cov_terms_sum[:, 5]
        eye = torch.eye(3, device=device, dtype=torch.float32).view(1, 3, 3)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance + eye * self.covariance_epsilon)
        eigenvalues = torch.flip(eigenvalues.clamp_min(self.covariance_epsilon), dims=[-1])
        rotation = torch.flip(eigenvectors, dims=[-1]).contiguous()

        # Keep a right-handed local frame, matching the measured-Gaussian renderer.
        cross = torch.cross(rotation[:, :, 0], rotation[:, :, 1], dim=-1)
        handed = (cross * rotation[:, :, 2]).sum(dim=-1)
        rotation[:, :, 2] = rotation[:, :, 2] * torch.where(handed[:, None] >= 0, torch.ones_like(handed[:, None]), -torch.ones_like(handed[:, None]))

        scales = self.sigma_scale_factor * torch.sqrt(eigenvalues)
        max_tangent = float(node_size) * self.max_scale_ratio
        max_normal = min(float(max_normal_scale), float(node_size) * self.max_normal_scale_ratio)
        scales[:, 0] = scales[:, 0].clamp(self.min_gaussian_scale, max_tangent)
        scales[:, 1] = scales[:, 1].clamp(self.min_gaussian_scale, max_tangent)
        scales[:, 2] = scales[:, 2].clamp(self.min_normal_scale, max_normal)
        normal_rms = torch.sqrt(eigenvalues[:, 2].clamp_min(self.covariance_epsilon))

        current_point_count = torch.zeros((num_groups,), device=device, dtype=torch.int32).index_add_(0, inv, is_current.to(torch.int32))
        current_weight = torch.zeros((num_groups,), device=device, dtype=torch.float32).index_add_(0, inv, weights * is_current.float())
        current_ratio = current_weight / denom
        strong_weight = torch.zeros((num_groups,), device=device, dtype=torch.float32).index_add_(0, inv, weights * strong_conflict.float())
        temporal_conflict_ratio = strong_weight / denom
        intensity_mean = torch.zeros((num_groups,), device=device, dtype=torch.float32).index_add_(0, inv, intensity * weights) / denom

        pair_key = inv * 32768 + sweep_id
        pair_unique = torch.unique(pair_key, sorted=False)
        pair_group = torch.div(pair_unique, 32768, rounding_mode="floor")
        distinct_sweeps_i32 = torch.zeros((num_groups,), device=device, dtype=torch.int32)
        if pair_group.numel() > 0:
            distinct_sweeps_i32.index_add_(0, pair_group.long(), torch.ones_like(pair_group, dtype=torch.int32))
        distinct_sweeps = distinct_sweeps_i32.clamp_max(32767).to(torch.int16)

        normal_axis = rotation[:, :, 2]
        normal_dist = torch.abs((centered * normal_axis[inv]).sum(dim=-1))
        off_mask = normal_dist > float(off_plane_distance)
        off_plane_point_count = torch.zeros((num_groups,), device=device, dtype=torch.int32).index_add_(0, inv, off_mask.to(torch.int32))
        off_plane_weighted_support = torch.zeros((num_groups,), device=device, dtype=torch.float32).index_add_(0, inv, weights * off_mask.float())
        off_plane_point_ratio = off_plane_point_count.float() / raw_point_count.float().clamp_min(1.0)
        off_plane_weight_ratio = off_plane_weighted_support / denom

        # GPU-friendly conservative complexity proxy.  If a group has enough
        # off-plane weighted support, treat it as one off-plane cluster.  This
        # preserves the original rejection intent without CPU clustering.
        off_plane_cluster_exists = (off_plane_point_count >= int(off_plane_min_cluster_points)) & (off_plane_weighted_support >= float(off_plane_min_cluster_weighted_support))
        off_plane_cluster_count = off_plane_cluster_exists.to(torch.int32)
        largest_off_plane_cluster_points = off_plane_point_count.clone()
        largest_off_plane_cluster_weight = off_plane_weighted_support.clone()
        connected_components = torch.where(off_plane_cluster_exists, torch.full((num_groups,), 2, device=device, dtype=torch.int32), torch.ones((num_groups,), device=device, dtype=torch.int32))

        # Robust residual summaries.  These are kept simple to avoid CPU loops;
        # the octree decisions use normal_rms and off-plane complexity above.
        normal_residual_p90 = torch.zeros((num_groups,), device=device, dtype=torch.float32).index_add_(0, inv, normal_dist * weights) / denom
        normal_residual_p95 = normal_residual_p90.clone()
        normal_residual_p99 = normal_residual_p90.clone()
        normal_residual_robust_max = torch.zeros((num_groups,), device=device, dtype=torch.float32)
        normal_residual_robust_max.scatter_reduce_(0, inv, normal_dist.float(), reduce="amax", include_self=False)
        valid = torch.isfinite(mean).all(dim=-1) & torch.isfinite(scales).all(dim=-1) & (raw_point_count > 0) & (weighted_support > 0)

        out = GaussianFitBatchResult(
            valid=valid,
            mean=mean.to(dtype=dtype),
            covariance=covariance.to(dtype=dtype),
            rotation=rotation.to(dtype=dtype),
            scales=scales.to(dtype=dtype),
            eigenvalues=eigenvalues.to(dtype=dtype),
            normal_rms=normal_rms.to(dtype=dtype),
            normal_residual_p90=normal_residual_p90.to(dtype=dtype),
            normal_residual_p95=normal_residual_p95.to(dtype=dtype),
            normal_residual_p99=normal_residual_p99.to(dtype=dtype),
            normal_residual_robust_max=normal_residual_robust_max.to(dtype=dtype),
            weighted_support=weighted_support.to(dtype=dtype),
            raw_point_count=raw_point_count,
            current_point_count=current_point_count,
            current_weight=current_weight.to(dtype=dtype),
            current_ratio=current_ratio.to(dtype=dtype),
            distinct_sweeps=distinct_sweeps,
            temporal_conflict_ratio=temporal_conflict_ratio.to(dtype=dtype),
            connected_components=connected_components,
            off_plane_point_count=off_plane_point_count,
            off_plane_point_ratio=off_plane_point_ratio.to(dtype=dtype),
            off_plane_weighted_support=off_plane_weighted_support.to(dtype=dtype),
            off_plane_weight_ratio=off_plane_weight_ratio.to(dtype=dtype),
            off_plane_cluster_count=off_plane_cluster_count,
            largest_off_plane_cluster_points=largest_off_plane_cluster_points,
            largest_off_plane_cluster_weight=largest_off_plane_cluster_weight.to(dtype=dtype),
            off_plane_cluster_exists=off_plane_cluster_exists,
        )
        out.intensity_mean = intensity_mean.to(dtype=dtype)
        return out

    def _empty(self, num_groups: int, device: torch.device, dtype: torch.dtype) -> GaussianFitBatchResult:
        n = int(num_groups)
        zf = torch.zeros((n,), device=device, dtype=dtype)
        zi = torch.zeros((n,), device=device, dtype=torch.int32)
        zb = torch.zeros((n,), device=device, dtype=torch.bool)
        out = GaussianFitBatchResult(
            valid=zb,
            mean=torch.zeros((n, 3), device=device, dtype=dtype),
            covariance=torch.zeros((n, 3, 3), device=device, dtype=dtype),
            rotation=torch.eye(3, device=device, dtype=dtype).view(1, 3, 3).expand(n, -1, -1).clone(),
            scales=torch.zeros((n, 3), device=device, dtype=dtype),
            eigenvalues=torch.zeros((n, 3), device=device, dtype=dtype),
            normal_rms=zf,
            normal_residual_p90=zf,
            normal_residual_p95=zf,
            normal_residual_p99=zf,
            normal_residual_robust_max=zf,
            weighted_support=zf,
            raw_point_count=zi,
            current_point_count=zi,
            current_weight=zf,
            current_ratio=zf,
            distinct_sweeps=zi.to(torch.int16),
            temporal_conflict_ratio=zf,
            connected_components=zi,
            off_plane_point_count=zi,
            off_plane_point_ratio=zf,
            off_plane_weighted_support=zf,
            off_plane_weight_ratio=zf,
            off_plane_cluster_count=zi,
            largest_off_plane_cluster_points=zi,
            largest_off_plane_cluster_weight=zf,
            off_plane_cluster_exists=zb,
        )
        out.intensity_mean = zf
        return out
