from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

from .gaussian_structures import TemporalSupportResult, VisibilityQueryResult
from .visibility_grid import MIXED, STRONG_FREE, SURFACE_HIT, UNKNOWN, WEAK_FREE, _pack_3d


def _quantiles(values: torch.Tensor) -> Dict[str, float]:
    values = values.reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"min": 0.0, "p01": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    q = torch.quantile(values.float(), torch.tensor([0.0, 0.01, 0.05, 0.50, 0.95, 1.0], device=values.device))
    return {"min": float(q[0].item()), "p01": float(q[1].item()), "p05": float(q[2].item()), "p50": float(q[3].item()), "p95": float(q[4].item()), "max": float(q[5].item())}


def query_visibility_points(visibility: Any, xyz: torch.Tensor, neighbor_radius: int = 1) -> VisibilityQueryResult:
    xyz = torch.as_tensor(xyz, device=visibility.coarse_state.device, dtype=torch.float32).reshape(-1, 3)
    n = xyz.shape[0]
    device = xyz.device
    grid_min = torch.tensor(visibility.grid_range[:3], device=device, dtype=torch.float32)
    grid_max = torch.tensor(visibility.grid_range[3:], device=device, dtype=torch.float32)
    coarse_size = float(visibility.coarse_voxel_size)
    refined_size = float(visibility.refined_voxel_size)
    coarse_shape = tuple(int(v) for v in visibility.coarse_shape)
    coarse_shape_t = torch.tensor(coarse_shape, device=device, dtype=torch.long)
    refined_shape_t = coarse_shape_t * 2
    valid = torch.isfinite(xyz).all(dim=1) & (xyz >= grid_min.view(1, 3)).all(dim=1) & (xyz < grid_max.view(1, 3)).all(dim=1)
    state = torch.full((n,), UNKNOWN, device=device, dtype=torch.uint8)
    free_score = torch.zeros((n,), device=device, dtype=torch.float32)
    free_count = torch.zeros((n,), device=device, dtype=torch.int32)
    hit_count = torch.zeros((n,), device=device, dtype=torch.int32)
    resolution = torch.zeros((n,), device=device, dtype=torch.float32)
    near_surface = torch.zeros((n,), device=device, dtype=torch.bool)
    if valid.any():
        rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
        cidx = torch.floor((xyz[rows] - grid_min.view(1, 3)) / coarse_size).long()
        # GPU float32 boundary arithmetic may produce index == shape for points
        # that passed the xyz < grid_max check. Clamp only the index used for
        # tensor addressing; this matches the CPU intent while avoiding CUDA
        # device-side asserts from out-of-range advanced indexing.
        cidx = torch.maximum(cidx, torch.zeros_like(cidx))
        cidx = torch.minimum(cidx, (coarse_shape_t - 1).view(1, 3))
        state[rows] = visibility.coarse_state[cidx[:, 0], cidx[:, 1], cidx[:, 2]]
        free_score[rows] = visibility.coarse_free_score[cidx[:, 0], cidx[:, 1], cidx[:, 2]]
        free_count[rows] = visibility.coarse_free_count[cidx[:, 0], cidx[:, 1], cidx[:, 2]]
        hit_count[rows] = visibility.coarse_hit_count[cidx[:, 0], cidx[:, 1], cidx[:, 2]]
        resolution[rows] = coarse_size
        surface = (visibility.coarse_hit_count > 0) | (visibility.coarse_state == SURFACE_HIT) | (visibility.coarse_state == MIXED)
        if neighbor_radius > 0:
            near = torch.zeros_like(surface)
            for dx in range(-neighbor_radius, neighbor_radius + 1):
                for dy in range(-neighbor_radius, neighbor_radius + 1):
                    for dz in range(-neighbor_radius, neighbor_radius + 1):
                        shifted = torch.zeros_like(surface)
                        xs = slice(max(0, dx), surface.shape[0] + min(0, dx))
                        ys = slice(max(0, dy), surface.shape[1] + min(0, dy))
                        zs = slice(max(0, dz), surface.shape[2] + min(0, dz))
                        xt = slice(max(0, -dx), surface.shape[0] - max(0, dx))
                        yt = slice(max(0, -dy), surface.shape[1] - max(0, dy))
                        zt = slice(max(0, -dz), surface.shape[2] - max(0, dz))
                        shifted[xt, yt, zt] = surface[xs, ys, zs]
                        near |= shifted
            surface = near
        near_surface[rows] = surface[cidx[:, 0], cidx[:, 1], cidx[:, 2]]
        parents = visibility.refined_parent_indices.long()
        if parents.numel() > 0:
            parents = parents.to(device=device, dtype=torch.long)
            parent_valid = (parents >= 0).all(dim=1) & (parents < coarse_shape_t.view(1, 3)).all(dim=1)
            parents = parents[parent_valid]
        if parents.numel() > 0:
            total = int(coarse_shape[0] * coarse_shape[1] * coarse_shape[2])
            parent_lookup = torch.full((total,), -1, device=device, dtype=torch.long)
            pflat = _pack_3d(parents, coarse_shape)
            parent_lookup[pflat] = torch.arange(parents.shape[0], device=device, dtype=torch.long)
            cflat = _pack_3d(cidx, coarse_shape)
            rr = parent_lookup[cflat]
            has = rr >= 0
            if has.any():
                tr = rows[has]
                ridx = torch.floor((xyz[tr] - grid_min.view(1, 3)) / refined_size).long()
                ridx = torch.maximum(ridx, torch.zeros_like(ridx))
                ridx = torch.minimum(ridx, (refined_shape_t - 1).view(1, 3))
                local = ridx % 2
                cid = local[:, 0] * 4 + local[:, 1] * 2 + local[:, 2]
                pr = rr[has]
                state[tr] = visibility.refined_state[pr, cid]
                free_score[tr] = visibility.refined_free_score[pr, cid]
                free_count[tr] = visibility.refined_free_count[pr, cid]
                hit_count[tr] = visibility.refined_hit_count[pr, cid]
                resolution[tr] = refined_size
    return VisibilityQueryResult(valid=valid, state=state, free_score=free_score, free_count=free_count, hit_count=hit_count, resolution=resolution, near_current_surface=near_surface)


class TemporalSupportWeighter:
    def __init__(
        self,
        current_weight: float = 1.0,
        min_point_weight: float = 0.01,
        strong_free_gamma: float = 4.0,
        weak_free_gamma: float = 2.0,
        mixed_factor: float = 0.7,
        visibility_neighbor_radius: int = 1,
        surface_guard_enabled: bool = True,
        surface_guard_min_factor: float = 0.15,
        time_bin_size: float = 0.02,
        xyz_dims: Sequence[int] = (0, 1, 2),
        intensity_dim: int = 3,
        time_lag_dim: int = 4,
        decay_dim: int = 5,
        current_flag_dim: int = 6,
    ) -> None:
        self.current_weight = float(current_weight)
        self.min_point_weight = float(min_point_weight)
        self.strong_free_gamma = float(strong_free_gamma)
        self.weak_free_gamma = float(weak_free_gamma)
        self.mixed_factor = float(mixed_factor)
        self.visibility_neighbor_radius = max(int(visibility_neighbor_radius), 0)
        self.surface_guard_enabled = bool(surface_guard_enabled)
        self.surface_guard_min_factor = float(surface_guard_min_factor)
        self.time_bin_size = float(time_bin_size)
        self.xyz_dims = tuple(int(v) for v in xyz_dims)
        self.intensity_dim = int(intensity_dim)
        self.time_lag_dim = int(time_lag_dim)
        self.decay_dim = int(decay_dim)
        self.current_flag_dim = int(current_flag_dim)

    def __call__(self, point_tensor: Any, visibility: Any) -> TemporalSupportResult:
        if hasattr(point_tensor, "tensor"):
            pts = point_tensor.tensor.float()
        else:
            pts = torch.as_tensor(point_tensor, device=getattr(point_tensor, "device", None), dtype=torch.float32)
        if pts.ndim != 2:
            raise ValueError(f"points must be [N,C], got {tuple(pts.shape)}")
        xyz = pts[:, list(self.xyz_dims)].contiguous()
        intensity = pts[:, self.intensity_dim].float()
        delta_t = torch.clamp(pts[:, self.time_lag_dim].float(), min=0.0)
        decay = torch.clamp(pts[:, self.decay_dim].float(), 0.0, 1.0)
        is_current = pts[:, self.current_flag_dim] > 0.5
        sweep_id = torch.round(delta_t / max(self.time_bin_size, 1e-6)).to(torch.int16)
        query = query_visibility_points(visibility, xyz, neighbor_radius=self.visibility_neighbor_radius)
        weights = decay.clone()
        weights[is_current] = self.current_weight
        history = ~is_current
        strong = history & (query.state == STRONG_FREE)
        weak = history & (query.state == WEAK_FREE)
        mixed = history & (query.state == MIXED)
        weights[strong] = decay[strong] * torch.exp(-self.strong_free_gamma * query.free_score[strong])
        weights[weak] = decay[weak] * torch.exp(-self.weak_free_gamma * query.free_score[weak])
        weights[mixed] = decay[mixed] * self.mixed_factor
        guarded = torch.zeros_like(history)
        if self.surface_guard_enabled:
            guard_candidates = (strong | weak) & query.near_current_surface
            guarded[guard_candidates] = True
            floor_weight = decay[guard_candidates] * self.surface_guard_min_factor
            weights[guard_candidates] = torch.maximum(weights[guard_candidates], floor_weight)
        weights = torch.clamp(weights, 0.0, self.current_weight).float()
        fit_mask = torch.isfinite(xyz).all(dim=1) & (weights >= self.min_point_weight)
        strong_conflict = history & (query.state == STRONG_FREE)
        history_weights = weights[history]
        history_decay = decay[history]
        debug = {
            "total_point_count": int(pts.shape[0]),
            "current_point_count": int(is_current.sum().item()),
            "history_point_count": int(history.sum().item()),
            "fit_point_count": int(fit_mask.sum().item()),
            "dropped_by_weight_count": int((history & ~fit_mask).sum().item()),
            "visibility_valid_count": int(query.valid.sum().item()),
            "visibility_invalid_count": int((~query.valid).sum().item()),
            "history_state_counts": {
                "unknown": int((history & (query.state == UNKNOWN)).sum().item()),
                "weak_free": int((history & (query.state == WEAK_FREE)).sum().item()),
                "strong_free": int((history & (query.state == STRONG_FREE)).sum().item()),
                "surface_hit": int((history & (query.state == SURFACE_HIT)).sum().item()),
                "mixed": int((history & (query.state == MIXED)).sum().item()),
            },
            "strong_conflict_count": int(strong_conflict.sum().item()),
            "surface_guarded_count": int(guarded.sum().item()),
            "distinct_sweep_ids": int(torch.unique(sweep_id).numel()),
            "history_decay_stats": _quantiles(history_decay),
            "history_weight_stats": _quantiles(history_weights),
            "all_weight_stats": _quantiles(weights),
            "weighted_current_support": float(weights[is_current].sum().item()),
            "weighted_history_support": float(weights[history].sum().item()),
            "weighted_total_support": float(weights.sum().item()),
            "strong_conflict_original_decay_sum": float(decay[strong_conflict].sum().item()),
            "strong_conflict_final_weight_sum": float(weights[strong_conflict].sum().item()),
        }
        return TemporalSupportResult(xyz=xyz, intensity=intensity, delta_t=delta_t, decay=decay, is_current=is_current, sweep_id=sweep_id, weights=weights, fit_mask=fit_mask, visibility_state=query.state, visibility_free_score=query.free_score, visibility_resolution=query.resolution, near_current_surface=query.near_current_surface, strong_conflict_mask=strong_conflict, debug_info=debug)
