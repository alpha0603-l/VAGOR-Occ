from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from .gaussian_fitter import WeightedGaussianFitter, _level_map
from .gaussian_structures import GaussianPrimitiveSet, TemporalSupportResult
from .temporal_support import query_visibility_points
from .visibility_grid import STRONG_FREE, _pack_3d


def _stats_t(values: torch.Tensor) -> Dict[str, float]:
    values = values.reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"min": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    q = torch.quantile(values.float(), torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0], device=values.device))
    return {"min": float(q[0].item()), "p05": float(q[1].item()), "p50": float(q[2].item()), "p95": float(q[3].item()), "max": float(q[4].item()), "mean": float(values.float().mean().item())}


class GaussianOctreeBuilder:
    """Top-down measured Gaussian octree implemented with CUDA tensors.

    This keeps the original accept/split/fallback decision order. It processes one
    octree level at a time on GPU: accepted nodes stop, rejected non-leaf nodes are
    forwarded to the next finer level. It does not generate independent Gaussians
    at every scale.
    """

    def __init__(
        self,
        octree_range: Sequence[float],
        levels: Sequence[float] = (3.2, 1.6, 0.8, 0.4),
        min_points_by_level: Mapping[Any, Any] = None,
        min_weighted_support_by_level: Mapping[Any, Any] = None,
        max_normal_rms_by_level: Mapping[Any, Any] = None,
        max_normal_scale_by_level: Mapping[Any, Any] = None,
        max_connected_components: int = 1,
        max_temporal_conflict_ratio: float = 0.35,
        history_only_min_sweeps: int = 3,
        history_only_min_weighted_support: float = 6.0,
        history_only_confidence_scale: float = 0.5,
        support_tau: float = 10.0,
        fit_sigma: float = 0.12,
        min_opacity: float = 0.10,
        max_opacity: float = 0.95,
        free_check_enabled: bool = True,
        free_check_num_samples: int = 32,
        free_overlap_warn: float = 0.10,
        free_overlap_reject: float = 0.35,
        free_overlap_shrink_factor: float = 0.8,
        free_overlap_max_shrink_steps: int = 3,
        free_query_neighbor_radius: int = 0,
        max_processed_nodes: int = 250000,
        max_gaussians: int = 50000,
        surface_complexity: Dict[str, Any] = None,
        ground_refinement: Dict[str, Any] = None,
        fitter: Dict[str, Any] = None,
    ) -> None:
        self.range = tuple(float(v) for v in octree_range)
        self.levels = tuple(float(v) for v in levels)
        min_points_by_level = min_points_by_level or {3.2: 20, 1.6: 12, 0.8: 7, 0.4: 3}
        min_weighted_support_by_level = min_weighted_support_by_level or {3.2: 12.0, 1.6: 8.0, 0.8: 4.0, 0.4: 2.0}
        max_normal_rms_by_level = max_normal_rms_by_level or {3.2: 0.12, 1.6: 0.09, 0.8: 0.07, 0.4: 0.05}
        max_normal_scale_by_level = max_normal_scale_by_level or {3.2: 0.25, 1.6: 0.16, 0.8: 0.10, 0.4: 0.06}
        self.min_points = {k: int(v) for k, v in _level_map(min_points_by_level, self.levels).items()}
        self.min_support = _level_map(min_weighted_support_by_level, self.levels)
        self.max_normal_rms = _level_map(max_normal_rms_by_level, self.levels)
        self.max_normal_scale = _level_map(max_normal_scale_by_level, self.levels)
        self.max_connected_components = max(int(max_connected_components), 1)
        self.max_temporal_conflict_ratio = float(max_temporal_conflict_ratio)
        self.history_only_min_sweeps = max(int(history_only_min_sweeps), 1)
        self.history_only_min_weighted_support = float(history_only_min_weighted_support)
        self.history_only_confidence_scale = float(history_only_confidence_scale)
        self.support_tau = float(support_tau)
        self.fit_sigma = float(fit_sigma)
        self.min_opacity = float(min_opacity)
        self.max_opacity = float(max_opacity)
        self.free_check_enabled = bool(free_check_enabled)
        self.free_check_num_samples = max(int(free_check_num_samples), 8)
        self.free_overlap_warn = float(free_overlap_warn)
        self.free_overlap_reject = float(free_overlap_reject)
        self.free_overlap_shrink_factor = float(free_overlap_shrink_factor)
        self.free_overlap_max_shrink_steps = max(int(free_overlap_max_shrink_steps), 0)
        self.free_query_neighbor_radius = max(int(free_query_neighbor_radius), 0)
        self.max_processed_nodes = max(int(max_processed_nodes), 1)
        self.max_gaussians = max(int(max_gaussians), 1)
        complexity = dict(surface_complexity or {})
        self.surface_complexity_enabled = bool(complexity.get("enabled", True))
        self.off_plane_distance = _level_map(complexity.get("off_plane_distance_by_level", {3.2: 0.15, 1.6: 0.10, 0.8: 0.07, 0.4: 0.05}), self.levels)
        self.max_off_plane_ratio = _level_map(complexity.get("max_off_plane_ratio_by_level", {3.2: 0.03, 1.6: 0.05, 0.8: 0.08, 0.4: 0.15}), self.levels)
        self.min_off_plane_points = {k: int(v) for k, v in _level_map(complexity.get("min_off_plane_points_by_level", {3.2: 4, 1.6: 3, 0.8: 2, 0.4: 2}), self.levels).items()}
        self.min_off_plane_weighted_support = _level_map(complexity.get("min_off_plane_weighted_support_by_level", {3.2: 2.0, 1.6: 1.5, 0.8: 1.0, 0.4: 0.5}), self.levels)
        self.off_plane_cluster_resolution = float(complexity.get("off_plane_cluster_resolution", 0.20))
        self.off_plane_min_cluster_points = max(int(complexity.get("min_cluster_points", 3)), 1)
        self.off_plane_min_cluster_weighted_support = float(complexity.get("min_cluster_weighted_support", 1.5))
        self.leaf_surface_complexity_confidence_scale = float(complexity.get("leaf_surface_complexity_confidence_scale", 0.45))

        # V24-G: geometry-aware refinement of large, near-horizontal low surfaces.
        # The decision itself is a single batched CUDA boolean mask inside the
        # existing fixed four-level octree loop.  No point/Gaussian is moved to
        # CPU and no per-node Python loop is introduced.
        ground_cfg = dict(ground_refinement or {})
        self.ground_refinement_enabled = bool(ground_cfg.get("enabled", False))
        source_levels = ground_cfg.get("source_levels", (3.2,))
        self.ground_refinement_source_levels = tuple(float(v) for v in source_levels)
        self.ground_refinement_min_abs_normal_z = float(ground_cfg.get("min_abs_normal_z", 0.85))
        self.ground_refinement_min_mean_z = float(ground_cfg.get("min_mean_z", self.range[2]))
        self.ground_refinement_max_mean_z = float(ground_cfg.get("max_mean_z", 0.60))
        self.ground_refinement_min_major_scale = float(ground_cfg.get("min_major_scale", 0.80))
        self.ground_refinement_min_minor_scale = float(ground_cfg.get("min_minor_scale", 0.45))

        self.fitter = WeightedGaussianFitter(**(fitter or {}))
        self._unit_samples_cache: Dict[torch.device, torch.Tensor] = {}

    def build(self, support: TemporalSupportResult, visibility: Any) -> GaussianPrimitiveSet:
        start = time.perf_counter()
        device = support.xyz.device
        rmin = torch.tensor(self.range[:3], device=device, dtype=torch.float32)
        rmax = torch.tensor(self.range[3:], device=device, dtype=torch.float32)
        valid = support.fit_mask & (support.xyz >= rmin.view(1, 3)).all(dim=1) & (support.xyz < rmax.view(1, 3)).all(dim=1)
        active_indices = torch.nonzero(valid, as_tuple=False).reshape(-1).long()
        if active_indices.numel() == 0:
            out = GaussianPrimitiveSet.empty(device)
            out.debug_info = {"elapsed_sec": 0.0, "reason": "no_valid_support"}
            return out
        rows = []
        active = active_indices
        processed_nodes_total = 0
        debug_level = {}
        for level_idx, node_size in enumerate(self.levels):
            if active.numel() == 0 or len(rows) >= self.max_gaussians or processed_nodes_total >= self.max_processed_nodes:
                break
            point_xyz = support.xyz[active]
            grid = torch.floor((point_xyz - rmin.view(1, 3)) / float(node_size)).long()
            flat = self._pack_grid(grid)
            uniq, inv = torch.unique(flat, sorted=True, return_inverse=True)
            num_groups = int(uniq.numel())
            processed_nodes_total += num_groups
            grid_index = self._unpack_grid(uniq)
            node_min = rmin.view(1, 3) + grid_index.float() * float(node_size)
            fit = self.fitter.fit_groups(
                support,
                active,
                inv,
                node_min,
                float(node_size),
                off_plane_distance=self.off_plane_distance[float(node_size)],
                off_plane_cluster_resolution=self.off_plane_cluster_resolution,
                off_plane_min_cluster_points=self.off_plane_min_cluster_points,
                off_plane_min_cluster_weighted_support=self.off_plane_min_cluster_weighted_support,
                max_normal_scale=self.max_normal_scale[float(node_size)],
            )
            surface_complexity = (
                self.surface_complexity_enabled
                & fit.off_plane_cluster_exists
                & (fit.off_plane_point_count >= self.min_off_plane_points[float(node_size)])
                & (fit.off_plane_weighted_support >= self.min_off_plane_weighted_support[float(node_size)])
                & (fit.off_plane_point_ratio > self.max_off_plane_ratio[float(node_size)])
            )
            history_only = fit.current_point_count == 0
            support_ok = (fit.raw_point_count >= self.min_points[float(node_size)]) & (fit.weighted_support >= self.min_support[float(node_size)])
            residual_ok = fit.normal_rms <= self.max_normal_rms[float(node_size)]
            components_ok = fit.connected_components <= self.max_connected_components
            surface_ok = ~surface_complexity
            conflict_ok = fit.temporal_conflict_ratio <= self.max_temporal_conflict_ratio
            history_ok = (~history_only) | ((fit.distinct_sweeps >= self.history_only_min_sweeps) & (fit.weighted_support >= self.history_only_min_weighted_support))
            scales, free_overlap, shrink_steps = self._check_free_overlap_batch(fit.mean, fit.scales, fit.rotation, visibility)
            free_ok = free_overlap < self.free_overlap_reject
            base_accept = fit.valid & support_ok & residual_ok & components_ok & surface_ok & conflict_ok & history_ok & free_ok

            # V24-G Ground-aware Octree Refinement.  A coarse node that is:
            #   1) otherwise geometrically acceptable,
            #   2) near horizontal,
            #   3) low in the ego coordinate system, and
            #   4) spatially broad in both tangent directions
            # is deliberately passed to the next finer level.  This prevents one
            # 3.2 m node / one semantic vector from spanning road, sidewalk and
            # other-flat boundaries.  Vertical walls retain the original coarse
            # representation.  All operations below stay on the current CUDA
            # device; source-level membership is a fixed Python scalar check.
            refine_source_level = self.ground_refinement_enabled and any(
                abs(float(node_size) - float(v)) < 1e-6
                for v in self.ground_refinement_source_levels
            )
            if refine_source_level:
                abs_normal_z = fit.rotation[:, 2, 2].abs()
                low_horizontal = (
                    (abs_normal_z >= self.ground_refinement_min_abs_normal_z)
                    & (fit.mean[:, 2] >= self.ground_refinement_min_mean_z)
                    & (fit.mean[:, 2] <= self.ground_refinement_max_mean_z)
                    & (scales[:, 0] >= self.ground_refinement_min_major_scale)
                    & (scales[:, 1] >= self.ground_refinement_min_minor_scale)
                )
                force_ground_split = base_accept & low_horizontal
            else:
                force_ground_split = torch.zeros_like(base_accept)

            accept = base_accept & (~force_ground_split)
            is_leaf = level_idx == len(self.levels) - 1
            leaf_support = (fit.raw_point_count >= max(3, self.min_points[float(node_size)] - 1)) & (fit.weighted_support >= 0.75 * self.min_support[float(node_size)])
            leaf_fit = fit.normal_rms <= 1.5 * self.max_normal_rms[float(node_size)]
            leaf_accept = is_leaf & (~accept) & (fit.current_point_count > 0) & leaf_support & leaf_fit & conflict_ok & free_ok & fit.valid
            if accept.any():
                rows.append(self._make_rows(fit, scales, free_overlap, float(node_size), history_only, accept, confidence_scale=None))
            if leaf_accept.any():
                scale = torch.where(surface_complexity, torch.full_like(fit.confidences if hasattr(fit, 'confidences') else fit.weighted_support, self.leaf_surface_complexity_confidence_scale), torch.full_like(fit.weighted_support, 0.65))
                rows.append(self._make_rows(fit, scales, free_overlap, float(node_size), torch.zeros_like(history_only), leaf_accept, confidence_scale=scale))
            reject_for_split = ~(accept | leaf_accept)
            if not is_leaf:
                keep_point = reject_for_split[inv]
                active = active[keep_point]
            else:
                active = active.new_empty((0,), dtype=torch.long)
            debug_level[f"{node_size:g}"] = {
                "nodes": int(num_groups),
                "accepted": int(accept.sum().item()),
                "leaf_fallback": int(leaf_accept.sum().item()),
                "split_points": int(active.numel()) if not is_leaf else 0,
                "surface_complexity": int(surface_complexity.sum().item()),
                "free_reject": int((~free_ok).sum().item()),
            }
        out = self._concat_rows(rows, device)
        if len(out) > self.max_gaussians:
            # Keep highest confidence without changing scale thresholds.
            keep = torch.topk(out.confidences, k=self.max_gaussians).indices
            out = self._take(out, keep)
        out.debug_info = {"octree_range": list(self.range), "levels": list(self.levels), "input_fit_point_count": int(active_indices.numel()), "processed_nodes": int(processed_nodes_total), "accepted_count": int(len(out)), "by_level": debug_level, "elapsed_sec": round(float(time.perf_counter() - start), 4)}
        return out

    def _make_rows(self, fit, scales, free_overlap, node_size: float, history_only: torch.Tensor, mask: torch.Tensor, confidence_scale=None):
        support_conf = 1.0 - torch.exp(-fit.weighted_support / max(self.support_tau, 1e-6))
        current_conf = 0.5 + 0.5 * fit.current_ratio
        fit_conf = torch.exp(-fit.normal_rms / max(self.fit_sigma, 1e-6))
        temporal_conf = torch.clamp(1.0 - fit.temporal_conflict_ratio, 0.0, 1.0)
        free_conf = torch.clamp(1.0 - free_overlap, 0.0, 1.0)
        confidence = support_conf * current_conf * fit_conf * temporal_conf * free_conf
        confidence = torch.where(history_only, confidence * self.history_only_confidence_scale, confidence)
        if confidence_scale is not None:
            confidence = confidence * confidence_scale
        confidence = torch.clamp(confidence, 0.0, 1.0)
        opacity = self.min_opacity + (self.max_opacity - self.min_opacity) * confidence
        idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
        return {
            "means": fit.mean[idx],
            "scales": scales[idx],
            "rotations": fit.rotation[idx],
            "opacities": opacity[idx],
            "confidences": confidence[idx],
            "levels": torch.full((idx.numel(),), float(node_size), device=fit.mean.device, dtype=torch.float32),
            "support_count": fit.raw_point_count[idx].to(torch.int32),
            "weighted_support": fit.weighted_support[idx].float(),
            "current_ratio": fit.current_ratio[idx].float(),
            "distinct_sweeps": fit.distinct_sweeps[idx].to(torch.int16),
            "fit_residual": fit.normal_rms[idx].float(),
            "temporal_conflict": fit.temporal_conflict_ratio[idx].float(),
            "free_overlap": free_overlap[idx].float(),
            "history_only": history_only[idx].bool(),
            "intensity_mean": getattr(fit, "intensity_mean", torch.zeros_like(fit.weighted_support))[idx].float(),
        }

    def _concat_rows(self, rows, device):
        if not rows:
            return GaussianPrimitiveSet.empty(device)
        cat = lambda k: torch.cat([r[k] for r in rows], dim=0)
        m = cat("means").shape[0]
        return GaussianPrimitiveSet(
            means=cat("means"), scales=cat("scales"), rotations=cat("rotations"), opacities=cat("opacities"), confidences=cat("confidences"), levels=cat("levels"), support_count=cat("support_count"), weighted_support=cat("weighted_support"), current_ratio=cat("current_ratio"), distinct_sweeps=cat("distinct_sweeps"), fit_residual=cat("fit_residual"), temporal_conflict=cat("temporal_conflict"), free_overlap=cat("free_overlap"), history_only=cat("history_only"), intensity_mean=cat("intensity_mean"), debug_info={}, sources=torch.zeros((m,),device=device,dtype=torch.uint8), parent_indices=torch.full((m,),-1,device=device,dtype=torch.long), child_indices=torch.full((m,),-1,device=device,dtype=torch.int16), completion_valid=torch.ones((m,),device=device,dtype=torch.bool))

    def _take(self, g: GaussianPrimitiveSet, keep: torch.Tensor) -> GaussianPrimitiveSet:
        d = {}
        for k, v in g.__dict__.items():
            if isinstance(v, torch.Tensor) and v.shape[:1] == g.means.shape[:1]:
                d[k] = v[keep]
            else:
                d[k] = v
        return GaussianPrimitiveSet(**d)

    def _unit_samples(self, device):
        if device in self._unit_samples_cache:
            return self._unit_samples_cache[device]
        base = [[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
        remaining = max(self.free_check_num_samples - len(base), 1)
        pts = list(base)
        golden = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0)))
        for i in range(remaining):
            z = 1.0 - 2.0 * (i + 0.5) / remaining
            radius = float(max(0.0, 1.0 - z*z)) ** 0.5
            theta = float(golden) * i
            pts.append([radius * torch.cos(torch.tensor(theta)).item(), radius * torch.sin(torch.tensor(theta)).item(), z])
        t = torch.tensor(pts[: self.free_check_num_samples], device=device, dtype=torch.float32)
        self._unit_samples_cache[device] = t
        return t

    def _check_free_overlap_batch(self, mean, scales, rotation, visibility):
        if not self.free_check_enabled or mean.numel() == 0:
            return scales, torch.zeros((mean.shape[0],),device=mean.device), torch.zeros((mean.shape[0],),device=mean.device,dtype=torch.int16)
        scales = scales.clone()
        overlap = self._free_overlap_batch(mean, scales, rotation, visibility)
        steps = torch.zeros_like(overlap, dtype=torch.int16)
        for _ in range(self.free_overlap_max_shrink_steps):
            mask = overlap >= self.free_overlap_warn
            if not bool(mask.any()):
                break
            previous = scales[:, 2].clone()
            scales[mask, 2] = torch.clamp(scales[mask, 2] * self.free_overlap_shrink_factor, min=self.fitter.min_normal_scale)
            overlap = self._free_overlap_batch(mean, scales, rotation, visibility)
            steps[mask & (scales[:, 2] < previous)] += 1
        return scales, overlap, steps

    def _free_overlap_batch(self, mean, scales, rotation, visibility):
        samples_unit = self._unit_samples(mean.device)
        samples = mean[:, None, :] + torch.einsum("gij,gsj->gsi", rotation, samples_unit[None, :, :] * scales[:, None, :])
        q = query_visibility_points(visibility, samples.reshape(-1, 3), neighbor_radius=self.free_query_neighbor_radius)
        valid = q.valid.view(mean.shape[0], -1)
        strong = (q.state.view(mean.shape[0], -1) == STRONG_FREE) & valid
        denom = valid.sum(dim=1).clamp_min(1)
        return strong.sum(dim=1).float() / denom.float()

    def _pack_grid(self, grid: torch.Tensor) -> torch.Tensor:
        # Safe packing for octree range with small indices.
        return ((grid[:, 0] + 100000).long() * 200000 + (grid[:, 1] + 100000).long()) * 200000 + (grid[:, 2] + 100000).long()

    def _unpack_grid(self, flat: torch.Tensor) -> torch.Tensor:
        z = flat % 200000 - 100000
        y = (flat // 200000) % 200000 - 100000
        x = flat // (200000 * 200000) - 100000
        return torch.stack([x, y, z], dim=-1).long()
