from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

UNKNOWN = 0
WEAK_FREE = 1
STRONG_FREE = 2
SURFACE_HIT = 3
MIXED = 4

STATE_NAMES = {UNKNOWN: "unknown", WEAK_FREE: "weak_free", STRONG_FREE: "strong_free", SURFACE_HIT: "surface_hit", MIXED: "mixed"}
STATE_COLORS = {UNKNOWN: (90, 90, 90), WEAK_FREE: (65, 160, 255), STRONG_FREE: (35, 215, 75), SURFACE_HIT: (255, 55, 55), MIXED: (255, 205, 25)}


@dataclass
class VisibilityGridSampleResult:
    grid_range: Tuple[float, float, float, float, float, float]
    coarse_voxel_size: float
    refined_voxel_size: float
    coarse_shape: Tuple[int, int, int]
    refined_shape: Tuple[int, int, int]
    coarse_free_count: torch.Tensor
    coarse_hit_count: torch.Tensor
    coarse_history_count: torch.Tensor
    coarse_free_score: torch.Tensor
    coarse_state: torch.Tensor
    refined_parent_indices: torch.Tensor
    refined_free_count: torch.Tensor
    refined_hit_count: torch.Tensor
    refined_history_count: torch.Tensor
    refined_free_score: torch.Tensor
    refined_state: torch.Tensor
    lidar_origin: torch.Tensor
    current_points: torch.Tensor
    history_points: torch.Tensor
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_lightweight_dict(self) -> Dict[str, Any]:
        return {
            "grid_range": self.grid_range,
            "coarse_voxel_size": self.coarse_voxel_size,
            "refined_voxel_size": self.refined_voxel_size,
            "coarse_shape": self.coarse_shape,
            "num_refined_parents": int(self.refined_parent_indices.shape[0]),
            "debug_info": dict(self.debug_info),
        }


def _as_tensor(x: Any, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach() if not x.requires_grad else x
        return t.to(device=device or t.device, dtype=dtype)
    if hasattr(x, "tensor"):
        return x.tensor.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _pack_3d(idx: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    return (idx[:, 0].long() * int(shape[1]) + idx[:, 1].long()) * int(shape[2]) + idx[:, 2].long()


def _unpack_3d(flat: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    z = flat % int(shape[2])
    y = (flat // int(shape[2])) % int(shape[1])
    x = flat // (int(shape[1]) * int(shape[2]))
    return torch.stack([x, y, z], dim=-1).long()


def _scatter_count(flat: torch.Tensor, total: int, dtype=torch.int32) -> torch.Tensor:
    out = torch.zeros((total,), device=flat.device, dtype=dtype)
    if flat.numel() > 0:
        out.scatter_add_(0, flat.long(), torch.ones_like(flat, dtype=dtype))
    return out


class VisibilityGridV1FullRange(nn.Module):
    """CUDA equivalent of the original visibility side branch.

    It keeps the original coarse/refined state semantics, but uses vectorized
    torch operations and parallel DDA over all active rays. No NumPy/CPU ray
    traversal is used during forward; tensors are moved to CPU only for optional
    PLY writing.
    """

    def __init__(
        self,
        enabled: bool = True,
        grid_range: Sequence[float] = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        coarse_voxel_size: float = 0.8,
        refined_voxel_size: float = 0.4,
        current_flag_dim: int = 6,
        endpoint_margin: float = 0.3,
        min_ray_range: float = 1.0,
        ray_count_tau: float = 3.0,
        strong_free_threshold: float = 0.75,
        refine_mixed_cells: bool = True,
        refine_temporal_conflict_cells: bool = True,
        max_refined_parents: int = 20000,
        occlusion_guard_enabled: bool = True,
        azimuth_bin_deg: float = 0.25,
        elevation_bin_deg: float = 0.5,
        occlusion_neighbor_radius: int = 1,
        occlusion_min_support_bins: int = 3,
        occlusion_depth_consistency_threshold: float = 1.0,
        occlusion_min_range_gap: float = 1.0,
        occlusion_barrier_margin: float = 0.4,
        always_compute: bool = False,
        retain_last_result: bool = False,
        debug: bool = False,
        debug_interval: int = 50,
        debug_max_print: int = 20,
        visualize: bool = False,
        visualize_interval: int = 200,
        visualize_max_outputs: int = 20,
        visualize_output_dir: str = "work_dirs/daocc_visibility_gpu_exact",
        visualize_sample_index: int = 0,
        voxel_mesh_scale: float = 0.88,
        current_point_color: Sequence[int] = (190, 0, 255),
        current_point_marker_size: float = 0.035,
        current_point_stride: int = 1,
        include_unknown_in_ply: bool = False,
        unknown_ply_stride: int = 64,
        startup_log: bool = True,
        dda_chunk_size: int = 65536,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        if len(grid_range) != 6:
            raise ValueError("grid_range must have 6 values")
        self.grid_range = tuple(float(v) for v in grid_range)
        self.coarse_voxel_size = float(coarse_voxel_size)
        self.refined_voxel_size = float(refined_voxel_size)
        self.current_flag_dim = int(current_flag_dim)
        self.endpoint_margin = float(endpoint_margin)
        self.min_ray_range = float(min_ray_range)
        self.ray_count_tau = float(ray_count_tau)
        self.strong_free_threshold = float(strong_free_threshold)
        self.refine_mixed_cells = bool(refine_mixed_cells)
        self.refine_temporal_conflict_cells = bool(refine_temporal_conflict_cells)
        self.max_refined_parents = int(max_refined_parents)
        self.always_compute = bool(always_compute)
        self.retain_last_result = bool(retain_last_result)
        self.last_result: Optional[List[VisibilityGridSampleResult]] = None
        self.debug = bool(debug)
        self.debug_interval = max(int(debug_interval), 1)
        self.debug_max_print = max(int(debug_max_print), 0)
        self._debug_batch_count = 0
        self._debug_print_count = 0
        self.visualize = bool(visualize)
        self.visualize_interval = max(int(visualize_interval), 1)
        self.visualize_max_outputs = max(int(visualize_max_outputs), 0)
        self._visualize_batch_count = 0
        self._visualize_write_count = 0
        self.visualize_output_dir = Path(visualize_output_dir)
        self.visualize_sample_index = max(int(visualize_sample_index), 0)
        self.voxel_mesh_scale = float(voxel_mesh_scale)
        self.current_point_color = tuple(int(v) for v in current_point_color)
        self.current_point_marker_size = float(current_point_marker_size)
        self.current_point_stride = max(int(current_point_stride), 1)
        self.include_unknown_in_ply = bool(include_unknown_in_ply)
        self.unknown_ply_stride = max(int(unknown_ply_stride), 1)
        self.dda_chunk_size = max(int(dda_chunk_size), 1024)

        gmin = torch.tensor(self.grid_range[:3], dtype=torch.float32)
        gmax = torch.tensor(self.grid_range[3:], dtype=torch.float32)
        extent = gmax - gmin
        coarse_shape = extent / self.coarse_voxel_size
        refined_shape = extent / self.refined_voxel_size
        if not torch.allclose(coarse_shape, coarse_shape.round(), atol=1e-5):
            raise ValueError("grid range must be divisible by coarse_voxel_size")
        if not torch.allclose(refined_shape, refined_shape.round(), atol=1e-5):
            raise ValueError("grid range must be divisible by refined_voxel_size")
        self.coarse_shape = tuple(int(v) for v in coarse_shape.round().tolist())
        self.refined_shape = tuple(int(v) for v in refined_shape.round().tolist())
        self.register_buffer("grid_min_buf", gmin, persistent=False)
        self.register_buffer("grid_max_buf", gmax, persistent=False)
        if startup_log and self._is_main_process():
            print(
                "[VisibilityGridV1FullRange GPU Config] "
                f"counter_unit=batch, grid={self.coarse_shape}@{self.coarse_voxel_size}m, "
                f"refined={self.refined_shape}@{self.refined_voxel_size}m, always_compute={self.always_compute}",
                flush=True,
            )

    @torch.no_grad()
    def forward(self, points: Sequence[Any], lidar_origins: Optional[Any] = None, metas: Optional[Any] = None) -> Optional[List[VisibilityGridSampleResult]]:
        if not self.enabled:
            return None
        if not isinstance(points, (list, tuple)) or len(points) == 0:
            return None
        self._debug_batch_count += 1
        self._visualize_batch_count += 1
        is_main = self._is_main_process()
        debug_due = self.debug and is_main and self._debug_print_count < self.debug_max_print and self._debug_batch_count % self.debug_interval == 0
        visualize_due = self.visualize and is_main and self._visualize_write_count < self.visualize_max_outputs and self._visualize_batch_count % self.visualize_interval == 0
        if not (self.always_compute or debug_due or visualize_due):
            return None
        origins = self._normalize_origins(lidar_origins, len(points), points[0])
        selected = min(self.visualize_sample_index, len(points) - 1)
        build_indices = list(range(len(points))) if self.always_compute else [selected]
        outputs = []
        by_idx = {}
        for b in build_indices:
            res = self._build_sample(_as_tensor(points[b]), origins[b])
            outputs.append(res)
            by_idx[b] = res
        display = by_idx.get(selected, outputs[0] if outputs else None)
        if debug_due and display is not None:
            print(f"[VisibilityGrid GPU Debug][batch={self._debug_batch_count}] {display.debug_info}", flush=True)
            self._debug_print_count += 1
        if visualize_due and display is not None:
            path = self._write_visualization(display, self._sample_tag(metas, selected, self._visualize_batch_count))
            display.debug_info["visualization_path"] = str(path)
            print(f"[VisibilityGrid GPU PLY][batch={self._visualize_batch_count}] {path}", flush=True)
            self._visualize_write_count += 1
        if self.retain_last_result:
            self.last_result = outputs
        return outputs or None

    def _normalize_origins(self, origins: Optional[Any], batch_size: int, first_point: Any) -> torch.Tensor:
        device = _as_tensor(first_point).device
        if origins is None:
            return torch.zeros((batch_size, 3), device=device, dtype=torch.float32)
        t = _as_tensor(origins, device=device)
        if t.ndim == 1:
            t = t.view(1, 3).repeat(batch_size, 1)
        return t[:, :3].float()

    def _build_sample(self, point_tensor: torch.Tensor, lidar_origin: torch.Tensor) -> VisibilityGridSampleResult:
        device = point_tensor.device
        pts = point_tensor.float()
        xyz = pts[:, :3]
        current = pts[:, self.current_flag_dim] > 0.5
        current_points = xyz[current]
        history_points = xyz[~current]
        grid_min = self.grid_min_buf.to(device)
        grid_max = self.grid_max_buf.to(device)
        total_coarse = int(self.coarse_shape[0] * self.coarse_shape[1] * self.coarse_shape[2])
        total_refined = int(self.refined_shape[0] * self.refined_shape[1] * self.refined_shape[2])
        coarse_hit_flat = self._point_counts(current_points, self.coarse_voxel_size, self.coarse_shape, total_coarse)
        coarse_history_flat = self._point_counts(history_points, self.coarse_voxel_size, self.coarse_shape, total_coarse)
        starts, ends, ray_valid, short_count = self._prepare_ray_segments(current_points, lidar_origin.to(device), grid_min, grid_max)
        coarse_free_flat = self._dda_count_full(starts, ends, self.coarse_voxel_size, self.coarse_shape, total_coarse)
        if ends.numel() > 0:
            end_idx = torch.floor((ends - grid_min) / self.coarse_voxel_size).long()
            inb = ((end_idx >= 0) & (end_idx < torch.tensor(self.coarse_shape, device=device))).all(dim=1)
            if inb.any():
                term = _pack_3d(end_idx[inb], self.coarse_shape)
                coarse_free_flat.scatter_add_(0, term, -torch.ones_like(term, dtype=coarse_free_flat.dtype))
                coarse_free_flat.clamp_min_(0)
        coarse_free = coarse_free_flat.view(self.coarse_shape)
        coarse_hit = coarse_hit_flat.view(self.coarse_shape)
        coarse_history = coarse_history_flat.view(self.coarse_shape)
        coarse_score = self._free_score(coarse_free)
        coarse_state = self._classify_state(coarse_free, coarse_hit, coarse_score)
        refine_mask = torch.zeros_like(coarse_state, dtype=torch.bool)
        if self.refine_mixed_cells:
            refine_mask |= (coarse_free > 0) & (coarse_hit > 0)
        if self.refine_temporal_conflict_cells:
            refine_mask |= (coarse_free > 0) & (coarse_history > 0)
        parent_flat = torch.nonzero(refine_mask.reshape(-1), as_tuple=False).reshape(-1)
        if self.max_refined_parents > 0 and parent_flat.numel() > self.max_refined_parents:
            score = (coarse_free.reshape(-1)[parent_flat].float() + 2 * coarse_hit.reshape(-1)[parent_flat].float() + coarse_history.reshape(-1)[parent_flat].float())
            keep = torch.topk(score, k=self.max_refined_parents).indices
            parent_flat = parent_flat[keep]
        refined_parent_indices = _unpack_3d(parent_flat, self.coarse_shape).int()
        if parent_flat.numel() > 0:
            refined_hit_full = self._point_counts(current_points, self.refined_voxel_size, self.refined_shape, total_refined)
            refined_history_full = self._point_counts(history_points, self.refined_voxel_size, self.refined_shape, total_refined)
            refined_free_full = self._dda_count_full(starts, ends, self.refined_voxel_size, self.refined_shape, total_refined)
            if ends.numel() > 0:
                end_idx = torch.floor((ends - grid_min) / self.refined_voxel_size).long()
                inb = ((end_idx >= 0) & (end_idx < torch.tensor(self.refined_shape, device=device))).all(dim=1)
                if inb.any():
                    term = _pack_3d(end_idx[inb], self.refined_shape)
                    refined_free_full.scatter_add_(0, term, -torch.ones_like(term, dtype=refined_free_full.dtype))
                    refined_free_full.clamp_min_(0)
            child_offsets = torch.tensor([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]], device=device, dtype=torch.long)
            child_idx = refined_parent_indices.long()[:, None, :] * 2 + child_offsets[None, :, :]
            child_flat = _pack_3d(child_idx.reshape(-1, 3), self.refined_shape).view(-1, 8)
            refined_free_count = refined_free_full[child_flat]
            refined_hit_count = refined_hit_full[child_flat]
            refined_history_count = refined_history_full[child_flat]
            refined_free_score = self._free_score(refined_free_count)
            refined_state = self._classify_state(refined_free_count, refined_hit_count, refined_free_score)
        else:
            refined_free_count = torch.empty((0, 8), device=device, dtype=torch.int32)
            refined_hit_count = torch.empty((0, 8), device=device, dtype=torch.int32)
            refined_history_count = torch.empty((0, 8), device=device, dtype=torch.int32)
            refined_free_score = torch.empty((0, 8), device=device, dtype=torch.float32)
            refined_state = torch.empty((0, 8), device=device, dtype=torch.uint8)
        debug = {
            "point_shape": tuple(int(v) for v in pts.shape),
            "current_point_count": int(current_points.shape[0]),
            "history_point_count": int(history_points.shape[0]),
            "valid_ray_count": int(starts.shape[0]),
            "short_ray_count": int(short_count),
            "coarse_nonzero_free": int((coarse_free > 0).sum().item()),
            "coarse_nonzero_hit": int((coarse_hit > 0).sum().item()),
            "num_refined_parents": int(refined_parent_indices.shape[0]),
        }
        return VisibilityGridSampleResult(
            grid_range=self.grid_range,
            coarse_voxel_size=self.coarse_voxel_size,
            refined_voxel_size=self.refined_voxel_size,
            coarse_shape=self.coarse_shape,
            refined_shape=self.refined_shape,
            coarse_free_count=coarse_free,
            coarse_hit_count=coarse_hit,
            coarse_history_count=coarse_history,
            coarse_free_score=coarse_score,
            coarse_state=coarse_state,
            refined_parent_indices=refined_parent_indices,
            refined_free_count=refined_free_count,
            refined_hit_count=refined_hit_count,
            refined_history_count=refined_history_count,
            refined_free_score=refined_free_score,
            refined_state=refined_state,
            lidar_origin=lidar_origin.to(device).float(),
            current_points=current_points,
            history_points=history_points,
            debug_info=debug,
        )

    def _point_counts(self, points: torch.Tensor, voxel_size: float, shape: Sequence[int], total: int) -> torch.Tensor:
        device = self.grid_min_buf.device if not isinstance(points, torch.Tensor) else points.device
        out = torch.zeros((total,), device=device, dtype=torch.int32)
        if points.numel() == 0:
            return out
        grid_min = self.grid_min_buf.to(points.device)
        idx = torch.floor((points[:, :3] - grid_min) / float(voxel_size)).long()
        shape_t = torch.tensor(shape, device=points.device, dtype=torch.long)
        valid = ((idx >= 0) & (idx < shape_t)).all(dim=1)
        if valid.any():
            flat = _pack_3d(idx[valid], shape)
            out.scatter_add_(0, flat, torch.ones_like(flat, dtype=out.dtype))
        return out

    def _prepare_ray_segments(self, current_points: torch.Tensor, origin: torch.Tensor, grid_min: torch.Tensor, grid_max: torch.Tensor):
        if current_points.numel() == 0:
            return current_points.new_empty((0, 3)), current_points.new_empty((0, 3)), current_points.new_empty((0,), dtype=torch.bool), 0
        vec = current_points[:, :3] - origin.view(1, 3)
        dist = torch.linalg.norm(vec, dim=1)
        finite = torch.isfinite(dist) & torch.isfinite(vec).all(dim=1)
        valid = finite & (dist >= self.min_ray_range)
        short_count = int((finite & ~valid).sum().item())
        if not valid.any():
            return current_points.new_empty((0, 3)), current_points.new_empty((0, 3)), valid, short_count
        dirs = vec[valid] / dist[valid].clamp_min(1e-6).view(-1, 1)
        end_dist = (dist[valid] - self.endpoint_margin).clamp_min(0.0)
        starts = origin.view(1, 3).expand_as(dirs)
        ends = origin.view(1, 3) + dirs * end_dist.view(-1, 1)
        in_end = ((ends >= grid_min.view(1, 3)) & (ends < grid_max.view(1, 3))).all(dim=1)
        return starts[in_end].contiguous(), ends[in_end].contiguous(), valid, short_count

    def _dda_count_full(self, starts: torch.Tensor, ends: torch.Tensor, voxel_size: float, shape: Sequence[int], total: int) -> torch.Tensor:
        device = starts.device if starts.numel() else self.grid_min_buf.device
        counts = torch.zeros((total,), device=device, dtype=torch.int32)
        if starts.numel() == 0:
            return counts
        grid_min = self.grid_min_buf.to(device)
        shape_t = torch.tensor(shape, device=device, dtype=torch.long)
        for s0 in range(0, starts.shape[0], self.dda_chunk_size):
            s = starts[s0:s0+self.dda_chunk_size]
            e = ends[s0:s0+self.dda_chunk_size]
            direction = e - s
            cur = torch.floor((s - grid_min) / voxel_size).long()
            end_idx = torch.floor((e - grid_min) / voxel_size).long()
            step = torch.sign(direction).long()
            abs_dir = direction.abs().clamp_min(1e-12)
            t_delta = torch.where(step == 0, torch.full_like(direction, 1e30), float(voxel_size) / abs_dir)
            boundary_offset = (step > 0).float()
            next_boundary = grid_min.view(1, 3) + (cur.float() + boundary_offset) * float(voxel_size)
            t_max = torch.where(step == 0, torch.full_like(direction, 1e30), (next_boundary - s) / direction.clamp(min=-1e30, max=1e30))
            t_max = torch.where(torch.isfinite(t_max), t_max, torch.full_like(t_max, 1e30))
            active = ((cur >= 0) & (cur < shape_t.view(1, 3))).all(dim=1)
            max_steps = int(shape[0] + shape[1] + shape[2] + 8)
            for _ in range(max_steps):
                if not bool(active.any()):
                    break
                rows = torch.nonzero(active, as_tuple=False).reshape(-1)
                flat = _pack_3d(cur[rows], shape)
                counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=counts.dtype))
                reached = (cur == end_idx).all(dim=1)
                active = active & ~reached
                if not bool(active.any()):
                    break
                axis = torch.argmin(t_max, dim=1)
                ar = torch.nonzero(active, as_tuple=False).reshape(-1)
                ax = axis[ar]
                cur[ar, ax] += step[ar, ax]
                t_max[ar, ax] += t_delta[ar, ax]
                active = active & ((cur >= 0) & (cur < shape_t.view(1, 3))).all(dim=1)
        return counts

    def _free_score(self, count: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.exp(-count.float() / max(float(self.ray_count_tau), 1e-6))

    def _classify_state(self, free_count: torch.Tensor, hit_count: torch.Tensor, free_score: torch.Tensor) -> torch.Tensor:
        state = torch.full_like(free_count, UNKNOWN, dtype=torch.uint8)
        state[(free_count > 0) & (free_score < self.strong_free_threshold) & (hit_count == 0)] = WEAK_FREE
        state[(free_count > 0) & (free_score >= self.strong_free_threshold) & (hit_count == 0)] = STRONG_FREE
        state[(hit_count > 0) & (free_count == 0)] = SURFACE_HIT
        state[(hit_count > 0) & (free_count > 0)] = MIXED
        return state

    def _write_visualization(self, result: VisibilityGridSampleResult, tag: str) -> Path:
        # Minimal point-only visibility file to avoid slowing training. GaussianProbe writes the full scene PLY.
        self.visualize_output_dir.mkdir(parents=True, exist_ok=True)
        path = self.visualize_output_dir / f"visibility_points_{tag}.ply"
        pts = result.current_points[::self.current_point_stride].detach().cpu()
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {pts.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for p in pts.tolist():
                f.write(f"{p[0]} {p[1]} {p[2]} {self.current_point_color[0]} {self.current_point_color[1]} {self.current_point_color[2]}\n")
        return path

    @staticmethod
    def _sample_tag(metas: Optional[Any], batch_idx: int, counter: int) -> str:
        token = None
        if isinstance(metas, (list, tuple)) and batch_idx < len(metas) and isinstance(metas[batch_idx], dict):
            token = metas[batch_idx].get("token") or metas[batch_idx].get("sample_idx")
        token = str(token) if token is not None else f"sample{batch_idx}"
        token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token)
        return f"{counter:06d}_{token}_batch{batch_idx}"

    @staticmethod
    def _is_main_process() -> bool:
        rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
        return rank in (None, "0")
