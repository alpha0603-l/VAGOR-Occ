from __future__ import annotations

import os
import pprint
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .gaussian_octree import GaussianOctreeBuilder
from .gaussian_structures import GaussianProbeSampleResult
from .temporal_support import TemporalSupportWeighter
from .visibility_grid import (
    MIXED,
    STATE_COLORS,
    STRONG_FREE,
    SURFACE_HIT,
    UNKNOWN,
    WEAK_FREE,
    VisibilityGridSampleResult,
    VisibilityGridV1FullRange,
    _unpack_3d,
)

MEASURED_LEVEL_COLORS = {
    3.2: (40, 210, 255),
    1.6: (45, 225, 95),
    0.8: (255, 170, 35),
    0.4: (255, 65, 65),
}


def _as_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "tensor") and isinstance(x.tensor, torch.Tensor):
        return x.tensor
    return torch.as_tensor(x)


class GaussianProbeGPU(nn.Module):
    """GPU-only DAOcc measured-Gaussian probe for GSF3D lifter.

    This file is the only probe entry used by OctreeGaussianLifter.  It does not
    import or call gaussian_probe.py.  The actual generation path is:

        7D points on CUDA
        -> VisibilityGridV1FullRange._build_sample  (0.8m coarse + 0.4m refined)
        -> TemporalSupportWeighter                  (current/history reliability)
        -> GaussianOctreeBuilder.build              (top-down accept/split octree)
        -> WeightedGaussianFitter                   (batched weighted PCA on CUDA)

    CPU conversion appears only in optional PLY/debug output.
    """

    def __init__(
        self,
        enabled: bool = True,
        always_compute: bool = True,
        retain_last_result: bool = False,
        process_sample_index: int = 0,
        compute_interval: int = 0,
        compute_max_outputs: int = -1,
        backend: str = "gpu",
        force_cuda: bool = True,
        debug: bool = False,
        debug_detail: str = "compact",
        debug_interval: int = 50,
        debug_max_print: int = 5,
        visualize: bool = False,
        visualize_interval: int = 50,
        visualize_max_outputs: int = 3,
        visualize_output_dir: str = "work_dirs/gsf_octree_debug_ply",
        visualize_sample_index: int = 0,
        point_stride: int = 1,
        current_point_color: Sequence[int] = (220, 0, 255),
        history_point_color: Sequence[int] = (105, 105, 115),
        ellipsoid_latitude_segments: int = 8,
        ellipsoid_longitude_segments: int = 12,
        max_visualized_gaussians: int = 6400,
        max_visualized_points: int = 0,
        startup_log: bool = True,
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
        dda_chunk_size: int = 65536,
        temporal_support: Optional[Dict[str, Any]] = None,
        octree: Optional[Dict[str, Any]] = None,
        **unused_kwargs: Any,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.always_compute = bool(always_compute)
        self.retain_last_result = bool(retain_last_result)
        self.last_result: Optional[List[GaussianProbeSampleResult]] = None
        self.process_sample_index = max(int(process_sample_index), 0)
        self.compute_interval = max(int(compute_interval), 0)
        self.compute_max_outputs = int(compute_max_outputs)
        self._compute_output_count = 0
        self.backend = str(backend).lower().strip()
        self.force_cuda = bool(force_cuda)

        self.debug = bool(debug)
        self.debug_detail = str(debug_detail).lower().strip()
        self.debug_interval = max(int(debug_interval), 1)
        self.debug_max_print = max(int(debug_max_print), 0)
        self._debug_batch_count = 0
        self._debug_print_count = 0

        self.visualize = bool(visualize)
        self.visualize_interval = max(int(visualize_interval), 1)
        self.visualize_max_outputs = max(int(visualize_max_outputs), 0)
        self._visualize_write_count = 0
        self.visualize_output_dir = Path(visualize_output_dir)
        self.visualize_sample_index = max(int(visualize_sample_index), 0)
        self.point_stride = max(int(point_stride), 1)
        self.max_visualized_points = max(int(max_visualized_points), 0)
        self.max_visualized_gaussians = max(int(max_visualized_gaussians), 0)
        self.current_point_color = self._rgb(current_point_color)
        self.history_point_color = self._rgb(history_point_color)
        self.unit_vertices, self.unit_faces = self._make_uv_sphere(
            max(int(ellipsoid_latitude_segments), 4),
            max(int(ellipsoid_longitude_segments), 6),
        )

        # DAOcc exact GPU visibility side branch: 0.8m coarse and 0.4m refined.
        self.visibility_builder = VisibilityGridV1FullRange(
            enabled=True,
            grid_range=grid_range,
            coarse_voxel_size=coarse_voxel_size,
            refined_voxel_size=refined_voxel_size,
            current_flag_dim=current_flag_dim,
            endpoint_margin=endpoint_margin,
            min_ray_range=min_ray_range,
            ray_count_tau=ray_count_tau,
            strong_free_threshold=strong_free_threshold,
            refine_mixed_cells=refine_mixed_cells,
            refine_temporal_conflict_cells=refine_temporal_conflict_cells,
            max_refined_parents=max_refined_parents,
            occlusion_guard_enabled=occlusion_guard_enabled,
            azimuth_bin_deg=azimuth_bin_deg,
            elevation_bin_deg=elevation_bin_deg,
            occlusion_neighbor_radius=occlusion_neighbor_radius,
            occlusion_min_support_bins=occlusion_min_support_bins,
            occlusion_depth_consistency_threshold=occlusion_depth_consistency_threshold,
            occlusion_min_range_gap=occlusion_min_range_gap,
            occlusion_barrier_margin=occlusion_barrier_margin,
            always_compute=True,
            retain_last_result=False,
            debug=False,
            visualize=False,
            startup_log=False,
            dda_chunk_size=dda_chunk_size,
        )
        self.temporal_weighter = TemporalSupportWeighter(**(temporal_support or {}))
        if octree is None:
            raise ValueError("gaussian_probe.octree configuration is required")
        self.octree_builder = GaussianOctreeBuilder(**dict(octree))

        if unused_kwargs and self._is_main_process():
            print(
                "[GaussianProbeGPU Strict Warning] Ignored unsupported keys: "
                f"{sorted(list(unused_kwargs.keys()))}",
                flush=True,
            )
        if startup_log and self._is_main_process():
            print(
                "[GaussianProbeGPU StrictTopDown Config] "
                f"backend=gpu, force_cuda={self.force_cuda}, "
                f"grid_range={tuple(float(v) for v in grid_range)}, "
                f"octree_range={self.octree_builder.range}, levels={self.octree_builder.levels}, "
                f"max_gaussians={self.octree_builder.max_gaussians}",
                flush=True,
            )

    @staticmethod
    def _rgb(value: Sequence[int]) -> Tuple[int, int, int]:
        if len(value) != 3:
            raise ValueError("RGB color must have three values")
        return tuple(max(0, min(255, int(v))) for v in value)

    @staticmethod
    def _is_main_process() -> bool:
        rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
        return rank in (None, "0")

    def _target_device(self, points: Sequence[Any]) -> torch.device:
        for p in points:
            try:
                dev = _as_tensor(p).device
                if dev.type == "cuda":
                    return dev
            except Exception:
                pass
        if self.force_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("GaussianProbeGPU requires CUDA when force_cuda=True")
            return torch.device("cuda", torch.cuda.current_device())
        for p in points:
            try:
                return _as_tensor(p).device
            except Exception:
                pass
        return torch.device("cpu")

    @staticmethod
    def _normalize_origins(origins: Any, batch_size: int, device: torch.device) -> torch.Tensor:
        if origins is None:
            out = torch.zeros((batch_size, 3), device=device, dtype=torch.float32)
        elif isinstance(origins, torch.Tensor):
            out = origins.to(device=device, dtype=torch.float32, non_blocking=True)
        else:
            out = torch.as_tensor(origins, device=device, dtype=torch.float32)
        if out.ndim == 1:
            out = out.view(1, -1).expand(batch_size, -1)
        if out.shape[0] < batch_size:
            out = torch.cat([out, out[-1:].expand(batch_size - out.shape[0], -1)], dim=0)
        return out[:, :3].contiguous()

    @torch.no_grad()
    def forward(
        self,
        points: Sequence[Union[torch.Tensor, Any]],
        lidar_origins: Optional[Union[torch.Tensor, Sequence[Sequence[float]]]] = None,
        metas: Optional[Any] = None,
        visibility_grid: Any = None,
    ) -> Optional[List[GaussianProbeSampleResult]]:
        if not self.enabled:
            return None
        if not isinstance(points, (list, tuple)) or len(points) == 0:
            return None

        self._debug_batch_count += 1
        is_main = self._is_main_process()
        debug_due = self.debug and is_main and self._debug_print_count < self.debug_max_print and self._debug_batch_count % self.debug_interval == 0
        visualize_due = self.visualize and is_main and self._visualize_write_count < self.visualize_max_outputs and self._debug_batch_count % self.visualize_interval == 0
        compute_due = self.compute_interval > 0 and self._debug_batch_count % self.compute_interval == 0 and (self.compute_max_outputs < 0 or self._compute_output_count < self.compute_max_outputs)
        if not (self.always_compute or compute_due or debug_due or visualize_due):
            return None
        if compute_due:
            self._compute_output_count += 1

        device = self._target_device(points)
        origins = self._normalize_origins(lidar_origins, len(points), device)
        selected = min(self.visualize_sample_index, len(points) - 1)
        build_indices = list(range(len(points))) if self.always_compute else [min(self.process_sample_index, len(points) - 1)]
        outputs: List[GaussianProbeSampleResult] = []
        t_all = time.perf_counter()

        for batch_idx in build_indices:
            sample_t = _as_tensor(points[batch_idx]).to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
            origin = origins[batch_idx].to(device=device, dtype=torch.float32)
            t0 = time.perf_counter()
            visibility = self.visibility_builder._build_sample(sample_t, origin)
            t_vis = time.perf_counter() - t0
            t0 = time.perf_counter()
            temporal = self.temporal_weighter(sample_t, visibility)
            t_support = time.perf_counter() - t0
            t0 = time.perf_counter()
            gaussians = self.octree_builder.build(temporal, visibility)
            t_octree = time.perf_counter() - t0
            info = self._compact_debug_info(batch_idx, sample_t, visibility, temporal, gaussians, (t_vis, t_support, t_octree))
            info["batch_counter"] = int(self._debug_batch_count)
            result = GaussianProbeSampleResult(
                visibility=visibility,
                temporal_support=temporal,
                gaussians=gaussians,
                debug_info=info,
                batch_index=int(batch_idx),
            )
            outputs.append(result)

            if debug_due and batch_idx == selected:
                tag = self._sample_tag(metas, batch_idx, self._debug_batch_count)
                print("[GaussianProbeGPU StrictTopDown Debug][{}]\n{}".format(tag, pprint.pformat(info, width=150, sort_dicts=False)), flush=True)
                self._debug_print_count += 1

        if len(outputs) == 0:
            return None
        if self.retain_last_result:
            self.last_result = outputs

        if visualize_due:
            chosen = None
            for item in outputs:
                if int(item.batch_index) == selected:
                    chosen = item
                    break
            if chosen is None:
                chosen = outputs[0]
            tag = self._sample_tag(metas, int(chosen.batch_index), self._visualize_write_count)
            paths = self._write_visualization_bundle(chosen, tag)
            chosen.debug_info.setdefault("visualization", {}).update({k: str(v) for k, v in paths.items()})
            print(f"[GaussianProbeGPU StrictTopDown PLY][{tag}] {paths}", flush=True)
            self._visualize_write_count += 1

        if debug_due and self.debug_detail == "summary" and self._debug_print_count < self.debug_max_print:
            total = sum(int(len(r.gaussians)) for r in outputs)
            print(f"[GaussianProbeGPU StrictTopDown Summary] batch={self._debug_batch_count}, total={total}, time={time.perf_counter()-t_all:.4f}s", flush=True)
            self._debug_print_count += 1
        return outputs

    def _compact_debug_info(self, batch_idx, points, visibility, temporal, gaussians, timing):
        return {
            "sample_batch_index": int(batch_idx),
            "device": str(points.device),
            "point_tensor_shape": tuple(int(v) for v in points.shape),
            "input": {
                "current_points": int(visibility.current_points.shape[0]),
                "history_points": int(visibility.history_points.shape[0]),
                "fit_points": int(temporal.fit_mask.sum().item()),
            },
            "timing_sec": {
                "gpu_visibility_0p8_0p4": round(float(timing[0]), 4),
                "gpu_temporal_support": round(float(timing[1]), 4),
                "gpu_topdown_octree": round(float(timing[2]), 4),
                "total": round(float(sum(timing)), 4),
            },
            "visibility": dict(getattr(visibility, "debug_info", {}) or {}),
            "temporal_support": dict(getattr(temporal, "debug_info", {}) or {}),
            "gaussians": {
                "count": int(len(gaussians)),
                "debug": dict(getattr(gaussians, "debug_info", {}) or {}),
            },
        }

    def _write_visualization_bundle(self, result: GaussianProbeSampleResult, tag: str) -> Dict[str, Path]:
        self.visualize_output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "visibility_0p8m": self.visualize_output_dir / f"visibility_0p8m_{tag}.ply",
            "visibility_0p4m_refined": self.visualize_output_dir / f"visibility_0p4m_refined_{tag}.ply",
            "measured_gaussians": self.visualize_output_dir / f"measured_gaussians_{tag}.ply",
        }
        self._write_visibility_coarse_ply(paths["visibility_0p8m"], result.visibility)
        self._write_visibility_refined_ply(paths["visibility_0p4m_refined"], result.visibility)
        self._write_measured_ply(paths["measured_gaussians"], result)
        return paths

    def _append_points(self, verts: List[Tuple[float, float, float, int, int, int]], pts: torch.Tensor, color: Tuple[int, int, int], max_points: int = 0) -> None:
        if pts is None or pts.numel() == 0:
            return
        pts = pts[:: self.point_stride]
        if max_points > 0 and pts.shape[0] > max_points:
            pts = pts[:max_points]
        for p in pts.detach().cpu().tolist():
            verts.append((float(p[0]), float(p[1]), float(p[2]), int(color[0]), int(color[1]), int(color[2])))

    def _write_visibility_coarse_ply(self, path: Path, visibility: VisibilityGridSampleResult) -> None:
        verts: List[Tuple[float, float, float, int, int, int]] = []
        self._append_points(verts, visibility.current_points, self.current_point_color, self.max_visualized_points)
        self._append_points(verts, visibility.history_points, self.history_point_color, self.max_visualized_points)
        device = visibility.coarse_state.device
        grid_min = torch.tensor(visibility.grid_range[:3], device=device, dtype=torch.float32)
        state = visibility.coarse_state.reshape(-1)
        keep = state != UNKNOWN
        flat = torch.nonzero(keep, as_tuple=False).reshape(-1)
        if flat.numel() > 0:
            if flat.numel() > 200000:
                flat = flat[:: max(int(flat.numel() // 200000), 1)]
            idx = _unpack_3d(flat, visibility.coarse_shape).float()
            centers = grid_min.view(1, 3) + (idx + 0.5) * float(visibility.coarse_voxel_size)
            states = state[flat].detach().cpu().tolist()
            for p, s in zip(centers.detach().cpu().tolist(), states):
                color = STATE_COLORS.get(int(s), (255, 255, 255))
                verts.append((float(p[0]), float(p[1]), float(p[2]), *color))
        self._write_point_ply(path, verts)

    def _write_visibility_refined_ply(self, path: Path, visibility: VisibilityGridSampleResult) -> None:
        verts: List[Tuple[float, float, float, int, int, int]] = []
        self._append_points(verts, visibility.current_points, self.current_point_color, self.max_visualized_points)
        self._append_points(verts, visibility.history_points, self.history_point_color, self.max_visualized_points)
        if visibility.refined_parent_indices.numel() > 0:
            device = visibility.refined_parent_indices.device
            grid_min = torch.tensor(visibility.grid_range[:3], device=device, dtype=torch.float32)
            offsets = torch.tensor([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]], device=device, dtype=torch.float32)
            parent = visibility.refined_parent_indices.float()
            child = parent[:, None, :] * 2.0 + offsets[None, :, :]
            centers = grid_min.view(1, 1, 3) + (child + 0.5) * float(visibility.refined_voxel_size)
            states = visibility.refined_state.reshape(-1)
            centers = centers.reshape(-1, 3)
            keep = states != UNKNOWN
            centers = centers[keep]
            states = states[keep]
            if centers.shape[0] > 200000:
                step = max(int(centers.shape[0] // 200000), 1)
                centers = centers[::step]
                states = states[::step]
            for p, s in zip(centers.detach().cpu().tolist(), states.detach().cpu().tolist()):
                color = STATE_COLORS.get(int(s), (255, 255, 255))
                verts.append((float(p[0]), float(p[1]), float(p[2]), *color))
        self._write_point_ply(path, verts)

    def _write_measured_ply(self, path: Path, result: GaussianProbeSampleResult) -> None:
        verts: List[Tuple[float, float, float, int, int, int]] = []
        faces: List[Tuple[int, int, int]] = []
        self._append_points(verts, result.visibility.current_points, self.current_point_color, self.max_visualized_points)
        self._append_points(verts, result.visibility.history_points, self.history_point_color, self.max_visualized_points)
        g = result.gaussians
        n = int(len(g))
        if self.max_visualized_gaussians > 0:
            n = min(n, self.max_visualized_gaussians)
        for i in range(n):
            lvl = float(g.levels[i].detach().item())
            color = self._level_color(lvl)
            self._append_ellipsoid(verts, faces, g.means[i], g.scales[i], g.rotations[i], color)
        self._write_mesh_ply(path, verts, faces)

    @staticmethod
    def _write_point_ply(path: Path, verts: List[Tuple[float, float, float, int, int, int]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(verts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for x, y, z, r, g, b in verts:
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    @staticmethod
    def _write_mesh_ply(path: Path, verts: List[Tuple[float, float, float, int, int, int]], faces: List[Tuple[int, int, int]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(verts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
            for x, y, z, r, g, b in verts:
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            for a, b, c in faces:
                f.write(f"3 {a} {b} {c}\n")

    def _append_ellipsoid(self, verts, faces, mean, scales, rotation, color):
        base = len(verts)
        uv = self.unit_vertices.to(device=mean.device, dtype=mean.dtype)
        xyz = mean.view(1, 3) + (uv * scales.view(1, 3)) @ rotation.T
        for p in xyz.detach().cpu().tolist():
            verts.append((float(p[0]), float(p[1]), float(p[2]), int(color[0]), int(color[1]), int(color[2])))
        for f in self.unit_faces:
            faces.append((base + int(f[0]), base + int(f[1]), base + int(f[2])))

    @staticmethod
    def _make_uv_sphere(lat: int, lon: int):
        verts = []
        faces = []
        for i in range(lat + 1):
            theta = torch.pi * i / lat
            st = torch.sin(torch.tensor(theta))
            ct = torch.cos(torch.tensor(theta))
            for j in range(lon):
                phi = 2 * torch.pi * j / lon
                verts.append([float(st * torch.cos(torch.tensor(phi))), float(st * torch.sin(torch.tensor(phi))), float(ct)])
        for i in range(lat):
            for j in range(lon):
                a = i * lon + j
                b = i * lon + (j + 1) % lon
                c = (i + 1) * lon + j
                d = (i + 1) * lon + (j + 1) % lon
                faces.append([a, c, b])
                faces.append([b, c, d])
        return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)

    @staticmethod
    def _level_color(level: float) -> Tuple[int, int, int]:
        if level >= 3.19:
            return MEASURED_LEVEL_COLORS[3.2]
        if level >= 1.59:
            return MEASURED_LEVEL_COLORS[1.6]
        if level >= 0.79:
            return MEASURED_LEVEL_COLORS[0.8]
        return MEASURED_LEVEL_COLORS[0.4]

    @staticmethod
    def _sample_tag(metas: Optional[Any], batch_idx: int, counter: int) -> str:
        token = None
        if isinstance(metas, (list, tuple)) and batch_idx < len(metas) and isinstance(metas[batch_idx], dict):
            token = metas[batch_idx].get("token") or metas[batch_idx].get("sample_idx")
        elif isinstance(metas, dict):
            token = metas.get("token") or metas.get("sample_idx")
        token = str(token) if token is not None else f"sample{batch_idx}"
        token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token)
        return f"{counter:06d}_{token}_batch{batch_idx}"
