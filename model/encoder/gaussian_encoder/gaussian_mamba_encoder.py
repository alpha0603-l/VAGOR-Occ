from __future__ import annotations

import os
import time
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmengine.model import BaseModule, xavier_init, constant_init
from mmengine.registry import MODELS

from ...utils.safe_ops import safe_sigmoid
from .gaussian_mamba import MixerModel


@MODELS.register_module()
class GaussianMambaFeatureAggregation3D(BaseModule):
    _instance_counter = 0

    """H2HE/Z-prior Mamba aggregation layer for sparse Gaussian tokens.

    This module is a drop-in replacement for ``DeformableFeatureAggregation3D``
    in ``GaussianOccEncoder3D``.  It deliberately keeps the same forward
    signature so the existing encoder operation order can remain unchanged:

        instance_feature, anchor, anchor_embed, feature_maps, dpt_feature_maps,
        metas, anchor_encoder -> instance_feature

    It does **not** perform image/depth multi-scale sampling.  Instead, it uses
    the OccMamba-style sparse Gaussian ordering implemented in the user's
    ``GaussianMambaCorrectionHead``: quantize each Gaussian center into an XY
    column, order XY columns by a 2D Hilbert/H2HE curve, and order tokens inside
    each XY column along Z.  Mamba then acts as a residual correction over the
    ordered Gaussian token sequence.

    The last projection is zero-initialized in ``init_weight`` so the module is
    an identity mapping at startup.  Occupancy loss still backpropagates through
    the residual path once the projection starts learning.
    """

    def __init__(
        self,
        embed_dims: int = 128,
        d_model: Optional[int] = None,
        n_layer: int = 2,
        ssm_cfg: Optional[Dict] = None,
        drop_path: float = 0.10,
        drop_out_in_block: float = 0.0,
        rms_norm: bool = False,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        norm_epsilon: float = 1.0e-5,
        pc_range: Sequence[float] = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        order_voxel_size: Sequence[float] = (0.4, 0.4, 0.2),
        order_method: str = "H2HE",
        coor_order: str = "xy",
        inverse: bool = False,
        phi_activation: str = "sigmoid",
        xyz_coordinate: str = "cartesian",
        use_anchor_embed: bool = True,
        use_pos_embed: bool = True,
        residual_scale: float = 0.10,
        learnable_residual_scale: bool = True,
        precompute_order_cache: bool = True,
        debug: bool = True,
        debug_interval: int = 50,
        debug_max_print: int = 5,
        startup_log: bool = True,
        init_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg)
        # 中文注释：避免 GaussianOccEncoder3D 的通用 Xavier 初始化覆盖 mamba_ssm 自带初始化。
        self.skip_encoder_xavier_init = True
        self.embed_dims = int(embed_dims)
        self.d_model = int(d_model if d_model is not None else embed_dims)
        self.n_layer = int(n_layer)
        self.ssm_cfg = dict(ssm_cfg or {})
        self.pc_range = tuple(float(v) for v in pc_range)
        if len(self.pc_range) != 6:
            raise ValueError("pc_range must have 6 values.")
        self.order_voxel_size = tuple(float(v) for v in order_voxel_size)
        if len(self.order_voxel_size) != 3:
            raise ValueError("order_voxel_size must have 3 values.")
        self.order_method = str(order_method).upper().strip()
        if self.order_method != "H2HE":
            raise ValueError("Only H2HE ordering is implemented.")
        self.coor_order = str(coor_order).lower().strip()
        if self.coor_order not in {"xy", "yx"}:
            raise ValueError("coor_order must be 'xy' or 'yx'.")
        self.inverse = bool(inverse)
        self.phi_activation = str(phi_activation)
        self.xyz_coordinate = str(xyz_coordinate)
        self.use_anchor_embed = bool(use_anchor_embed)
        self.use_pos_embed = bool(use_pos_embed)
        self.debug = bool(debug)
        self.debug_interval = max(int(debug_interval), 1)
        self.debug_max_print = max(int(debug_max_print), 0)
        self._call_count = 0
        self._debug_print_count = 0
        # 中文注释：encoder 里通常有 4 个 mamba 层。只让第 0 个层按 batch 计数打印，
        # 保持“每 50 个 batch 打印一次、最多 5 次”的日志量，避免每层都刷屏。
        self._mamba_layer_index = int(GaussianMambaFeatureAggregation3D._instance_counter)
        GaussianMambaFeatureAggregation3D._instance_counter += 1
        self._xy_rank_cache: Dict[Tuple[int, int, str, bool], torch.Tensor] = {}

        pc_min = torch.tensor(self.pc_range[:3], dtype=torch.float32)
        pc_max = torch.tensor(self.pc_range[3:], dtype=torch.float32)
        order_vs = torch.tensor(self.order_voxel_size, dtype=torch.float32)
        self.register_buffer("pc_min", pc_min, persistent=False)
        self.register_buffer("pc_max", pc_max, persistent=False)
        self.register_buffer("order_vs", order_vs, persistent=False)

        grid = torch.ceil((pc_max - pc_min) / torch.clamp(order_vs, min=1.0e-6)).long()
        self.grid_x = int(grid[0].item())
        self.grid_y = int(grid[1].item())
        self.grid_z = int(grid[2].item())

        # 中文注释：输入仍然是高斯 token 特征；anchor_embed 只作为几何先验融合，不改变原接口。
        self.in_norm = nn.LayerNorm(self.embed_dims)
        self.in_proj = nn.Linear(self.embed_dims, self.d_model)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.mamba = MixerModel(
            d_model=self.d_model,
            n_layer=self.n_layer,
            ssm_cfg=self.ssm_cfg,
            norm_epsilon=float(norm_epsilon),
            rms_norm=bool(rms_norm),
            fused_add_norm=bool(fused_add_norm),
            residual_in_fp32=bool(residual_in_fp32),
            drop_out_in_block=float(drop_out_in_block),
            drop_path=float(drop_path),
        )
        self.out_proj = nn.Linear(self.d_model, self.embed_dims)
        self.out_drop = nn.Dropout(float(drop_out_in_block)) if drop_out_in_block > 0.0 else nn.Identity()
        if learnable_residual_scale:
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale), dtype=torch.float32))
        else:
            self.register_buffer("residual_scale", torch.tensor(float(residual_scale), dtype=torch.float32), persistent=False)

        if precompute_order_cache:
            self._get_xy_rank_cpu(self.grid_x, self.grid_y, self.coor_order, self.inverse)

        if startup_log and self._is_main_process() and self._mamba_layer_index == 0:
            print(
                "[GaussianMambaFeatureAggregation3D Startup] "
                f"embed_dims={self.embed_dims}, d_model={self.d_model}, n_layer={self.n_layer}, "
                f"ssm_cfg={self.ssm_cfg}, order=H2HE_z_prior_sparse, "
                f"grid=[{self.grid_x},{self.grid_y},{self.grid_z}], "
                f"order_voxel_size={list(self.order_voxel_size)}, pc_range={list(self.pc_range)}, "
                f"use_anchor_embed={self.use_anchor_embed}, use_pos_embed={self.use_pos_embed}, "
                f"residual_scale={float(self.residual_scale.detach().cpu().item()):.4f}, "
                f"trainable_params_per_layer={sum(p.numel() for p in self.parameters())}, "
                f"num_mamba_layers_in_encoder≈{GaussianMambaFeatureAggregation3D._instance_counter}",
                flush=True,
            )

    @staticmethod
    def _is_main_process() -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return int(os.environ.get("RANK", "0")) == 0

    def init_weight(self) -> None:
        # 中文注释：先常规初始化输入/位置投影，再将输出投影置零，保证替换 attention 后初始为恒等残差。
        xavier_init(self.in_proj, distribution="uniform", bias=0.0)
        for m in self.pos_embed.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution="uniform", bias=0.0)
        constant_init(self.out_proj, val=0.0, bias=0.0)

    def forward(
        self,
        instance_feature: Tensor,
        anchor: Tensor,
        anchor_embed: Tensor,
        feature_maps=None,
        dpt_feature_maps=None,
        metas=None,
        anchor_encoder=None,
        **kwargs,
    ) -> Tensor:
        if instance_feature is None or anchor is None:
            return instance_feature
        self._call_count += 1
        start = time.perf_counter()

        residual = instance_feature
        token = instance_feature
        if self.use_anchor_embed and anchor_embed is not None:
            token = token + anchor_embed.to(dtype=token.dtype)

        means = self._anchor_to_xyz(anchor).to(dtype=torch.float32)
        norm_xyz = self._normalize_xyz(means)
        order = self._batched_h2he_sparse_order(means)

        token_ordered = self._batch_gather(token, order)
        pos_ordered = self._batch_gather(norm_xyz, order)

        x = self.in_norm(token_ordered).float()
        x = self.in_proj(x)
        pos = self.pos_embed(pos_ordered.float()) if self.use_pos_embed else None
        x = self.mamba(x, pos=pos)
        delta = self.out_proj(x).to(dtype=residual.dtype)
        delta = self.out_drop(delta)

        inv_order = torch.empty_like(order)
        arange = torch.arange(order.shape[1], device=order.device, dtype=order.dtype).view(1, -1).expand_as(order)
        inv_order.scatter_(1, order, arange)
        delta = self._batch_gather(delta, inv_order)

        scale = self.residual_scale.to(device=residual.device, dtype=residual.dtype)
        out = residual + scale * delta

        if (
            self.debug
            and self._is_main_process()
            and self._mamba_layer_index == 0
            and self._debug_print_count < self.debug_max_print
            and self._call_count % self.debug_interval == 0
        ):
            elapsed = time.perf_counter() - start
            with torch.no_grad():
                print(
                    "[GaussianMambaFeatureAggregation3D Debug] "
                    f"call={self._call_count}, layer={self._mamba_layer_index}, shape={tuple(instance_feature.shape)}, "
                    f"order=H2HE_z_prior_sparse, grid=[{self.grid_x},{self.grid_y},{self.grid_z}], "
                    f"ffn_input_dim={int(out.shape[-1])}, residual_scale={float(scale.detach().cpu().item()):.6f}, "
                    f"token_abs_mean={float(token.detach().abs().mean().cpu().item()):.6f}, "
                    f"delta_abs_mean={float(delta.detach().abs().mean().cpu().item()):.6f}, "
                    f"out_abs_mean={float(out.detach().abs().mean().cpu().item()):.6f}, "
                    f"time={elapsed:.6f}s",
                    flush=True,
                )
            self._debug_print_count += 1
        return out

    def _anchor_to_xyz(self, anchor: Tensor) -> Tensor:
        if self.phi_activation == "sigmoid":
            xyz = safe_sigmoid(anchor[..., :3])
        elif self.phi_activation == "loop":
            xy = safe_sigmoid(anchor[..., :2])
            z = torch.remainder(anchor[..., 2:3], 1.0)
            xyz = torch.cat([xy, z], dim=-1)
        else:
            raise NotImplementedError(f"Unsupported phi_activation={self.phi_activation}")

        pc = anchor.new_tensor(self.pc_range)
        if self.xyz_coordinate == "polar":
            rrr = xyz[..., 0] * (pc[3] - pc[0]) + pc[0]
            theta = xyz[..., 1] * (pc[4] - pc[1]) + pc[1]
            phi = xyz[..., 2] * (pc[5] - pc[2]) + pc[2]
            xxx = rrr * torch.sin(theta) * torch.cos(phi)
            yyy = rrr * torch.sin(theta) * torch.sin(phi)
            zzz = rrr * torch.cos(theta)
        elif self.xyz_coordinate == "cartesian":
            xxx = xyz[..., 0] * (pc[3] - pc[0]) + pc[0]
            yyy = xyz[..., 1] * (pc[4] - pc[1]) + pc[1]
            zzz = xyz[..., 2] * (pc[5] - pc[2]) + pc[2]
        else:
            raise NotImplementedError(f"Unsupported xyz_coordinate={self.xyz_coordinate}")
        return torch.stack([xxx, yyy, zzz], dim=-1)

    def _normalize_xyz(self, means: Tensor) -> Tensor:
        pc_min = self.pc_min.to(device=means.device, dtype=means.dtype)
        pc_max = self.pc_max.to(device=means.device, dtype=means.dtype)
        extent = torch.clamp(pc_max - pc_min, min=1.0e-6)
        return 2.0 * (means - pc_min.view(1, 1, 3)) / extent.view(1, 1, 3) - 1.0

    @staticmethod
    def _batch_gather(x: Tensor, index: Tensor) -> Tensor:
        # x: [B, N, C], index: [B, N]
        expand_index = index.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return torch.gather(x, dim=1, index=expand_index)

    def _batched_h2he_sparse_order(self, means: Tensor) -> Tensor:
        orders = [self._h2he_sparse_order(means[b]) for b in range(int(means.shape[0]))]
        return torch.stack(orders, dim=0)

    def _h2he_sparse_order(self, means: Tensor) -> Tensor:
        device = means.device
        pc_min = self.pc_min.to(device=device, dtype=means.dtype)
        order_vs = self.order_vs.to(device=device, dtype=means.dtype)
        idx = torch.floor((means - pc_min.view(1, 3)) / torch.clamp(order_vs.view(1, 3), min=1.0e-6)).long()
        x = idx[:, 0].clamp(0, self.grid_x - 1)
        y = idx[:, 1].clamp(0, self.grid_y - 1)
        z = idx[:, 2].clamp(0, self.grid_z - 1)
        xy_rank_cpu = self._get_xy_rank_cpu(self.grid_x, self.grid_y, self.coor_order, self.inverse)
        xy_rank = xy_rank_cpu.to(device=device, non_blocking=True)
        xy_order = xy_rank[x, y].long()
        # 中文注释：严格沿用 correction 头的 H2HE 思路：XY Hilbert 柱顺序 + 柱内 Z 轴顺序。
        keys = xy_order * int(self.grid_z) + z
        row = torch.arange(means.shape[0], device=device, dtype=torch.long)
        keys = keys * (means.shape[0] + 1) + row
        return torch.argsort(keys)

    def _get_xy_rank_cpu(self, max_x: int, max_y: int, coor_order: str, inverse: bool) -> torch.Tensor:
        key = (int(max_x), int(max_y), str(coor_order), bool(inverse))
        cached = self._xy_rank_cache.get(key)
        if cached is not None:
            return cached
        order_flat = self._h2he_order_index_within_range_cpu(max_x, max_y, coor_order=coor_order, inverse=inverse)
        rank = torch.empty((max_x * max_y,), dtype=torch.long)
        rank[order_flat] = torch.arange(order_flat.numel(), dtype=torch.long)
        rank = rank.view(max_x, max_y).contiguous()
        self._xy_rank_cache[key] = rank
        return rank

    @staticmethod
    def _hilbert2d_points(limit: Tuple[int, int], n: int, x: int = 0, y: int = 0, order=None):
        # Same simple recursive policy as the user's GaussianMambaCorrectionHead.
        if n == 1:
            if x >= limit[0] or y >= limit[1] or x < 0 or y < 0:
                return []
            return [(x, y)]
        if order is None:
            order = [(0, 0), (0, 1), (1, 1), (1, 0)]
        n //= 2
        points = []
        for dx, dy in order:
            points += GaussianMambaFeatureAggregation3D._hilbert2d_points(limit, n, x + dx * n, y + dy * n, order=order)
        return points

    @classmethod
    def _h2he_order_index_within_range_cpu(
        cls,
        max_x: int,
        max_y: int,
        coor_order: str = "xy",
        inverse: bool = False,
    ) -> torch.Tensor:
        index_map = {"x": 0, "y": 1}
        base_order = [(0, 0), (0, 1), (1, 1), (1, 0)] if not inverse else [(0, 0), (0, -1), (-1, -1), (-1, 0)]
        desired_order_indices = [index_map[ch] for ch in coor_order]
        order = [tuple(coord[idx] for idx in desired_order_indices) for coord in base_order]
        cube_side = 1
        while cube_side < max(max_x, max_y):
            cube_side *= 2
        if not inverse:
            points2d = cls._hilbert2d_points((max_x, max_y), cube_side, order=order)
        else:
            points2d = cls._hilbert2d_points((max_x, max_y), cube_side, max_x - 1, max_y - 1, order=order)
        points2d = [p for p in points2d if 0 <= p[0] < max_x and 0 <= p[1] < max_y]
        return torch.tensor([x * max_y + y for x, y in points2d], dtype=torch.long)
