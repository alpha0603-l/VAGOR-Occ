from __future__ import annotations

from typing import Optional, Sequence, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.registry import MODELS

from .gaussian_mamba_encoder import GaussianMambaFeatureAggregation3D


@MODELS.register_module()
class GaussianGAFMambaFeatureAggregation3D(GaussianMambaFeatureAggregation3D):
    """Gau-Occ style GAF-lite image sampler + Gaussian Mamba token mixer.

    Design goals for GSF3D:
      1) keep the original GaussianFormer3D differentiable Gaussian path;
      2) do not restore the heavy 3D deformable attention / depth volume;
      3) inject multi-view image semantics into Gaussian tokens with geometry-guided 2D sampling;
      4) let Mamba scan all image-enhanced Gaussian tokens afterwards.

    The sampler is intentionally lightweight:
      - fpc is built from Gaussian token/anchor embedding plus optional LiDAR voxel context;
      - fpc predicts local 2D offsets around each Gaussian projection on every FPN level;
      - image tokens are bilinearly sampled by grid_sample, so gradients flow to image features,
        sampling offsets, and the projected Gaussian coordinates;
      - LiDAR voxel lookup is a hard sparse context lookup, but it is only a conditioning signal and
        does not detach or replace the main Gaussian gradient path.
    """

    def __init__(
        self,
        *args,
        num_cams: int = 6,
        num_levels: int = 4,
        num_offsets: int = 4,
        img_feat_channels: int = 128,
        lidar_feat_dim: int = 6,
        use_lidar_fpc: bool = True,
        fpc_voxel_size: Sequence[float] = (0.4, 0.4, 0.4),
        lidar_pc_range: Optional[Sequence[float]] = None,
        fpc_aggregation: str = "soft",
        fpc_radius: float = 3.2,
        fpc_max_voxels: int = 4096,
        fpc_chunk_size: int = 512,
        offset_radius: Sequence[float] = (2.0, 2.0, 2.0, 2.0),
        sampling_active_offsets: int = 0,
        sampling_scale_factors: Sequence[float] = (1.0,),
        min_depth: float = 1.0e-3,
        gaf_residual_scale: float = 1.0,
        learnable_gaf_scale: bool = True,
        gaf_apply_layer_indices: Optional[Sequence[int]] = None,
        use_uvd_sampling: bool = False,
        depth_channels: int = 112,
        depth_bound: Sequence[float] = (2.0, 58.0, 0.5),
        num_depth_offsets: int = 3,
        depth_offset_radius: float = 1.0,
        min_depth_prob_weight: float = 0.0,
        # Memory root fix for 8k/10k+ Gaussian tokens:
        #   1) memory_efficient_uvd avoids materializing
        #      [B, G, Cam, Noff, Nd, C] sampled_uvd tensors.
        #   2) gaf_token_chunk_size bounds grid_sample/depth_sample peak memory
        #      along the Gaussian-token dimension. 0 keeps old full-token path.
        memory_efficient_uvd: bool = True,
        gaf_token_chunk_size: int = 0,
        use_source_embed: bool = True,
        num_source_types: int = 4,
        source_embed_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_cams = int(num_cams)
        self.num_levels = int(num_levels)
        self.num_offsets = int(num_offsets)
        self.img_feat_channels = int(img_feat_channels)
        self.lidar_feat_dim = int(lidar_feat_dim)
        self.use_lidar_fpc = bool(use_lidar_fpc)
        self.fpc_aggregation = str(fpc_aggregation).lower().strip()
        self.fpc_radius = float(fpc_radius)
        self.fpc_max_voxels = max(int(fpc_max_voxels), 1)
        self.fpc_chunk_size = max(int(fpc_chunk_size), 1)
        self.min_depth = float(min_depth)
        self.gaf_apply_layer_indices = None if gaf_apply_layer_indices is None else set(int(x) for x in gaf_apply_layer_indices)
        # Sparse-UVD options.  This path does not build a dense [B,Cam,C,D,H,W]
        # feature volume.  It samples sparse (u,v) image tokens and uses the
        # predicted depth distribution at those sparse locations to weight a few
        # nearby depth bins.
        self.use_uvd_sampling = bool(use_uvd_sampling)
        self.depth_channels = int(depth_channels)
        db = list(float(x) for x in depth_bound)
        if len(db) != 3:
            raise ValueError("depth_bound must be [min_depth, max_depth, step].")
        self.depth_min = float(db[0])
        self.depth_max = float(db[1])
        self.depth_step = float(db[2])
        self.num_depth_offsets = max(int(num_depth_offsets), 1)
        self.depth_offset_radius = float(depth_offset_radius)
        self.min_depth_prob_weight = float(min_depth_prob_weight)
        self.memory_efficient_uvd = bool(memory_efficient_uvd)
        self.gaf_token_chunk_size = max(int(gaf_token_chunk_size), 0)
        self.use_source_embed = bool(use_source_embed)
        self.num_source_types = max(int(num_source_types), 1)
        self.source_embed_scale_value = float(source_embed_scale)
        if self.use_source_embed:
            self.source_embedding = nn.Embedding(self.num_source_types, self.embed_dims)
        else:
            self.source_embedding = None

        if lidar_pc_range is None:
            lidar_pc_range = self.pc_range
        self.register_buffer("lidar_pc_min", torch.tensor(lidar_pc_range[:3], dtype=torch.float32), persistent=False)
        self.register_buffer("fpc_vs", torch.tensor(fpc_voxel_size, dtype=torch.float32), persistent=False)

        radius = list(float(x) for x in offset_radius)
        if len(radius) < self.num_levels:
            radius = radius + [radius[-1]] * (self.num_levels - len(radius))
        self.register_buffer("offset_radius", torch.tensor(radius[:self.num_levels], dtype=torch.float32), persistent=False)

        # Checkpoint-compatible multi-ring sampling:
        # no trainable tensor shape is changed. A subset of the existing learned
        # directions is evaluated at multiple fixed radii.
        active_offsets = int(sampling_active_offsets)
        if active_offsets <= 0:
            active_offsets = self.num_offsets
        self.sampling_active_offsets = min(max(active_offsets, 1), self.num_offsets)

        scale_factors = [float(x) for x in sampling_scale_factors]
        if not scale_factors:
            scale_factors = [1.0]
        if any(x <= 0.0 for x in scale_factors):
            raise ValueError(
                f"sampling_scale_factors must contain only positive values, got {scale_factors}"
            )
        self.num_sampling_scales = len(scale_factors)
        self.register_buffer(
            "sampling_scale_factors",
            torch.tensor(scale_factors, dtype=torch.float32),
            persistent=False,
        )

        if self.num_depth_offsets == 1:
            depth_base_offsets = torch.zeros(1, dtype=torch.float32)
        else:
            depth_base_offsets = torch.linspace(
                -self.depth_offset_radius,
                self.depth_offset_radius,
                self.num_depth_offsets,
                dtype=torch.float32,
            )
        self.register_buffer("depth_base_offsets", depth_base_offsets, persistent=False)

        # fpc follows Gau-Occ's role: a geometry-conditioned anchor descriptor.
        # We use current Gaussian token + anchor_embed + LiDAR sparse context as first implementation.
        self.lidar_context_proj = nn.Sequential(
            nn.Linear(self.lidar_feat_dim + 1, self.embed_dims),
            nn.GELU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.fpc_norm = nn.LayerNorm(self.embed_dims)
        self.offset_mlp = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.GELU(),
            nn.Linear(self.embed_dims, self.num_levels * self.num_offsets * 2),
        )
        self.sample_weight_mlp = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.GELU(),
            nn.Linear(self.embed_dims, self.num_levels * self.num_offsets),
        )
        self.depth_offset_mlp = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.GELU(),
            nn.Linear(self.embed_dims, self.num_levels * self.num_offsets * self.num_depth_offsets),
        )
        self.depth_embed = nn.Embedding(self.depth_channels, self.img_feat_channels)
        self.img_proj = nn.Linear(self.img_feat_channels, self.embed_dims)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.embed_dims * 3, self.embed_dims),
            nn.GELU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        if learnable_gaf_scale:
            self.gaf_residual_scale = nn.Parameter(torch.tensor(float(gaf_residual_scale), dtype=torch.float32))
        else:
            self.register_buffer("gaf_residual_scale", torch.tensor(float(gaf_residual_scale), dtype=torch.float32), persistent=False)

    def init_weight(self) -> None:
        super().init_weight()
        if self.source_embedding is not None:
            nn.init.normal_(self.source_embedding.weight, mean=0.0, std=0.02)
        # Keep image injection close to zero at startup for stable training, while preserving gradients.
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)

    def forward(
        self,
        instance_feature: Tensor,
        anchor: Tensor,
        anchor_embed: Tensor,
        feature_maps=None,
        dpt_feature_maps=None,
        metas=None,
        anchor_encoder=None,
        voxel_lidar_feats: Optional[Tensor] = None,
        coors_batch: Optional[Tensor] = None,
        gaussian_source_types: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        if instance_feature is None or anchor is None:
            return instance_feature
        self._call_count += 1
        start_time = time.perf_counter() if self.debug else 0.0

        residual = instance_feature
        token = instance_feature
        if self.use_anchor_embed and anchor_embed is not None:
            token = token + anchor_embed.to(dtype=token.dtype)
        if self.source_embedding is not None and gaussian_source_types is not None:
            st = gaussian_source_types.to(device=token.device, dtype=torch.long)
            if st.ndim == 1:
                st = st.view(1, -1).expand(token.shape[0], -1)
            if st.shape[0] == token.shape[0] and st.shape[1] == token.shape[1]:
                st = st.clamp(0, self.num_source_types - 1)
                src = self.source_embedding(st).to(dtype=token.dtype)
                token = token + float(self.source_embed_scale_value) * src

        means = self._anchor_to_xyz(anchor).to(dtype=torch.float32)
        norm_xyz = self._normalize_xyz(means)

        # GAF-lite image feature injection before Mamba.  Only selected layers run GAF to control cost.
        debug_img_abs_mean = None
        debug_gate_mean = None
        apply_gaf = self.gaf_apply_layer_indices is None or int(self._mamba_layer_index) in self.gaf_apply_layer_indices
        if apply_gaf and feature_maps is not None and metas is not None:
            if isinstance(feature_maps, torch.Tensor):
                feature_maps = [feature_maps]
            if len(feature_maps) > 0 and feature_maps[0] is not None:
                fpc = self._build_fpc(
                    token=token,
                    anchor_embed=anchor_embed,
                    means=means,
                    voxel_lidar_feats=voxel_lidar_feats,
                    coors_batch=coors_batch,
                )
                img_token = self._sample_multiview_image_tokens(
                    means=means,
                    fpc=fpc,
                    feature_maps=feature_maps,
                    metas=metas,
                    dtype=token.dtype,
                    dpt_feature_maps=dpt_feature_maps,
                )
                if img_token is not None:
                    img_token = self.img_proj(img_token.float()).to(dtype=token.dtype)
                    gate_in = torch.cat([token.float(), fpc.float(), img_token.float()], dim=-1)
                    gate = torch.sigmoid(self.gate_mlp(gate_in)).to(dtype=token.dtype)
                    if self.debug and self._mamba_layer_index == 0:
                        debug_img_abs_mean = img_token.detach().abs().mean()
                        debug_gate_mean = gate.detach().mean()
                    scale = self.gaf_residual_scale.to(device=token.device, dtype=token.dtype)
                    token = token + scale * gate * img_token

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
            elapsed = time.perf_counter() - start_time
            with torch.no_grad():
                img_abs = float(debug_img_abs_mean.detach().cpu().item()) if debug_img_abs_mean is not None else 0.0
                gate_mean = float(debug_gate_mean.detach().cpu().item()) if debug_gate_mean is not None else 0.0
                print(
                    "[GaussianGAFMambaFeatureAggregation3D Debug] "
                    f"call={self._call_count}, layer={self._mamba_layer_index}, shape={tuple(instance_feature.shape)}, "
                    f"apply_gaf={bool(apply_gaf)}, feature_maps={0 if feature_maps is None else len(feature_maps)}, "
                    f"num_offsets={self.num_offsets}, active_offsets={self.sampling_active_offsets}, scales={self.num_sampling_scales}, num_levels={self.num_levels}, fpc={self.fpc_aggregation}, "
                    f"uvd={self.use_uvd_sampling}, depth_offsets={self.num_depth_offsets}, source_embed={self.use_source_embed}, "
                    f"residual_scale={float(scale.detach().cpu().item()):.6f}, "
                    f"gaf_scale={float(self.gaf_residual_scale.detach().cpu().item()):.6f}, "
                    f"token_abs_mean={float(token.detach().abs().mean().cpu().item()):.6f}, "
                    f"img_abs_mean={img_abs:.6f}, gate_mean={gate_mean:.6f}, "
                    f"delta_abs_mean={float(delta.detach().abs().mean().cpu().item()):.6f}, "
                    f"out_abs_mean={float(out.detach().abs().mean().cpu().item()):.6f}, "
                    f"time={elapsed:.6f}s",
                    flush=True,
                )
            self._debug_print_count += 1
        return out

    def _build_fpc(
        self,
        token: Tensor,
        anchor_embed: Optional[Tensor],
        means: Tensor,
        voxel_lidar_feats: Optional[Tensor],
        coors_batch: Optional[Tensor],
    ) -> Tensor:
        # `token` already contains anchor_embed when use_anchor_embed=True.
        # Do not add anchor_embed again here, otherwise fpc becomes
        # instance_feature + 2 * anchor_embed + lidar_context.
        base = token.float()
        if not self.use_lidar_fpc:
            return self.fpc_norm(base).to(dtype=token.dtype)
        context = self._lookup_lidar_context(means, voxel_lidar_feats, coors_batch)
        context_token = self.lidar_context_proj(context.float())
        return self.fpc_norm(base + context_token).to(dtype=token.dtype)

    def _lookup_lidar_context(
        self,
        means: Tensor,
        voxel_lidar_feats: Optional[Tensor],
        coors_batch: Optional[Tensor],
    ) -> Tensor:
        B, G, _ = means.shape
        device = means.device
        out = means.new_zeros((B, G, self.lidar_feat_dim + 1), dtype=torch.float32)
        if voxel_lidar_feats is None or coors_batch is None or voxel_lidar_feats.numel() == 0:
            return out
        feats = voxel_lidar_feats.to(device=device, dtype=torch.float32)
        coors = coors_batch.to(device=device)
        feat_dim = min(int(feats.shape[-1]), self.lidar_feat_dim)

        for b in range(B):
            mask = coors[:, 0].long() == b
            if not torch.any(mask):
                continue
            f = feats[mask]
            if f.numel() == 0:
                continue
            if self.fpc_aggregation == "hard":
                out[b] = self._lookup_lidar_context_hard(means[b], f, feat_dim)
            else:
                out[b] = self._lookup_lidar_context_soft(means[b], f, feat_dim)
        return out

    def _lookup_lidar_context_soft(self, means_b: Tensor, feats_b: Tensor, feat_dim: int) -> Tensor:
        # Gau-Occ style soft geometry context: exponential distance kernel around each Gaussian.
        # The voxel subset cap controls cost; within the subset, the weighted average is differentiable
        # w.r.t. Gaussian centers and LiDAR voxel features.
        G = int(means_b.shape[0])
        out = means_b.new_zeros((G, self.lidar_feat_dim + 1), dtype=torch.float32)
        if feats_b.shape[0] > self.fpc_max_voxels:
            idx = torch.linspace(0, feats_b.shape[0] - 1, self.fpc_max_voxels, device=feats_b.device).long()
            feats_b = feats_b.index_select(0, idx)
        xyz = feats_b[:, :3].float()
        vals = feats_b[:, :feat_dim].float()
        radius = max(float(self.fpc_radius), 1.0e-3)
        sigma2 = radius * radius
        chunks = []
        valids = []
        for start in range(0, G, self.fpc_chunk_size):
            end = min(start + self.fpc_chunk_size, G)
            q = means_b[start:end].float()
            dist2 = torch.cdist(q, xyz, p=2.0).pow(2)
            weights = torch.exp(-dist2 / (2.0 * sigma2))
            if self.fpc_radius > 0:
                weights = weights * (dist2 <= sigma2).float()
            denom = weights.sum(dim=-1, keepdim=True)
            ctx = weights @ vals / denom.clamp_min(1.0e-6)
            chunks.append(ctx)
            valids.append((denom > 1.0e-6).float())
        ctx = torch.cat(chunks, dim=0)
        valid = torch.cat(valids, dim=0)
        out[:, :feat_dim] = ctx
        out[:, self.lidar_feat_dim:self.lidar_feat_dim + 1] = valid
        return out

    def _lookup_lidar_context_hard(self, means_b: Tensor, feats_b: Tensor, feat_dim: int) -> Tensor:
        # Optional fast sparse coarse-cell lookup.  This is less smooth than soft aggregation,
        # so the default is fpc_aggregation="soft".
        G = int(means_b.shape[0])
        out = means_b.new_zeros((G, self.lidar_feat_dim + 1), dtype=torch.float32)
        pc_min = self.lidar_pc_min.to(device=means_b.device, dtype=torch.float32)
        vs = self.fpc_vs.to(device=means_b.device, dtype=torch.float32).clamp_min(1.0e-6)
        xyz = feats_b[:, :3]
        v_idx = torch.floor((xyz - pc_min.view(1, 3)) / vs.view(1, 3)).long()
        g_idx = torch.floor((means_b.float() - pc_min.view(1, 3)) / vs.view(1, 3)).long()
        v_hash = v_idx[:, 0] * 73856093 + v_idx[:, 1] * 19349663 + v_idx[:, 2] * 83492791
        g_hash = g_idx[:, 0] * 73856093 + g_idx[:, 1] * 19349663 + g_idx[:, 2] * 83492791
        order = torch.argsort(v_hash)
        v_hash_sorted = v_hash[order]
        f_sorted = feats_b[order]
        pos = torch.searchsorted(v_hash_sorted, g_hash)
        valid = (pos < v_hash_sorted.numel()) & (v_hash_sorted[pos.clamp_max(max(v_hash_sorted.numel() - 1, 0))] == g_hash)
        if torch.any(valid):
            chosen = f_sorted[pos[valid].clamp_max(f_sorted.shape[0] - 1)]
            out[valid, :feat_dim] = chosen[:, :feat_dim]
            out[valid, self.lidar_feat_dim:self.lidar_feat_dim + 1] = 1.0
        return out


    def _linear_sample_depth_prob(self, depth_dist: Tensor, depth_index: Tensor) -> Tuple[Tensor, Tensor]:
        """Linearly sample depth probabilities at sparse depth-channel indices.

        Args:
            depth_dist: [B, G, Cam, Noff, D]
            depth_index: [B, G, Cam, Noff, Nd]
        Returns:
            prob and valid, both [B, G, Cam, Noff, Nd].
        """
        D = int(depth_dist.shape[-1])
        d0f = torch.floor(depth_index)
        d0 = d0f.long()
        d1 = d0 + 1
        valid = (depth_index >= 0.0) & (depth_index <= float(D - 1))
        d0c = d0.clamp(0, max(D - 1, 0))
        d1c = d1.clamp(0, max(D - 1, 0))
        p0 = torch.gather(depth_dist, dim=-1, index=d0c)
        p1 = torch.gather(depth_dist, dim=-1, index=d1c)
        alpha = (depth_index - d0f).clamp(0.0, 1.0)
        prob = p0 * (1.0 - alpha) + p1 * alpha
        return prob * valid.float(), valid

    def _sample_multiview_image_tokens(
        self,
        means: Tensor,
        fpc: Tensor,
        feature_maps,
        metas,
        dtype: torch.dtype,
        dpt_feature_maps=None,
    ) -> Optional[Tensor]:
        if not isinstance(metas, dict) or "projection_mat" not in metas or "image_wh" not in metas:
            return None
        projection = metas["projection_mat"].to(device=means.device, dtype=torch.float32)
        image_wh = metas["image_wh"].to(device=means.device, dtype=torch.float32)
        if projection.ndim == 3:
            projection = projection.unsqueeze(0)
        if image_wh.ndim == 2:
            image_wh = image_wh.unsqueeze(0)

        B, G, _ = means.shape
        gaf_chunk = int(getattr(self, "gaf_token_chunk_size", 0))
        if gaf_chunk > 0 and G > gaf_chunk:
            # Chunk only the Gaussian-token dimension. This keeps the mathematical
            # output identical to the full-token path because each token samples
            # image/depth features independently before the later Mamba mixing.
            chunks = []
            for start in range(0, G, gaf_chunk):
                end = min(start + gaf_chunk, G)
                out = self._sample_multiview_image_tokens(
                    means=means[:, start:end],
                    fpc=fpc[:, start:end],
                    feature_maps=feature_maps,
                    metas=metas,
                    dtype=dtype,
                    dpt_feature_maps=dpt_feature_maps,
                )
                if out is None:
                    return None
                chunks.append(out)
            return torch.cat(chunks, dim=1)

        num_levels = min(self.num_levels, len(feature_maps))
        if isinstance(dpt_feature_maps, torch.Tensor):
            dpt_feature_maps = [dpt_feature_maps]
        has_depth = (
            self.use_uvd_sampling
            and dpt_feature_maps is not None
            and len(dpt_feature_maps) > 0
            and dpt_feature_maps[0] is not None
        )
        if has_depth:
            num_levels = min(num_levels, len(dpt_feature_maps))
        fpc_float = fpc.float()
        offsets = torch.tanh(self.offset_mlp(fpc_float)).view(
            B, G, self.num_levels, self.num_offsets, 2
        )
        raw_weights = F.softplus(self.sample_weight_mlp(fpc_float)).view(
            B, G, self.num_levels, self.num_offsets
        )
        depth_offsets = torch.tanh(self.depth_offset_mlp(fpc_float)).view(
            B, G, self.num_levels, self.num_offsets, self.num_depth_offsets
        )
        depth_offsets = (
            depth_offsets * self.depth_offset_radius
            + self.depth_base_offsets.view(1, 1, 1, 1, self.num_depth_offsets)
        )

        # Use the same three learned directions as the former V24-I branch,
        # evaluated at two fixed rings. Learned weights and depth offsets are
        # shared across rings, which keeps the epoch-4 checkpoint compatible.
        active = self.sampling_active_offsets
        scale = self.sampling_scale_factors.to(
            device=means.device, dtype=torch.float32
        )
        num_samples = active * self.num_sampling_scales

        xyz1 = torch.cat([means.float(), torch.ones_like(means[..., :1]).float()], dim=-1)
        # [B, cam, G, 4]
        proj = torch.einsum("bcij,bgj->bcgi", projection, xyz1)
        depth = proj[..., 2]
        u = proj[..., 0] / depth.clamp_min(self.min_depth)
        v = proj[..., 1] / depth.clamp_min(self.min_depth)
        W_img = image_wh[..., 0].unsqueeze(-1).clamp_min(1.0)
        H_img = image_wh[..., 1].unsqueeze(-1).clamp_min(1.0)
        valid_base = (depth > self.min_depth) & (u >= 0.0) & (v >= 0.0) & (u <= (W_img - 1.0)) & (v <= (H_img - 1.0))

        accum = means.new_zeros((B, G, self.img_feat_channels), dtype=torch.float32)
        denom = means.new_zeros((B, G, 1), dtype=torch.float32)
        radius = self.offset_radius.to(device=means.device, dtype=torch.float32)

        for lvl in range(num_levels):
            feat = feature_maps[lvl]
            if feat is None:
                continue
            if feat.ndim != 5:
                raise ValueError(f"feature_maps[{lvl}] must be [B,N,C,H,W], got {tuple(feat.shape)}")
            Bf, Cam, C, H, W = feat.shape
            if Bf != B:
                raise ValueError(f"feature_maps batch mismatch: {Bf} vs {B}")
            C_use = min(int(C), self.img_feat_channels)
            feat = feat[:, :self.num_cams, :C_use]
            Cam = feat.shape[1]

            u_l = u[:, :Cam] / W_img[:, :Cam].clamp_min(1.0) * max(W - 1, 1)
            v_l = v[:, :Cam] / H_img[:, :Cam].clamp_min(1.0) * max(H - 1, 1)
            base_off = offsets[:, :, lvl, :active]
            off = (
                base_off.unsqueeze(-2)
                * scale.view(1, 1, 1, self.num_sampling_scales, 1)
                * radius[lvl]
            ).reshape(B, G, num_samples, 2)

            grid_x = u_l.unsqueeze(-1) + off[:, None, :, :, 0]
            grid_y = v_l.unsqueeze(-1) + off[:, None, :, :, 1]
            valid = (
                valid_base[:, :Cam].unsqueeze(-1)
                & (grid_x >= 0.0)
                & (grid_x <= max(W - 1, 1))
                & (grid_y >= 0.0)
                & (grid_y <= max(H - 1, 1))
            )
            gx = 2.0 * grid_x / max(W - 1, 1) - 1.0
            gy = 2.0 * grid_y / max(H - 1, 1) - 1.0
            grid = torch.stack([gx, gy], dim=-1).reshape(
                B * Cam, G * num_samples, 1, 2
            )
            sampled = F.grid_sample(
                feat.reshape(B * Cam, C_use, H, W).float(),
                grid.float(),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled = sampled.view(
                B, Cam, C_use, G, num_samples
            ).permute(0, 3, 1, 4, 2).contiguous()

            # Share each direction's learned confidence across the two rings.
            # The final aggregation is normalized by the accumulated denominator.
            ring_weights = (
                raw_weights[:, :, lvl, :active]
                .unsqueeze(-1)
                .expand(B, G, active, self.num_sampling_scales)
                .reshape(B, G, num_samples)
            )
            w_uv = (
                ring_weights.unsqueeze(2)
                * valid.permute(0, 2, 1, 3).float()
            )
            if C_use < self.img_feat_channels:
                sampled = F.pad(sampled, (0, self.img_feat_channels - C_use))

            if has_depth:
                dpt = dpt_feature_maps[lvl]
                if dpt is not None:
                    if dpt.ndim != 5:
                        raise ValueError(f"dpt_feature_maps[{lvl}] must be [B,N,D,H,W], got {tuple(dpt.shape)}")
                    Bd, CamD, D, Hd, Wd = dpt.shape
                    if Bd != B:
                        raise ValueError(f"dpt_feature_maps batch mismatch: {Bd} vs {B}")
                    CamUse = min(Cam, CamD, self.num_cams)
                    dpt = dpt[:, :CamUse]
                    # Sample depth distribution at the same sparse uv positions.  The depth maps are
                    # normally interpolated to each FPN level, but this supports mismatched sizes too.
                    u_d = u[:, :CamUse] / W_img[:, :CamUse].clamp_min(1.0) * max(Wd - 1, 1)
                    v_d = v[:, :CamUse] / H_img[:, :CamUse].clamp_min(1.0) * max(Hd - 1, 1)
                    grid_x_d = u_d.unsqueeze(-1) + off[:, None, :, :, 0]
                    grid_y_d = v_d.unsqueeze(-1) + off[:, None, :, :, 1]
                    gx_d = 2.0 * grid_x_d / max(Wd - 1, 1) - 1.0
                    gy_d = 2.0 * grid_y_d / max(Hd - 1, 1) - 1.0
                    grid_d = torch.stack([gx_d, gy_d], dim=-1).reshape(
                        B * CamUse, G * num_samples, 1, 2
                    )
                    sampled_dpt = F.grid_sample(
                        dpt.reshape(B * CamUse, D, Hd, Wd).float(),
                        grid_d.float(),
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=True,
                    )
                    sampled_dpt = sampled_dpt.view(
                        B, CamUse, D, G, num_samples
                    ).permute(0, 3, 1, 4, 2).contiguous()
                    # Convert metric camera depth to the same depth-channel convention used by DepthHead.
                    # For dbound=[2,58,0.5], z=2.0m maps to channel 0.
                    base_depth_bin = (depth[:, :CamUse].permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1) - self.depth_min) / max(self.depth_step, 1.0e-6)
                    ring_depth_offsets = (
                        depth_offsets[:, :, lvl, :active]
                        .unsqueeze(-2)
                        .expand(
                            B,
                            G,
                            active,
                            self.num_sampling_scales,
                            self.num_depth_offsets,
                        )
                        .reshape(B, G, num_samples, self.num_depth_offsets)
                    )
                    d_index = base_depth_bin + ring_depth_offsets.unsqueeze(2)
                    d_prob, d_valid = self._linear_sample_depth_prob(sampled_dpt, d_index)
                    if self.min_depth_prob_weight > 0.0:
                        d_prob = d_prob + self.min_depth_prob_weight * d_valid.float()
                    w = w_uv[:, :, :CamUse].unsqueeze(-1) * d_prob
                    d_nearest = torch.round(d_index).long().clamp(0, max(self.depth_channels - 1, 0))

                    if self.memory_efficient_uvd:
                        # Old code built:
                        #   sampled_uvd = sampled_uv.unsqueeze(-2) + depth_embed[d_nearest]
                        #   accum += (sampled_uvd * w.unsqueeze(-1)).sum(Cam,Noff,Nd)
                        # That materializes [B,G,Cam,Noff,Nd,C], which becomes a
                        # multi-GB activation around G=8k/10k.  Algebraically:
                        #   sum((sampled_uv + emb) * w)
                        # = sampled_uv * sum_Nd(w) + sum(emb * w).
                        # We compute the depth-embedding term through a compact
                        # [B,G,D] histogram, then a matmul with the embedding table.
                        uv_weight = w.sum(dim=-1)  # [B,G,Cam,Noff]
                        accum = accum + (sampled[:, :, :CamUse] * uv_weight.unsqueeze(-1)).sum(dim=(2, 3))

                        depth_hist = sampled.new_zeros((B, G, self.depth_channels), dtype=torch.float32)
                        depth_hist.scatter_add_(
                            2,
                            d_nearest.reshape(B, G, -1),
                            w.reshape(B, G, -1).to(dtype=depth_hist.dtype),
                        )
                        depth_token = torch.matmul(
                            depth_hist,
                            self.depth_embed.weight.to(device=sampled.device, dtype=depth_hist.dtype),
                        )
                        accum = accum + depth_token.to(dtype=accum.dtype)
                    else:
                        # Original full materialization path, kept for exact A/B debugging.
                        d_emb = self.depth_embed(d_nearest).to(device=sampled.device, dtype=sampled.dtype)
                        sampled_uvd = sampled[:, :, :CamUse].unsqueeze(-2) + d_emb
                        accum = accum + (sampled_uvd * w.unsqueeze(-1)).sum(dim=(2, 3, 4))

                    denom = denom + w.sum(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)
                    continue

            w = w_uv
            accum = accum + (sampled * w.unsqueeze(-1)).sum(dim=(2, 3))
            denom = denom + w.sum(dim=(2, 3), keepdim=False).unsqueeze(-1)

        return (accum / denom.clamp_min(1.0e-6)).to(dtype=dtype)
