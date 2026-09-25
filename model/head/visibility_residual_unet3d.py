from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def _num_groups(channels: int, preferred: int = 8) -> int:
    channels = int(channels)
    for groups in range(min(int(preferred), channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SeparableConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Tuple[int, int, int] = (1, 1, 1),
    ):
        super().__init__()
        padding = int(kernel_size) // 2
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = SeparableConv3D(in_channels, out_channels)
        self.norm1 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = SeparableConv3D(out_channels, out_channels)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout3d(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.proj = (
            nn.Identity()
            if int(in_channels) == int(out_channels)
            else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.proj(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.act(x + identity)


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: Tuple[int, int, int]):
        super().__init__()
        self.conv = SeparableConv3D(in_channels, out_channels, stride=stride)
        self.norm = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class VisibilityResidualUNet3D(nn.Module):
    """Lightweight post-render 3D U-Net for Occ3D refinement.

    The upstream Mamba in this project operates on Gaussian tokens. GaussianHead
    then renders those post-Mamba Gaussian semantics onto the dense Occ3D grid.
    This module therefore refines the *post-Mamba rendered voxel logits* instead
    of inventing new Gaussians.

    Two residuals are predicted:
      1. semantic residuals for non-empty classes only;
      2. one geometry residual that shifts all non-empty logits against empty.

    Visibility is used conservatively. In STRONG_FREE cells, the geometry branch
    may only suppress occupancy, never create it. This protects precision while
    still allowing UNKNOWN/MIXED regions to be completed from spatial context.
    """

    def __init__(
        self,
        num_classes: int = 18,
        empty_label: int = 17,
        base_channels: int = 16,
        mid_channels: int = 24,
        bottleneck_channels: int = 32,
        visibility_channels: int = 5,
        use_camera_mask_channel: bool = True,
        use_checkpoint: bool = True,
        detach_base_input: bool = True,
        dropout: float = 0.0,
        semantic_residual_init: float = 0.05,
        geometry_residual_init: float = 0.05,
        semantic_strong_free_scale: float = 0.10,
        semantic_weak_free_scale: float = 0.50,
        residual_regularization: float = 0.002,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.empty_label = int(empty_label)
        if not (0 <= self.empty_label < self.num_classes):
            raise ValueError(f"empty_label={empty_label} is outside num_classes={num_classes}")
        self.num_nonempty = self.num_classes - 1
        self.visibility_channels = int(visibility_channels)
        self.use_camera_mask_channel = bool(use_camera_mask_channel)
        self.use_checkpoint = bool(use_checkpoint)
        self.detach_base_input = bool(detach_base_input)
        self.semantic_strong_free_scale = float(semantic_strong_free_scale)
        self.semantic_weak_free_scale = float(semantic_weak_free_scale)
        self.residual_regularization = float(residual_regularization)

        # Input: class probabilities + non-empty probability + normalized entropy
        #        + visibility one-hot + optional camera mask.
        in_channels = self.num_classes + 2 + self.visibility_channels
        if self.use_camera_mask_channel:
            in_channels += 1

        c0 = int(base_channels)
        c1 = int(mid_channels)
        c2 = int(bottleneck_channels)

        self.stem = ResidualBlock3D(in_channels, c0, dropout=dropout)
        # Tensor order is [B,C,D,H,W]. Preserve vertical resolution first.
        self.down1 = Downsample3D(c0, c1, stride=(1, 2, 2))
        self.enc1 = ResidualBlock3D(c1, c1, dropout=dropout)
        self.down2 = Downsample3D(c1, c2, stride=(2, 2, 2))
        self.bottleneck = ResidualBlock3D(c2, c2, dropout=dropout)

        self.up2_proj = nn.Conv3d(c2, c1, 1, bias=False)
        self.dec1 = ResidualBlock3D(c1 + c1, c1, dropout=dropout)
        self.up1_proj = nn.Conv3d(c1, c0, 1, bias=False)
        self.dec0 = ResidualBlock3D(c0 + c0, c0, dropout=dropout)

        self.semantic_head = nn.Conv3d(c0, self.num_nonempty, 1)
        self.geometry_head = nn.Conv3d(c0, 1, 1)
        sem_init = max(float(semantic_residual_init), 1.0e-4)
        geo_init = max(float(geometry_residual_init), 1.0e-4)
        self.semantic_scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(sem_init))))
        self.geometry_scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(geo_init))))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Start as a near-identity residual branch, but keep tiny non-zero weights
        # so gradients reach the U-Net trunk from the first update.
        nn.init.normal_(self.semantic_head.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.semantic_head.bias)
        nn.init.normal_(self.geometry_head.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.geometry_head.bias)

    def _run(self, module: nn.Module, *args: Tensor) -> Tensor:
        if not (self.use_checkpoint and self.training and any(x.requires_grad for x in args)):
            return module(*args)
        try:
            return checkpoint(module, *args, use_reentrant=False)
        except TypeError:
            return checkpoint(module, *args)

    def _build_input(
        self,
        base_logits: Tensor,
        visibility_onehot: Tensor,
        camera_mask: Optional[Tensor],
    ) -> Tensor:
        # A detached conditioning path keeps the original Gaussian/Mamba branch
        # stable; the main semantic loss still reaches base_logits through the
        # identity residual connection in final_logits.
        probs = torch.softmax(base_logits, dim=1)
        if self.detach_base_input:
            probs = probs.detach()
        empty_prob = probs[:, self.empty_label : self.empty_label + 1]
        nonempty_prob = 1.0 - empty_prob
        entropy = -(probs.clamp_min(1.0e-6).log() * probs).sum(dim=1, keepdim=True)
        entropy = entropy / max(math.log(float(self.num_classes)), 1.0)

        parts = [probs, nonempty_prob, entropy, visibility_onehot.to(dtype=probs.dtype)]
        if self.use_camera_mask_channel:
            if camera_mask is None:
                camera_mask = torch.ones_like(nonempty_prob)
            parts.append(camera_mask.to(dtype=probs.dtype))
        return torch.cat(parts, dim=1)

    def forward(
        self,
        base_logits: Tensor,
        visibility_onehot: Tensor,
        camera_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if base_logits.ndim != 5:
            raise ValueError(f"base_logits must be [B,C,D,H,W], got {tuple(base_logits.shape)}")
        if visibility_onehot.ndim != 5:
            raise ValueError(
                f"visibility_onehot must be [B,V,D,H,W], got {tuple(visibility_onehot.shape)}"
            )
        if visibility_onehot.shape[1] != self.visibility_channels:
            raise ValueError(
                f"visibility channels mismatch: got {visibility_onehot.shape[1]}, "
                f"expected {self.visibility_channels}"
            )

        x = self._build_input(base_logits, visibility_onehot, camera_mask)
        e0 = self._run(self.stem, x)
        e1 = self._run(self.enc1, self.down1(e0))
        e2 = self._run(self.bottleneck, self.down2(e1))

        u1 = F.interpolate(e2, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        u1 = self.up2_proj(u1)
        u1 = self._run(self.dec1, torch.cat([u1, e1], dim=1))
        u0 = F.interpolate(u1, size=e0.shape[-3:], mode="trilinear", align_corners=False)
        u0 = self.up1_proj(u0)
        u0 = self._run(self.dec0, torch.cat([u0, e0], dim=1))

        semantic_delta_nonempty = self.semantic_head(u0)
        geometry_delta = self.geometry_head(u0)

        # Visibility channels follow [UNKNOWN, WEAK_FREE, STRONG_FREE,
        # SURFACE_HIT, MIXED]. Strong-free geometry can only reduce occupancy.
        weak_free = visibility_onehot[:, 1:2]
        strong_free = visibility_onehot[:, 2:3]
        semantic_visibility_scale = (
            1.0
            - weak_free * (1.0 - self.semantic_weak_free_scale)
            - strong_free * (1.0 - self.semantic_strong_free_scale)
        ).clamp_min(0.0)
        semantic_delta_nonempty = semantic_delta_nonempty * semantic_visibility_scale
        geometry_delta = torch.where(
            strong_free > 0.5,
            -F.softplus(geometry_delta),
            geometry_delta,
        )

        semantic_delta = torch.zeros_like(base_logits)
        nonempty_ids = [i for i in range(self.num_classes) if i != self.empty_label]
        semantic_delta[:, nonempty_ids] = semantic_delta_nonempty

        # +/- half shift means the occupied-vs-empty log-odds changes by exactly
        # geometry_scale * geometry_delta.
        geometry_shift = torch.zeros_like(base_logits)
        geometry_shift[:, nonempty_ids] = 0.5 * geometry_delta
        geometry_shift[:, self.empty_label : self.empty_label + 1] = -0.5 * geometry_delta

        semantic_scale = F.softplus(self.semantic_scale_raw)
        geometry_scale = F.softplus(self.geometry_scale_raw)
        final_logits = (
            base_logits
            + semantic_scale * semantic_delta
            + geometry_scale * geometry_shift
        )

        nonempty_logits = final_logits[:, nonempty_ids]
        occupied_logit = torch.logsumexp(nonempty_logits, dim=1, keepdim=True) - final_logits[
            :, self.empty_label : self.empty_label + 1
        ]
        residual_reg = self.residual_regularization * (
            semantic_delta_nonempty.float().abs().mean() + geometry_delta.float().abs().mean()
        )

        return {
            "final_logits": final_logits,
            "occupied_logit": occupied_logit,
            "semantic_delta": semantic_delta_nonempty,
            "geometry_delta": geometry_delta,
            "residual_regularization": residual_reg,
            "semantic_scale": semantic_scale,
            "geometry_scale": geometry_scale,
        }
