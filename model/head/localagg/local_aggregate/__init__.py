#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import os
import torch
import torch.nn as nn
from . import _C


class _LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pts,
        points_int,
        means3D,
        means3D_int,
        opacities,
        semantics,
        radii,
        cov3D,
        H, W, D,
    ):
        """Batch-aware CUDA local aggregation.

        Expected shapes:
            pts:         [B, N, 3]
            points_int:  [B, N, 3]
            means3D:     [B, G, 3]
            means3D_int: [B, G, 3]
            opacities:   [B, G]
            semantics:   [B, G, C]
            radii:       [B, G]
            cov3D:       [B, G, 6]

        The CUDA extension keeps each sample in its own voxel namespace by
        using key = batch_id * H * W * D + x * W * D + y * D + z.
        It does NOT move tensors to CPU and does NOT enlarge H/W/D.
        """
        args = (
            pts.contiguous(),
            points_int.contiguous(),
            means3D.contiguous(),
            means3D_int.contiguous(),
            opacities.contiguous(),
            semantics.contiguous(),
            radii.contiguous(),
            cov3D.contiguous(),
            H, W, D,
        )
        num_rendered, logits, geomBuffer, binningBuffer, imgBuffer = _C.local_aggregate(*args)

        ctx.num_rendered = num_rendered
        ctx.H = H
        ctx.W = W
        ctx.D = D
        ctx.save_for_backward(
            geomBuffer,
            binningBuffer,
            imgBuffer,
            means3D,
            pts,
            points_int,
            cov3D,
            opacities,
            semantics,
        )
        return logits

    @staticmethod
    def backward(ctx, out_grad):
        H = ctx.H
        W = ctx.W
        D = ctx.D
        num_rendered = ctx.num_rendered
        geomBuffer, binningBuffer, imgBuffer, means3D, pts, points_int, cov3D, opacities, semantics = ctx.saved_tensors

        args = (
            geomBuffer,
            binningBuffer,
            imgBuffer,
            H, W, D,
            num_rendered,
            means3D,
            pts,
            points_int,
            cov3D,
            opacities,
            semantics,
            out_grad.contiguous(),
        )
        means3D_grad, opacity_grad, semantics_grad, cov3D_grad = _C.local_aggregate_backward(*args)

        return (
            None,
            None,
            means3D_grad,
            None,
            opacity_grad,
            semantics_grad,
            None,
            cov3D_grad,
            None, None, None,
        )


class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, inv_softmax=False):
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H = int(H)
        self.W = int(W)
        self.D = int(D)
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float32).view(1, 1, 3))
        self.grid_size = float(grid_size)
        self.inv_softmax = inv_softmax
        self._debug_printed = 0

    def forward(
        self,
        pts,
        means3D,
        opacities,
        semantics,
        scales,
        cov3D,
    ):
        if self.inv_softmax:
            raise NotImplementedError('inv_softmax=True is not implemented for LocalAggregator.')
        if not pts.is_cuda:
            raise RuntimeError('LocalAggregator expects pts on CUDA. Do not run localagg on CPU.')
        if not means3D.is_cuda:
            raise RuntimeError('LocalAggregator expects means3D on CUDA. Do not run localagg on CPU.')

        input_was_unbatched = False
        if pts.dim() == 2:
            pts = pts.unsqueeze(0)
            input_was_unbatched = True
        if means3D.dim() == 2:
            means3D = means3D.unsqueeze(0)
        if opacities.dim() == 1:
            opacities = opacities.unsqueeze(0)
        if semantics.dim() == 2:
            semantics = semantics.unsqueeze(0)
        if scales.dim() == 2:
            scales = scales.unsqueeze(0)
        if cov3D.dim() == 3:
            cov3D = cov3D.unsqueeze(0)

        assert pts.dim() == 3 and pts.shape[-1] == 3, f'pts must be [B,N,3], got {tuple(pts.shape)}'
        assert means3D.dim() == 3 and means3D.shape[-1] == 3, f'means3D must be [B,G,3], got {tuple(means3D.shape)}'
        assert opacities.dim() == 2, f'opacities must be [B,G], got {tuple(opacities.shape)}'
        assert semantics.dim() == 3, f'semantics must be [B,G,C], got {tuple(semantics.shape)}'
        assert scales.dim() == 3 and scales.shape[-1] == 3, f'scales must be [B,G,3], got {tuple(scales.shape)}'
        assert cov3D.dim() == 4 and cov3D.shape[-2:] == (3, 3), f'cov3D must be [B,G,3,3], got {tuple(cov3D.shape)}'
        assert not pts.requires_grad

        bs, n = pts.shape[:2]
        bs_g, g = means3D.shape[:2]
        if bs_g != bs:
            raise RuntimeError(f'Batch mismatch: pts batch={bs}, means batch={bs_g}')

        pc_min = self.pc_min.to(device=pts.device, dtype=pts.dtype)
        points_int = ((pts - pc_min) / self.grid_size).to(torch.int32)
        means3D_int = ((means3D.detach() - pc_min) / self.grid_size).to(torch.int32)

        # Do not run per-iteration min/max asserts here.  Those reductions synchronize
        # CUDA with CPU and are a common reason for CPU spikes.  Enable only for
        # one-off debugging.
        if os.environ.get('LOCALAGG_STRICT_CHECK', '0').lower() in {'1', 'true', 'yes', 'on'}:
            if not (points_int[..., 0].min() >= 0 and points_int[..., 0].max() < self.H and
                    points_int[..., 1].min() >= 0 and points_int[..., 1].max() < self.W and
                    points_int[..., 2].min() >= 0 and points_int[..., 2].max() < self.D):
                raise RuntimeError(
                    f'pts voxel indices out of range: min={points_int.amin(dim=(0, 1)).tolist()}, '
                    f'max={points_int.amax(dim=(0, 1)).tolist()}, grid=({self.H},{self.W},{self.D})'
                )
            if not (means3D_int[..., 0].min() >= 0 and means3D_int[..., 0].max() < self.H and
                    means3D_int[..., 1].min() >= 0 and means3D_int[..., 1].max() < self.W and
                    means3D_int[..., 2].min() >= 0 and means3D_int[..., 2].max() < self.D):
                raise RuntimeError(
                    f'means3D voxel indices out of range: min={means3D_int.amin(dim=(0, 1)).tolist()}, '
                    f'max={means3D_int.amax(dim=(0, 1)).tolist()}, grid=({self.H},{self.W},{self.D})'
                )

        radii = torch.ceil(scales.detach().max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int32)
        radii = torch.clamp(radii, min=1)
        cov3D_compact = cov3D.flatten(-2)[:, :, [0, 4, 8, 1, 5, 2]].contiguous()

        if os.environ.get('LOCALAGG_DEBUG', '0').lower() in {'1', 'true', 'yes', 'on'} and self._debug_printed < 5:
            self._debug_printed += 1
            print(
                f'[LocalAggregator BatchCUDA Debug] bs={bs}, pts={tuple(pts.shape)}, '
                f'means={tuple(means3D.shape)}, semantics={tuple(semantics.shape)}, '
                f'grid=({self.H},{self.W},{self.D}), device={pts.device}',
                flush=True,
            )

        logits = _LocalAggregate.apply(
            pts.contiguous(),
            points_int.contiguous(),
            means3D.contiguous(),
            means3D_int.contiguous(),
            opacities.contiguous(),
            semantics.contiguous(),
            radii.contiguous(),
            cov3D_compact,
            self.H, self.W, self.D,
        )

        # Keep backward compatibility for any external direct call with unbatched [N,3].
        if input_was_unbatched:
            return logits.squeeze(0)
        return logits
