import torch, torch.nn as nn

from mmengine.registry import MODELS
from .base_head import BaseTaskHead
from .localagg.local_aggregate import LocalAggregator
from ..utils.utils import get_rotation_matrix
from .visibility_residual_unet3d import VisibilityResidualUNet3D


@MODELS.register_module()
class GaussianHead(BaseTaskHead):
    def __init__(
        self,
        init_cfg=None,
        apply_loss_type=None,
        auxiliary_voxel_stride=1,
        num_classes=18,
        empty_args=None,
        with_empty=False,
        cuda_kwargs=None,
        dataset_type='nusc',
        empty_label=17,
        debug=False,
        debug_interval=50,
        debug_max_print=40,
        debug_class_ids=(1, 2, 5, 6, 7, 8),
        debug_layer_semantics=True,
        debug_voxel_hist=True,
        debug_gt_voxel_topk=True,
        debug_gt_voxel_topk_k=3,
        debug_max_voxels_per_class=4096,
        v21_d0_enabled=False,
        v21_d0_modes=None,
        use_completion_render_gate=True,
        refine_unet=None,
        **kwargs,
    ):
        super().__init__(init_cfg)

        self.num_classes = num_classes
        self.aggregator = LocalAggregator(**cuda_kwargs)
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float))
            self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :])
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, None, :])
        self.with_emtpy = with_empty
        self.empty_args = empty_args
        self.dataset_type = dataset_type
        self.empty_label = empty_label
        self.debug = bool(debug)
        self.debug_interval = max(int(debug_interval), 1)
        self.debug_max_print = max(int(debug_max_print), 0)
        self.debug_class_ids = tuple(int(x) for x in debug_class_ids)
        self.debug_layer_semantics = bool(debug_layer_semantics)
        self.debug_voxel_hist = bool(debug_voxel_hist)
        self.debug_gt_voxel_topk = bool(debug_gt_voxel_topk)
        self.debug_gt_voxel_topk_k = max(int(debug_gt_voxel_topk_k), 1)
        self.debug_max_voxels_per_class = max(int(debug_max_voxels_per_class), 1)
        self._forward_count = 0
        self._debug_print_count = 0
        self.v21_d0_enabled = bool(v21_d0_enabled)
        self.use_completion_render_gate = bool(use_completion_render_gate)
        self.v21_d0_modes = tuple(v21_d0_modes or (
            'base_no_completion', 'without_object', 'without_local',
            'without_generic', 'without_fallback',
        ))

        refine_cfg = dict(refine_unet or {})
        self.use_refine_unet = bool(refine_cfg.pop('enabled', False))
        self.refine_grid_shape = tuple(int(x) for x in refine_cfg.pop('grid_shape', (200, 200, 16)))
        if len(self.refine_grid_shape) != 3:
            raise ValueError(f'refine_unet.grid_shape must be [H,W,D], got {self.refine_grid_shape}')
        self.refine_geometry_loss_weight = float(refine_cfg.pop('geometry_loss_weight', 1.0))
        self.refine_debug = bool(refine_cfg.pop('debug', False))
        self.refine_debug_interval = max(int(refine_cfg.pop('debug_interval', 100)), 1)
        self.refine_debug_max_print = max(int(refine_cfg.pop('debug_max_print', 12)), 0)
        self._refine_debug_count = 0
        if self.use_refine_unet:
            refine_cfg.setdefault('num_classes', self.num_classes)
            refine_cfg.setdefault('empty_label', self.empty_label)
            self.refine_unet = VisibilityResidualUNet3D(**refine_cfg)
        else:
            self.refine_unet = None

        self.auxiliary_voxel_stride = max(int(auxiliary_voxel_stride), 1)
        if apply_loss_type == 'all':
            self.apply_loss_type = 'all'
            self.last_apply_loss_layers = 0
        elif isinstance(apply_loss_type, str) and apply_loss_type.startswith('last_'):
            self.apply_loss_type = 'last'
            self.last_apply_loss_layers = max(int(apply_loss_type.split('_')[1]), 1)
        elif isinstance(apply_loss_type, str) and apply_loss_type.startswith('random_'):
            # Legacy compatibility only. Selection is now performed with torch on
            # the current device rather than NumPy/CPU.
            self.apply_loss_type = 'random'
            self.random_apply_loss_layers = max(int(apply_loss_type.split('_')[1]), 1)
            self.last_apply_loss_layers = 0
        else:
            raise NotImplementedError(f'Unknown apply_loss_type={apply_loss_type!r}')
        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _sampling(self, gt_xyz, gt_label, gt_mask=None):
        if gt_mask is None:
            gt_label = gt_label.flatten(1)
            gt_xyz = gt_xyz.flatten(1, 3)
        else:
            assert gt_label.shape[0] == 1, "OccLoss does not support masked bs > 1"
            gt_label = gt_label[gt_mask].reshape(1, -1)
            gt_xyz = gt_xyz[gt_mask].reshape(1, -1, 3)
        return gt_xyz, gt_label

    @staticmethod
    def _expand_batch_buffer(buffer, batch_size, dtype, device):
        return buffer.to(device=device, dtype=dtype).expand(batch_size, -1, -1).contiguous()

    @staticmethod
    def _is_rank0() -> bool:
        try:
            import torch.distributed as dist
            return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        except Exception:
            return True

    def _hist_selected(self, values: torch.Tensor) -> dict:
        values = values.detach().reshape(-1).long()
        return {int(c): int((values == int(c)).sum().item()) for c in self.debug_class_ids}

    def _maybe_debug_layer_semantics(self, representation, apply_loss_layers):
        if not (self.debug and self.debug_layer_semantics and self._is_rank0()):
            return
        if self._debug_print_count >= self.debug_max_print or self._forward_count % self.debug_interval != 0:
            return
        reports = []
        with torch.no_grad():
            for i, rep in enumerate(representation):
                if not isinstance(rep, dict) or 'gaussian' not in rep:
                    continue
                gsem = getattr(rep['gaussian'], 'semantics', None)
                if gsem is None or gsem.numel() == 0:
                    continue
                pred = gsem.argmax(dim=-1)
                layer_tag = f"L{i}{'*' if i in apply_loss_layers else ''}"
                reports.append(f"{layer_tag}:{self._hist_selected(pred)}")
        if reports:
            print(
                "[GaussianHead Gaussian SEM] "
                f"call={self._forward_count}, classes={list(self.debug_class_ids)}, "
                + "; ".join(reports),
                flush=True,
            )

    def _mask_like_labels(self, sampled_label, occ_mask=None):
        if occ_mask is None:
            return torch.ones_like(sampled_label, dtype=torch.bool)
        mask = occ_mask.to(device=sampled_label.device, dtype=torch.bool)
        if mask.ndim > 2:
            mask = mask.flatten(1)
        if mask.shape == sampled_label.shape:
            return mask
        if mask.numel() == sampled_label.numel():
            return mask.reshape_as(sampled_label)
        return torch.ones_like(sampled_label, dtype=torch.bool)

    def _maybe_debug_voxel_logits(self, semantics, sampled_label, layer_idx, occ_mask=None):
        if not (self.debug and self._is_rank0()):
            return
        if self._debug_print_count >= self.debug_max_print or self._forward_count % self.debug_interval != 0:
            return
        with torch.no_grad():
            pred = semantics.argmax(dim=1)
            valid_mask = self._mask_like_labels(sampled_label, occ_mask)
            if self.debug_voxel_hist:
                pred_hist = self._hist_selected(pred[valid_mask])
                gt_hist = self._hist_selected(sampled_label[valid_mask])
                logit_stats = {}
                for c in self.debug_class_ids:
                    if 0 <= int(c) < semantics.shape[1]:
                        v = semantics[:, int(c)].detach()
                        logit_stats[int(c)] = (float(v.max().cpu()), float(v.mean().cpu()))
                print(
                    "[GaussianHead Voxel SEM] "
                    f"call={self._forward_count}, layer={layer_idx}, classes={list(self.debug_class_ids)}, "
                    f"pred_hist={pred_hist}, gt_hist={gt_hist}, logit_max_mean={logit_stats}",
                    flush=True,
                )

            if self.debug_gt_voxel_topk:
                k = min(int(self.debug_gt_voxel_topk_k), int(semantics.shape[1]))
                reports = []
                # semantics: [B, C, N], sampled_label: [B, N]
                sem_bn_c = semantics.detach().transpose(1, 2)  # [B, N, C]
                for c in self.debug_class_ids:
                    c = int(c)
                    if c < 0 or c >= semantics.shape[1]:
                        continue
                    cmask = (sampled_label.long() == c) & valid_mask
                    n = int(cmask.sum().item())
                    if n <= 0:
                        reports.append(f"c{c}:n=0")
                        continue
                    logits_c = sem_bn_c[cmask]
                    if logits_c.shape[0] > self.debug_max_voxels_per_class:
                        # Deterministic sub-sampling keeps debug bounded and reproducible.
                        take = torch.linspace(
                            0,
                            logits_c.shape[0] - 1,
                            steps=self.debug_max_voxels_per_class,
                            device=logits_c.device,
                        ).round().long()
                        logits_c = logits_c[take]
                    topv, topi = torch.topk(logits_c, k=k, dim=-1)
                    top1 = topi[:, 0].long()
                    top1_hist = self._hist_selected(top1)
                    target_logit = logits_c[:, c]
                    # Rank is 1 for best class. Lower is better.
                    rank = (logits_c > target_logit[:, None]).sum(dim=-1).float() + 1.0
                    in_topk = (topi == c).any(dim=-1).float().mean().item()
                    reports.append(
                        f"c{c}:n={n},used={int(logits_c.shape[0])},"
                        f"rank_mean={float(rank.mean().cpu()):.2f},rank_min={float(rank.min().cpu()):.0f},"
                        f"in_top{k}={in_topk:.3f},"
                        f"target=({float(target_logit.max().cpu()):.3f},{float(target_logit.mean().cpu()):.3f}),"
                        f"top1={top1_hist}"
                    )
                if reports:
                    print(
                        "[GaussianHead GT-VOXEL TOPK] "
                        f"call={self._forward_count}, layer={layer_idx}, classes={list(self.debug_class_ids)}, "
                        + " | ".join(reports),
                        flush=True,
                    )
            self._debug_print_count += 1

    def prepare_gaussian_args(self, gaussians):
        means = gaussians.means             # B, G, 3
        scales = gaussians.scales           # B, G, 3
        rotations = gaussians.rotations     # B, G, 4
        opacities = gaussians.semantics     # B, G, C-1 before empty class
        origi_opa = gaussians.opacities     # B, G, 1

        bs = int(means.shape[0])
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)

        if self.with_emtpy:
            assert opacities.shape[-1] == self.num_classes - 1
            if 'kitti' in self.dataset_type:
                opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1)
            else:
                opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)

            empty_mean = self._expand_batch_buffer(self.empty_mean, bs, means.dtype, means.device)
            empty_scale = self._expand_batch_buffer(self.empty_scale, bs, scales.dtype, scales.device)
            empty_rot = self._expand_batch_buffer(self.empty_rot, bs, rotations.dtype, rotations.device)
            empty_opa = self._expand_batch_buffer(self.empty_opa, bs, origi_opa.dtype, origi_opa.device)
            empty_sem = self.empty_sem.to(device=opacities.device, dtype=opacities.dtype).expand(bs, -1, -1).clone()
            empty_sem[..., self.empty_label] += self.empty_scalar.to(device=opacities.device, dtype=opacities.dtype).view(1, 1)

            means = torch.cat([means, empty_mean], dim=1)
            scales = torch.cat([scales, empty_scale], dim=1)
            rotations = torch.cat([rotations, empty_rot], dim=1)
            opacities = torch.cat([opacities, empty_sem], dim=1)
            origi_opa = torch.cat([origi_opa, empty_opa], dim=1)

        bs, g, _ = means.shape
        R = get_rotation_matrix(rotations)  # B, G, 3, 3

        # Analytic inverse for Cov = R^T diag(scale^2) R.
        # This avoids materializing Cov and calling torch.linalg.inv on B*G 3x3 matrices,
        # which saves memory/time and is more stable under AMP/DDP.
        scales_safe = scales.float().clamp_min(1.0e-4)
        inv_scale2 = 1.0 / (scales_safe * scales_safe)
        R_float = R.float()
        # Dinv @ R without materializing Dinv.  This saves one [B, G, 3, 3]
        # tensor compared with torch.diag_embed/zero-fill construction.
        Dinv_R = inv_scale2[..., :, None] * R_float
        CovInv = torch.matmul(R_float.transpose(-1, -2), Dinv_R).to(dtype=means.dtype)
        return means, origi_opa, opacities, scales, CovInv


    def _aggregate_semantics(self, gaussians, sampled_xyz, opacity_mask=None):
        """Render Gaussians with an optional persistent per-Gaussian opacity gate."""
        if opacity_mask is not None:
            mask = opacity_mask.to(gaussians.opacities.device, gaussians.opacities.dtype)
            if mask.ndim == 2:
                mask = mask[..., None]
            gaussians = gaussians._replace(opacities=gaussians.opacities * mask)
        means, origi_opa, opacities, scales, CovInv = self.prepare_gaussian_args(gaussians)
        bs, g = means.shape[:2]
        logits = self.aggregator(sampled_xyz.clone().float(), means,
                                 origi_opa.reshape(bs, g), opacities, scales, CovInv)
        if logits.dim() == 2:
            return logits[None].transpose(1, 2)
        if logits.dim() == 3:
            return logits.transpose(1, 2)
        raise RuntimeError(f'Unexpected LocalAggregator output shape: {tuple(logits.shape)}')

    @staticmethod
    def _v21_d0_keep_mask(branch, mode):
        if mode == 'base_no_completion': return (branch == 0) | (branch == 5)
        if mode == 'measured_only': return branch == 0
        if mode == 'object_only': return branch == 1
        if mode == 'local_only': return branch == 2
        if mode == 'generic_only': return branch == 3
        if mode == 'fallback_only': return branch == 4
        if mode == 'without_object': return branch != 1
        if mode == 'without_local': return branch != 2
        if mode == 'without_generic': return branch != 3
        if mode == 'without_fallback': return branch != 4
        raise ValueError(f'Unknown V21-D0 mode: {mode}')

    def _sample_visibility_onehot(
        self,
        sampled_xyz,
        state_map=None,
        valid_batch=None,
        grid_min=None,
        voxel_size=None,
    ):
        """Nearest-sample coarse visibility states at Occ3D voxel centers."""
        B, N, _ = sampled_xyz.shape
        device = sampled_xyz.device
        out = torch.zeros((B, 5, N), device=device, dtype=torch.float32)
        # Missing visibility is treated as UNKNOWN, not free.
        out[:, 0] = 1.0
        if not isinstance(state_map, torch.Tensor) or state_map.ndim != 4:
            return out
        state_map = state_map.to(device=device, dtype=torch.uint8, non_blocking=True)
        if int(state_map.shape[0]) != B:
            return out
        if grid_min is None or voxel_size is None:
            return out
        gmin = torch.as_tensor(grid_min, device=device, dtype=torch.float32).reshape(-1)[:3]
        if int(gmin.numel()) != 3:
            return out
        vs = torch.as_tensor(voxel_size, device=device, dtype=torch.float32).reshape(-1)
        if int(vs.numel()) == 1:
            vs = vs.expand(3)
        else:
            vs = vs[:3]
        vs = vs.clamp_min(1.0e-6)

        idx = torch.floor((sampled_xyz.float() - gmin.view(1, 1, 3)) / vs.view(1, 1, 3)).long()
        sx, sy, sz = (int(state_map.shape[1]), int(state_map.shape[2]), int(state_map.shape[3]))
        inside = (
            (idx[..., 0] >= 0) & (idx[..., 0] < sx)
            & (idx[..., 1] >= 0) & (idx[..., 1] < sy)
            & (idx[..., 2] >= 0) & (idx[..., 2] < sz)
        )
        if isinstance(valid_batch, torch.Tensor):
            vb = valid_batch.to(device=device, dtype=torch.bool).reshape(B, -1)[:, 0]
            inside = inside & vb[:, None]
        lin = ((idx[..., 0].clamp(0, sx - 1) * sy + idx[..., 1].clamp(0, sy - 1)) * sz
               + idx[..., 2].clamp(0, sz - 1))
        state = torch.gather(state_map.reshape(B, -1), 1, lin)
        state = torch.where(inside, state, torch.zeros_like(state))
        out = torch.stack([(state == i).float() for i in range(5)], dim=1)
        return out

    def _refine_dense_logits(
        self,
        semantics,
        sampled_xyz,
        sampled_label,
        occ_cam_mask,
        **kwargs,
    ):
        if not self.use_refine_unet or self.refine_unet is None:
            zero = semantics.sum() * 0.0
            return semantics, zero, None

        B, C, N = semantics.shape
        H, W, D = self.refine_grid_shape
        expected = int(H * W * D)
        if int(N) != expected:
            raise RuntimeError(
                f'[C2ResidualGeoUNet] rendered voxel count {N} does not match '
                f'grid_shape={self.refine_grid_shape} ({expected})'
            )
        dense = semantics.reshape(B, C, H, W, D).permute(0, 1, 4, 2, 3).contiguous()
        vis = self._sample_visibility_onehot(
            sampled_xyz,
            state_map=kwargs.get('gaussian_visibility_state_map', None),
            valid_batch=kwargs.get('gaussian_visibility_valid_batch', None),
            grid_min=kwargs.get('gaussian_visibility_grid_min', None),
            voxel_size=kwargs.get('gaussian_visibility_voxel_size', None),
        )
        vis_dense = vis.reshape(B, 5, H, W, D).permute(0, 1, 4, 2, 3).contiguous()

        cam_dense = None
        if isinstance(occ_cam_mask, torch.Tensor):
            cam = occ_cam_mask.to(device=dense.device).reshape(B, H, W, D)
            cam_dense = cam.permute(0, 3, 1, 2).unsqueeze(1).to(dtype=dense.dtype)

        refine = self.refine_unet(dense, vis_dense, cam_dense)
        final_dense = refine['final_logits']
        final = final_dense.permute(0, 1, 3, 4, 2).reshape(B, C, N).contiguous()

        labels = sampled_label.reshape(B, H, W, D).permute(0, 3, 1, 2)
        valid = labels != 255
        if isinstance(occ_cam_mask, torch.Tensor):
            valid = valid & occ_cam_mask.to(device=labels.device, dtype=torch.bool).reshape(B, H, W, D).permute(0, 3, 1, 2)
        target = (labels != int(self.empty_label)).to(dtype=refine['occupied_logit'].dtype)
        occ_logit = refine['occupied_logit'].squeeze(1)
        if bool(valid.any()):
            geometry_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                occ_logit[valid], target[valid], reduction='mean'
            )
        else:
            geometry_loss = occ_logit.sum() * 0.0
        geometry_loss = self.refine_geometry_loss_weight * geometry_loss + refine['residual_regularization']

        if (
            self.refine_debug and self._is_rank0()
            and self._refine_debug_count < self.refine_debug_max_print
            and self._forward_count % self.refine_debug_interval == 0
        ):
            with torch.no_grad():
                pred_occ = (occ_logit > 0) & valid
                gt_occ = (target > 0.5) & valid
                inter = (pred_occ & gt_occ).sum().float()
                precision = inter / pred_occ.sum().clamp_min(1).float()
                recall = inter / gt_occ.sum().clamp_min(1).float()
                strong = vis_dense[:, 2:3] > 0.5
                strong_add = ((refine['geometry_delta'] > 0) & strong).sum().item()
                print(
                    '[C2ResidualGeoUNet] '
                    f'call={self._forward_count}, shape={tuple(final_dense.shape)}, '
                    f'scaleSem={float(refine["semantic_scale"].detach().cpu()):.4f}, '
                    f'scaleGeo={float(refine["geometry_scale"].detach().cpu()):.4f}, '
                    f'geoLoss={float(geometry_loss.detach().cpu()):.4f}, '
                    f'occP/R={float(precision.cpu()):.3f}/{float(recall.cpu()):.3f}, '
                    f'strongPositiveDelta={int(strong_add)}',
                    flush=True,
                )
            self._refine_debug_count += 1
        return final, geometry_loss, refine

    def forward(
        self,
        representation,
        metas=None,
        **kwargs,
    ):
        self._forward_count += 1
        num_decoder = len(representation)
        if not self.training:
            apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == "all":
            apply_loss_layers = list(range(num_decoder))
        elif self.apply_loss_type == "last":
            first = max(num_decoder - self.last_apply_loss_layers, 0)
            apply_loss_layers = list(range(first, num_decoder))
        elif self.apply_loss_type == "random":
            if self.random_apply_loss_layers > 1 and num_decoder > 1:
                k = min(self.random_apply_loss_layers - 1, num_decoder - 1)
                chosen = torch.randperm(num_decoder - 1, device=self.zero_tensor.device)[:k]
                apply_loss_layers = torch.sort(chosen).values.tolist() + [num_decoder - 1]
            else:
                apply_loss_layers = [num_decoder - 1]
        else:
            raise NotImplementedError

        self._maybe_debug_layer_semantics(representation, apply_loss_layers)

        prediction = []
        auxiliary_prediction = []
        v21_d0_predictions = {}
        loss_unet_geometry = self.zero_tensor.sum() * 0.0
        render_gate = kwargs.get('gaussian_completion_render_gate', None)
        if not self.use_completion_render_gate:
            render_gate = None
        elif isinstance(render_gate, torch.Tensor):
            # Encoder refinement preserves Gaussian count/order.  Keep the gate
            # detached and on-device; it is a physical visibility constraint,
            # not a learnable shortcut.
            render_gate = render_gate.detach()
        else:
            render_gate = None
        occ_xyz = metas['occ_xyz'].to(self.zero_tensor.device)
        occ_label = metas['occ_label'].to(self.zero_tensor.device)
        occ_cam_mask = metas['occ_cam_mask'].to(self.zero_tensor.device)
        sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)
        final_layer_idx = num_decoder - 1

        # The auxiliary decoder uses a deterministic 3D checkerboard subset.
        # A flat ``arange(..., stride)`` would repeatedly choose only a few height
        # slices because D is the fastest dimension in [H,W,D]. The checkerboard
        # distributes supervision across X/Y/Z while retaining exactly about 1/S
        # of the voxels. All index construction stays on the current GPU.
        flat_occ_mask = occ_cam_mask.flatten(1)
        if self.training and len(apply_loss_layers) > 1 and self.auxiliary_voxel_stride > 1:
            H, W, D = (int(x) for x in occ_cam_mask.shape[-3:])
            h = torch.arange(H, device=sampled_xyz.device, dtype=torch.long).view(H, 1, 1)
            w = torch.arange(W, device=sampled_xyz.device, dtype=torch.long).view(1, W, 1)
            d = torch.arange(D, device=sampled_xyz.device, dtype=torch.long).view(1, 1, D)
            aux_keep = ((h + w + d) % self.auxiliary_voxel_stride == 0).reshape(-1)
            aux_indices = torch.nonzero(aux_keep, as_tuple=False).squeeze(1)
            sampled_xyz_aux = sampled_xyz.index_select(1, aux_indices)
            sampled_label_aux = sampled_label.index_select(1, aux_indices)
            occ_mask_aux = flat_occ_mask.index_select(1, aux_indices)
        else:
            sampled_xyz_aux = sampled_xyz
            sampled_label_aux = sampled_label
            occ_mask_aux = flat_occ_mask

        for idx in apply_loss_layers:
            gaussians = representation[idx]['gaussian']
            layer_gate = render_gate
            if isinstance(layer_gate, torch.Tensor) and layer_gate.shape[:2] != gaussians.means.shape[:2]:
                layer_gate = None
            is_final = idx == final_layer_idx
            layer_xyz = sampled_xyz if is_final else sampled_xyz_aux
            semantics = self._aggregate_semantics(gaussians, layer_xyz, layer_gate)
            if is_final and self.use_refine_unet:
                semantics, loss_unet_geometry, _ = self._refine_dense_logits(
                    semantics, sampled_xyz, sampled_label, occ_cam_mask, **kwargs
                )
            if is_final:
                self._maybe_debug_voxel_logits(semantics, sampled_label, idx, occ_cam_mask)
                if self.v21_d0_enabled and not self.training:
                    branch = kwargs.get('gaussian_diagnostic_branch_types', None)
                    if isinstance(branch, torch.Tensor) and branch.shape[:2] == gaussians.means.shape[:2]:
                        v21_d0_predictions['full'] = semantics.argmax(dim=1).to(torch.uint8)
                        for mode in self.v21_d0_modes:
                            keep = self._v21_d0_keep_mask(branch, mode)
                            if isinstance(layer_gate, torch.Tensor):
                                keep = keep.to(layer_gate.dtype) * layer_gate
                            sem_mode = self._aggregate_semantics(gaussians, sampled_xyz, keep)
                            v21_d0_predictions[str(mode)] = sem_mode.argmax(dim=1).to(torch.uint8)
                prediction.append(semantics)
            else:
                auxiliary_prediction.append(semantics)

        return {
            'pred_occ': prediction,
            'sampled_label': sampled_label,
            'sampled_xyz': sampled_xyz,
            'occ_mask': occ_cam_mask,
            'pred_occ_aux': auxiliary_prediction,
            'sampled_label_aux': sampled_label_aux,
            'sampled_xyz_aux': sampled_xyz_aux,
            'occ_mask_aux': occ_mask_aux,
            'gaussian': representation[-1]['gaussian'],
            'v21_d0_pred_occ': v21_d0_predictions,
            'loss_unet_geometry': loss_unet_geometry,
        }
