
from mmseg.models import SEGMENTORS
from mmseg.models import build_backbone
from mmseg.models import build_head

from model.utils.grid_mask import GridMask, GridMaskHybrid

from .base_segmentor import CustomBaseSegmentor
from mmcv.ops.voxelize import Voxelization
from mmdet3d.registry import MODELS
import torch
from torch.nn import functional as F
from model.detector.yolo26_online import OnlineFrozenYOLO26
from model.detector.yolo_2d_to_3d_lifter import YOLO2DTo3DHypothesisLifter
from model.detector.object_center_query_lifter import ObjectCenterQueryLifter
from model.detector.frustum_geometry_lifter import FrustumGeometryLifter
from model.detector.object_gaussian_seed_completion import ObjectGaussianSeedCompletionHead
from model.detector.box_gaussian_completion import BoxGaussianCompletionHead
from model.visualizer.gaussian_completion_visualizer import GaussianCompletionVisualizer

@SEGMENTORS.register_module()
class BEVSegmentorLiDAR3D(CustomBaseSegmentor):

    def __init__(
        self,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        voxelize_lidar=None,
        lidar_voxel_encoder=None,
        use_grid_mask=False,
        d_bound=[2.0, 58, 0.5],
        pts_dpt_head=None,
        online_yolo=None,
        offline_yolo2d=None,
        gt_projected_2d=None,
        yolo_2d_to_3d_lifter=None,
        object_center_query_lifter=None,
        frustum_geometry_lifter=None,
        object_gaussian_seed_completion=None,
        box_hypothesis_mamba=None,
        box_candidate_ranker=None,
        box_gaussian_completion=None,
        gt_box_gaussian_completion=None,
        gaussian_completion_visualizer=None,
        skip_img_depth_branch=False,
        use_img_branch=None,
        use_depth_branch=None,
        dpt_debug=False,
        dpt_debug_interval=200,
        dpt_debug_max_print=3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # self.fp16_enabled = False
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices
        self.use_grid_mask = use_grid_mask
        self.d_bound = d_bound
        self.dpt_debug = bool(dpt_debug)
        self.dpt_debug_interval = max(int(dpt_debug_interval), 1)
        self.dpt_debug_max_print = max(int(dpt_debug_max_print), 0)
        self._dpt_debug_print_count = 0
        # 中文注释：兼容旧配置 skip_img_depth_branch，同时把 image 与 depth 分支拆开。
        # GAF-Mamba 第一版需要 image feature，但不需要 depth_gt / depth head。
        self.skip_img_depth_branch = bool(skip_img_depth_branch)
        if use_img_branch is None:
            use_img_branch = not self.skip_img_depth_branch
        if use_depth_branch is None:
            use_depth_branch = (not self.skip_img_depth_branch) and (pts_dpt_head is not None)
        self.use_img_branch = bool(use_img_branch)
        self.use_depth_branch = bool(use_depth_branch) and (pts_dpt_head is not None)
        self.grid_mask = GridMaskHybrid(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if extra_img_backbone is not None:
            self.extra_img_backbone = build_backbone(extra_img_backbone)
        if pts_dpt_head is not None:
            self.pts_dpt_head = build_head(pts_dpt_head)
        self.use_online_yolo = bool(online_yolo is not None and online_yolo.get("enabled", True))
        if self.use_online_yolo:
            yolo_cfg = dict(online_yolo)
            yolo_cfg.pop("enabled", None)
            self.online_yolo = OnlineFrozenYOLO26(**yolo_cfg)

        offline_yolo_cfg = dict(offline_yolo2d or {})
        self.use_offline_yolo2d = bool(offline_yolo_cfg.get("enabled", False))
        self.offline_yolo2d_debug = bool(offline_yolo_cfg.get("debug", False))
        self.offline_yolo2d_debug_interval = max(int(offline_yolo_cfg.get("debug_interval", 50)), 1)
        self.offline_yolo2d_debug_max_print = max(int(offline_yolo_cfg.get("debug_max_print", 10)), 0)
        # YOLO cache can contain many false positives.  Capping before OQG keeps
        # the true YOLO2D -> OQG -> boxhyp route practical without changing the
        # cache file itself.  Negative/zero values disable the cap.
        self.offline_yolo2d_max_boxes_per_sample = int(offline_yolo_cfg.get("max_boxes_per_sample", -1))
        self.offline_yolo2d_max_boxes_per_cam = int(offline_yolo_cfg.get("max_boxes_per_cam", -1))
        # v8.7.4: when using real YOLO2D cache, project GT 3D boxes into each
        # camera and match YOLO boxes to the projected GT boxes.  The matched GT
        # index is passed to OQG as yolo_gt_indices_2d so loss_oqg is no longer
        # zero in the YOLO-cache route.  This is training/eval supervision only;
        # inference can run without GT and the matcher will simply return no loss.
        self.offline_yolo2d_gt_supervision = bool(offline_yolo_cfg.get("gt_supervision", False))
        self.offline_yolo2d_gt_match_min_iou = float(offline_yolo_cfg.get("gt_match_min_iou", 0.05))
        self.offline_yolo2d_gt_match_class_aware = bool(offline_yolo_cfg.get("gt_match_class_aware", True))
        self.offline_yolo2d_gt_match_min_area = float(offline_yolo_cfg.get("gt_match_min_area", 4.0))
        self.offline_yolo2d_gt_match_clip = bool(offline_yolo_cfg.get("gt_match_clip", True))
        self.offline_yolo2d_gt_match_debug = bool(offline_yolo_cfg.get("gt_match_debug", True))
        self.offline_yolo2d_gt_match_debug_interval = max(int(offline_yolo_cfg.get("gt_match_debug_interval", 200)), 1)
        self.offline_yolo2d_gt_match_debug_max_print = max(int(offline_yolo_cfg.get("gt_match_debug_max_print", 20)), 0)
        self._offline_yolo2d_gt_match_print_count = 0
        self._offline_yolo2d_call_count = 0
        self._offline_yolo2d_debug_print_count = 0

        # Step-0 diagnostic proposal source: project GT 3D boxes to perfect 2D
        # boxes, then feed them into the same 2D->3D lifter used by YOLO.
        # This isolates the quality of the 2D-to-3D lifting strategy from YOLO
        # detection noise.  It is a diagnostic path only; normal GT-box oracle
        # completion still uses gt_bboxes_3d directly unless box completion
        # source is changed to 'boxhyp'.
        gt_proj_cfg = dict(gt_projected_2d or {})
        self.use_gt_projected_2d = bool(gt_proj_cfg.get("enabled", False))
        self.gt_projected_2d_max_det = int(gt_proj_cfg.get("max_det", 80))
        self.gt_projected_2d_score = float(gt_proj_cfg.get("score", 1.0))
        self.gt_projected_2d_min_depth = float(gt_proj_cfg.get("min_depth", 1.0e-3))
        self.gt_projected_2d_min_visible_corners = int(gt_proj_cfg.get("min_visible_corners", 1))
        self.gt_projected_2d_min_box_area = float(gt_proj_cfg.get("min_box_area", 4.0))
        self.gt_projected_2d_clip = bool(gt_proj_cfg.get("clip", True))
        self.gt_projected_2d_debug = bool(gt_proj_cfg.get("debug", False))
        self.gt_projected_2d_debug_interval = max(int(gt_proj_cfg.get("debug_interval", 50)), 1)
        self.gt_projected_2d_debug_max_print = max(int(gt_proj_cfg.get("debug_max_print", 10)), 0)
        self._gt_projected_2d_call_count = 0
        self._gt_projected_2d_debug_print_count = 0

        # Step-1a legacy rule-based image proposal lifting: online YOLO or
        # diagnostic perfect-2D boxes -> multiple coarse 3D box hypotheses.
        # This is kept as a fallback/baseline, but the preferred path is the
        # OQG-lite ObjectCenterQueryLifter below.
        self.use_yolo_2d_to_3d_lifter = bool(
            (self.use_online_yolo or self.use_offline_yolo2d or self.use_gt_projected_2d)
            and yolo_2d_to_3d_lifter is not None
            and yolo_2d_to_3d_lifter.get("enabled", True)
        )
        if self.use_yolo_2d_to_3d_lifter:
            lifter_cfg = dict(yolo_2d_to_3d_lifter)
            lifter_cfg.pop("enabled", None)
            self.yolo_2d_to_3d_lifter = YOLO2DTo3DHypothesisLifter(**lifter_cfg)

        # Step-1b OQG-lite lifter: 2D object boxes + sparse image/depth features
        # -> object-centered 3D query candidates.  It runs after image/depth
        # features are extracted, but it consumes the same yolo_* 2D proposal
        # tensors.  This is the route inspired by STUR3D's Object-center Query
        # Generator rather than hand-written depth anchors.
        self.use_object_center_query_lifter = bool(
            (self.use_online_yolo or self.use_offline_yolo2d or self.use_gt_projected_2d)
            and object_center_query_lifter is not None
            and object_center_query_lifter.get("enabled", True)
        )
        if self.use_object_center_query_lifter:
            oqg_cfg = dict(object_center_query_lifter)
            oqg_cfg.pop("enabled", None)
            self.object_center_query_lifter = ObjectCenterQueryLifter(**oqg_cfg)

        self.use_frustum_geometry_lifter = bool(
            self.use_object_center_query_lifter
            and frustum_geometry_lifter is not None
            and frustum_geometry_lifter.get("enabled", True)
        )
        if self.use_frustum_geometry_lifter:
            frustum_cfg = dict(frustum_geometry_lifter)
            frustum_cfg.pop("enabled", None)
            self.frustum_geometry_lifter = FrustumGeometryLifter(**frustum_cfg)

        # Main route: 2D boxes select real measured Gaussians as object seeds.
        # The head runs after the measured Gaussian lifter and before the encoder.
        # V22-C1 uses ROI/depth ownership and never consumes OQG geometry.
        self.use_object_gaussian_seed_completion = bool(
            object_gaussian_seed_completion is not None
            and object_gaussian_seed_completion.get("enabled", True)
            and (self.use_online_yolo or self.use_offline_yolo2d or self.use_gt_projected_2d)
        )
        if self.use_object_gaussian_seed_completion:
            seed_cfg = dict(object_gaussian_seed_completion)
            seed_cfg.pop("enabled", None)
            self.object_gaussian_seed_completion = ObjectGaussianSeedCompletionHead(**seed_cfg)

        # v8.5 direct-OQG route: no BoxHypMamba and no GeoRanker.
        # ObjectCenterQueryLifter outputs yolo_box3d_selected_* directly.

        # Box-driven Gaussian completion.
        # source='gt'   : oracle branch using GT 3D boxes, used now to verify
        #                 completion Gaussians.
        # source='boxhyp': final branch using YOLO + Box-Hypothesis Mamba
        #                 selected 3D boxes after the nuScenes10 YOLO is ready.
        # source='auto' : prefer GT when available, otherwise use BoxHyp boxes.
        if box_gaussian_completion is None and gt_box_gaussian_completion is not None:
            # Backward-compatible alias for older oracle configs.
            box_gaussian_completion = gt_box_gaussian_completion
        self.use_box_gaussian_completion = bool(
            box_gaussian_completion is not None
            and box_gaussian_completion.get("enabled", True)
        )
        if self.use_box_gaussian_completion:
            comp_cfg = dict(box_gaussian_completion)
            comp_cfg.pop("enabled", None)
            self.box_gaussian_completion = BoxGaussianCompletionHead(**comp_cfg)
        self.use_gaussian_completion_visualizer = bool(
            gaussian_completion_visualizer is not None
            and gaussian_completion_visualizer.get("enabled", True)
        )
        if self.use_gaussian_completion_visualizer:
            # Keep the public config switch and the visualizer's internal switch
            # in sync.  Previously `enabled` was popped and not passed into
            # GaussianCompletionVisualizer, whose default is enabled=False; this
            # caused logs such as:
            #   [GaussianCompletionVis][skip] ... reason=disabled
            # even when config.gaussian_completion_visualizer.enabled=True.
            vis_cfg = dict(gaussian_completion_visualizer)
            vis_enabled = bool(vis_cfg.pop("enabled", True))
            self.gaussian_completion_visualizer = GaussianCompletionVisualizer(
                enabled=vis_enabled,
                **vis_cfg,
            )
        if voxelize_lidar is not None:
            self.voxelize_lidar = Voxelization(**voxelize_lidar)
        if lidar_voxel_encoder is not None:
            self.lidar_voxel_encoder = MODELS.build(lidar_voxel_encoder)

        # Debug/visualization fallback counter. train.py normally passes global_iter,
        # but keeping a module-side counter prevents visualization/debug code from
        # silently disabling itself in eval, standalone forward tests, or older
        # training scripts.
        self._fallback_global_iter = 0

    def _resolve_global_iter(self, kwargs):
        """Return a stable integer global_iter and update kwargs in-place."""
        global_iter = kwargs.get('global_iter', None)
        if global_iter is None:
            global_iter = int(self._fallback_global_iter)
            kwargs['global_iter'] = global_iter
        else:
            try:
                if isinstance(global_iter, torch.Tensor):
                    global_iter = int(global_iter.detach().cpu().reshape(-1)[0].item())
                else:
                    global_iter = int(global_iter)
            except Exception:
                global_iter = int(self._fallback_global_iter)
            kwargs['global_iter'] = global_iter
        self._fallback_global_iter = max(int(self._fallback_global_iter) + 1, int(global_iter) + 1)
        return global_iter

    @staticmethod
    def _is_rank0():
        try:
            import torch.distributed as dist
            return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        except Exception:
            return True

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return {'ms_img_feats': img_feats_reshaped}
    
    @torch.no_grad()
    def voxelize(self, points, **kwargs):
        """Apply dynamic voxelization to points.
        Args:
            points (list[torch.Tensor]): Points of each sample.
        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxelize_lidar(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        
        return voxels, num_points, coors_batch
    
    def extract_lidar_feat(self, points, **kwargs):
        """Extract features of lidar."""
        voxels, num_points, coors_batch = self.voxelize(points)
        voxel_lidar_feats = self.lidar_voxel_encoder(voxels, num_points, coors_batch)
        
        return {'voxel_lidar_feats': voxel_lidar_feats, 'coors_batch': coors_batch}
    
    def extract_img_dpt_feat(self, imgs, dpt, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)
        if imgs is not None:
            if imgs.dim() == 5 and imgs.size(0) == 1:
                imgs.squeeze_(0)
                if not (dpt is None):
                    dpt.squeeze_(0)
            elif imgs.dim() == 5 and imgs.size(0) > 1:
                B, N, C, H, W = imgs.size()
                imgs = imgs.reshape(B * N, C, H, W)
                if not (dpt is None):
                    _, _, C_dpt, _, _ = dpt.size()
                    dpt = dpt.reshape(B * N, C_dpt, H, W)
            
            # data augmentation
            if self.use_grid_mask:
                if not (dpt is None):
                    imgs, dpt = self.grid_mask(imgs, dpt)
                else:
                    imgs = self.grid_mask(imgs)

            img_feats = self.img_backbone(imgs)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
            
        return {'ms_img_feats': img_feats_reshaped, 'dpt_masked': dpt}
    
    def extract_multiscale_dpt(self, ms_img_feats, dpt_masked=None, metas=None, **kwargs):
        """Extract multi-scale depth distributions for Sparse-UVD sampling.

        Supported modes:
          1. DepthHead / learned depth heads: image features predict a depth
             distribution.  dpt_masked is used only for optional auxiliary
             depth supervision.
          2. DepthHead_GTDpt: dpt_masked is the LiDAR-projected depth_gt.  It is
             converted directly into a multi-scale one-hot/soft depth
             distribution, matching the GaussianFormer3D LiDAR-depth-prior
             route.  No depth prediction loss is produced in this mode.

        The encoder/GAF still performs sparse UVD sampling: conceptually it
        samples F_img(u, v) * P_depth(u, v, d), without materializing a dense
        [B, Cam, C, D, H, W] feature volume.
        """
        lidar2img = metas["projection_mat"].to(ms_img_feats[0].device)
        dpt_head_name = self.pts_dpt_head.__class__.__name__
        use_gt_dpt_forward = dpt_head_name == "DepthHead_GTDpt"

        if use_gt_dpt_forward:
            if dpt_masked is None:
                raise RuntimeError(
                    "DepthHead_GTDpt requires dpt/depth_gt from "
                    "LoadMultiViewDepthFromFiles, but dpt_masked is None."
                )
            dpt_dist, out_dpt_multiscale = self.pts_dpt_head(
                ms_img_feats,
                lidar2img.flatten(2),
                gt_dpt=dpt_masked,
                return_dpt=True,
            )
        else:
            dpt_dist, out_dpt_multiscale = self.pts_dpt_head(
                ms_img_feats,
                lidar2img.flatten(2),
                return_dpt=True,
            )

        out_dpt_multiscale = [
            outdpt.view(*lidar2img.shape[:2], *outdpt.shape[1:])
            for outdpt in out_dpt_multiscale
        ]

        outs = {
            'dpt_dist': dpt_dist,
            'out_dpt_multiscale': out_dpt_multiscale,
        }

        if (not use_gt_dpt_forward) and dpt_masked is not None and hasattr(self.pts_dpt_head, 'loss'):
            outs.update(self.pts_dpt_head.loss(dpt_masked, dpt_dist))

        # DEBUG: verify depth supervision / GT-depth conversion after downsampling into depth bins.
        # For DepthHead_GTDpt, loss_dpt is intentionally absent because depth_gt is used as input prior.
        global_iter = kwargs.get('global_iter', -1)
        if (
            self.dpt_debug
            and self._dpt_debug_print_count < self.dpt_debug_max_print
            and isinstance(global_iter, int)
            and global_iter % self.dpt_debug_interval == 0
        ):
            is_rank0 = True
            try:
                import torch.distributed as dist
                is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
            except Exception:
                is_rank0 = True
            if is_rank0:
                if dpt_masked is None:
                    print("[SEG DPT DEBUG] dpt_masked=None", flush=True)
                else:
                    with torch.no_grad():
                        gt_depth_bins = self.pts_dpt_head.get_downsampled_gt_depth(dpt_masked)
                        fg_mask = torch.max(gt_depth_bins, dim=1).values > 0.0
                        loss_dpt_val = outs.get('loss_dpt', None)
                        print(
                            "[SEG DPT DEBUG] "
                            f"dpt_shape={tuple(dpt_masked.shape)}, "
                            f"dpt_min={float(dpt_masked.detach().float().min().cpu()):.6f}, "
                            f"dpt_max={float(dpt_masked.detach().float().max().cpu()):.6f}, "
                            f"dpt_nonzero={int((dpt_masked.detach().float() > 0).sum().cpu())}, "
                            f"gt_bins_shape={tuple(gt_depth_bins.shape)}, "
                            f"gt_fg={int(fg_mask.sum().cpu())}, "
                            f"dpt_dist_shape={tuple(dpt_dist.shape)}, "
                            f"dpt_dist_min={float(dpt_dist.detach().float().min().cpu()):.8f}, "
                            f"dpt_dist_max={float(dpt_dist.detach().float().max().cpu()):.8f}, "
                            f"loss_dpt={None if loss_dpt_val is None else float(loss_dpt_val.detach().float().cpu()):}",
                            flush=True,
                        )
                        self._dpt_debug_print_count += 1

        return outs    
    
    def forward_extra_img_backbone(self, imgs, **kwargs):
        """Extract features of images."""
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.extra_img_backbone(imgs)

        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())

        img_feats_backbone_reshaped = []
        for img_feat_backbone in img_feats_backbone:
            BN, C, H, W = img_feat_backbone.size()
            img_feats_backbone_reshaped.append(
                img_feat_backbone.view(B, int(BN / B), C, H, W))
        return img_feats_backbone_reshaped

    @staticmethod
    def _as_batch_tensor_list(value, B, device, dtype=None):
        """Normalize variable-length batch fields to a list of tensors."""
        if value is None:
            return [None for _ in range(B)]
        if isinstance(value, torch.Tensor):
            if value.ndim >= 3 and int(value.shape[0]) == B:
                out = [value[b] for b in range(B)]
            elif B == 1:
                out = [value]
            else:
                # Ambiguous collate result.  Keep a safe empty value for each sample
                # rather than silently assigning the same boxes to every batch item.
                out = [None for _ in range(B)]
            if dtype is not None:
                out = [None if x is None else x.to(device=device, dtype=dtype) for x in out]
            else:
                out = [None if x is None else x.to(device=device) for x in out]
            return out
        if isinstance(value, (list, tuple)):
            out = []
            for i in range(B):
                if i >= len(value) or value[i] is None:
                    out.append(None)
                else:
                    x = value[i]
                    if not isinstance(x, torch.Tensor):
                        x = torch.as_tensor(x)
                    if dtype is not None:
                        x = x.to(device=device, dtype=dtype)
                    else:
                        x = x.to(device=device)
                    out.append(x)
            return out
        return [None for _ in range(B)]

    @staticmethod
    def _box_corners_lwh_yaw(boxes):
        """Build 8 corners for boxes [N,7] = x,y,z,l,w,h,yaw."""
        centers = boxes[:, 0:3]
        lwh = boxes[:, 3:6].clamp_min(1.0e-4)
        yaw = boxes[:, 6]
        device, dtype = boxes.device, boxes.dtype
        sx = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], device=device, dtype=dtype)
        sy = torch.tensor([1, 1, -1, -1, 1, 1, -1, -1], device=device, dtype=dtype)
        sz = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1], device=device, dtype=dtype)
        local = torch.stack([
            sx * lwh[:, 0:1] * 0.5,
            sy * lwh[:, 1:2] * 0.5,
            sz * lwh[:, 2:3] * 0.5,
        ], dim=-1)  # [N,8,3]
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        x = local[..., 0] * c[:, None] - local[..., 1] * s[:, None]
        y = local[..., 0] * s[:, None] + local[..., 1] * c[:, None]
        z = local[..., 2]
        return torch.stack([x, y, z], dim=-1) + centers[:, None, :]

    @staticmethod
    def _box_iou_2d(boxes1, boxes2):
        """Pairwise IoU for xyxy boxes: boxes1 [N,4], boxes2 [M,4]."""
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
        x11, y11, x12, y12 = boxes1.unbind(dim=-1)
        x21, y21, x22, y22 = boxes2.unbind(dim=-1)
        area1 = (x12 - x11).clamp_min(0.0) * (y12 - y11).clamp_min(0.0)
        area2 = (x22 - x21).clamp_min(0.0) * (y22 - y21).clamp_min(0.0)
        lt_x = torch.maximum(x11[:, None], x21[None, :])
        lt_y = torch.maximum(y11[:, None], y21[None, :])
        rb_x = torch.minimum(x12[:, None], x22[None, :])
        rb_y = torch.minimum(y12[:, None], y22[None, :])
        inter = (rb_x - lt_x).clamp_min(0.0) * (rb_y - lt_y).clamp_min(0.0)
        union = area1[:, None] + area2[None, :] - inter
        return inter / union.clamp_min(1.0e-6)

    def _match_offline_yolo2d_to_gt(self, imgs=None, metas=None, yolo_boxes_2d=None,
                                    yolo_scores_2d=None, yolo_labels_2d=None,
                                    yolo_valid_2d=None, gt_bboxes_3d=None,
                                    gt_labels_3d=None, global_iter=None):
        """Match cached YOLO 2D boxes to projected GT 3D boxes.

        This produces yolo_gt_indices_2d [B,Cam,M], the same supervision field
        used by the Perfect2D diagnostic path.  For each YOLO box, we project all
        GT 3D boxes to that camera, compute same-class 2D IoU, and keep the GT
        index when IoU >= gt_match_min_iou.  Unmatched YOLO boxes remain -1 and
        are treated as weak negatives inside OQG.
        """
        if not self.offline_yolo2d_gt_supervision:
            return {}
        if yolo_boxes_2d is None or yolo_labels_2d is None or yolo_valid_2d is None:
            return {}
        if metas is None or not isinstance(metas, dict) or "projection_mat" not in metas:
            return {}
        if gt_bboxes_3d is None:
            gt_bboxes_3d = metas.get('gt_bboxes_3d', None)
        if gt_labels_3d is None:
            gt_labels_3d = metas.get('gt_labels_3d', None)
        if gt_bboxes_3d is None or gt_labels_3d is None:
            return {}
        if imgs is not None and isinstance(imgs, torch.Tensor) and imgs.ndim == 5:
            B, Cam, _, H_img, W_img = imgs.shape
            device = imgs.device
        else:
            if not isinstance(yolo_boxes_2d, torch.Tensor):
                yolo_boxes_2d = torch.as_tensor(yolo_boxes_2d)
            B, Cam = int(yolo_boxes_2d.shape[0]), int(yolo_boxes_2d.shape[1])
            H_img, W_img = 256, 704
            device = yolo_boxes_2d.device
        dtype = torch.float32
        boxes2d = torch.as_tensor(yolo_boxes_2d, device=device, dtype=dtype)
        labels2d = torch.as_tensor(yolo_labels_2d, device=device, dtype=torch.long)
        valid2d = torch.as_tensor(yolo_valid_2d, device=device, dtype=torch.bool)
        proj = metas["projection_mat"]
        if not isinstance(proj, torch.Tensor):
            proj = torch.as_tensor(proj)
        proj = proj.to(device=device, dtype=dtype, non_blocking=True)
        if proj.ndim != 4 or int(proj.shape[0]) != B or int(proj.shape[1]) != Cam:
            return {}

        M = int(boxes2d.shape[2])
        out_gt_indices = torch.full((B, Cam, M), -1, device=device, dtype=torch.long)
        out_gt_ious = torch.zeros((B, Cam, M), device=device, dtype=dtype)
        boxes_list = self._as_batch_tensor_list(gt_bboxes_3d, B, device, dtype=dtype)
        labels_list = self._as_batch_tensor_list(gt_labels_3d, B, device, dtype=None)

        total_valid_yolo = int(valid2d.sum().detach().cpu()) if valid2d.numel() else 0
        total_projected_gt = 0
        total_matched = 0
        matched_ious = []
        per_sample_matched = []
        for b in range(B):
            matched_b = 0
            gt = boxes_list[b]
            gt_lab = labels_list[b]
            if gt is None or gt_lab is None or gt.numel() == 0 or gt_lab.numel() == 0:
                per_sample_matched.append(0)
                continue
            gt = gt.reshape(-1, gt.shape[-1])[:, :7].to(device=device, dtype=dtype)
            gt_lab = gt_lab.reshape(-1).to(device=device, dtype=torch.long)
            n = min(int(gt.shape[0]), int(gt_lab.numel()))
            gt, gt_lab = gt[:n], gt_lab[:n]
            gt_ok = torch.isfinite(gt).all(dim=-1) & (gt_lab >= 0) & (gt_lab < 10)
            if not bool(gt_ok.any()):
                per_sample_matched.append(0)
                continue
            orig_idx = torch.arange(n, device=device, dtype=torch.long)[gt_ok]
            gt = gt[gt_ok]
            gt_lab = gt_lab[gt_ok]
            corners = self._box_corners_lwh_yaw(gt)
            ones = torch.ones((corners.shape[0], corners.shape[1], 1), device=device, dtype=dtype)
            corners_h = torch.cat([corners, ones], dim=-1)
            for cidx in range(Cam):
                img_h = torch.einsum('ij,nkj->nki', proj[b, cidx], corners_h)
                depth = img_h[..., 2]
                in_front = depth > float(self.gt_projected_2d_min_depth)
                depth_safe = depth.clamp_min(float(self.gt_projected_2d_min_depth))
                u = img_h[..., 0] / depth_safe
                v = img_h[..., 1] / depth_safe
                inf = torch.full_like(u, float('inf'))
                ninf = torch.full_like(u, -float('inf'))
                u_min_raw = torch.where(in_front, u, inf).min(dim=-1).values
                v_min_raw = torch.where(in_front, v, inf).min(dim=-1).values
                u_max_raw = torch.where(in_front, u, ninf).max(dim=-1).values
                v_max_raw = torch.where(in_front, v, ninf).max(dim=-1).values
                if self.offline_yolo2d_gt_match_clip:
                    u_min = u_min_raw.clamp(0.0, float(W_img - 1))
                    u_max = u_max_raw.clamp(0.0, float(W_img - 1))
                    v_min = v_min_raw.clamp(0.0, float(H_img - 1))
                    v_max = v_max_raw.clamp(0.0, float(H_img - 1))
                else:
                    u_min, u_max = u_min_raw, u_max_raw
                    v_min, v_max = v_min_raw, v_max_raw
                proj_boxes = torch.stack([u_min, v_min, u_max, v_max], dim=-1)
                area = (u_max - u_min).clamp_min(0.0) * (v_max - v_min).clamp_min(0.0)
                raw_overlap = (u_max_raw >= 0.0) & (u_min_raw <= float(W_img - 1)) & (v_max_raw >= 0.0) & (v_min_raw <= float(H_img - 1))
                proj_ok = (
                    (in_front.sum(dim=-1) >= max(int(self.gt_projected_2d_min_visible_corners), 1))
                    & raw_overlap
                    & torch.isfinite(proj_boxes).all(dim=-1)
                    & torch.isfinite(area)
                    & (area >= float(self.offline_yolo2d_gt_match_min_area))
                )
                if not bool(proj_ok.any()):
                    continue
                gt_boxes_c = proj_boxes[proj_ok]
                gt_labels_c = gt_lab[proj_ok]
                gt_orig_c = orig_idx[proj_ok]
                total_projected_gt += int(gt_boxes_c.shape[0])
                yidx = torch.nonzero(valid2d[b, cidx], as_tuple=False).flatten()
                if yidx.numel() == 0:
                    continue
                y_boxes = boxes2d[b, cidx, yidx]
                y_labels = labels2d[b, cidx, yidx]
                iou = self._box_iou_2d(y_boxes, gt_boxes_c)
                if self.offline_yolo2d_gt_match_class_aware:
                    same_cls = y_labels[:, None] == gt_labels_c[None, :]
                    iou = iou.masked_fill(~same_cls, -1.0)
                best_iou, best_j = iou.max(dim=-1)
                keep = best_iou >= float(self.offline_yolo2d_gt_match_min_iou)
                if not bool(keep.any()):
                    continue
                sel_y = yidx[keep]
                sel_gt = gt_orig_c[best_j[keep].long()]
                out_gt_indices[b, cidx, sel_y] = sel_gt.long()
                out_gt_ious[b, cidx, sel_y] = best_iou[keep].to(dtype=dtype)
                matched_b += int(keep.sum().detach().cpu())
                matched_ious.append(best_iou[keep].detach())
            per_sample_matched.append(matched_b)
            total_matched += matched_b

        if (
            self.offline_yolo2d_gt_match_debug
            and self._offline_yolo2d_gt_match_print_count < self.offline_yolo2d_gt_match_debug_max_print
            and self._offline_yolo2d_call_count % self.offline_yolo2d_gt_match_debug_interval == 0
            and self._is_rank0()
        ):
            if len(matched_ious):
                miou = torch.cat(matched_ious).float()
                iou_msg = f"iou_mean={float(miou.mean().cpu()):.3f}, iou_p50={float(miou.median().cpu()):.3f}"
            else:
                iou_msg = "iou_mean=0.000, iou_p50=0.000"
            print(
                "[YOLO2DGTMatch] "
                f"call={self._offline_yolo2d_call_count}, yolo_valid={total_valid_yolo}, "
                f"projected_gt={total_projected_gt}, matched={total_matched}, "
                f"match_ratio={(float(total_matched) / max(float(total_valid_yolo), 1.0)):.3f}, "
                f"min_iou={self.offline_yolo2d_gt_match_min_iou:.3f}, "
                f"class_aware={self.offline_yolo2d_gt_match_class_aware}, "
                f"per_sample={per_sample_matched}, {iou_msg}",
                flush=True,
            )
            self._offline_yolo2d_gt_match_print_count += 1

        return {
            'yolo_gt_indices_2d': out_gt_indices,
            'yolo_gt_match_iou_2d': out_gt_ious,
            'yolo_source_has_gt_match': torch.ones((B,), device=device, dtype=torch.bool),
        }

    def _build_offline_yolo2d_proposals(self, imgs=None, metas=None, global_iter=None, **kwargs):
        """Read fixed-shape offline YOLO2D tensors from the batch.

        Dataset returns these fields as top-level keys; depending on the train
        script they may arrive through kwargs or inside metas.  The tensors are
        already in network image coordinates [B,Cam,M,4] xyxy, so this function
        only moves them to the image device and returns the same names consumed
        by OQG.
        """
        def pick(name):
            if name in kwargs:
                return kwargs[name]
            if isinstance(metas, dict) and name in metas:
                return metas[name]
            return None

        boxes = pick('yolo_boxes_2d')
        scores = pick('yolo_scores_2d')
        labels = pick('yolo_labels_2d')
        valid = pick('yolo_valid_2d')
        if boxes is None or scores is None or labels is None or valid is None:
            return {}
        device = imgs.device if isinstance(imgs, torch.Tensor) else None
        boxes = torch.as_tensor(boxes, device=device, dtype=torch.float32)
        scores = torch.as_tensor(scores, device=device, dtype=torch.float32)
        labels = torch.as_tensor(labels, device=device, dtype=torch.long)
        valid = torch.as_tensor(valid, device=device, dtype=torch.bool)
        valid = valid & torch.isfinite(boxes).all(dim=-1) & torch.isfinite(scores) & (labels >= 0) & (labels < 10)

        # Cheap top-score filtering on the fixed-shape cache tensors.  This is
        # deliberately done after moving to GPU so downstream OQG sees fewer
        # valid candidates while tensor shapes remain unchanged for collation.
        # It mainly removes duplicate/low-confidence YOLO boxes and directly
        # reduces OQG K-candidate expansion and box completion cost.
        if self.offline_yolo2d_max_boxes_per_cam > 0 and valid.ndim == 3:
            cap_cam = int(self.offline_yolo2d_max_boxes_per_cam)
            B, Cam, M = valid.shape
            new_valid = torch.zeros_like(valid)
            for b in range(B):
                for c in range(Cam):
                    idx = torch.nonzero(valid[b, c], as_tuple=False).flatten()
                    if idx.numel() <= 0:
                        continue
                    if idx.numel() > cap_cam:
                        top = torch.topk(scores[b, c, idx], k=cap_cam, largest=True, sorted=False).indices
                        idx = idx[top]
                    new_valid[b, c, idx] = True
            valid = new_valid

        if self.offline_yolo2d_max_boxes_per_sample > 0 and valid.ndim == 3:
            cap_sample = int(self.offline_yolo2d_max_boxes_per_sample)
            B, Cam, M = valid.shape
            flat_valid = valid.reshape(B, Cam * M)
            flat_scores = scores.reshape(B, Cam * M)
            new_flat = torch.zeros_like(flat_valid)
            for b in range(B):
                idx = torch.nonzero(flat_valid[b], as_tuple=False).flatten()
                if idx.numel() <= 0:
                    continue
                if idx.numel() > cap_sample:
                    top = torch.topk(flat_scores[b, idx], k=cap_sample, largest=True, sorted=False).indices
                    idx = idx[top]
                new_flat[b, idx] = True
            valid = new_flat.reshape(B, Cam, M)

        self._offline_yolo2d_call_count += 1
        if (
            self.offline_yolo2d_debug
            and self._offline_yolo2d_debug_print_count < self.offline_yolo2d_debug_max_print
            and self._offline_yolo2d_call_count % self.offline_yolo2d_debug_interval == 0
        ):
            is_rank0 = True
            try:
                import torch.distributed as dist
                is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
            except Exception:
                is_rank0 = True
            if is_rank0:
                hist = {}
                if bool(valid.any()):
                    labs = labels[valid].detach().cpu()
                    uniq, cnt = torch.unique(labs, return_counts=True)
                    hist = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
                print(
                    f'[OfflineYOLO2D] call={self._offline_yolo2d_call_count}, '
                    f'boxes2d={int(valid.sum().detach().cpu())}, shape={tuple(boxes.shape)}, '
                    f'cap_sample={self.offline_yolo2d_max_boxes_per_sample}, '
                    f'cap_cam={self.offline_yolo2d_max_boxes_per_cam}, hist={hist}',
                    flush=True,
                )
                self._offline_yolo2d_debug_print_count += 1
        out = {
            'yolo_boxes_2d': boxes,
            'yolo_scores_2d': scores,
            'yolo_labels_2d': labels,
            'yolo_valid_2d': valid,
        }
        flag = pick('yolo_source_is_offline_cache')
        if flag is not None:
            try:
                out['yolo_source_is_offline_cache'] = torch.as_tensor(flag, device=device, dtype=torch.bool)
            except Exception:
                pass
        return out

    def _build_gt_projected_2d_proposals(self, imgs=None, metas=None, global_iter=None, **kwargs):
        """Project GT 3D boxes to per-camera perfect 2D boxes.

        Returned tensors intentionally match OnlineFrozenYOLO26 output names:
            yolo_boxes_2d / yolo_scores_2d / yolo_labels_2d / yolo_valid_2d.
        An extra yolo_gt_indices_2d tensor stores the original GT box index so
        the lifter can print perfect-2D lifting error against the source 3D box.
        """
        if metas is None or not isinstance(metas, dict) or "projection_mat" not in metas:
            return {}
        if imgs is None or not isinstance(imgs, torch.Tensor) or imgs.ndim != 5:
            return {}
        gt_boxes_obj = kwargs.get('gt_bboxes_3d', metas.get('gt_bboxes_3d', None))
        gt_labels_obj = kwargs.get('gt_labels_3d', metas.get('gt_labels_3d', None))
        if gt_boxes_obj is None or gt_labels_obj is None:
            return {}

        B, Cam, _, H, W = imgs.shape
        device = imgs.device
        dtype = torch.float32
        proj = metas["projection_mat"]
        if not isinstance(proj, torch.Tensor):
            proj = torch.as_tensor(proj)
        proj = proj.to(device=device, dtype=dtype, non_blocking=True)
        if proj.ndim != 4 or int(proj.shape[0]) != B or int(proj.shape[1]) != Cam:
            return {}

        max_det = max(int(self.gt_projected_2d_max_det), 1)
        out_boxes = torch.zeros((B, Cam, max_det, 4), device=device, dtype=dtype)
        out_scores = torch.zeros((B, Cam, max_det), device=device, dtype=dtype)
        out_labels = torch.full((B, Cam, max_det), -1, device=device, dtype=torch.long)
        out_valid = torch.zeros((B, Cam, max_det), device=device, dtype=torch.bool)
        out_gt_indices = torch.full((B, Cam, max_det), -1, device=device, dtype=torch.long)

        boxes_list = self._as_batch_tensor_list(gt_boxes_obj, B, device, dtype=dtype)
        labels_list = self._as_batch_tensor_list(gt_labels_obj, B, device, dtype=None)
        per_sample_counts = []
        per_cam_counts = []
        total_gt = 0
        for b in range(B):
            boxes = boxes_list[b]
            labels = labels_list[b]
            if boxes is None or labels is None or boxes.numel() == 0 or labels.numel() == 0:
                per_sample_counts.append(0)
                per_cam_counts.append([0 for _ in range(Cam)])
                continue
            boxes = boxes.reshape(-1, boxes.shape[-1])[:, :7].to(device=device, dtype=dtype)
            labels = labels.reshape(-1).to(device=device, dtype=torch.long)
            n = min(int(boxes.shape[0]), int(labels.numel()))
            boxes, labels = boxes[:n], labels[:n]
            valid_label = (labels >= 0) & (labels < 10) & torch.isfinite(boxes).all(dim=-1)
            if not bool(valid_label.any()):
                per_sample_counts.append(0)
                per_cam_counts.append([0 for _ in range(Cam)])
                continue
            orig_indices = torch.arange(n, device=device, dtype=torch.long)[valid_label]
            boxes = boxes[valid_label]
            labels = labels[valid_label]
            total_gt += int(boxes.shape[0])
            corners = self._box_corners_lwh_yaw(boxes)  # [N,8,3]
            ones = torch.ones((corners.shape[0], corners.shape[1], 1), device=device, dtype=dtype)
            corners_h = torch.cat([corners, ones], dim=-1)  # [N,8,4]
            sample_count = 0
            cam_counts = []
            for cidx in range(Cam):
                img_h = torch.einsum('ij,nkj->nki', proj[b, cidx], corners_h)
                depth = img_h[..., 2]
                in_front = depth > float(self.gt_projected_2d_min_depth)
                depth_safe = depth.clamp_min(float(self.gt_projected_2d_min_depth))
                u = img_h[..., 0] / depth_safe
                v = img_h[..., 1] / depth_safe
                # Robust visibility for diagnostic perfect-2D boxes.  The old
                # version required at least one 3D box corner to be already
                # inside the image before clipping.  That drops valid nearby or
                # partially visible boxes whose projected 2D rectangle overlaps
                # the image although all projected corners lie outside.  For the
                # lifting diagnostic we only need a faithful 2D proposal source,
                # so keep boxes with enough front-facing corners and a non-empty
                # clipped overlap with the image.
                front_count = in_front.sum(dim=-1)
                keep = front_count >= max(int(self.gt_projected_2d_min_visible_corners), 1)
                inf = torch.full_like(u, float('inf'))
                ninf = torch.full_like(u, -float('inf'))
                u_min_raw = torch.where(in_front, u, inf).min(dim=-1).values
                v_min_raw = torch.where(in_front, v, inf).min(dim=-1).values
                u_max_raw = torch.where(in_front, u, ninf).max(dim=-1).values
                v_max_raw = torch.where(in_front, v, ninf).max(dim=-1).values
                if self.gt_projected_2d_clip:
                    u_min = u_min_raw.clamp(0.0, float(W - 1))
                    u_max = u_max_raw.clamp(0.0, float(W - 1))
                    v_min = v_min_raw.clamp(0.0, float(H - 1))
                    v_max = v_max_raw.clamp(0.0, float(H - 1))
                else:
                    u_min, u_max = u_min_raw, u_max_raw
                    v_min, v_max = v_min_raw, v_max_raw
                w2d = (u_max - u_min).clamp_min(0.0)
                h2d = (v_max - v_min).clamp_min(0.0)
                area = w2d * h2d
                # Require the raw projected rectangle to overlap the image.
                # This avoids accepting boxes entirely left/right/above/below
                # whose clipped min/max collapse to a line.
                raw_overlap = (u_max_raw >= 0.0) & (u_min_raw <= float(W - 1)) & (v_max_raw >= 0.0) & (v_min_raw <= float(H - 1))
                keep = keep & raw_overlap & torch.isfinite(area) & (area >= float(self.gt_projected_2d_min_box_area))
                idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
                if idx.numel() > max_det:
                    # Keep the largest projected boxes when a camera sees too many objects.
                    top = torch.topk(area[idx], k=max_det, largest=True, sorted=False).indices
                    idx = idx[top]
                m = int(idx.numel())
                cam_counts.append(m)
                if m <= 0:
                    continue
                out_boxes[b, cidx, :m, 0] = u_min[idx]
                out_boxes[b, cidx, :m, 1] = v_min[idx]
                out_boxes[b, cidx, :m, 2] = u_max[idx]
                out_boxes[b, cidx, :m, 3] = v_max[idx]
                out_scores[b, cidx, :m] = float(self.gt_projected_2d_score)
                out_labels[b, cidx, :m] = labels[idx]
                out_valid[b, cidx, :m] = True
                out_gt_indices[b, cidx, :m] = orig_indices[idx]
                sample_count += m
            per_sample_counts.append(sample_count)
            per_cam_counts.append(cam_counts)

        self._gt_projected_2d_call_count += 1
        if (
            self.gt_projected_2d_debug
            and self._gt_projected_2d_debug_print_count < self.gt_projected_2d_debug_max_print
            and self._gt_projected_2d_call_count % self.gt_projected_2d_debug_interval == 0
        ):
            is_rank0 = True
            try:
                import torch.distributed as dist
                is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
            except Exception:
                is_rank0 = True
            if is_rank0:
                total_2d = int(out_valid.sum().detach().cpu())
                label_hist = {}
                if total_2d > 0:
                    labs = out_labels[out_valid].detach().cpu()
                    uniq, cnt = torch.unique(labs, return_counts=True)
                    label_hist = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
                print(
                    "[GTProjected2D] "
                    f"call={self._gt_projected_2d_call_count}, gt3d={total_gt}, boxes2d={total_2d}, "
                    f"per_sample={per_sample_counts}, per_cam={per_cam_counts}, hist={label_hist}",
                    flush=True,
                )
                self._gt_projected_2d_debug_print_count += 1

        return {
            'yolo_boxes_2d': out_boxes,
            'yolo_scores_2d': out_scores,
            'yolo_labels_2d': out_labels,
            'yolo_valid_2d': out_valid,
            'yolo_gt_indices_2d': out_gt_indices,
            'yolo_source_is_gt_projected_2d': torch.ones((B,), device=device, dtype=torch.bool),
        }

    def forward(self,
                imgs=None,
                metas=None,
                points=None,
                dpt=None,
                extra_backbone=False,
                occ_only=False,
                rep_only=False,
                **kwargs,
        ):
        """Forward training function.
        """
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=imgs)

        global_iter = self._resolve_global_iter(kwargs)
        
        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points,
            'dpt': dpt,
        }
        # Keep auxiliary geometry fields available to the lifter. In the
        # current training loop, all dataset keys that are not popped out
        # (e.g. visibility_points / ego2lidar) remain inside metas.
        if isinstance(metas, dict):
            for aux_key in ['visibility_points', 'ego2lidar', 'sample_idx', 'scene_token', 'scene_name', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_names', 'gt_bboxes_3d_has_ann', 'yolo_boxes_2d', 'yolo_scores_2d', 'yolo_labels_2d', 'yolo_valid_2d', 'yolo_source_is_offline_cache']:
                if aux_key in metas:
                    results[aux_key] = metas[aux_key]
        results.update(kwargs)

        # Image proposal source.  For normal final runs this will be offline/online
        # YOLO.  For the first diagnostic step, gt_projected_2d generates perfect
        # 2D boxes from GT 3D boxes so we can evaluate 2D->3D lifting quality
        # without YOLO detection noise.
        yolo_outs = None
        if getattr(self, 'use_online_yolo', False):
            yolo_outs = self.online_yolo(imgs=imgs, metas=metas, **kwargs)
        elif getattr(self, 'use_offline_yolo2d', False):
            offline_keys = [
                'yolo_boxes_2d', 'yolo_scores_2d', 'yolo_labels_2d',
                'yolo_valid_2d', 'yolo_source_is_offline_cache',
            ]
            offline_kwargs = {k: results.get(k, None) for k in offline_keys if k in results}
            yolo_outs = self._build_offline_yolo2d_proposals(
                imgs=imgs, metas=metas, global_iter=global_iter, **offline_kwargs
            )
            if isinstance(yolo_outs, dict) and len(yolo_outs) > 0 and getattr(self, 'offline_yolo2d_gt_supervision', False):
                match_outs = self._match_offline_yolo2d_to_gt(
                    imgs=imgs,
                    metas=metas,
                    yolo_boxes_2d=yolo_outs.get('yolo_boxes_2d', None),
                    yolo_scores_2d=yolo_outs.get('yolo_scores_2d', None),
                    yolo_labels_2d=yolo_outs.get('yolo_labels_2d', None),
                    yolo_valid_2d=yolo_outs.get('yolo_valid_2d', None),
                    gt_bboxes_3d=results.get('gt_bboxes_3d', None),
                    gt_labels_3d=results.get('gt_labels_3d', None),
                    global_iter=global_iter,
                )
                if isinstance(match_outs, dict) and len(match_outs) > 0:
                    yolo_outs.update(match_outs)
        elif getattr(self, 'use_gt_projected_2d', False):
            # Do not pass **results here.  results already contains imgs/metas,
            # and passing it together with explicit imgs/metas causes:
            #   TypeError: got multiple values for keyword argument 'imgs'
            # Only pass the GT fields needed by the projection diagnostic.
            yolo_outs = self._build_gt_projected_2d_proposals(
                imgs=imgs,
                metas=metas,
                global_iter=global_iter,
                gt_bboxes_3d=results.get('gt_bboxes_3d', None),
                gt_labels_3d=results.get('gt_labels_3d', None),
            )
        if isinstance(yolo_outs, dict) and len(yolo_outs) > 0:
            results.update(yolo_outs)
            if getattr(self, 'use_yolo_2d_to_3d_lifter', False):
                # Pass only the fields consumed by the lifter.  Do NOT pass
                # **results here, because results already contains imgs/metas and
                # would cause "got multiple values for keyword argument 'imgs'".
                yolo3d_outs = self.yolo_2d_to_3d_lifter(
                    yolo_boxes_2d=yolo_outs.get('yolo_boxes_2d', None),
                    yolo_scores_2d=yolo_outs.get('yolo_scores_2d', None),
                    yolo_labels_2d=yolo_outs.get('yolo_labels_2d', None),
                    yolo_valid_2d=yolo_outs.get('yolo_valid_2d', None),
                    yolo_gt_indices_2d=yolo_outs.get('yolo_gt_indices_2d', None),
                    gt_bboxes_3d=results.get('gt_bboxes_3d', None),
                    gt_labels_3d=results.get('gt_labels_3d', None),
                    metas=metas,
                    imgs=imgs,
                )
                results.update(yolo3d_outs)

        if self.use_img_branch:
            # GAF-Mamba needs multi-scale image features.  When the depth branch
            # is disabled, do not pass dpt.  In GTDpt mode the depth branch is
            # enabled and dpt_masked is the LiDAR-projected depth prior.
            img_results = dict(results)
            if not self.use_depth_branch:
                img_results['dpt'] = None
            outs = self.extract_img_dpt_feat(**img_results)
            if outs is None:
                results.update({'ms_img_feats': None, 'dpt_masked': None})
            else:
                results.update(outs)
        else:
            results.update({'ms_img_feats': None, 'dpt_masked': None})

        if self.use_depth_branch:
            outs = self.extract_multiscale_dpt(**results)
            results.update(outs)
        else:
            # Depth branch disabled: keep only image features, no UVD depth prior.
            results.update({
                'dpt_masked': None,
                'dpt_dist': None,
                'out_dpt_multiscale': None,
            })

        # OQG-lite 2D->3D lifting after image/depth features are available.
        # It consumes yolo_* 2D proposal tensors produced earlier by either
        # online YOLO or the perfect-2D diagnostic path, and outputs the same
        # yolo_box3d_* tensors expected by BoxHyp/box completion.
        if getattr(self, 'use_object_center_query_lifter', False) and isinstance(yolo_outs, dict) and len(yolo_outs) > 0:
            oqg_outs = self.object_center_query_lifter(
                yolo_boxes_2d=results.get('yolo_boxes_2d', None),
                yolo_scores_2d=results.get('yolo_scores_2d', None),
                yolo_labels_2d=results.get('yolo_labels_2d', None),
                yolo_valid_2d=results.get('yolo_valid_2d', None),
                yolo_gt_indices_2d=results.get('yolo_gt_indices_2d', None),
                yolo_source_has_gt_match=results.get('yolo_source_has_gt_match', None),
                yolo_source_is_gt_projected_2d=results.get('yolo_source_is_gt_projected_2d', None),
                gt_bboxes_3d=results.get('gt_bboxes_3d', None),
                gt_labels_3d=results.get('gt_labels_3d', None),
                metas=metas,
                imgs=imgs,
                ms_img_feats=results.get('ms_img_feats', None),
                out_dpt_multiscale=results.get('out_dpt_multiscale', None),
            )
            results.update(oqg_outs)

            if getattr(self, 'use_frustum_geometry_lifter', False):
                frustum_outs = self.frustum_geometry_lifter(
                    yolo_box3d_hypotheses=results.get('yolo_box3d_hypotheses', None),
                    yolo_box3d_base_scores=results.get('yolo_box3d_base_scores', None),
                    yolo_box3d_scores=results.get('yolo_box3d_scores', None),
                    yolo_box3d_labels=results.get('yolo_box3d_labels', None),
                    yolo_box3d_valid=results.get('yolo_box3d_valid', None),
                    yolo_box3d_depths=results.get('yolo_box3d_depths', None),
                    yolo_boxes_2d=results.get('yolo_boxes_2d', None),
                    yolo_scores_2d=results.get('yolo_scores_2d', None),
                    yolo_labels_2d=results.get('yolo_labels_2d', None),
                    yolo_valid_2d=results.get('yolo_valid_2d', None),
                    yolo_gt_indices_2d=results.get('yolo_gt_indices_2d', None),
                    gt_bboxes_3d=results.get('gt_bboxes_3d', None),
                    visibility_points=results.get('visibility_points', None),
                    metas=metas,
                    imgs=imgs,
                )
                results.update(frustum_outs)

        # v8.5: OQG-lite already selects top-k candidates directly, so no
        # external Mamba/GeoRanker scorer is run in this diagnostic route.

        outs = self.extract_lidar_feat(**results)
        results.update(outs)
        outs = self.lifter(**results)
        results.update(outs)

        # V11 object-seed completion: project the measured Gaussian centers to
        # each 2D proposal, inject object semantics into reliable measured
        # seeds, and create free-space-gated Local children around those real
        # 3D seeds.  V22-C1 does not consume OQG outputs.
        if getattr(self, 'use_object_gaussian_seed_completion', False):
            seed_outs = self.object_gaussian_seed_completion(
                representation=results.get('representation', None),
                rep_features=results.get('rep_features', None),
                gaussian_filled_count=results.get('gaussian_filled_count', None),
                gaussian_source_types=results.get('gaussian_source_types', None),
                gaussian_visibility_state_map=results.get('gaussian_visibility_state_map', None),
                gaussian_visibility_strong_any_map=results.get('gaussian_visibility_strong_any_map', None),
                gaussian_visibility_strong_ratio_map=results.get('gaussian_visibility_strong_ratio_map', None),
                gaussian_visibility_weak_ratio_map=results.get('gaussian_visibility_weak_ratio_map', None),
                yolo_boxes_2d=results.get('yolo_boxes_2d', None),
                yolo_scores_2d=results.get('yolo_scores_2d', None),
                yolo_labels_2d=results.get('yolo_labels_2d', None),
                yolo_valid_2d=results.get('yolo_valid_2d', None),
                yolo_gt_indices_2d=results.get('yolo_gt_indices_2d', None),
                gt_bboxes_3d=results.get('gt_bboxes_3d', None),
                metas=metas,
                imgs=imgs,
            )
            if isinstance(seed_outs, dict) and len(seed_outs) > 0:
                results.update(seed_outs)

        # C2 + post-Mamba residual 3D U-Net: Local completion consumes the
        # neighborhood maps first. Keep only the compact coarse state map and
        # its metadata for the dense head, and release the three pooled maps
        # before the Gaussian encoder. The retained tensors are passed only to
        # GaussianHead, not through Mamba.
        refine_visibility = {
            'gaussian_visibility_state_map': results.pop('gaussian_visibility_state_map', None),
            'gaussian_visibility_valid_batch': results.pop('gaussian_visibility_valid_batch', None),
            'gaussian_visibility_grid_min': results.pop('gaussian_visibility_grid_min', None),
            'gaussian_visibility_voxel_size': results.pop('gaussian_visibility_voxel_size', None),
        }
        results.pop('gaussian_visibility_strong_any_map', None)
        results.pop('gaussian_visibility_strong_ratio_map', None)
        results.pop('gaussian_visibility_weak_ratio_map', None)

        # Box/GT driven completion fills unused anchor slots with completed
        # Gaussians.  The encoder then sees measured + completed Gaussians
        # together.  With source='gt' this is an oracle development branch; with
        # source='boxhyp' it uses YOLO + Box-Hypothesis Mamba proposals.
        if getattr(self, 'use_box_gaussian_completion', False):
            measured_representation = results.get('representation', None)
            comp_outs = self.box_gaussian_completion(**results)
            if getattr(self, 'use_gaussian_completion_visualizer', False):
                source_tensor = comp_outs.get('box_completion_source', None)
                vis_source = 'unknown'
                if isinstance(source_tensor, torch.Tensor) and source_tensor.numel() > 0:
                    vis_source = 'gt' if int(source_tensor.reshape(-1)[0].detach().cpu().item()) == 0 else 'boxhyp'
                vis_outs = self.gaussian_completion_visualizer(
                    measured_representation=measured_representation,
                    completed_representation=comp_outs.get('representation', measured_representation),
                    gaussian_filled_count=results.get('gaussian_filled_count', None),
                    box_completion_count=comp_outs.get('box_completion_count', None),
                    gaussian_measured_levels=results.get('gaussian_measured_levels', None),
                    points=results.get('visibility_points', results.get('points', points)),
                    global_iter=global_iter,
                    metas=metas,
                    source=vis_source,
                )
                if isinstance(vis_outs, dict) and len(vis_outs) > 0:
                    comp_outs.update(vis_outs)
            results.update(comp_outs)

        outs = self.encoder(**results)
        if rep_only:
            return outs['representation']
        results.update(outs)
        head_inputs = dict(results)
        head_inputs.update(refine_visibility)
        if occ_only and hasattr(self.head, "forward_occ"):
            outs = self.head.forward_occ(**head_inputs)
        else:
            outs = self.head(**head_inputs)
        results.update(outs)
        return results
