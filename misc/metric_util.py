import numpy as np
from mmengine import MMLogger
import torch.distributed as dist
import torch


logger = MMLogger.get_instance('gf3d')


OCC3D_OFFICIAL_LABEL_STR = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
]

OCC3D_OFFICIAL_CLASS_INDICES = list(range(17))


def _dist_ready():
    return dist.is_available() and dist.is_initialized()


def _is_main_process():
    return (not _dist_ready()) or dist.get_rank() == 0


class MeanIoU:

    def __init__(self,
                 class_indices,
                #  ignore_label: int,
                 empty_label,
                 label_str,
                 use_mask=False,
                 dataset_empty_label=17,
                 filter_minmax=True,
                 name='none',
                 training=True):
        # Route-A / Occ3D official evaluation: 17 semantic classes + empty/free.
        # Some older SurroundOcc/GSF configs pass 16 labels without the Occ3D
        # "others" class.  When empty_label/dataset_empty_label is 17, the
        # tensor label space is official Occ3D, so silently promote the metric to
        # 17-class evaluation to stay comparable with Occ3D papers.
        class_indices = list(class_indices)
        label_str = list(label_str)
        looks_like_old_occ16 = (
            int(empty_label) == 17
            and len(class_indices) == 16
            and len(label_str) == 16
            and len(label_str) > 0
            and str(label_str[0]).lower() == 'barrier'
        )
        if looks_like_old_occ16:
            class_indices = OCC3D_OFFICIAL_CLASS_INDICES
            label_str = OCC3D_OFFICIAL_LABEL_STR

        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        # self.ignore_label = ignore_label
        self.empty_label = empty_label
        self.dataset_empty_label = dataset_empty_label
        self.label_str = label_str
        self.use_mask = use_mask
        self.filter_minmax = filter_minmax
        self.name = name
        self.training = training
        # 中文注释：本地保存上一轮 eval 的逐类指标，供 train.py 写 TensorBoard 和 eval 文件。
        self.last_ious = []
        self.last_precs = []
        self.last_recas = []
        self.last_miou = 0.0
        self.last_occ_iou = 0.0

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes + 1).cuda()
        self.total_correct = torch.zeros(self.num_classes + 1).cuda()
        self.total_positive = torch.zeros(self.num_classes + 1).cuda()

    def _after_step(self, outputs, targets, mask=None):
        # outputs = outputs[targets != self.ignore_label]
        # targets = targets[targets != self.ignore_label]
        if not isinstance(targets, (torch.Tensor, np.ndarray)):  # for occ3d but we don't use it
            assert mask is None
            labels = torch.from_numpy(targets['semantics']).cuda()
            masks = torch.from_numpy(targets['mask_camera']).bool().cuda()
            targets = labels
            targets[targets == self.dataset_empty_label] = self.empty_label
            if self.filter_minmax:
                max_z = (targets != self.empty_label).nonzero()[:, 2].max()
                min_z = (targets != self.empty_label).nonzero()[:, 2].min()
                outputs[..., (max_z + 1):] = self.empty_label
                outputs[..., :min_z] = self.empty_label
            if self.use_mask:
                outputs = outputs[masks]
                targets = targets[masks]
        else:
            if mask is not None:
                outputs = outputs[mask]
                targets = targets[mask]

        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += torch.sum(targets == c).item()
            self.total_correct[i] += torch.sum((targets == c) & (outputs == c)).item()
            self.total_positive[i] += torch.sum(outputs == c).item()

        self.total_seen[-1] += torch.sum(targets != self.empty_label).item()
        self.total_correct[-1] += torch.sum((targets != self.empty_label) & (outputs != self.empty_label)).item()
        self.total_positive[-1] += torch.sum(outputs != self.empty_label).item()

    def _after_epoch(self):
        if _dist_ready():
            dist.all_reduce(self.total_seen)
            dist.all_reduce(self.total_correct)
            dist.all_reduce(self.total_positive)
            dist.barrier()

        ious = []
        precs = []
        recas = []

        for i in range(self.num_classes):
            if self.total_positive[i] == 0:
                precs.append(0.0)
            else:
                cur_prec = self.total_correct[i] / self.total_positive[i]
                precs.append(float(cur_prec.item()))

            if self.total_seen[i] == 0:
                ious.append(1.0)
                recas.append(1.0)
            else:
                cur_iou = self.total_correct[i] / (
                    self.total_seen[i] + self.total_positive[i] - self.total_correct[i]
                )
                cur_reca = self.total_correct[i] / self.total_seen[i]
                ious.append(float(cur_iou.item()))
                recas.append(float(cur_reca.item()))

        miou = float(np.mean(ious))

        # 中文注释：彻底移除外部实验平台记录；只在主进程写本地日志，避免 DDP 多进程重复输出。
        if _is_main_process():
            logger.info(f'Validation per class iou {self.name}:')
            for iou, prec, reca, label_str in zip(ious, precs, recas, self.label_str):
                logger.info('%s : %.2f%%, %.2f, %.2f' % (label_str, iou * 100, prec, reca))

            logger.info(self.total_seen.int())
            logger.info(self.total_correct.int())
            logger.info(self.total_positive.int())

        occ_iou = self.total_correct[-1] / (
            self.total_seen[-1] + self.total_positive[-1] - self.total_correct[-1]
        )
        occ_iou = float(occ_iou.item())

        self.last_ious = list(ious)
        self.last_precs = list(precs)
        self.last_recas = list(recas)
        self.last_miou = miou * 100.0
        self.last_occ_iou = occ_iou * 100.0

        return self.last_miou, self.last_occ_iou
