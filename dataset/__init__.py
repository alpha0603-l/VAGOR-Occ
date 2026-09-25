from mmengine.registry import Registry

OPENOCC_DATASET = Registry('openocc_dataset')
OPENOCC_DATAWRAPPER = Registry('openocc_datawrapper')
OPENOCC_TRANSFORMS = Registry('openocc_transforms')

from .dataset import NuScenesDataset
from .dataset_occ3d import NuScenesOcc3DDataset
from .dataset_wildocc import WildOccDataset
from .transform_3d import *
from .sampler import CustomDistributedSampler
from .utils import custom_collate_fn_temporal
from .loading_utils import *

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader


def _build_dataloader_runtime_kwargs(loader_cfg):
    """Build DataLoader runtime kwargs from config.

    Notes:
    - pin_memory / prefetch_factor / persistent_workers only affect data loading
      and CPU->GPU transfer behavior; they do not change sample contents,
      model forward, loss, or gradients.
    - prefetch_factor and persistent_workers are valid only when num_workers > 0.
      Passing them with num_workers=0 can raise errors in some PyTorch versions.
    """
    num_workers = int(loader_cfg.get("num_workers", 0))
    kwargs = dict(
        num_workers=num_workers,
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
    )

    if num_workers > 0:
        prefetch_factor = loader_cfg.get("prefetch_factor", None)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = bool(loader_cfg.get("persistent_workers", False))

    return kwargs


def get_dataloader(
    train_dataset_config,
    val_dataset_config,
    train_loader,
    val_loader,
    dist=False,
    iter_resume=False,
    train_sampler_config=dict(shuffle=True, drop_last=True),
    val_sampler_config=dict(shuffle=False, drop_last=False),
    val_only=False,
):
    if val_only:
        val_wrapper = OPENOCC_DATASET.build(val_dataset_config)

        val_sampler = None
        if dist:
            val_sampler = DistributedSampler(val_wrapper, **val_sampler_config)

        val_dataset_loader = DataLoader(
            dataset=val_wrapper,
            batch_size=val_loader["batch_size"],
            collate_fn=custom_collate_fn_temporal,
            shuffle=False,
            sampler=val_sampler,
            **_build_dataloader_runtime_kwargs(val_loader),
        )
        return None, val_dataset_loader

    train_wrapper = OPENOCC_DATASET.build(train_dataset_config)
    val_wrapper = OPENOCC_DATASET.build(val_dataset_config)

    train_sampler = val_sampler = None
    if dist:
        if iter_resume:
            train_sampler = CustomDistributedSampler(train_wrapper, **train_sampler_config)
        else:
            train_sampler = DistributedSampler(train_wrapper, **train_sampler_config)
        val_sampler = DistributedSampler(val_wrapper, **val_sampler_config)

    train_dataset_loader = DataLoader(
        dataset=train_wrapper,
        batch_size=train_loader["batch_size"],
        collate_fn=custom_collate_fn_temporal,
        shuffle=False if dist else train_loader["shuffle"],
        sampler=train_sampler,
        **_build_dataloader_runtime_kwargs(train_loader),
    )
    val_dataset_loader = DataLoader(
        dataset=val_wrapper,
        batch_size=val_loader["batch_size"],
        collate_fn=custom_collate_fn_temporal,
        shuffle=False,
        sampler=val_sampler,
        **_build_dataloader_runtime_kwargs(val_loader),
    )
    return train_dataset_loader, val_dataset_loader
