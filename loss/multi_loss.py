import torch, torch.nn as nn
from . import OPENOCC_LOSS
from misc.tb_wrapper import WrappedTBWriter
if 'gf3d' in WrappedTBWriter._instance_dict:
    writer = WrappedTBWriter.get_instance('gf3d')
else:
    writer = None

@OPENOCC_LOSS.register_module()
class MultiLoss(nn.Module):

    def __init__(self, loss_cfgs):
        super().__init__()
        
        assert isinstance(loss_cfgs, list)
        self.num_losses = len(loss_cfgs)
        
        losses = []
        for loss_cfg in loss_cfgs:
            losses.append(OPENOCC_LOSS.build(loss_cfg))
        self.losses = nn.ModuleList(losses)
        self.iter_counter = 0

    def forward(self, inputs):

        tot_loss = None
        loss_tensors = []
        loss_names = []
        name_counts = {}

        for loss_func in self.losses:
            loss = loss_func(inputs)
            if not hasattr(loss, 'detach'):
                raise TypeError(
                    f'{loss_func.__class__.__name__} returned {type(loss).__name__}; '
                    'every loss must return a scalar tensor.'
                )
            tot_loss = loss if tot_loss is None else (tot_loss + loss)
            loss_tensors.append(loss.detach().float().reshape(()))

            base_name = loss_func.__class__.__name__
            count = name_counts.get(base_name, 0)
            name_counts[base_name] = count + 1
            loss_names.append(base_name if count == 0 else f'{base_name}_{count + 1}')

        if tot_loss is None:
            raise RuntimeError('MultiLoss requires at least one configured loss.')

        # One synchronization for all scalar logging values, instead of one
        # .item() call per loss plus duplicate writer .item() calls.
        detached_values = torch.stack(loss_tensors)
        host_values = detached_values.cpu().tolist()
        loss_dict = {name: value for name, value in zip(loss_names, host_values)}

        if writer and self.iter_counter % 10 == 0:
            for name, value in loss_dict.items():
                writer.add_scalar(f'loss/{name}', value, self.iter_counter)
            writer.add_scalar('loss/total', float(sum(host_values)), self.iter_counter)

        self.iter_counter += 1
        return tot_loss, loss_dict
