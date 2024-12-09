from seg.utils.registry import Registry, build_from_cfg
from torch.utils.data import DataLoader

DATALOADERS = Registry('dataloader')
DATALOADERS.register_module('DataLoader', module=DataLoader)


def build_dataloader(cfg, num_gpus, distributed, default_args=None):
    cfg_ = cfg.copy()
    samples_per_gpu = cfg_.pop('samples_per_gpu')
    workers_per_gpu = cfg_.pop('workers_per_gpu')
    if distributed:
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = samples_per_gpu * num_gpus
        num_workers = workers_per_gpu * num_gpus

    cfg_.update({'batch_size': batch_size, 'num_workers': num_workers})

    dataloader = build_from_cfg(cfg_, DATALOADERS, default_args)

    return dataloader
