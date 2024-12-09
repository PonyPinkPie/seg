import torch
from seg.utils.registry import Registry, build_from_cfg

LR_SCHEDULERS = Registry('LR_SCHEDULERS')


for module_name in dir(torch.optim.lr_scheduler):
    if 'LR' not in module_name or module_name.startswith('_'):
        continue
    optim = getattr(torch.optim.lr_scheduler, module_name)
    LR_SCHEDULERS.register_module(module_name, module=optim)
    LR_SCHEDULERS.register_module(module_name.lower(), module=optim)


def build_lr_scheduler(cfg, default_args=None):
    return build_from_cfg(cfg, LR_SCHEDULERS, default_args=default_args)
