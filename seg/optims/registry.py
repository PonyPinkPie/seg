import torch
from seg.utils import Registry, build_from_cfg

OPTIMIZERS = Registry('optimizers')

for module_name in dir(torch.optim):
    if module_name.startswith('_') or module_name.islower():
        continue
    optim = getattr(torch.optim, module_name)
    OPTIMIZERS.register_module(module_name, module=optim)
    OPTIMIZERS.register_module(module_name.lower(), module=optim, force=True)


def build_optimizer(cfg, default_args=None):
    return build_from_cfg(cfg, OPTIMIZERS, default_args=default_args)
