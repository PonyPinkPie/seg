import torch
from seg.utils import Registry, build_from_cfg

OPTIMIZERS = Registry('optimizers')


def build_optimizer(cfg, default_args=None):
    return build_from_cfg(cfg, OPTIMIZERS, default_args=default_args)
