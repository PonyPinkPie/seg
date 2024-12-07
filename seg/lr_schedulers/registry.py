from seg.utils.registry import Registry, build_from_cfg

LR_SCHEDULERS = Registry('LR_SCHEDULERS')

def build_lr_scheduler(cfg, default_args=None):
    return build_from_cfg(cfg, LR_SCHEDULERS, default_args)