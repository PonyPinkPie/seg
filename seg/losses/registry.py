from seg.utils.registry import Registry, build_from_cfg

LOSSES = Registry('losses')


def build_loss(cfg, default_args=None):
    return build_from_cfg(cfg, LOSSES, default_args=default_args)
