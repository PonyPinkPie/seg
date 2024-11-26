from seg.utils.registry import Registry, build_from_cfg

TRANSFORMS = Registry('transform')


def build_transform(cfg):
    return build_from_cfg(cfg, TRANSFORMS)
