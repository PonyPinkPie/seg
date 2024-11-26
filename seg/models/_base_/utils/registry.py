from seg.utils.registry import Registry, build_from_cfg, build_from_registry, build_from_module

BACKBONES = Registry('backbone')


def build_backbone(cfg, default_args=None):
    return build_from_cfg(cfg, BACKBONES, default_args)


ACTIVATIONS = Registry('activation')


def build_activation(cfg, default_args=None):
    return build_from_cfg(cfg, ACTIVATIONS, default_args)


BLOCKS = Registry('block')


def build_block(cfg, default_args=None):
    return build_from_cfg(cfg, BLOCKS, default_args)


LAYERS = Registry('layer')


def build_layer(cfg, default_args=None):
    return build_from_cfg(cfg, LAYERS, default_args)


NORMS = Registry('norm')


def build_norm(cfg, default_args=None):
    return build_from_cfg(cfg, NORMS, default_args)


HEADS = Registry('head')


def build_head(cfg, default_args=None):
    return build_from_cfg(cfg, HEADS, default_args)


NECKS = Registry('neck')


def build_neck(cfg, default_args=None):
    return build_from_cfg(cfg, NECKS, default_args)
