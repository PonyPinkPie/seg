from seg.utils.registry import Registry, build_from_cfg

# _bricks_
ACTIVATIONS = Registry('activations')
BLOCKS = Registry('blocks')
CONVOLUTIONS = Registry('convolution')
LAYERS = Registry('layers')
NORMS = Registry('norms')
PADDINGS = Registry('paddings')
UPSAMPLES = Registry('upsamples')

# backbones heads necks
BACKBONES = Registry('backbones')
HEADS = Registry('heads')
NECKS = Registry('necks')


def build_activation(cfg, default_args=None):
    return build_from_cfg(cfg, ACTIVATIONS, default_args)


def build_block(cfg, default_args=None):
    return build_from_cfg(cfg, BLOCKS, default_args)

def build_convolution(cfg, default_args=None):
    return build_from_cfg(cfg, CONVOLUTIONS, default_args)

def build_layer(cfg, default_args=None):
    return build_from_cfg(cfg, LAYERS, default_args)


def build_norm(cfg, default_args=None):
    return build_from_cfg(cfg, NORMS, default_args)


def build_padding(cfg, default_args=None):
    return build_from_cfg(cfg, PADDINGS, default_args)


def build_upsample(cfg, default_args=None):
    return build_from_cfg(cfg, UPSAMPLES, default_args)


def build_head(cfg, default_args=None):
    return build_from_cfg(cfg, HEADS, default_args)


def build_neck(cfg, default_args=None):
    return build_from_cfg(cfg, NECKS, default_args)


def build_backbone(cfg, default_args=None):
    return build_from_cfg(cfg, BACKBONES, default_args)
