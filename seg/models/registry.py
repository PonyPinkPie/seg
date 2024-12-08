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

# segmentations
SEGMENTATIONS = Registry('segmentations')

def build_backbone(cfg, default_args=None):
    return build_from_cfg(cfg, BACKBONES, default_args)


def build_head(cfg, default_args=None):
    return build_from_cfg(cfg, HEADS, default_args)


def build_neck(cfg, default_args=None):
    return build_from_cfg(cfg, NECKS, default_args)


def build_segmentation(cfg, default_args=None):
    return build_from_cfg(cfg, SEGMENTATIONS, default_args)
