import torch.nn as nn
from seg.models.registry import PADDINGS

PADDINGS.register_module('zero', module=nn.ZeroPad2d)
PADDINGS.register_module('reflect', module=nn.ReflectionPad2d)
PADDINGS.register_module('replicate', module=nn.ReplicationPad2d)


def build_padding_layer(cfg, *args, **kwargs):
    """Build padding layer.

    Parameters
    ----------
    cfg : dict, optional
        The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns
    -------
    padding_layer : nn.Module
        Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type not in PADDINGS:
        raise KeyError(f'Unrecognized padding type {padding_type}.')
    else:
        padding_layer = PADDINGS.get(padding_type)

    layer = padding_layer(*args, **kwargs, **cfg_)

    return layer