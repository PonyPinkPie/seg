import math
import torch.nn as nn
import torch.nn.functional as F
from seg.models.registry import CONVOLUTIONS

CONVOLUTIONS.register_module('Conv1d', module=nn.Conv1d)
CONVOLUTIONS.register_module('Conv2d', module=nn.Conv2d)
CONVOLUTIONS.register_module('Conv3d', module=nn.Conv3d)
CONVOLUTIONS.register_module('Conv', module=nn.Conv2d)

def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Parameters
    ----------
    cfg : dict, optional
        The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.
    args : argument list
        Arguments passed to the `__init__` method of the corresponding conv layer.
    kwargs : keyword arguments
        Keyword arguments passed to the `__init__` method of the corresponding conv layer.

    Returns
    -------
    conv_layer : nn.Module
        Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in CONVOLUTIONS:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        conv_layer = CONVOLUTIONS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


@CONVOLUTIONS.register_module()
class Conv2dAdaptivePadding(nn.Conv2d):
    """Implementation of 2D convolution in tensorflow with `padding` as "same",
    which applies padding to input (if needed) so that input image gets fully
    covered by filter and stride you specified. For stride 1, this will ensure
    that output image size is same as input. For stride of 2, output dimensions
    will be half, for example.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements.
            Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
                         dilation, groups, bias)

    def forward(self, x):
        img_h, img_w = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        pad_h = (
            max((output_h - 1) * self.stride[0] +
                (kernel_h - 1) * self.dilation[0] + 1 - img_h, 0))
        pad_w = (
            max((output_w - 1) * self.stride[1] +
                (kernel_w - 1) * self.dilation[1] + 1 - img_w, 0))
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [
                pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2
            ])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
