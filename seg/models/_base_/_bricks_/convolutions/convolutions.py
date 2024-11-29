import warnings
import math
import torch.nn as nn
import torch.nn.functional as F
from seg.models._base_ import (
    CONVOLUTIONS,
    build_padding, build_convolution, build_norm, build_activation,
    kaiming_init, constant_init
)

CONVOLUTIONS.register_module('Conv1d', module=nn.Conv1d)
CONVOLUTIONS.register_module('Conv2d', module=nn.Conv2d)
CONVOLUTIONS.register_module('Conv3d', module=nn.Conv3d)
CONVOLUTIONS.register_module('Conv', module=nn.Conv2d)


@CONVOLUTIONS.register_module()
class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Parameters
    ----------
    dtype : str, optional
        Whether to store matrix in C- or Fortran-contiguous order,
        default is 'C'.
    in_channels : int
        Same as nn.Conv2d.
    out_channels : int
        Same as nn.Conv2d.
    kernel_size : {int, tuple[int]}
        Same as nn.Conv2d.
    stride : {int, tuple[int]}
        Same as nn.Conv2d.
    padding : {int, tuple[int]}
        Same as nn.Conv2d.
    dilation : {int, tuple[int]}
        Same as nn.Conv2d.
    groups : int
        Same as nn.Conv2d.
    bias : {bool, str}
        If specified as `auto`, it will be decided by the
        norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
        False. Default: "auto".
    conv_cfg : dict
        Config dict for convolution layer. Default: None,
        which means using conv2d.
    norm_cfg : dict
        Config dict for normalization layer. Default: None.
    act_cfg : dict
        Config dict for activation layer.
        Default: dict(type='ReLU').
    inplace : bool
        Whether to use inplace mode for activation.
        Default: True.
    with_spectral_norm : bool
        Whether use spectral norm in conv module.
        Default: False.
    padding_mode : str
        If the `padding_mode` has not been supported by
        current `Conv2d` in PyTorch, we will use our own padding layer
        instead. Currently, we support ['zeros', 'circular'] with official
        implementation and ['reflect'] with our own implementation.
        Default: 'zeros'.
    order : tuple[str]
        The order of conv/norm/activation layers. It is a
        sequence of "conv", "norm" and "act". Common examples are
        ("conv", "norm", "act") and ("act", "conv", "norm").
        Default: ('conv', 'norm', 'act').
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act'), **kwargs):
        super(ConvModule, self).__init__()
        # assert conv_cfg is None or isinstance(conv_cfg, dict)
        # assert norm_cfg is None or isinstance(norm_cfg, dict)
        # assert act_cfg is None or isinstance(act_cfg, dict)
        if isinstance(conv_cfg, str):
            conv_cfg = dict(type=conv_cfg)
        if isinstance(norm_cfg, str):
            norm_cfg = dict(type=norm_cfg)
        if isinstance(act_cfg, str):
            act_cfg = dict(type=act_cfg)

        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding(pad_cfg, padding)

        conv_padding = 0 if self.with_explicit_padding else padding
        self.conv = build_convolution(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        if self.with_norm:
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            if act_cfg_['type'] in ['ReLU', 'LeakyReLU', 'RReLU', 'ReLU6', 'ELU']:
                act_cfg_.setdefault('inplace', inplace)

            elif act_cfg_['type'] in ['MetaAconC', 'AconC', 'FReLU', 'xUnitD', 'xUnitS', 'KAF']:
                act_cfg_.setdefault('c1', self.out_channels)

            self.activate = build_activation(act_cfg_)

        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x


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
