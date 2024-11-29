import warnings
import math
import torch.nn as nn
import torch.nn.functional as F
from seg.models._base_ import PADDINGS

PADDINGS.register_module('zero', module=nn.ZeroPad2d)
PADDINGS.register_module('reflect', module=nn.ReflectionPad2d)
PADDINGS.register_module('replicate', module=nn.ReplicationPad2d)