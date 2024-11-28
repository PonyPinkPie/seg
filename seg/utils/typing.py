from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from .easy_dict import EasyDict

# TODO: Need to avoid circular import with assigner and sampler
# Type hint of config data
ConfigType = Union[EasyDict, dict]
OptConfigType = Optional[ConfigType]
# Type hint of one or more config data
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]
InitConfigType = Union[Dict, List[Dict]]
OptInitConfigType = Optional[InitConfigType]


RangeType = Sequence[Tuple[int, int]]

# Data
OptTensor = Optional[torch.Tensor]


# Visualization
ColorType = Union[str, Tuple, List[str], List[Tuple]]

ArrayLike = 'ArrayLike'
ForwardInputs = Tuple[Dict[str, Union[Tensor, str, int]], Tensor]
NoiseVar = Union[Tensor, Callable, None]
LabelVar = Union[Tensor, Callable, List[int], None]