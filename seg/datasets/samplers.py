import torch
from torch.utils.data import Sampler
from torch.utils.data import DistributedSampler
from .registry import SAMPLERS

from seg.utils.distribute import get_dist_info


@SAMPLERS.register_module()
class DefaultSampler(Sampler):
    """Default non-distributed sampler."""

    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            return iter(torch.randperm(len(self.dataset)).tolist())
        else:
            return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)


@SAMPLERS.register_module()
class DistributSampler(DistributedSampler):
    """Default distributed sampler."""

    def __init__(self, dataset, shuffle=True):
        rank, num_replicas = get_dist_info()
        super().__init__(dataset, num_replicas, rank, shuffle)