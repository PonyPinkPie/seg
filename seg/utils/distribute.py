import os
import torch
import torch.distributed as dist

def cuda_is_available():
    return torch.cuda.is_available()

def devices_count():
    return torch.cuda.device_count()

def init_dist_pytorch(**kwargs):
    # rank = kwargs.get('rank', 0)
    # num_gpus = torch.cuda.device_count()
    # torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(**kwargs)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def reduce_value(value, average=True):
    world_size = dist.get_world_size() if is_dist_avail_and_initialized() else 1
    if world_size < 2:
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value