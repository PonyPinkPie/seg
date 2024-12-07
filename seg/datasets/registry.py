from seg.utils.registry import Registry, build_from_cfg

DATASETS = Registry('datasets')
SAMPLERS = Registry('samplers')


def build_dataset(cfg, default_args=None):
    return build_from_cfg(cfg, DATASETS, default_args=default_args)

def build_sampler(cfg, default_args=None):
    return build_from_cfg(cfg, SAMPLERS, default_args)