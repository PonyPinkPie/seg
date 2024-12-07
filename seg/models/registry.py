from seg.utils.registry import Registry, build_from_cfg

MODELS = Registry('models')


def build_model(cfg, default_args=None):
    return build_from_cfg(cfg, MODELS, default_args)