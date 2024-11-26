from seg.utils import Config

cfg = dict(a='1', b=2, kwargs=dict(c='ccc', b=[1, 2, 3]))
cfg = Config(cfg)
print(cfg.a)
print(cfg.b)
print(cfg.kwargs)