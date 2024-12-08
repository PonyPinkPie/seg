import os

import torch
from torch.backends import cudnn
import random
import numpy as np

from seg.utils.distribute import init_dist_pytorch, get_dist_info, cuda_is_available, devices_count
from seg.loggers import build_logger


class BaseRunner(object):
    def __init__(self, cfg):
        self.workdir = cfg.get('workdir', 'workdir')
        os.makedirs(self.workdir, exist_ok=True)
        logger_cfg = cfg.get('logger')
        if logger_cfg is None:
            logger_cfg = dict(
                handlers=(
                    dict(type='StreamHandler', level='INFO'),
                    dict(type='FileHandler', level='INFO'),
                )
            )
        self.logger = self._build_logger(logger_cfg)

        self.distribute = cfg.get('distribute', False)

        self.gpu_num = devices_count()
        self.ues_gpu = cuda_is_available()

        if self.distribute and self.ues_gpu:
            init_dist_pytorch(**cfg.dist_params)

        self.rank, self.world_size = get_dist_info()

        self._set_cudnn(
            cfg.get('cudnn_deterministic', False),
            cfg.get('cudnn_benchmark', False)
        )

        self._set_seed(cfg.get('seed', 0))

        # self.metrics = self._build_metrics(cfg.get('metrics', {}))

    def _build_logger(self, cfg):
        return build_logger(cfg, dict(workdir=self.workdir))

    def _set_cudnn(self, deterministic, benchmark):
        self.logger.info('Set cudnn deterministic {}'.format(deterministic))
        cudnn.deterministic = deterministic

        self.logger.info('Set cudnn benchmark {}'.format(benchmark))
        cudnn.benchmark = benchmark

    def _set_seed(self, seed):
        self.logger.info('Set seed {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    # def _build_metrics(self, cfg):
    #     return build_metrics(cfg)