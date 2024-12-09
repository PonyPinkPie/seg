from .inference_runner import InferenceRunner

from seg.dataloaders import build_dataloader
from seg.datasets import build_dataset
from seg.optimizers import build_optimizer
from seg.lr_schedulers import build_lr_scheduler
from collections.abc import Iterable
import numpy as np


class TrainRunner(InferenceRunner):
    def __init__(self, train_cfg, inference_cfg, base_cfg=None):
        super().__init__(inference_cfg, base_cfg)

        self.train_dataloader = self._build_dataloader(train_cfg['train'])
        self.valid_dataloader = self._build_dataloader(train_cfg['valid'])

        self.optimizer = self._build_optimizer(train_cfg['optimizer'])
        self.lr_scheduler = self._build_lr_scheduler(train_cfg['lr_scheduler'])
        self.max_epochs = train_cfg['max_epochs']
        self.log_interval = train_cfg.get('log_interval', 10)
        self.train_valid_interval = train_cfg.get('train_valid_interval', 1)

    def _build_dataloader(self, cfg):
        transform = self._build_transform(cfg['transform'])
        dataset = build_dataset(cfg['dataset'], dict(transform=transform, logger=self.logger))
        shuffle = cfg['dataloader'].get('shuffle', False)
        dataloader = build_dataloader(
            cfg['dataloader'], self.gpu_num, self.distribute, dict(dataset=dataset, shuffle=shuffle)
        )
        return dataloader

    def _build_optimizer(self, cfg):
        return build_optimizer(cfg, dict(params=self.model.parameters()))
        # return build_optimizer(cfg, dict(params=[{'params':self.model.parameters(), 'lr':0.01}]))

    def _build_lr_scheduler(self, cfg):
        return build_lr_scheduler(cfg, dict(optimizer=self.optimizer))

    @property
    def epoch(self):
        """int: Current epoch."""
        return self.lr_scheduler.last_epoch

    @epoch.setter
    def epoch(self, val):
        """int: Current epoch."""
        self.lr_scheduler.last_epoch = val

    @property
    def lr(self):
        lr = [x['lr'] for x in self.optimizer.param_groups]
        return np.array(lr)

    @lr.setter
    def lr(self, val):
        for idx, param in enumerate(self.optimizer.param_groups):
            if isinstance(val, Iterable):
                param['lr'] = val[idx]
            else:
                param['lr'] = val

    def _train(self):
        self.model.train()
        self.logger.info(f'Epoch {self.epoch + 1}, start training Lr: {self.lr[0]:.6f}')

        self.lr_scheduler.step()
        pass

    def _valid(self):
        self.model.eval()
        pass

    def __call__(self, *args, **kwargs):
        for _ in range(self.epoch, self.max_epochs):
            if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(self.epoch)

            self._train()

            if self.epoch % self.train_valid_interval == 0:
                self._valid()
