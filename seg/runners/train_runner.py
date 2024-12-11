import time

import torch

from .inference_runner import InferenceRunner

from seg.dataloaders import build_dataloader
from seg.datasets import build_dataset
from seg.optimizers import build_optimizer
from seg.lr_schedulers import build_lr_scheduler
from collections.abc import Iterable
import numpy as np
from seg.utils.gpu import get_gpu_memroy
import datetime
from seg.metrics.common import calculate_metric_for_more, calculate_metric_for_one, parse_seg_metrics, parse_seg_metrics_to_table


class TrainRunner(InferenceRunner):
    def __init__(self, train_cfg, inference_cfg, base_cfg=None):
        super().__init__(inference_cfg, base_cfg)

        self.train_dataloader = self._build_dataloader(train_cfg['train'])

        self.valid_dataloader = self._build_dataloader(train_cfg['valid'])
        self.shape_labels = self.valid_dataloader.dataset.shape_labels
        self.class2label = self.valid_dataloader.dataset.class2label
        self.label2class = self.valid_dataloader.dataset.label2class

        self.optimizer = self._build_optimizer(train_cfg['optimizer'])
        self.lr_scheduler = self._build_lr_scheduler(train_cfg['lr_scheduler'])
        self.max_epochs = train_cfg['max_epochs']
        self.iters = len(self.train_dataloader)
        self.log_interval = train_cfg.get('log_interval', self.iters // 10)
        self.train_valid_interval = train_cfg.get('train_valid_interval', 1)

        self.best_value = 0

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

    def echo_info(self):
        iter_info = f"{self.iter}/{self.iters}".ljust(8)
        time_info = f"{(self.used_time * self.log_interval):.2f} sec".ljust(10)
        loss_info = f"{self.losses['loss']:.4f}".ljust(10)
        lr_info = f"{self.lr[0]:.6f}".ljust(10)
        gpu_info = f"{(float(get_gpu_memroy([self.image.device.index])[0]['memory_used']) / 1024):.2f} GB".ljust(8)
        self.logger.info(f"Step:{iter_info} Time:{time_info} Loss:{loss_info} Lr:{lr_info} GPU:{gpu_info}")

    def _train(self):
        self.iter = 0
        self.model.train()
        self.logger.info(f'Epoch {self.epoch + 1}/{self.max_epochs}')
        for batch_idx, batch_data in enumerate(self.train_dataloader):
            t1 = time.time()
            self.optimizer.zero_grad()
            self.image = batch_data['image'].cuda()
            self.mask = batch_data['mask'].cuda()
            self.losses = self.model(self.image, return_metrics=True, ground_truth=self.mask)
            self.losses['loss'].backward()
            self.optimizer.step()
            self.iter += 1
            self.used_time = time.time() - t1
            if batch_idx % self.log_interval == 0 and batch_idx // self.log_interval > 0:
                self.echo_info()

        self.lr_scheduler.step()
        pass

    def _valid(self):
        self.logger.info(f"Start Valid")
        self.model.eval()

        all_seg_metrics_dict = dict()
        for i in range(len(self.label2class)):
            all_seg_metrics_dict[self.label2class[i]] = []

        with torch.no_grad():
            for idx, batch_data in enumerate(self.valid_dataloader):
                self.image = batch_data['image'].cuda()
                self.mask = batch_data['mask'].numpy().astype(np.uint8)
                probs = self.model(self.image).cpu().numpy()
                if len(self.class2label) > 1:
                    # 多分类评估
                    y_probs = np.transpose(probs, (0, 2, 3, 1))  # [B C H W] -> [B H W C]
                    y_preds = np.argmax(y_probs, axis=-1)
                    y_trues = self.mask
                    for class_idx in self.class2label.values():
                        metrics = []
                        for y_pred, y_prob, y_true in zip(y_preds, y_probs, y_trues):
                            mask = (y_true == class_idx).astype(np.uint8)
                            if mask.sum() == 0:
                                # GT 不存在， 计算召回是都为0，因此不参与计算
                                continue
                            prob = y_prob[..., class_idx]
                            pred = (y_pred == mask).astype(np.uint8)
                            metric, threshold_list = calculate_metric_for_more(prob, pred, mask)
                            metrics.append(metric)
                        all_seg_metrics_dict[self.label2class[class_idx]] += metrics
                else:
                    if 'background' not in all_seg_metrics_dict:
                        all_seg_metrics_dict['background'] = []

                    y_probs = probs
                    y_trues = self.mask
                    for y_prob, y_true in zip(y_probs, y_trues):
                        metric, threshold_list = calculate_metric_for_one(y_prob, y_true)
                        metrics.append(metric)
                        all_seg_metrics_dict["foreground"] += [metric]

        curr_metrics, best_index = parse_seg_metrics(all_seg_metrics_dict)

        curr_mean_f1 = curr_metrics[-1, 5]
        curr_mean_iou = curr_metrics[-1, 6]
        if self.best_value < curr_mean_f1:
            self.best_value = curr_mean_f1
            self.best_metrics = curr_metrics
            threshold = threshold_list[best_index]

        parse_seg_metrics_to_table(curr_metrics, self.best_metrics[-1, :], self.label2class, self.logger)

    def __call__(self, *args, **kwargs):

        # self._valid()

        for _ in range(self.epoch, self.max_epochs):
            if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(self.epoch)
            t1 = time.time()
            self._train()

            if self.epoch % self.train_valid_interval == 0:
                self._valid()
            train_valid_time = time.time() - t1

            eta_string = str(datetime.timedelta(seconds=int(train_valid_time * (self.max_epochs - self.epoch))))
            self.logger.info(f"ETA:{eta_string}")