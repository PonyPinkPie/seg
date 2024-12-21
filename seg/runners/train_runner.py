import os
import time
import traceback
from os.path import join as opj
import torch

from .inference_runner import InferenceRunner

from seg.dataloaders import build_dataloader
from seg.datasets import build_dataset
from seg.optimizers import build_optimizer
from seg.lr_schedulers import build_lr_scheduler
from collections.abc import Iterable
import numpy as np
from seg.utils import get_gpu_memroy, save_json, save_checkpoint, load_checkpoint
import datetime
from seg.metrics.common import calculate_metric_for_more, calculate_metric_for_one, parse_seg_metrics, \
    parse_seg_metrics_to_table
from seg.export.converters import TRTModel, torch2onnx
from seg.statistics.statistics import ClsStatistics


class TrainRunner(InferenceRunner):
    def __init__(self, export_cfg, train_cfg, inference_cfg, base_cfg=None):
        super().__init__(inference_cfg, base_cfg)
        self.export_cfg, self.train_cfg, self.inference_cfg, self.base_cfg = \
            export_cfg.copy(), train_cfg.copy(), inference_cfg.copy(), base_cfg.copy()

        imageNetMeanStd = train_cfg.get('ImageNetMeanStd', True)
        if not imageNetMeanStd:
            root = opj(self.train_cfg['train']['dataset']['root'], 'train')
            statistics = ClsStatistics(root, self.logger)
            mean, std = statistics.mean, statistics.std
            mean_std = dict(mean=mean, std=std)

            self.inference_cfg['transform'][-2].update(mean_std)
            self.train_cfg['train']['transform'][-2].update(mean_std)
            self.train_cfg['valid']['transform'][-2].update(mean_std)

        cfg = {'common': self.base_cfg, 'inference': self.inference_cfg, 'data': self.train_cfg,
               'export': self.export_cfg}
        jp = opj(self.workdir, self.timestamp + '.json')
        save_json(cfg, jp)

        self.train_dataloader = self._build_dataloader(self.train_cfg['train'])

        self.valid_dataloader = self._build_dataloader(self.train_cfg['valid'])
        self.shape_labels = self.valid_dataloader.dataset.shape_labels
        self.class2label = self.valid_dataloader.dataset.class2label
        self.label2class = self.valid_dataloader.dataset.label2class

        self.optimizer = self._build_optimizer(self.train_cfg['optimizer'])
        self.lr_scheduler = self._build_lr_scheduler(self.train_cfg['lr_scheduler'])
        self.max_epochs = self.train_cfg['max_epochs']
        self.iters = len(self.train_dataloader)
        self.log_interval = self.train_cfg.get('log_interval', self.iters // 10)
        self.train_valid_interval = self.train_cfg.get('train_valid_interval', 1)

        self.best_value = 0
        self.best_pth_path = opj(self.workdir, self.timestamp + '.pth')
        self.select_metric = train_cfg.get('select_metric', 'f1').lower()
        assert self.select_metric in ['f1', 'iou', 'b_f1_iou', 'b_p_r_iou'], \
            f"select_metric must be one of 'f1', 'iou', 'b_f1_iou', 'b_p_r_iou', but got {self.select_metric}"


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
        line = '-' * 40
        self.logger.info(f'{line} Train Epoch {self.epoch + 1}/{self.max_epochs} {line}')
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

    def _valid(self, is_valid=True):
        line = '-' * 40
        if is_valid:
            self.logger.info(f"{line} Valid Epoch {self.epoch}/{self.max_epochs} {line}")
        else:
            self.logger.info(f"{line} Valid Infer Using TRT Model {line}")
        self.model.eval()

        all_seg_metrics_dict = dict()
        for i in range(len(self.label2class)):
            all_seg_metrics_dict[self.label2class[i]] = []

        valid_iters = len(self.valid_dataloader)
        interval = valid_iters // 10
        with torch.no_grad():
            for idx, batch_data in enumerate(self.valid_dataloader):
                self.image = batch_data['image'].cuda()
                self.mask = batch_data['mask'].numpy().astype(np.uint8)
                if is_valid:
                    probs = self.model(self.image).cpu().numpy()
                else:
                    probs = self.model(self.image)[0].cpu().numpy()
                if len(self.class2label) > 1:
                    # 多分类评估
                    y_probs = np.transpose(probs, (0, 2, 3, 1))  # [B C H W] -> [B H W C]
                    y_preds = np.argmax(y_probs, axis=-1).astype(np.uint8)
                    y_trues = self.mask
                    for class_idx in range(len(self.class2label)):
                        metrics = []
                        for y_pred, y_prob, y_true in zip(y_preds, y_probs, y_trues):
                            mask = (y_true == class_idx).astype(np.uint8)
                            if mask.sum() == 0:
                                # GT 不存在， 计算召回是都为0，因此不参与计算
                                continue
                            prob = y_prob[..., class_idx]
                            pred = (y_pred == class_idx).astype(np.uint8)
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
                if idx % interval == 0 and idx // interval > 0:
                    self.logger.info(f"Step:{idx}/{valid_iters}")
        curr_metrics, best_index = parse_seg_metrics(all_seg_metrics_dict)

        if self.select_metric == 'f1':
            curr_mean_metric = curr_metrics[-1, 5]
        elif self.select_metric == 'iou':
            curr_mean_metric = curr_metrics[-1, 6]
        else:
            curr_mean_metric = curr_metrics[-1, 5]

        if self.best_value < curr_mean_metric and is_valid:
            self.best_value = curr_mean_metric
            self.best_metrics = curr_metrics
            self.threshold = threshold_list[best_index]
            meta = {'threshold': self.threshold}
            save_checkpoint(self.model, self.best_pth_path, meta=meta)
            self.logger.info(f'Saved best model to {self.best_pth_path} Threshold: {self.threshold:.2f}')
        parse_seg_metrics_to_table(curr_metrics, self.best_metrics[-1, :], self.label2class, self.logger)

    def _torch2onnx(self):
        self.logger.info(f"Load model from {self.best_pth_path}")
        height, width = (self.train_cfg['valid']['transform'][0]['height'],
                         self.train_cfg['valid']['transform'][0]['width'])
        ckpt = load_checkpoint(self.model, self.best_pth_path)
        threshold = ckpt['meta']['threshold']
        onnx_cfg = dict(
            model=self.model,
            dummy_input=torch.ones(1, 3, height, width).cuda(),
            onnx_model_name=self.best_pth_path.replace('.pth', '.onnx'),
            opset_version=17
        )
        try:
            self.logger.info(f"Convert onnx.")
            torch2onnx(**onnx_cfg)
            self.logger.info(f"Convert onnx successfully!")
        except Exception as e:
            self.logger.error(f"Convert onnx failed! {e} \n{traceback.format_exc()}")

    def _onnx2trt(self):
        onnx_path = self.best_pth_path.replace('.pth', '.onnx')
        # best_pth_path = '/workspace/mycode/03-seg/seg/workdir/test/20241213_111019/20241213_111019.pth'
        trt_cfg = dict(
            max_batch_size=32,
            mode='fp16'
        )
        try:
            self.logger.info(f"Convert trt. It takes about 10 minutes for segmentation model.")
            self.model = TRTModel(onnx_path, **trt_cfg)
            self.logger.info(f"Convert trt successfully!")
        except Exception as e:
            self.logger.error(f"Convert trt failed! {e} \n{traceback.format_exc()}")

    def _export(self):
        # 1. torch to onnx
        try:
            self.logger.info(f"Load model from {self.best_pth_path}")
            height, width = (self.train_cfg['valid']['transform'][0]['height'],
                             self.train_cfg['valid']['transform'][0]['width'])
            ckpt = load_checkpoint(self.model, self.best_pth_path)
            # threshold = ckpt['meta']['threshold']
            onnx_path = self.best_pth_path.replace('.pth', '.onnx')
            onnx_cfg = dict(
                model=self.model,
                dummy_input=torch.ones(1, 3, height, width).cuda(),
                onnx_model_name=onnx_path,
                opset_version=self.export_cfg['onnx']['opset_version']
            )
        except Exception as e:
            self.logger.error(f"Export info error! {e} \n{traceback.format_exc()}")
        try:
            self.logger.info(f"Convert onnx.")
            torch2onnx(**onnx_cfg)
            self.logger.info(f"Convert onnx successfully!")
        except Exception as e:
            self.logger.error(f"Convert onnx failed! {e} \n{traceback.format_exc()}")

        # 2. onnx to trt
        mode = self.export_cfg['trt']['mode']
        max_batch_size = self.export_cfg['trt']['max_batch_size']
        trt_cfg = dict(
            mode=mode,
            max_batch_size=max_batch_size
        )
        try:
            self.logger.info(f"Convert trt. It takes about 10 minutes for segmentation model.")
            self.model = TRTModel(onnx_path, **trt_cfg)
            self.logger.info(f"Convert trt successfully!")
        except Exception as e:
            self.logger.error(f"Convert trt failed! {e} \n{traceback.format_exc()}")

        # 3. valid using engine
        try:
            self.logger.info(f"TRT model valid.")
            self._valid(is_valid=False)
            self.logger.info(f"TRT model valid done.")
        except Exception as e:
            self.logger.error(f"TRT model valid failed! {e} \n{traceback.format_exc()}")

        # 4. write trt cfg
        try:
            self.logger.info(f"Save TRT json.")
            self.export_cfg['trt']['engine'] = onnx_path.replace(".onnx", ".engine")
            self.export_cfg['transform'] = self.train_cfg['valid']['transform']
            self.export_cfg['shape_labels'] = self.train_cfg['valid']['dataset']['shape_labels']
            trt_cfg_path = onnx_path.replace('.onnx', '_trt.json')
            save_json(self.export_cfg, trt_cfg_path)
            self.logger.info(f"Save TRT json to {trt_cfg_path}")
        except Exception as e:
            self.logger.error(f"Save TRT json failed. {e}\n{traceback.format_exc()}")

    def __call__(self, *args, **kwargs):
        start_time = time.time()
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

        total_time = datetime.timedelta(seconds=int(time.time() - start_time))
        self.logger.info(f'End training. Total training time: {total_time}')

        self._export()
