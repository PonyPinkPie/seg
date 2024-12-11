import torch
from .base_runner import BaseRunner
from seg.transforms.compose import Compose

from seg.models.registry import build_segmentation
from seg.utils.distribute import init_dist_pytorch, get_dist_info, cuda_is_available, devices_count
# from seg.metrics import build_metrics

class InferenceRunner(BaseRunner):
    def __init__(self, inference_cfg, base_cfg=None):
        super().__init__(base_cfg)

        self.multi_label = inference_cfg.get('multi_label', False)

        self.transform = self._build_transform(inference_cfg['transform'])

        self._build_model(inference_cfg['model'])

        self.model.eval()


    def _build_model(self, cfg):
        self.logger.info(f"Building model.")
        self.model = build_segmentation(cfg)
        # if cuda_is_available():
        #     if self.distribute:
        #         self.model = torch.nn.parallel.DistributedDataParallel(
        #             self.model.cuda(),
        #             device_ids=[1],
        #             broadcast_buffers=True,
        #         )
        #         self.logger.info('Using DistributedDataParallel Training.')
        #     else:
        #         if devices_count() > 1:
        #             self.model = torch.nn.DataParallel(self.model)
        #             self.logger.info('Using DataParallel Training.')
        self.model.cuda()
        self.logger.info(f"Building model Done.")

    def _build_transform(self, cfg):
        return Compose(cfg)


    # def compute(self, output):
    #     if self.multi_label:
    #         output = output.sigmoid()
    #         output = torch.where(output >= 0.5,
    #                              torch.full_like(output, 1),
    #                              torch.full_like(output, 0)).long()
    #
    #     else:
    #         output = output.softmax(dim=1)
    #         _, output = torch.max(output, dim=1)
    #     return output

    def __call__(self, image, mask):
        with torch.no_grad():
            data = {'image': image, 'mask': mask}
            image = self.transform(**data)['image']
            image = image.unsqueeze(0)
            if self.ues_gpu:
                image = image.cuda()

            output = self.model(image)
            # output = self.compute(output)
            output = output.squeeze().cpu().numpy()
        return output

