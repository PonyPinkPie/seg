import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseSegmentor
from seg.models.registry import *
from seg.models.utils.common import add_prefix


@SEGMENTATIONS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone=None,
                 decoder_head=None,
                 num_classes=None,
                 neck=None,
                 auxiliary_head=None,
                 pretrained=None,
                 **kwargs):
        super(EncoderDecoder, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if decoder_head is not None:
            self._init_decode_head(decoder_head, decoder_head['num_classes'])
        if auxiliary_head is not None:
            self._init_auxiliary_head(auxiliary_head, decoder_head['num_classes'])


        self.init_weights(pretrained=pretrained)

        # assert self.with_decode_head                                             # modify

    def _init_decode_head(self, decode_head, num_classes):
        """Initialize ``decode_head``"""
        self.decode_head = build_head(decode_head, default_args=(dict(num_classes=num_classes)))
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head, num_classes):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    head_cfg['num_classes'] = num_classes
                    self.auxiliary_head.append(build_head(head_cfg))
            else:
                auxiliary_head['num_classes'] = num_classes
                self.auxiliary_head = build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Parameters
        ----------
        pretrained : str, optional
            Path to pre-trained weights.
            Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.decode_head:
            self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, inputs):
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs, return_feat=False):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        feat = self.extract_feat(inputs)
        out = self._decode_head_forward_infer(feat)
        out = F.interpolate(
            input=out,
            size=inputs.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return (out, feat) if return_feat else out

    def _decode_head_forward_train(self, x, ground_truth, **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, ground_truth, **kwargs)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_infer(self, x, return_feat=False):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_infer(x, return_feat=return_feat)

        return seg_logits

    def _auxiliary_head_forward_train(self, x, ground_truth, **kwargs):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, ground_truth, **kwargs)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, ground_truth, **kwargs)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_train(self, inputs, ground_truth, return_feat=False, **kwargs):
        """Forward function for training.

        Parameters
        ----------
        inputs : Tensor
            Input images.
        ground_truth : Tensor
            Semantic segmentation masks
            used if the architecture supports semantic segmentation task.

        Returns
        -------
        dict[str, Tensor]
            a dictionary of loss components
        """

        feat = self.extract_feat(inputs)

        losses = dict()
        gt_masks = ground_truth['mask'].to(inputs.device, dtype=inputs.dtype)
        weight = ground_truth['weight'].to(inputs.device, dtype=inputs.dtype) if "weight" in ground_truth else None
        loss_decode = self._decode_head_forward_train(feat, gt_masks, weight=weight, **kwargs)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                feat, gt_masks, **kwargs)
            losses.update(loss_aux)

        return (losses, feat) if return_feat else losses

    def forward_infer(self, inputs, return_feat=False, **kwargs):
        if return_feat:
            seg_logit, feat = self.encode_decode(inputs, return_feat=True)
        else:
            seg_logit = self.encode_decode(inputs, return_feat=False)
        if self.num_classes > 1:
            if kwargs.get("with_softmax", True):
                seg_probs = torch.softmax(seg_logit, dim=1)
            else:
                seg_probs = seg_logit
        else:
            seg_probs = torch.sigmoid(seg_logit)

        return (seg_probs, feat) if return_feat else seg_probs






