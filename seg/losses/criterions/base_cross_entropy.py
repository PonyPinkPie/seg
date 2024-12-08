from ..registry import LOSSES
from ._base_loss import BaseLoss
from .functional import *

@LOSSES.register_module()
class CrossEntropyLoss(BaseLoss):
    def __init__(self, class_weight=None, ignore_label=-100, use_sigmoid=False, label_smoothing=0.0, **kwargs):
        super(CrossEntropyLoss, self).__init__(loss_name='loss_ce', **kwargs)
        self.class_weight = class_weight
        self.ignore_label = ignore_label
        self.use_sigmoid = use_sigmoid
        self.label_smoothing = label_smoothing

        if use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_label=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = self.class_weight.to(pred.device)
        else:
            class_weight = None

        if ignore_label is None:
            ignore_label = self.ignore_label

        target = target.long()
        if self.use_sigmoid:
            loss_cls = self.loss_weight * self.cls_criterion(
                pred,
                target,
                weight=weight,
                class_weight=class_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                ignore_index=ignore_label)
        else:
            loss_cls = self.loss_weight * self.cls_criterion(
                pred,
                target,
                weight=weight,
                class_weight=class_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                ignore_index=ignore_label,
                label_smoothing=self.label_smoothing)

        return loss_cls


@LOSSES.register_module()
class FocalLoss(BaseLoss):
    def __init__(self, use_sigmoid=True, alpha=0.25, gamma=2.0, **kwargs):
        super(FocalLoss, self).__init__(loss_name='loss_focal_loss', **kwargs)
        self.alpha = alpha
        self.gamma = gamma

        if use_sigmoid:
            self.criterion = focal_loss_with_logits
        else:
            self.criterion = softmax_focal_loss_with_logits

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_label=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        if ignore_label is None:
            ignore_label = self.ignore_label
        loss_cls = self.loss_weight * self.criterion(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_label)

        return loss_cls


@LOSSES.register_module()
class FFocalLoss(BaseLoss):
    def __init__(self, lam=1, alpha=-1, gamma=4, reduction="mean", **kwargs):
        super(FFocalLoss, self).__init__(loss_name='FFocal_loss', **kwargs)
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean() * self.lam
        elif self.reduction == "sum":
            loss = loss.sum() * self.lam

        return loss
