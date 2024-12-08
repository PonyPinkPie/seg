import torch.nn as nn
from ._base_loss import BaseLoss
from .functional import *
from ..registry import LOSSES

@LOSSES.register_module()
class SmoothL1Loss(BaseLoss):
    def __init__(self, beta=1.0, **kwargs):
        super(SmoothL1Loss, self).__init__(loss_name='loss_l1', **kwargs)
        self.beta = beta

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            ignore_label=self.ignore_label,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss


@LOSSES.register_module()
class L1Loss(BaseLoss):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduce loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False, use_sigmoid=False):
        super().__init__(loss_name="l1_loss")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.use_sigmoid = use_sigmoid

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        if self.use_sigmoid:
            pred = pred.sigmoid()

        if target.dim() == 3:
            target = target.unsqueeze(1)

        return self.loss_weight * l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class CosineSimilarityLoss(BaseLoss):
    """Cosine similarity loss function.

    Compute the similarity between two features and optimize that similarity as
    loss.

    Args:
        shift_factor (float): The shift factor of cosine similarity.
            Default: 0.0.
        scale_factor (float): The scale factor of cosine similarity.
            Default: 1.0.
    """

    def __init__(self,
                 dim: int = 1,
                 shift_factor: float = 0.0,
                 scale_factor: float = 1.0,
                 **kwargs
                 ) -> None:
        super().__init__(loss_name='cos_loss', **kwargs)
        self.shift_factor = shift_factor
        self.scale_factor = scale_factor
        self.dim = dim

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor = None,
                dim: int = None,
                ) -> torch.Tensor:
        """Forward function of cosine similarity loss.

        Args:
            pred (torch.Tensor): The predicted features.
            target (torch.Tensor): The target features.

        Returns:
            torch.Tensor: The cosine similarity loss.
        """
        if dim is None:
            dim = self.dim

        pred_norm = nn.functional.normalize(pred, dim=dim)
        target_norm = nn.functional.normalize(target, dim=dim)
        loss = self.shift_factor - self.scale_factor * (
                pred_norm * target_norm).sum(dim=dim)

        if mask is None:
            loss = loss.mean()
        else:
            loss = (loss * mask).sum() / mask.sum()
        return self.loss_weight * loss


@LOSSES.register_module()
class CSUMLoss(BaseLoss):
    def __init__(self, lam=1, **kwargs):
        super().__init__(loss_name='CSUM_loss', **kwargs)
        self.lam = lam

    def forward(self, input):
        loss = 0
        for instance in input:
            _, _, h, w = instance.shape
            loss += torch.sum(instance) / (h * w) * self.lam
        return loss
