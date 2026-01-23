# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from .loss_util import weighted_loss
from .lpips import LPIPS

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class MRIPSNRLoss(nn.Module):
    """
    PSNR Loss for single-channel MRI reconstruction.
    Computes: loss = -PSNR  (so minimizing loss = maximizing PSNR)
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)  # for converting ln → log10

    def forward(self, pred, target):
        # pred, target: [N, 1, H, W]
        assert pred.ndim == 4 and pred.size(1) == 1
        term = {}

        # compute MSE per sample
        mse = F.mse_loss(pred, target, reduction='none')
        mse = mse.mean(dim=(1,2,3)) + 1e-8

        # compute MAX_I using target dynamic range (per sample)
        max_val = torch.amax(target, dim=(1,2,3)) - torch.amin(target, dim=(1,2,3))
        max_val = max_val.clamp(min=1e-6)

        # PSNR = 20*log10(MAX_I) - 10*log10(MSE)
        psnr = 20 * torch.log(max_val) * self.scale - self.scale * torch.log(mse)
        # negative PSNR as loss
        loss = self.loss_weight * (-psnr.mean())
        term['loss'] = loss

        # return
        return term


class MRIPerceptualLoss(nn.Module):
    """
    L1 loss and perceptual loss for single-channel MRI reconstruction.
    """

    def __init__(self, perceptual_weight=1.0, logvar_init=0.0, reduction='mean', device='cpu'):
        super().__init__()
        assert reduction == 'mean'
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init).to(device)
        self.perceptual_loss = LPIPS().to(device).eval()
        self.perceptual_weight = perceptual_weight

    def forward(self, pred, target, weights=None):
        # pred, target: [N, 1, H, W]
        assert pred.ndim == 4 and pred.size(1) == 1
        term = {}

        max_val = torch.amax(target, dim=(1, 2, 3))
        max_val = max_val.clamp(min=1e-6)
        normalize_target = target / max_val.view(-1, 1, 1, 1)
        normalize_pred = pred / max_val.view(-1, 1, 1, 1)

        rec_loss = torch.abs(normalize_target.contiguous() - normalize_pred.contiguous())
        term['rec_loss'] = rec_loss.detach().mean()
        if self.perceptual_weight > 0:
            normalize_target_3c = normalize_target.repeat(1, 3, 1, 1)
            normalize_pred_3c = normalize_pred.repeat(1, 3, 1, 1)
            p_loss = self.perceptual_loss(normalize_target_3c.contiguous(), normalize_pred_3c.contiguous())
            term['p_loss'] = p_loss.detach().mean()
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        term['loss'] = weighted_nll_loss

        # return
        return term


class DISTSLoss(nn.Module):
    """DISTS loss.
    """

    def __init__(self, dists_weight=1.0, logvar_init=0.0, reg_type='l1', device='cpu'):
        super().__init__()
        from DISTS_pytorch import DISTS
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init).to(device)
        self.dists_loss = DISTS().to(device).eval()
        self.dists_weight = dists_weight
        self.reg_type = reg_type

    def forward(self, pred, target, weights=None):
        # pred, target: [N, 1, H, W]
        assert pred.ndim == 4 and pred.size(1) == 1
        term = {}

        max_val = torch.amax(target, dim=(1, 2, 3))
        max_val = max_val.clamp(min=1e-6)
        normalize_target = target / max_val.view(-1, 1, 1, 1)
        normalize_pred = pred / max_val.view(-1, 1, 1, 1)

        if self.reg_type == 'l1':
            rec_loss = torch.abs(normalize_target.contiguous() - normalize_pred.contiguous())
        elif self.reg_type == 'l2':
            rec_loss = (normalize_target.contiguous() - normalize_pred.contiguous()) ** 2
        else:
            raise NotImplementedError(f'Unsupported loss type: {self.reg_type}. Supported ones are: l1, l2.')
        term['rec_loss'] = rec_loss.detach().mean()

        if self.dists_weight > 0:
            normalize_target_3c = normalize_target.repeat(1, 3, 1, 1)
            normalize_pred_3c = normalize_pred.repeat(1, 3, 1, 1)
            dists_loss = self.dists_loss(normalize_target_3c, normalize_pred_3c, require_grad=True, batch_average=False)
            term['dists_loss'] = dists_loss.detach().mean()
            rec_loss = rec_loss + self.dists_weight * dists_loss.view(-1, 1, 1, 1)

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        term['loss'] = weighted_nll_loss

        return term