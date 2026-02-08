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


###################################################################################
#                             For training GAN                                    #
###################################################################################

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus( - logits_real)) + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss

def smoothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = dx*dx
    dy = dy*dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d
    return grad

class Transformer_2D(nn.Module):
    def __init__(self):
        super(Transformer_2D, self).__init__()

    def forward(self, src, flow):
        b = flow.shape[0]
        h = flow.shape[2]
        w = flow.shape[3]

        size = (h, w)

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b,1,1,1).cuda()
        new_locs = grid+flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1 , 0]]
        warped = F.grid_sample(src,new_locs,align_corners=True,padding_mode="border")
        # ctx.save_for_backward(src,flow)
        return warped


class DISTSLossWithDiscriminatorAndRegistration(nn.Module):
    """
    DISTS loss with discriminator and registration network.
    """
    def __init__(self, disc_start, logvar_init=0.0, disc_factor=1.0, disc_weight=1.0, dists_weight=1.0, disc_loss="hinge",
                 corr_weight=20., smooth_weight=10., device='cpu', weight_dtype=torch.float32):
        super().__init__()
        from DISTS_pytorch import DISTS
        from src.module.reggan.discriminator import Discriminator
        from src.module.reggan.registration import Reg

        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init).to(device)
        self.dists_loss = DISTS().to(device).eval()
        self.dists_weight = dists_weight

        self.discriminator = Discriminator(input_nc=1).to(device).to(weight_dtype)
        self.corr_weight = corr_weight
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

        self.reg = Reg(height=256, width=256, in_channels_a=1, in_channels_b=1).to(device).to(weight_dtype)
        self.spatial_transform = Transformer_2D().to(device)
        self.smooth_weight = smooth_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1.e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, pred, target, optimizer_idx, global_step, last_layer=None, split="train"):
        term = {}

        max_val = torch.amax(target, dim=(1, 2, 3))
        max_val = max_val.clamp(min=1e-6)
        normalize_target = target / max_val.view(-1, 1, 1, 1)
        normalize_pred = pred / max_val.view(-1, 1, 1, 1)

        if optimizer_idx == 0:
            # rec_loss = torch.abs(normalize_target.contiguous() - normalize_pred.contiguous())
            trans = self.reg(normalize_pred, normalize_target)
            sm_loss = smoothing_loss(trans) # TODO: add sm_loss to total loss
            regist_normalize_pred = self.spatial_transform(normalize_pred, trans)
            rec_loss = torch.abs(regist_normalize_pred.contiguous() - normalize_target.contiguous())
            if self.dists_weight > 0:
                normalize_target_3c = normalize_target.repeat(1, 3, 1, 1)
                normalize_pred_3c = normalize_pred.repeat(1, 3, 1, 1)
                p_loss = self.dists_loss(normalize_target_3c, normalize_pred_3c, require_grad=True, batch_average=False)
                rec_loss = rec_loss + self.dists_weight * p_loss.view(-1, 1, 1, 1)

            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

            # now the GAN part
            # generator update
            logits_fake = self.discriminator(normalize_pred.contiguous())
            g_loss = - torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.smooth_weight * sm_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/sm_loss".format(split): sm_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(normalize_target.contiguous().detach())
            logits_fake = self.discriminator(normalize_pred.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

