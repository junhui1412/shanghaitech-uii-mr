import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from DISTS_pytorch import DISTS
from tqdm.autonotebook import tqdm
import numpy as np

from src.module.bbdm.utils import extract, default
from src.module.bbdm.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
from src.module.elatentlpips.elatentlpips import ELatentLPIPS


# from src.module.bbdm.BrownianBridge.base.modules.encoders.modules import SpatialRescaler


class BrownianBridgeDiffusion(nn.Module):
    def __init__(self,
                mt_type='linear',  # options {'linear', 'sin'}
                objective='grad',  # options {'grad', 'noise', 'ysubx'}
                loss_type='l1',  # options {'l1', 'l2', 'l1_lpips1', 'l1_lpips2', 'l1_pixel_dists'}

                skip_sample=True,
                sample_type='linear',  # options {"linear", "sin"}
                sample_step=200,

                num_timesteps=1000,  # timesteps
                eta=1.0,  # DDIM reverse process eta
                max_var=1.0,  # maximum variance
                denormalize_latents_fn=None,  # function to denormalize latents for lpips calculation
                vae=None,  # VAE model for decoding latents to pixels
                loss_weight=1.0,
                device='cpu',
        ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.mt_type = mt_type
        self.max_var = max_var
        self.eta = eta
        self.skip_sample = skip_sample
        self.sample_type = sample_type
        self.sample_step = sample_step
        self.steps = None
        self.register_schedule()

        # loss and objective
        self.loss_type = loss_type
        self.objective = objective
        if loss_type in ['l1_lpips1', 'l1_lpips2']:
            # self.elatentlpips = ELatentLPIPS(encoder="flux", augment='bg').to(device).eval()
            self.elatentlpips = ELatentLPIPS(encoder="flux", augment=None).to(device).eval()
        elif loss_type in ['l1_pixel_dists']:
            self.loss_weight = loss_weight
            if denormalize_latents_fn and vae: # latent space
                self.vae = vae
                self.denormalize_latents_fn = denormalize_latents_fn
                self.dists_loss = DISTS().to(device).eval()
            else: # pixel space
                self.dists_loss = DISTS().to(device).eval()

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def forward(self, model, x, y, t, context=None):
        if model.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        # t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(model, x, y, context, t)

    def p_losses(self, model, x0, y, context, t, noise=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param y_ori: original source domain image
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))

        x_t, objective = self.q_sample(x0, y, t, noise)
        objective_recon = model(x_t, timesteps=t, context=context)

        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        log_dict = {
            "x0_recon": x0_recon.detach(),
        }

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        elif self.loss_type == 'l1_lpips1':
            recloss = self.elatentlpips(x0, x0_recon, ensembling=False, add_l1_loss=True)
        elif self.loss_type == 'l1_lpips2':
            recloss = (objective - objective_recon).abs()
            lpips_loss = self.elatentlpips(x0, x0_recon, ensembling=False, add_l1_loss=False)
            log_dict["lpips_loss"] = lpips_loss.mean().detach()
            recloss = (recloss + lpips_loss).mean()
        elif self.loss_type == 'l1_pixel_dists':
            recloss = (objective - objective_recon).abs().mean()

            if hasattr(self, 'vae'):
                x0_denorm = self.denormalize_latents_fn(x0)
                x0_recon_denorm = self.denormalize_latents_fn(x0_recon)
                x0_denorm_pixel, x0_recon_denorm_pixel = self.vae.decode(x0_denorm).sample, self.vae.decode(x0_recon_denorm).sample
                x0_denorm_pixel = torch.clip(x0_denorm_pixel * 0.5 + 0.5, 0., 1.)
                x0_recon_denorm_pixel = x0_recon_denorm_pixel * 0.5 + 0.5
                dists_loss = self.dists_loss(x0_denorm_pixel, x0_recon_denorm_pixel, require_grad=True, batch_average=True)
                log_dict["dists_loss"] = dists_loss.mean().detach()
            else:
                x0_denorm_pixel = torch.clip(x0 * 0.5 + 0.5, 0., 1.)
                x0_recon_denorm_pixel = x0_recon * 0.5 + 0.5
                dists_loss = self.dists_loss(x0_denorm_pixel, x0_recon_denorm_pixel, require_grad=True, batch_average=True)
                log_dict["dists_loss"] = dists_loss.mean().detach()
            recloss = recloss + dists_loss * self.loss_weight
        else:
            raise NotImplementedError()

        # x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        log_dict["loss"] = recloss
        return recloss, log_dict

    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, model, x_t, y, context, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = model(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            objective_recon = model(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, model, y, context=None, clip_denoised=True, sample_mid_step=False):
        if model.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(model, x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample(model, x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
            return img

    @torch.no_grad()
    def sample(self, model, y, context=None, clip_denoised=True, sample_mid_step=False):
        return self.p_sample_loop(model, y, context, clip_denoised, sample_mid_step)