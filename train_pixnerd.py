import argparse
import re
from copy import deepcopy
import logging
import os

from pathlib import Path
from collections import OrderedDict
import shutil

import torch
from omegaconf import OmegaConf
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from src.data_utils.dicom_dataset import MRIDicomDataset

import wandb
import math
from torchvision.utils import make_grid

from src.module.bbdm.BrownianBridge.BrownianBridgeModel import BrownianBridgeDiffusion
from src.module.pixnerd.encoder import DINOv2
from src.module.pixnerd.pixnerd_c2i import PixNerDiT


# from src.module.diffusion.models.unet import UNetModel


def normalize_torch(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= torch.amin(img, dim=(1, 2, 3), keepdim=True)  # np.min(img)
    img /= torch.amax(img, dim=(1, 2, 3), keepdim=True)  # np.max(img)
    return img

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    # x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    x = x.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x

@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device

    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

def load_pretrained_parameters(model, pretrained, logger=None, verbose=True):
    ckpt = torch.load(pretrained, map_location='cpu')
    # if 'ema' in ckpt:
    #     ckpt = ckpt['ema']
    if 'model' in ckpt:
        ckpt = ckpt['model']
    if logger is not None:
        logger.info(f"Loading pretrained model parameters from '{pretrained}'. ")
    # csd = csd_copy  # checkpoint state_dict
    cmsd = model.state_dict()  # current model state_dict with required grad
    csd = intersect_dicts(ckpt, cmsd)  # intersect
    unmatching_keys = [k for k, v in cmsd.items() if k not in csd.keys()] if len(csd) != len(cmsd) else []
    model.load_state_dict(csd, strict=False)  # load
    if verbose and logger is not None:
        logger.info(f'Transferred {len(csd)}/{len(cmsd)} items from pretrained weights, unmatching keys are:{unmatching_keys}')
    return model

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def replace_specific_parent(path: Path, old_name: str, new_name: str) -> Path:
    parts = list(path.parts)
    parts = [new_name if part == old_name else part for part in parts]
    return Path(*parts)

def rename_with_increment(filename):
    # Match files that end with _three-digit-number.npy
    match = re.match(r"^(.*_)(\d{3})(\.npy)$", filename)
    prefix, index, suffix = match.groups()
    new_index = f"{int(index) + 1:03d}"  # Increment and format as three digits
    new_name = f"{prefix}{new_index}{suffix}"
    return new_name

def create_dataloader(args, accelerator, logger=None, is_train=True):
    if is_train:
        dataset = MRIDicomDataset(
            root=args.train_data_path,
            resolution=args.resolution,
            normalize_type=args.normalize_type,
            use_csv=getattr(args, "use_hq_data", False),
        )
    else:
        dataset = MRIDicomDataset(
            root=args.val_data_path,
            resolution=args.resolution,
            normalize_type=args.normalize_type,
        )

    if accelerator.is_main_process and logger is not None:
        logger.info(f"{'Train' if is_train else 'Val'} Dataset contains {len(dataset):,} images")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True if is_train else False,
        collate_fn=getattr(dataset, 'collate_fn', None),
        persistent_workers=True,
    )
    return dataloader


#################################################################################
#                                    EDM Solver                                 #
#################################################################################


class EDMPrecondWithREAP(nn.Module):
    def __init__(self,
                 is_train,
                 label_dim       = 0,                # Number of class labels, 0 = unconditional.
                 use_fp16        = False,            # Execute the underlying model at FP16 precision?
                 sigma_min       = 0,                # Minimum supported noise level.
                 sigma_max       = float('inf'),     # Maximum supported noise level.
                 sigma_data      = 0.5,              # Expected standard deviation of the training data.
                 P_mean=-1.2,
                 P_std=1.2,
                 model_path='',
                 proj_denoiser_dim=1152,
                 proj_hidden_dim=1152,
                 proj_encoder_dim=768,
                 device='cpu'
    ):
        super().__init__()
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.P_mean = P_mean
        self.P_std = P_std

        # REAP # Just for training
        if is_train:
            self.encoder = DINOv2(weight_path=model_path).to(device)
            requires_grad(self.encoder, False)

            self.proj = nn.Sequential(
                nn.Sequential(
                    nn.Linear(proj_denoiser_dim, proj_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(proj_hidden_dim, proj_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(proj_hidden_dim, proj_encoder_dim),
                )
            ).to(device)
            self.proj.requires_grad_(True)

    def forward(self, model, x, sigma, context=None, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = model((c_in * x).to(dtype), timesteps=c_noise.flatten(), context=context, y=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def training_losses(self, net, images, context=None, labels=None, augment_pipe=None):
        loss_dict = {}
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = self.forward(net, y + n, sigma=sigma, context=context, class_labels=labels)
        loss = weight * ((D_yn - y) ** 2)
        loss_dict['loss'] = loss
        return loss_dict

    def edm_sampler(
        self, net, latents, context=None, class_labels=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, dtype=torch.float64,
    ):
        # Adjust noise levels based on what's supported by the network.
        # sigma_min = max(sigma_min, net.sigma_min)
        # sigma_max = min(sigma_max, net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=dtype, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        x_next = latents.to(dtype) * t_steps[0]
        for i, (t_cur, t_next) in tqdm(enumerate(zip(t_steps[:-1], t_steps[1:])), total=num_steps): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, 2 ** 0.5 - 1) if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = self.forward(net, x_hat, sigma=t_hat, context=context, class_labels=class_labels).to(dtype)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.forward(net, x_next, sigma=t_next, context=context, class_labels=class_labels).to(dtype)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main():
    args = OmegaConf.load(parse_args())
    # set accelerator
    logging_dir = Path(args.output_dir, args.project_name, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=str(Path(args.output_dir, args.project_name)), logging_dir=str(logging_dir)
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # cpu=True, # for debugging.
    )

    save_dir = Path(args.output_dir, args.project_name, args.exp_name)
    checkpoint_dir = save_dir / "checkpoints"  # Stores saved model checkpoints
    args.save_dir = str(save_dir)
    if accelerator.is_main_process:
        save_dir.mkdir(mode=0o777, parents=True, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        # Save to a yaml file
        OmegaConf.save(args, save_dir / 'args.yaml')
        checkpoint_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    else:
        logger = None
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    model = PixNerDiT(
        in_channels=2,
        out_channels=1,
        patch_size=16,
        num_groups=16,
        hidden_size=1152,
        hidden_size_x=64,
        num_blocks=30,
        num_cond_blocks=26,
        nerf_mlpratio=2,
    )
    # diffusion = EDMPrecondWithREAP()
    diffusion = BrownianBridgeDiffusion(loss_type=getattr(args, "loss_type", "l1"), sample_step=args.sample_steps, loss_weight=getattr(args, "loss_weight", 1.0), device=device)

    # load pre-trained model
    if args.pretrained is not None:
        model = model.from_pretrained(args.pretrained) if Path(args.pretrained).is_dir() else load_pretrained_parameters(model, args.pretrained, logger)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora dit)
    # to half-precision as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model = model.to(device).to(weight_dtype)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    if accelerator.is_main_process:
        logger.info(f"Diffusion Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Setup mri data:
    dataloader = create_dataloader(args, accelerator, logger=logger, is_train=True)
    val_dataloader = create_dataloader(args, accelerator, logger=logger, is_train=False)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Setup optimizer and learning rate scheduler:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=10000,
        cooldown=10000,
        min_lr=5.e-7,
        verbose=True
    )

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    global_step = 0
    first_epoch = 0

    model, optimizer, lr_scheduler, dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, dataloader, val_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = OmegaConf.to_container(args, resolve=True)
        accelerator.init_trackers(
            project_name=args.project_name,
            config=tracker_config,
            init_kwargs={
                "wandb": {
                    "name": f"{args.exp_name}",
                    "dir": save_dir,
                }
            },
        )

    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(checkpoint_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(str(checkpoint_dir / str(path)))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # # Create sampling noise (feel free to change):
    # total_batch_size = 64 if 8 * accelerator.num_processes >= 64 else 8 * accelerator.num_processes
    # sample_batch_size = total_batch_size // accelerator.num_processes
    # xT = torch.randn((sample_batch_size, 2, args.resolution, args.resolution), device=device)

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(dataloader):
            # data preparation
            input_images, output_images, _, _ = batch # [*, 2, h, w]
            input_images, output_images = input_images.to(device), output_images.to(device)
            model.train()

            with accelerator.accumulate(model):
                t = torch.randint(0, diffusion.num_timesteps, (input_images.shape[0],), device=device)
                _, loss_dict = diffusion(model, output_images, input_images, t)
                loss = loss_dict['loss'].mean()

                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step(loss.item())

                if accelerator.sync_gradients:
                    update_ema(ema, model, decay=0.9999)  # change ema function

            # calculate gpu memory usage
            mem = f'{torch.cuda.memory_reserved() / 2 ** 30 if torch.cuda.is_available() else 0.0:.3g}G'
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                model.eval()

                if global_step % args.checkpointing_steps == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(checkpoint_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = checkpoint_dir / removing_checkpoint
                                    shutil.rmtree(removing_checkpoint)

                        checkpoint_path = checkpoint_dir / f"checkpoint-{global_step}"
                        accelerator.save_state(str(checkpoint_path))

                        accelerator.unwrap_model(model).save_pretrained(f"{checkpoint_dir}/model")
                        ema.save_pretrained(f"{checkpoint_dir}/ema_model")
                        logger.info(f"Saved checkpoint to '{checkpoint_path}', model to '{checkpoint_dir}/model', ema model to '{checkpoint_dir}/ema_model' ")

                if (global_step == 1 or (global_step % args.sample_every == 0 and global_step > 0)):
                    with torch.no_grad():

                        out_samples_list = []
                        gt_images_list = []
                        for val_step, val_batch in enumerate(val_dataloader):
                            if val_step >= 1:
                                break
                            val_input_images, val_output_images, input_normalize_value, output_normalize_value = val_batch
                            val_input_images, val_output_images, input_normalize_value, output_normalize_value = val_input_images.to(device), val_output_images.to(device), input_normalize_value.to(device), output_normalize_value.to(device)
                            samples = diffusion.p_sample_loop(model, val_input_images, clip_denoised=False)
                            samples = samples.to(torch.float32) # [B, 1, H, W]

                            # denormalize samples
                            if args.normalize_type in ['minmax', 'mean_minmax']:
                                samples = samples * 0.5 + 0.5
                            samples = samples * input_normalize_value[..., None, None, None]
                            samples = normalize_torch(samples)
                            out_samples = accelerator.gather(samples.to(torch.float32))
                            out_samples_list.append(out_samples)
                            # denormalize gt images
                            if args.normalize_type in ['minmax', 'mean_minmax']:
                                val_output_images = val_output_images * 0.5 + 0.5
                            gt_images = val_output_images * output_normalize_value[..., None, None, None]
                            gt_images = normalize_torch(gt_images)
                            gt_images = accelerator.gather(gt_images.to(torch.float32))
                            gt_images_list.append(gt_images)
                    out_samples = torch.cat(out_samples_list, dim=0)
                    gt_images = torch.cat(gt_images_list, dim=0)
                    if accelerator.is_main_process:
                        accelerator.log(
                            {
                                "samples": wandb.Image(array2grid(out_samples)),
                                "gt_images": wandb.Image(array2grid(gt_images)),
                            },
                            step=global_step,
                        )
                    if logger is not None:
                        logger.info("Generating samples done.")

            logs = {
                "loss": accelerator.gather(loss).mean().detach().item(),
                "lr": optimizer.param_groups[0]['lr'],
                "gpu_memory": mem,
            }
            if accelerator.is_main_process:
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        # default=None,
        default="cfg/train_configs/train_pixnerd.yaml",
        # required=True,
        help="path to config",
    )
    args = parser.parse_args()
    return args.config


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0' # TODO: delete after testing
    # for wandb logging
    os.environ["WANDB_API_KEY"] = 'd41703d5fb88dbc5ac4d7bdc7bef1f590f2d0460'
    os.environ["WANDB_MODE"] = 'dryrun'
    main()
