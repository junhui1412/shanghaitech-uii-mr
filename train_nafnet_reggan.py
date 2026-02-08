import argparse
from copy import deepcopy
import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3' # TODO: delete after testing
from pathlib import Path
from collections import OrderedDict
import shutil

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from src.data_utils.dicom_dataset import MRIDicomDataset

import wandb
import math
from typing import Any, Dict
from torchvision.utils import make_grid

from src.module.nafnet.models.archs import define_network
from src.module.nafnet.models.losses.losses import DISTSLossWithDiscriminatorAndRegistration
# import matplotlib.pyplot as plt

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
def update_ema(ema_model, model, decay=0.995):
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

def build_scheduler(
    optimizer: Optimizer,
    steps_per_epoch: int,
    scheduler_cfg: Dict[str, Any],
) -> tuple[LambdaLR, str]:
    """
    Create a learning rate scheduler with optional warmup. Supports 'linear' and 'cosine'.
    """
    schedule_type = scheduler_cfg.get("type", "linear").lower()

    base_lr = float(optimizer.param_groups[0]["lr"])
    final_lr = float(scheduler_cfg.get("final_lr", base_lr))
    final_ratio = final_lr / base_lr if base_lr > 0 else 1.0

    warmup_steps_cfg = scheduler_cfg.get("warmup_steps")
    warmup_from_zero = bool(scheduler_cfg.get("warmup_from_zero", False))
    if warmup_steps_cfg is not None:
        warmup_steps = int(warmup_steps_cfg)
    else:
        warmup_epochs = float(scheduler_cfg.get("warmup_epochs", 0))
        warmup_steps = int(warmup_epochs * steps_per_epoch)

    decay_end_steps_cfg = scheduler_cfg.get("decay_end_steps")
    if decay_end_steps_cfg is not None:
        decay_end_steps = int(decay_end_steps_cfg)
    else:
        decay_end_epoch = float(scheduler_cfg.get("decay_end_epoch", warmup_steps / steps_per_epoch if steps_per_epoch else 0))
        decay_end_steps = int(decay_end_epoch * steps_per_epoch)

    warmup_steps = max(warmup_steps, 0)
    decay_end_steps = max(decay_end_steps, warmup_steps)
    total_decay_steps = max(decay_end_steps - warmup_steps, 1)

    for group in optimizer.param_groups:
        group["lr"] = base_lr

    if schedule_type == "linear":

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                if warmup_from_zero:
                    return (step + 1) / warmup_steps
                else:
                    return 1.0 # constant lr during warmup
            if step >= decay_end_steps:
                return final_ratio
            progress = (step - warmup_steps) / total_decay_steps
            return 1.0 - (1.0 - final_ratio) * progress

    elif schedule_type == "cosine":

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                if warmup_from_zero:
                    return (step + 1) / warmup_steps
                else:
                    return 1.0 # constant lr during warmup
            if step >= decay_end_steps:
                return final_ratio
            progress = (step - warmup_steps) / total_decay_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final_ratio + (1.0 - final_ratio) * cosine
    else:
        raise ValueError(f"Unsupported scheduler '{schedule_type}'. Choose from ['linear', 'cosine'].")
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    # return some debug msg for optimizer/scheduler
    sched_msg = f"Scheduler: {schedule_type} with warmup_steps={warmup_steps}, decay_end_steps={decay_end_steps}, final_lr={final_lr}, base_lr={base_lr}, warmup_from_zero={warmup_from_zero}"
    return scheduler, sched_msg

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
        drop_last=True,
    )
    return dataloader


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
    ## Generator
    model_kwargs = {
        'type': 'NAFNet',
        'img_channel': 1,
        'width': 64,
        'enc_blk_nums': [2, 2, 4, 8],
        'middle_blk_num': 12,
        'dec_blk_nums': [2, 2, 2, 2],
        'upsampler_mode': 'interpolate',
    }
    model = define_network(model_kwargs)
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

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Setup mri data:
    dataloader = create_dataloader(args, accelerator, logger=logger, is_train=True)
    val_dataloader = create_dataloader(args, accelerator, logger=logger, is_train=False)

    dataloader, val_dataloader = accelerator.prepare(dataloader, val_dataloader)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if getattr(args, "max_train_steps", None) is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Setup loss function:
    loss_func = DISTSLossWithDiscriminatorAndRegistration(disc_start=getattr(args, "disc_start", 0) * num_update_steps_per_epoch, disc_weight=getattr(args, "disc_weight", 0.75), dists_weight=getattr(args, "dists_weight", 0.05), device=device, weight_dtype=weight_dtype)
    requires_grad(loss_func.discriminator, True)
    requires_grad(loss_func.reg, True)

    if accelerator.is_main_process:
        logger.info(f"NAFNet Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Discriminator Parameters: {sum(p.numel() for p in loss_func.discriminator.parameters()):,}")
        logger.info(f"Registration Network Parameters: {sum(p.numel() for p in loss_func.reg.parameters()):,}")

    # Setup optimizer and learning rate scheduler:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_D_B = torch.optim.AdamW(
        loss_func.discriminator.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_R_A = torch.optim.AdamW(
        loss_func.reg.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler, msg_sched = build_scheduler(optimizer, num_update_steps_per_epoch, args.lr_scheduler)
    if accelerator.is_main_process:
        logger.info(msg_sched)
    lr_scheduler_D_B, msg_sched_D_B = build_scheduler(optimizer_D_B, num_update_steps_per_epoch, args.lr_scheduler)
    if accelerator.is_main_process:
        logger.info(msg_sched_D_B)
    lr_scheduler_R_A, msg_sched_R_A = build_scheduler(optimizer_R_A, num_update_steps_per_epoch, args.lr_scheduler)
    if accelerator.is_main_process:
        logger.info(msg_sched_R_A)
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    global_step = 0
    first_epoch = 0

    (
        model, loss_func.discriminator, loss_func.reg,
        optimizer, optimizer_D_B, optimizer_R_A,
        lr_scheduler, lr_scheduler_D_B, lr_scheduler_R_A,
        # dataloader, val_dataloader
    ) = accelerator.prepare(
        model, loss_func.discriminator, loss_func.reg,
        optimizer, optimizer_D_B, optimizer_R_A,
        lr_scheduler, lr_scheduler_D_B, lr_scheduler_R_A,
        # dataloader, val_dataloader
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

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(dataloader):
            # data preparation
            input_images, output_images, _, _ = batch # [*, 2, h, w]
            input_images, output_images = input_images.to(device), output_images.to(device)
            model.train()

            with accelerator.accumulate([model, loss_func.reg, loss_func.discriminator]):
                loss_func.reg.train()
                loss_func.discriminator.eval()
                optimizer.zero_grad()
                optimizer_R_A.zero_grad()
                pred_images = model(input_images)
                loss_gen, gen_log_dict = loss_func(pred_images, output_images, optimizer_idx=0, global_step=global_step, last_layer=accelerator.unwrap_model(model).ending.weight)

                ## optimization
                ### generator, registration
                accelerator.backward(loss_gen)
                if accelerator.sync_gradients:
                    if args.max_grad_norm > 0.0:
                        params_to_clip = model.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer_R_A.step()
                lr_scheduler.step()
                lr_scheduler_R_A.step()

                if accelerator.sync_gradients:
                    update_ema(ema, model, decay=args.ema_decay)  # change ema function

                ### discriminator
            # with accelerator.accumulate(loss_func.discriminator):
                loss_func.reg.eval()
                loss_func.discriminator.train()
                optimizer_D_B.zero_grad()
                with torch.no_grad():
                    pred_images = model(input_images)
                loss_disc, disc_log_dict = loss_func(pred_images, output_images, optimizer_idx=1, global_step=global_step, last_layer=None)
                ## optimization
                accelerator.backward(loss_disc)
                optimizer_D_B.step()
                lr_scheduler_D_B.step()

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

                                logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
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

                            samples = accelerator.unwrap_model(model)(val_input_images)
                            samples = samples.to(torch.float32) # [B, 1, H, W]

                            # denormalize
                            if args.normalize_type in ['minmax', 'mean_minmax']:
                                samples = samples * 0.5 + 0.5
                            samples = samples * input_normalize_value[..., None, None, None]
                            samples = normalize_torch(samples)
                            out_samples = accelerator.gather(samples.to(torch.float32))
                            out_samples_list.append(out_samples)
                            # gt images
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
                    "loss_gen": accelerator.gather(loss_gen).mean().detach().item(),
                    "loss_disc": accelerator.gather(loss_disc).mean().detach().item(),
                    "lr": optimizer.param_groups[0]['lr'],
                    "lr_R_A": optimizer_R_A.param_groups[0]['lr'],
                    "lr_D_B": optimizer_D_B.param_groups[0]['lr'],
                    "gpu_memory": mem,
                }
                if accelerator.is_main_process:
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                    accelerator.log({**gen_log_dict, **disc_log_dict}, step=global_step)

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
        default="cfg/train_configs/train_nafnet_dists_loss_reggan_uii_all_data.yaml",
        # required=True,
        help="path to config",
    )
    args = parser.parse_args()
    return args.config


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2' # TODO: delete after testing
    # for wandb logging
    os.environ["WANDB_API_KEY"] = 'd41703d5fb88dbc5ac4d7bdc7bef1f590f2d0460'
    os.environ["WANDB_MODE"] = 'dryrun'
    main()
