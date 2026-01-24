# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained model.
"""
import math
import shutil
from pathlib import Path

import pandas as pd
import pydicom
import torch
import torchmetrics
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_utils.dicom_dataset import pad_to_multiple_centered, MRIVolumeTestDicomDataset, pad_to_target_size_centered, SingleMRIVolumeTestDicomDataset
from src.module.nafnet.models.archs import define_network

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse

name2seriesID = {
    'nafnet': 6,
    'nafnet_dists_loss_1e-3': 17,
    'nafnet_dists_loss_1e-4': 18,
    'nafnet_dists_loss_1e-6': 19,
    'nafnet_dists_loss_1e-4_l2_reg': 23,
    'nafnet_dists_loss_1e-4_hq': 24,
    'nafnet_dists_loss_1e-4_hq_lr_1e-4': 25,
    'nafnet_dists_loss_1.0_hq_lr_1e-4': 26,
    'nafnet_dists_loss_0.1_hq_lr_1e-4': 27,
    'nafnet_dists_loss_1e-2_hq_lr_1e-4': 28,
    'nafnet_dists_loss_1e-3_hq_lr_1e-4': 29,
    'nafnet_dists_loss_5e-2_hq_lr_1e-4': 30,
    'nafnet_l1_only_hq_lr_1e-4': 31,
    'nafnet_dists_loss_5e-2_hq_uii_all_data': 32,
    'nafnet_mix': 10,
    'nafnet_perceptual_loss': 22,
    'nafnet_max_normalize': 20,
}

def display_all_images(input_image, sample, output_image, fname, idx, all_image_path):
    # create figures
    fig, axes = plt.subplots(1, 3, dpi=300)
    axes[0].imshow(input_image.cpu().numpy(), cmap='gray')
    axes[0].axis('off')
    axes[0].set_title("Low Quality", fontsize=16)

    axes[1].imshow(sample.cpu().numpy(), cmap='gray')
    axes[1].axis('off')
    axes[1].set_title("Model Predict", fontsize=16)

    axes[2].imshow(output_image.cpu().numpy(), cmap='gray')
    axes[2].axis('off')
    axes[2].set_title("High Quality", fontsize=16)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    # plt.show()
    plt.savefig(all_image_path / f"{fname}_{idx:03d}.png", bbox_inches='tight', pad_inches=0.1, dpi=300)

def display_single_image(image, fname, idx, all_image_path):
    # create figures
    fig, axes = plt.subplots(1, 1, dpi=300)
    axes.imshow(image.cpu().numpy(), cmap='gray')
    axes.axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    # plt.show()
    plt.savefig(all_image_path / f"{fname}_{idx:03d}.png", bbox_inches='tight', pad_inches=0.0, dpi=300)

def normalize_torch(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= torch.amin(img, dim=(1, 2, 3), keepdim=True)  # np.min(img)
    img /= torch.amax(img, dim=(1, 2, 3), keepdim=True)  # np.max(img)
    return img

def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

def load_pretrained_parameters(model, pretrained, verbose=True):
    ckpt = torch.load(pretrained, map_location='cpu')
    # if 'ema' in ckpt:
    #     ckpt = ckpt['ema']
    if 'model' in ckpt:
        ckpt = ckpt['model']
    if verbose:
        print(f"Loading pretrained model parameters from '{pretrained}'. ")
    # csd = csd_copy  # checkpoint state_dict
    cmsd = model.state_dict()  # current model state_dict with required grad
    csd = intersect_dicts(ckpt, cmsd)  # intersect
    unmatching_keys = [k for k, v in cmsd.items() if k not in csd.keys()] if len(csd) != len(cmsd) else []
    model.load_state_dict(csd, strict=False)  # load
    if verbose:
        print(f'Transferred {len(csd)}/{len(cmsd)} items from pretrained weights, unmatching keys are:{unmatching_keys}')
    return model

def create_dataloader(args, verbose=True, input_only=False):
    dataset = MRIVolumeTestDicomDataset(
        root=args.data_path,
        normalize_type=args.normalize_type,
    ) if not input_only else SingleMRIVolumeTestDicomDataset(
        root=args.data_path,
        normalize_type=args.normalize_type,
    )

    if verbose:
        print(f"Test Dataset contains {len(dataset):,} subjects.")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=getattr(dataset, 'collate_fn', None),
        persistent_workers=True,
    )
    return dataloader

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # save path
    save_path = Path(args.save) / f"test_{args.model_type}"
    save_path.mkdir(parents=True, exist_ok=True)

    measurement_path = save_path / 'measurement'
    sample_path = save_path / 'sample'
    reference_path = save_path / 'reference'
    all_image_path = save_path / 'all_image'
    measurement_path.mkdir(parents=True, exist_ok=True)
    sample_path.mkdir(parents=True, exist_ok=True)
    reference_path.mkdir(parents=True, exist_ok=True)
    all_image_path.mkdir(parents=True, exist_ok=True)

    # Load model:
    if args.model_type in ['nafnet', 'nafnet_mix', 'nafnet_perceptual_loss', 'nafnet_max_normalize']:
        model_kwargs = {
            'type': 'NAFNet',
            'img_channel': 1,
            'width': 64,
            'enc_blk_nums': [2, 2, 4, 8],
            'middle_blk_num': 12,
            'dec_blk_nums': [2, 2, 2, 2],
            # 'upsampler_mode': 'interpolate',
        }
    else:
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
    ckpt_path = args.ckpt
    model = model.from_pretrained(ckpt_path) if Path(ckpt_path).is_dir() else load_pretrained_parameters(model, ckpt_path)
    model.to(device)
    model.eval()  # important!

    # Setup mri data:
    dataloader = create_dataloader(args, verbose=True)

    # Set up metrics
    psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
    lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity()
    ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
    # save metrics
    columns = ['FileName', 'Measurement_PSNR', 'Measurement_SSIM', 'Measurement_LPIPS', 'Recon_PSNR', 'Recon_SSIM', 'Recon_LPIPS']
    all_df = pd.DataFrame(columns=columns)
    df_output_file = save_path / 'metrics.csv'
    if not Path(df_output_file).exists():
        all_df.to_csv(df_output_file, index=False)

    # Sampling loop:
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_images, output_images, input_normalize_value, output_normalize_value, fname_lq, fname_hq = batch
        fname_lq, fname_hq = fname_lq[0], fname_hq[0]
        fname_lq, fname_hq = Path(fname_lq), Path(fname_hq)
        # establish new subject dir
        if len(fname_lq.parts) >= 5 and 'Siemens' in fname_lq.parts[-5]:
            new_subject_dir = fname_lq.parent.parent / f"AI_{args.model_type}_{fname_lq.parent.stem}" / fname_lq.stem
        else:
            new_subject_dir = fname_lq.parent / f"AI_{args.model_type}_{fname_lq.stem}"
        # if new_subject_dir.exists(): # already processed
        #     continue
        if args.save_dicom:
            new_subject_dir.mkdir(parents=True, exist_ok=True)

        input_images, output_images, input_normalize_value, output_normalize_value = input_images[0].to(device), output_images[0].to(device), input_normalize_value[0].to(device), output_normalize_value[0].to(device)
        # Pre process
        input_images, pad_info = pad_to_multiple_centered(input_images, mode='reflect', return_pad_info=True)

        slice_per_volume = input_images.size(0)
        if args.sample_middle_slices != 0 and slice_per_volume > args.sample_middle_slices:
            start_idx = (slice_per_volume - args.sample_middle_slices) // 2
            end_idx = start_idx + args.sample_middle_slices
            input_images = input_images[start_idx: end_idx]
            output_images = output_images[start_idx: end_idx]
            input_normalize_value = input_normalize_value[start_idx: end_idx]
            output_normalize_value = output_normalize_value[start_idx: end_idx]
        else:
            start_idx, end_idx = 0, slice_per_volume

        # There are too many slices of volume, so we split the batch to avoid a memory issue.
        if args.split_batch != 0 and input_images.size(0) > args.split_batch:
            num_splits = int(math.ceil(input_images.size(0) / args.split_batch))
            samples_list = []
            ssi = 0  # start slice idx
            for split in range(num_splits):
                # Sample images:
                micro_input_immages = input_images[ssi: ssi+args.split_batch]
                samples = model(micro_input_immages)
                samples = samples.to(torch.float32)  # [B, 1, H, W]
                ssi += args.split_batch
                samples_list.append(samples)
            samples = torch.cat(samples_list, dim=0)
        else:
            # sample images:
            samples = model(input_images)
            samples = samples.to(torch.float32)  # [B, 1, H, W]

        # # Post process
        if args.normalize_type in ['minmax', 'mean_minmax']:
            samples = samples * 0.5 + 0.5
        samples = samples[..., pad_info[0]: samples.shape[-2] - pad_info[1], pad_info[2]: samples.shape[-1] - pad_info[3]]
        samples = (torch.clip(samples, 0) * input_normalize_value[..., None, None, None]).to(torch.float32)
        # measurement
        if args.normalize_type in ['minmax', 'mean_minmax']:
            input_images = input_images * 0.5 + 0.5
        input_images = input_images[..., pad_info[0]: input_images.shape[-2] - pad_info[1], pad_info[2]: input_images.shape[-1] - pad_info[3]]
        input_images = (input_images * input_normalize_value[..., None, None, None]).to(torch.float32)
        # reference
        if args.normalize_type in ['minmax', 'mean_minmax']:
            output_images = output_images * 0.5 + 0.5
        output_images = (output_images * output_normalize_value[..., None, None, None]).to(torch.float32)

        if args.model_type == 'nafnet_mix':
            samples = samples * 0.8 + input_images * 0.2

        # Save dicom files:
        if args.save_dicom:
            batch_size = input_images.size(0)
            dicom_lq_files, dicom_hq_files = sorted(list(fname_lq.glob('IM*')), key=lambda s: int(s.stem[2:])) + sorted(list(fname_lq.glob('*.dcm'))), sorted(list(fname_hq.glob('IM*')), key=lambda s: int(s.stem[2:])) + sorted(list(fname_hq.glob('*.dcm')))

            for idx in range(batch_size):
                slice_ds = pydicom.dcmread(dicom_lq_files[idx + start_idx], force=True)
                slice_ds.PixelData = torch.clip(samples[idx, 0], 0).cpu().numpy().astype(slice_ds.pixel_array.dtype).tobytes() # ds.PixelData = image.tobytes()
                # modify some fields
                slice_ds.Rows, slice_ds.Columns = samples.shape[-2:]
                slice_ds.SeriesDescription = f"AI_{args.model_type}:{slice_ds.SeriesDescription}"
                slice_ds.SeriesInstanceUID = f'1.{name2seriesID[args.model_type]}.' + slice_ds.SeriesInstanceUID
                slice_ds.SOPInstanceUID = f'1.{name2seriesID[args.model_type]}.' + slice_ds.SOPInstanceUID
                slice_ds.SeriesNumber = slice_ds.SeriesNumber + 100 * name2seriesID[args.model_type]
                # save modified dicom
                slice_ds.save_as(new_subject_dir / dicom_lq_files[idx + start_idx].name, write_like_original=True)

        # normalize to [0, 1]
        samples = torch.clip(normalize_torch(samples), min=0, max=1)
        input_images = torch.clip(normalize_torch(input_images), min=0, max=1)
        output_images = torch.clip(normalize_torch(output_images), min=0, max=1)

        # Save results and display images:
        if args.display_image:
            batch_size = input_images.size(0)
            for idx in range(batch_size):
                # display images
                display_all_images(input_images[idx, 0], samples[idx, 0], output_images[idx, 0], '_'.join(fname_lq.parts[-5:]), idx + start_idx, all_image_path) # '_'.join(fname_lq.parts[-5:])
                # display_single_image(input_images[idx, 0], 'measurement_' + '_'.join(fname_lq.parts[-5:]), idx + start_idx, measurement_path)
                # display_single_image(samples[idx, 0], 'sample_' + '_'.join(fname_lq.parts[-5:]), idx + start_idx, sample_path)
                # display_single_image(output_images[idx, 0], 'reference_' + '_'.join(fname_lq.parts[-5:]), idx + start_idx, reference_path)

                # compute metrics
                measurement_image = input_images[idx: idx + 1].cpu()
                recon_image = samples[idx: idx + 1].detach().cpu()
                target_image = output_images[idx: idx + 1].cpu()

                # pad to target size
                measurement_image = pad_to_target_size_centered(measurement_image, target_image.shape[-2:])
                recon_image = pad_to_target_size_centered(recon_image, target_image.shape[-2:])

                recon_psnr_value = psnr(recon_image, target_image).item()
                recon_ssim_value = ssim(recon_image, target_image).item()
                measurement_psnr_value = psnr(measurement_image, target_image).item()
                measurement_ssim_value = ssim(measurement_image, target_image).item()

                measurement_image = torch.repeat_interleave(measurement_image, 3, dim=1) * 2 - 1
                recon_image = torch.repeat_interleave(recon_image, 3, dim=1) * 2 - 1
                target_image = torch.repeat_interleave(target_image, 3, dim=1) * 2 - 1

                recon_lpips_value = lpips(
                    torch.clip(recon_image, min=-1, max=1),
                    torch.clip(target_image, min=-1, max=1)
                ).item()
                measurement_lpips_value = lpips(
                    torch.clip(measurement_image, min=-1, max=1),
                    torch.clip(target_image, min=-1, max=1)
                ).item()

                # all_df.loc[len(all_df)] = [fname[idx], psnr_value, ssim_value, lpips_value]
                new_row = pd.DataFrame([[f"{fname_lq}_{idx + start_idx:03d}", measurement_psnr_value, measurement_ssim_value, measurement_lpips_value, recon_psnr_value, recon_ssim_value, recon_lpips_value,]], columns=columns)
                # append to csv
                new_row.to_csv(df_output_file, mode='a', header=False, index=False)

            # # TODO: delete after testing
            # if i >= 20:
            #     break

def main_single_input(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # save path
    save_path = Path(args.save) / f"test_{args.model_type}"
    save_path.mkdir(parents=True, exist_ok=True)

    measurement_path = save_path / 'measurement'
    sample_path = save_path / 'sample'
    all_image_path = save_path / 'all_image'
    measurement_path.mkdir(parents=True, exist_ok=True)
    sample_path.mkdir(parents=True, exist_ok=True)
    all_image_path.mkdir(parents=True, exist_ok=True)

    # Load model:
    if args.model_type in ['nafnet', 'nafnet_mix', 'nafnet_perceptual_loss', 'nafnet_max_normalize']:
        model_kwargs = {
            'type': 'NAFNet',
            'img_channel': 1,
            'width': 64,
            'enc_blk_nums': [2, 2, 4, 8],
            'middle_blk_num': 12,
            'dec_blk_nums': [2, 2, 2, 2],
            # 'upsampler_mode': 'interpolate',
        }
    else:
        model_kwargs = {
            'type': 'NAFNet',
            'img_channel': 1,
            'width': 64,
            'enc_blk_nums': [2, 2, 4, 8],
            'middle_blk_num': 12,
            'dec_blk_nums': [2, 2, 2, 2],
            'upsampler_mode': 'interpolate',
        }
    model = define_network(model_kwargs).to(device)
    ckpt_path = args.ckpt
    model = model.from_pretrained(ckpt_path) if Path(ckpt_path).is_dir() else load_pretrained_parameters(model, ckpt_path)
    model.eval()  # important!

    # Setup mri data:
    dataloader = create_dataloader(args, verbose=True, input_only=True)

    # Sampling loop:
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_images, input_normalize_value, fname_lq = batch
        fname_lq = fname_lq[0]
        fname_lq = Path(fname_lq)
        # establish new subject dir
        if len(fname_lq.parts) >= 5 and 'Siemens' in fname_lq.parts[-5]:
            new_subject_dir = fname_lq.parent.parent / f"AI_{args.model_type}_{fname_lq.parent.stem}" / fname_lq.stem
        else:
            new_subject_dir = fname_lq.parent / f"AI_{args.model_type}_{fname_lq.stem}"
        # if new_subject_dir.exists(): # already processed
        #     continue
        if args.save_dicom:
            new_subject_dir.mkdir(parents=True, exist_ok=True)

        input_images, input_normalize_value = input_images[0].to(device), input_normalize_value[0].to(device)
        # Pre process
        input_images, pad_info = pad_to_multiple_centered(input_images, mode='reflect', return_pad_info=True)

        slice_per_volume = input_images.size(0)
        if args.sample_middle_slices != 0 and slice_per_volume > args.sample_middle_slices:
            start_idx = (slice_per_volume - args.sample_middle_slices) // 2
            end_idx = start_idx + args.sample_middle_slices
            input_images = input_images[start_idx: end_idx]
            input_normalize_value = input_normalize_value[start_idx: end_idx]
        else:
            start_idx, end_idx = 0, slice_per_volume

        # There are too many slices of volume, so we split the batch to avoid a memory issue.
        if args.split_batch != 0 and input_images.size(0) > args.split_batch:
            num_splits = int(math.ceil(input_images.size(0) / args.split_batch))
            samples_list = []
            ssi = 0  # start slice idx
            for split in range(num_splits):
                # Sample images:
                micro_input_immages = input_images[ssi: ssi + args.split_batch]
                samples = model(micro_input_immages)
                samples = samples.to(torch.float32)  # [B, 1, H, W]
                ssi += args.split_batch
                samples_list.append(samples)
            samples = torch.cat(samples_list, dim=0)
        else:
            # sample images:
            samples = model(input_images)
            samples = samples.to(torch.float32)  # [B, 1, H, W]

        # # Post process
        if args.normalize_type in ['minmax', 'mean_minmax']:
            samples = samples * 0.5 + 0.5
        samples = samples[..., pad_info[0]: samples.shape[-2] - pad_info[1], pad_info[2]: samples.shape[-1] - pad_info[3]]
        samples = (samples * input_normalize_value[..., None, None, None]).to(torch.float32)
        # measurement
        if args.normalize_type in ['minmax', 'mean_minmax']:
            input_images = input_images * 0.5 + 0.5
        input_images = input_images[..., pad_info[0]: input_images.shape[-2] - pad_info[1], pad_info[2]: input_images.shape[-1] - pad_info[3]]
        input_images = (input_images * input_normalize_value[..., None, None, None]).to(torch.float32)

        if args.normalize_type in ['minmax', 'mean_minmax']:
            samples = samples * 0.8 + input_images * 0.2

        # Save dicom files:
        if args.save_dicom:
            batch_size = input_images.size(0)
            dicom_lq_files = sorted(list(fname_lq.glob('I*')))

            for idx in range(batch_size):
                slice_ds = pydicom.dcmread(dicom_lq_files[idx + start_idx], force=True)
                slice_ds.PixelData = torch.clip(samples[idx, 0], 0).cpu().numpy().astype(
                    slice_ds.pixel_array.dtype).tobytes()  # ds.PixelData = image.tobytes()
                # modify some fields
                slice_ds.Rows, slice_ds.Columns = samples.shape[-2:]
                slice_ds.SeriesDescription = f"AI_{args.model_type}:{slice_ds.SeriesDescription}"
                slice_ds.SeriesInstanceUID = f'1.{name2seriesID[args.model_type]}.' + slice_ds.SeriesInstanceUID
                slice_ds.SOPInstanceUID = f'1.{name2seriesID[args.model_type]}.' + slice_ds.SOPInstanceUID
                slice_ds.SeriesNumber = slice_ds.SeriesNumber + 100 * name2seriesID[args.model_type]
                # save modified dicom
                slice_ds.save_as(new_subject_dir / dicom_lq_files[idx + start_idx].name, write_like_original=True)

        # # TODO: delete after testing
        # if i >= 20:
        #     break

if __name__ == "__main__":
    parser = argparse.ArgumentParser() # /mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/ACA_data_transfer_organized_test # badGT_normal_reorganized # priority_test_data # ACA_test2_mini
    # data # /data/yuning/zhongjian/Data/aca_test_20260202/
    parser.add_argument("--data-path", default='/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/aca_test_20260202', type=str, help="Path to the dataset.")
    parser.add_argument("--num-workers", default=8, type=int, help="Number of dataloader workers.")
    parser.add_argument("--resolution", default=256, type=int, choices=[256, 320, 512], help="Image size.")
    parser.add_argument("--normalize-type", default='mean', type=str, choices=['mean', 'minmax', 'mean_minmax'], help="Normalization type.")
    parser.add_argument("--split-batch", default=6, type=int, help="Split batch size to avoid memory issue. 0 means no split.")
    parser.add_argument("--sample-middle-slices", default=0, type=int, help="If >0, only sample the middle N slices of each volume to save time.")
    # model
    parser.add_argument("--ckpt", default="./runs/train_nafnet/dists_loss_5e-2_hq_lr_1e-4/checkpoints/model_ema.pt", type=str, help="Optional path to a model checkpoint.")
    parser.add_argument("--model-type", default='nafnet_dists_loss_5e-2_hq_lr_1e-4', type=str, choices=['nafnet', 'nafnet_l1_only_hq_lr_1e-4', 'nafnet_dists_loss_1e-3', 'nafnet_dists_loss_1e-4', 'nafnet_dists_loss_1e-4_l2_reg', 'nafnet_dists_loss_1e-4_hq', 'nafnet_dists_loss_1e-4_hq_lr_1e-4', 'nafnet_dists_loss_1.0_hq_lr_1e-4', 'nafnet_dists_loss_0.1_hq_lr_1e-4', 'nafnet_dists_loss_1e-2_hq_lr_1e-4', 'nafnet_dists_loss_5e-2_hq_lr_1e-4', 'nafnet_dists_loss_1e-3_hq_lr_1e-4', 'nafnet_dists_loss_5e-2_hq_uii_all_data', 'nafnet_dists_loss_1e-6', 'nafnet_mix', 'nafnet_perceptual_loss', 'nafnet_max_normalize'], help="Type of diffusion model.")
    # general
    parser.add_argument("--save", default='./runs', type=str, help="Path to save sampled images.")
    parser.add_argument("--save-dicom", default=True, type=bool, help="Whether to save the sampled images as dicom files.")
    parser.add_argument("--display-image", default=False, type=bool, help="Whether to display and save the sampled images as png files.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    args = parser.parse_args()
    main(args)
    # main_single_input(args)
    # CUDA_VISIBLE_DEVICES=2 python nafnet_sample.py --model-type 'nafnet_dists_loss_5e-2_hq_uii_all_data' --split-batch 4 --ckpt ./runs/train_nafnet/dists_loss_5e-2_hq_uii_all_data/checkpoints/model/ --data-path /data/yuning/zhongjian/Data/aca_test_20260202/ --save ./runs --save-dicom True --display-image False
