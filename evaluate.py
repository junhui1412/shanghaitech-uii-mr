import argparse
import math
import os
from pathlib import Path

import clip
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import pandas as pd
import torch
import numpy as np
import random
import cv2
from src.data_utils.dicom_dataset import MRISubjectEvaluateDicomDataset
from src.module.maniqa.models.maniqa import MANIQA
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.module.musiq.model.backbone import resnet50_backbone
from src.module.musiq.model.model_main import IQARegression
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

method_type = [
    '',
    'UII',
    'nafnet_dists_loss_5e-2_hq_uii_all_data',
    'nafnet_dists_loss_5e-2_reggan_hq_test',
    'nafnet_dists_loss_5e-2_gan_uii_all_data',
]

""" configuration json """
class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Image(torch.utils.data.Dataset):
    def __init__(self, image_path, transform, num_crops=20):
        super(Image, self).__init__()
        self.img_name = image_path.split('/')[-1]
        self.img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = np.array(self.img).astype('float32') / 255
        self.img = np.transpose(self.img, (2, 0, 1))

        self.transform = transform

        c, h, w = self.img.shape
        print(self.img.shape)
        new_h = 224
        new_w = 224

        self.img_patches = []
        for i in range(num_crops):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            patch = self.img[:, top: top + new_h, left: left + new_w]
            self.img_patches.append(patch)

        self.img_patches = np.array(self.img_patches)

    def get_patch(self, idx):
        patch = self.img_patches[idx]
        sample = {'d_img_org': patch, 'score': 0, 'd_name': self.img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample

def create_dataloader(args, verbose=True):
    dataset = MRISubjectEvaluateDicomDataset(
        root=args.data_path,
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

def maniqa_evaluate(args):
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # save path
    save_path = Path(args.save) / f"evaluate_{args.model_type}"
    save_path.mkdir(parents=True, exist_ok=True)

    # config file
    # config = Config({
    #     # image path
    #     "image_path": "./test_images/kunkun.png",
    #
    #     # valid times
    #     "num_crops": 20,
    #
    #     # model
    #     "patch_size": 8,
    #     "img_size": 224,
    #     "embed_dim": 768,
    #     "dim_mlp": 768,
    #     "num_heads": [4, 4],
    #     "window_size": 4,
    #     "depths": [2, 2],
    #     "num_outputs": 1,
    #     "num_tab": 2,
    #     "scale": 0.8,
    #
    #     # checkpoint path
    #     "ckpt_path": "./ckpt_koniq10k.pt",
    # })

    # data load
    # Img = Image(image_path=config.image_path,
    #             transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
    #             num_crops=20)

    # create dataloader
    dataloader = create_dataloader(args, verbose=True)

    # model defination
    net = MANIQA(embed_dim=768, num_outputs=1, dim_mlp=768,
                 patch_size=8, img_size=224, window_size=4,
                 depths=[2, 2], num_heads=[4, 4], num_tab=2, scale=0.8)

    net.load_state_dict(torch.load('./model_weight/maniqa/ckpt_koniq10k.pt'), strict=False)
    print(f"NAFNet Model Parameters: {sum(p.numel() for p in net.parameters()):,}")
    net = net.to(device)
    net.requires_grad_(False) # important!

    # define random crop transform
    trans = transforms.Compose([
        transforms.RandomCrop(224, pad_if_needed=True, padding_mode='reflect'),
    ])

    # save metrics
    columns = ['FileName', 'MANIQA']
    all_df = pd.DataFrame(columns=columns)
    df_output_file = save_path / 'maniqa_metrics.csv'
    if not Path(df_output_file).exists():
        all_df.to_csv(df_output_file, index=False)

    # evaluate
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if i <=1:
            continue
        images, fname = batch
        images, fname = images[0], Path(fname[0])
        # patchify
        image_patch_list = []
        for i in range(args.num_crops):
            patch = trans(images)
            image_patch_list.append(patch)
        images = torch.stack(image_patch_list, dim=0)  # [num_crops, b, c, h, w]
        # file name
        filepath_name = '_'.join(fname.parts[-4:])
        # record score
        avg_score = torch.zeros((images.shape[1],), device=device)

        # There are too many slices of volume, so we split the batch to avoid a memory issue.
        if args.split_batch != 0 and images.size(1) > args.split_batch:
            num_splits = int(math.ceil(images.size(1) / args.split_batch))
            ssi = 0  # start slice idx
            for split in range(num_splits):
                micro_images = images[:, ssi: ssi + args.split_batch]
                for nc in tqdm(range(args.num_crops)):
                    micro_patch_images = micro_images[nc].to(device)
                    micro_patch_images_3c = torch.repeat_interleave(micro_patch_images, 3, dim=1)
                    micro_score = net(micro_patch_images_3c)
                    avg_score[ssi: ssi + args.split_batch] += micro_score
                ssi += args.split_batch
        else:
            for nc in tqdm(range(args.num_crops)):
                patch_images = images[nc].to(device)
                patch_images_3c = torch.repeat_interleave(patch_images, 3, dim=1)
                score = net(patch_images_3c)
                avg_score += score

        final_score = (avg_score / args.num_crops).mean().cpu().numpy()
        print("Image {} score: {}".format(filepath_name, final_score))
        new_row = pd.DataFrame([[f"{filepath_name}", final_score]], columns=columns)
        # append to csv
        new_row.to_csv(df_output_file, mode='a', header=False, index=False)
        if i >= 2: # TODO: delete after test
            break

def musiq_evaluate(args):
    setup_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # save path
    save_path = Path(args.save) / f"evaluate_{args.model_type}"
    save_path.mkdir(parents=True, exist_ok=True)

    # create dataloader
    dataloader = create_dataloader(args, verbose=True)

    # model defination
    config = Config({
        # ViT structure
        'n_enc_seq': 32 * 24 + 12 * 9 + 7 * 5,  # input feature map dimension (N = H*W) from backbone
        'n_layer': 14,  # number of encoder layers
        'd_hidn': 384,  # input channel of encoder (input: C x N)
        'i_pad': 0,
        'd_ff': 384,  # feed forward hidden layer dimension
        'd_MLP_head': 1152,  # hidden layer of final MLP
        'n_head': 6,  # number of head (in multi-head attention)
        'd_head': 384,  # channel of each head -> same as d_hidn
        'dropout': 0.1,  # dropout ratio
        'emb_dropout': 0.1,  # dropout ratio of input embedding
        'layer_norm_epsilon': 1e-12,
        'n_output': 1,  # dimension of output
        'Grid': 10,  # grid of 2D spatial embedding
        'scale_1': 384,  # multi-scale
        'scale_2': 224,  # multi-scale
    })
    model_backbone = resnet50_backbone()
    model_transformer = IQARegression(config)
    ckpt = torch.load('./model_weight/musiq/ckpt_koniq10k.pt')
    model_backbone.load_state_dict(ckpt['model_backbone_state_dict'])
    model_transformer.load_state_dict(ckpt['model_transformer_state_dict'])
    print(f"Model Backbone Parameters: {sum(p.numel() for p in model_backbone.parameters()):,}")
    print(f"Model Transformer Parameters: {sum(p.numel() for p in model_transformer.parameters()):,}")
    model_backbone = model_backbone.to(device)
    model_transformer = model_transformer.to(device)
    model_backbone.requires_grad_(False)  # important!
    model_transformer.requires_grad_(False)  # important!

    # save metrics
    columns = ['FileName', 'MUSIQ']
    all_df = pd.DataFrame(columns=columns)
    df_output_file = save_path / 'musiq_metrics.csv'
    if not Path(df_output_file).exists():
        all_df.to_csv(df_output_file, index=False)

    # transforms
    trans = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # evaluate
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if i <= 1:
            continue
        images, fname = batch
        images, fname = images[0], Path(fname[0])

        w, h = images.shape[-2:]
        # Scale 1: Calculate height based on config.scale_1 (width)
        h_1 = int(h * (config.scale_1 / w))
        # F.resize takes size as (height, width)
        images_scale_1 = F.resize(images, size=[h_1, config.scale_1], interpolation=InterpolationMode.BICUBIC)

        # Scale 2: Calculate height based on config.scale_2 (width)
        h_2 = int(h * (config.scale_2 / w))
        images_scale_2 = F.resize(images, size=[h_2, config.scale_2], interpolation=InterpolationMode.BICUBIC)

        filepath_name = '_'.join(fname.parts[-4:])
        avg_score = torch.zeros((images.shape[0],), device=device)

        # input mask (batch_size x len_sqe+1)
        mask_inputs = torch.ones(images.shape[0], config.n_enc_seq + 1).to(config.device)

        # There are too many slices of volume, so we split the batch to avoid a memory issue.
        if args.split_batch != 0 and images.size(0) > args.split_batch:
            num_splits = int(math.ceil(images.size(0) / args.split_batch))
            ssi = 0  # start slice idx
            for split in range(num_splits):
                micro_images = images[ssi: ssi + args.split_batch].to(device)
                micro_images_scale_1 = images_scale_1[ssi: ssi + args.split_batch].to(device)
                micro_images_scale_2 = images_scale_2[ssi: ssi + args.split_batch].to(device)
                micro_images_3c = trans(micro_images.repeat(1, 3, 1, 1))
                micro_images_scale_1_3c = trans(micro_images_scale_1.repeat(1, 3, 1, 1))
                micro_images_scale_2_3c = trans(micro_images_scale_2.repeat(1, 3, 1, 1))
                feat_dis_org = model_backbone(micro_images_3c)
                feat_dis_scale_1 = model_backbone(micro_images_scale_1_3c)
                feat_dis_scale_2 = model_backbone(micro_images_scale_2_3c)
                # quality prediction
                pred_score = model_transformer(mask_inputs[ssi: ssi + args.split_batch], feat_dis_org, feat_dis_scale_1, feat_dis_scale_2)
                avg_score[ssi: ssi + args.split_batch] += pred_score
                ssi += args.split_batch
        else:
            images = images.to(device)
            images_scale_1 = images_scale_1.to(device)
            images_scale_2 = images_scale_2.to(device)
            images_3c = torch.repeat_interleave(images, 3, dim=1)
            images_scale_1_3c = torch.repeat_interleave(images_scale_1, 3, dim=1)
            images_scale_2_3c = torch.repeat_interleave(images_scale_2, 3, dim=1)
            feat_dis_org = model_backbone(images_3c)
            feat_dis_scale_1 = model_backbone(images_scale_1_3c)
            feat_dis_scale_2 = model_backbone(images_scale_2_3c)
            pred_score = model_transformer(mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2)
            avg_score += pred_score

        final_score = avg_score.mean().cpu().numpy()
        print("Image {} score: {}".format(filepath_name, final_score))
        new_row = pd.DataFrame([[f"{filepath_name}", final_score]], columns=columns)
        # append to csv
        new_row.to_csv(df_output_file, mode='a', header=False, index=False)
        if i >= 2:  # TODO: delete after test
            break

def clip_iqa_evaluate(args):
    setup_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # save path
    save_path = Path(args.save) / f"evaluate_metrics_clip_iqa"
    save_path.mkdir(parents=True, exist_ok=True)

    # create dataloader
    dataloader = create_dataloader(args, verbose=True)

    # model defination
    model, preprocess = clip.load(args.model_name, device=device)
    model.requires_grad_(False)  # important!

    target_image_size = 512
    if "ViT" in args.model_name:
        # Dynamic Adaptive Position Encoding
        patch_size = model.visual.conv1.kernel_size[0]
        target_grid = target_image_size // patch_size
        pos_embed = model.visual.positional_embedding.data
        embed_dim = pos_embed.shape[1]
        orig_grid = int(math.sqrt(pos_embed.shape[0] - 1))
        spatial_pos_resized = torch.nn.functional.interpolate(
            pos_embed[1:].transpose(0, 1).reshape(1, embed_dim, orig_grid, orig_grid),
            size=(target_grid, target_grid),
            mode='bicubic', align_corners=False
        )
        new_pos = torch.cat([pos_embed[0:1], spatial_pos_resized.reshape(1, embed_dim, -1).transpose(1, 2).squeeze(0)], dim=0)
        model.visual.positional_embedding = torch.nn.Parameter(new_pos)
        print(f"Adapted ViT Positional Embedding to {target_image_size}px (Grid: {target_grid})")

    elif "RN" in args.model_name:
        # === ResNet 系列的处理逻辑 ===
        feature_size = target_image_size // 32  # RN 的总下采样率通常是 32
        attnpool = model.visual.attnpool
        pos_embed = attnpool.positional_embedding.data
        embed_dim = pos_embed.shape[1]

        class_pos = pos_embed[0:1, :]
        spatial_pos = pos_embed[1:, :]

        # 原生 RN50 的 attnpool 空间是 7x7
        spatial_pos = spatial_pos.transpose(0, 1).reshape(1, embed_dim, 7, 7)

        spatial_pos_resized = torch.nn.functional.interpolate(
            spatial_pos, size=(feature_size, feature_size), mode='bicubic', align_corners=False
        )

        num_patches = feature_size ** 2
        spatial_pos_resized = spatial_pos_resized.reshape(1, embed_dim, num_patches).transpose(1, 2).squeeze(0)

        new_pos_embed = torch.cat([class_pos, spatial_pos_resized], dim=0)
        attnpool.positional_embedding = torch.nn.Parameter(new_pos_embed)
        print(f" RN Series: Positional embedding adapted for {target_image_size}x{target_image_size}.")

    # pre-compute text features (Text Anchors)
    # prompts = ["A clean photo", "A noisy photo"]
    # prompts = ["A clean, clear, noise-free magnetic resonance image with more detail.", "A dirty, blurry, noisy magnetic resonance image"]
    prompts = ["A clean magnetic resonance image.", "A noisy magnetic resonance image."]
    text_inputs = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # save metrics
    metrics_output_file = save_path / 'clip_iqa_metrics.txt'

    # transforms
    trans = transforms.Compose([
        transforms.Resize(512, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(512),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # evaluate
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if i <= 1:
            continue
        volumes, volume_name_list = batch
        volumes, volume_name_list = volumes[0], [volume_name[0] for volume_name in volume_name_list]

        num_volumes, num_slices = volumes.size(0), volumes.size(1)
        all_scores = torch.zeros((num_volumes, num_slices), device=device)

        for num_volume in range(num_volumes):
            volume = volumes[num_volume]
            # There are too many slices of volume, so we split the batch to avoid a memory issue.
            split_batch = args.split_batch if args.split_batch != 0 else num_slices
            num_splits = int(math.ceil(num_slices / split_batch))

            with torch.no_grad():
                for s in range(num_splits):
                    start_idx = s * split_batch
                    end_idx = min((s + 1) * split_batch, num_slices)

                    # slices choice
                    micro_batch = volume[start_idx:end_idx].to(device)

                    # convert to 3 channels (to accommodate CLIP)
                    if micro_batch.shape[1] == 1:
                        micro_batch = (micro_batch * 255.0).to(torch.uint8) / 255.0
                        micro_batch = trans(micro_batch.repeat(1, 3, 1, 1))

                    # image features extraction
                    image_features = model.encode_image(micro_batch)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    # compute cosine similarity and apply Softmax
                    # similarity shape: [batch, 2]
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                    # select pos prompt score
                    all_scores[num_volume, start_idx:end_idx] = similarity[:, 0]

        avg_volume_score = all_scores.mean(dim=1).cpu()
        metrics_dict = {str(volume_name): avg_volume_score[i].item() for i, volume_name in enumerate(volume_name_list)}
        print("Score: {}".format(metrics_dict))
        with open(metrics_output_file, 'a', encoding='utf-8') as f:
            for i, volume_name in enumerate(volume_name_list):
                f.write(str(volume_name) + f': {avg_volume_score[i].item()}\n')
            f.write('\n')
        if i >= 2:  # TODO: delete after test
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser() # aca_test_20260202 # test_data
    parser.add_argument("--data-path", default='/data/yuning/zhongjian/Data/test_data', type=str, help="Path to the dataset.")
    parser.add_argument("--num-workers", default=1, type=int, help="Number of dataloader workers.")
    # parser.add_argument("--normalize-type", default='mean_std', type=str, choices=['mean', 'minmax', 'mean_std'], help="Normalization type.")
    parser.add_argument("--split-batch", default=40, type=int, help="Split batch size to avoid memory issue. 0 means no split.")
    parser.add_argument("--num-crops", default=20, type=int, help="Number of crops.")
    # model  # nafnet_dists_loss_5e-2_gan_uii_all_data # UII
    parser.add_argument("--model-name", default="RN50", choices=["ViT-B/32", "RN50"], type=str, help="Name of the evaluate model.")
    # parser.add_argument("--model-type", default='', type=str, choices=method_type, help="Type of diffusion model.")
    # general
    parser.add_argument("--evaluate_type", default="clip_iqa", type=str, choices=["maniqa", "musiq", "clip_iqa"], help="Type of evaluation.")
    parser.add_argument("--save", default='./runs', type=str, help="Path to save evaluation results.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    args = parser.parse_args()
    if args.evaluate_type == "maniqa":
        maniqa_evaluate(args)
    elif args.evaluate_type == "musiq":
        musiq_evaluate(args)
    elif args.evaluate_type == "clip_iqa":
        clip_iqa_evaluate(args)
    else:
        raise ValueError("Unknown evaluation type.")

