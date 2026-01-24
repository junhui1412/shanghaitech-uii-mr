# This is a repository for MR image denoising that includes multiple models.


## 💻 Local Setup

### 1. Prepare the environment


- python 3.10  
- PyTorch 2.5.1  
- CUDA 12.1  

Other versions of PyTorch with proper CUDA should work but are not fully tested.

```bash
# In shanghaitech-uii-mr folder
conda create -n rddm python=3.10
conda activate rddm

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install accelerate einops ema-pytorch pydicom wandb omegaconf timm torchmetrics pandas matplotlib # SimpleITK Augmentor

pip install scipy scikit-image opencv-python # NAFNet

pip install dists-pytorch # Deep Image Structure and Texture Similarity (DISTS Loss)

pip install diffusers # latent bbdm

pip install transformers # For PixNerDiT

```

### 2. Prepare the **MRI** dataset

use `preprocess_mr_dicom.py` to preprocess the dataset. 

This process will precompute the 'volume mean', 'volume max' and 'volume percentile max' for each subject and save them in each Dicom file's metadata. 

```python
# uncomment and run the following command in preprocess_mr_dicom.py
# You can modify the parameters such as 'root_path' in the function 'multi_process' as needed.
if __name__ == '__main__':
      multi_process(num_processes=1, split='training') # training, validation, testing
```

### 3. Training

You can train different models using the corresponding training scripts and configuration files (find in the folder `./cfg/train_configs/*.yaml`).

- Train NAFNet for MR image denoising using `train_nafnet.py`.
```bash
accelerate launch --config_file ./cfg/accelerate_configs/accelerate_config.yaml --num_processes 1 train_nafnet.py --config ./cfg/train_configs/train_nafnet.yaml

```

- Train BBDM for MR image denoising using `train_bbdm.py`.
```bash
accelerate launch --config_file ./cfg/accelerate_configs/accelerate_config.yaml --num_processes 1 train_bbdm.py --config ./cfg/train_configs/train_bbdm.yaml
```

- Train EDM for MR image denoising using `train_edm.py`.
```bash
accelerate launch --config_file ./cfg/accelerate_configs/accelerate_config.yaml --num_processes 1 train_edm.py --config ./cfg/train_configs/train_edm.yaml
```

### 4. Inference

You can perform inference using the corresponding inference scripts.

- Inference NAFNet for MR image denoising using `nafnet_sample.py`.
```bash
python nafnet_sample.py --model-type 'nafnet_dists_loss_5e-2_hq_lr_1e-4' --split-batch 4 --ckpt /public_bme2/bme-dgshen/ZhongjianJiang/projects/shanghaitech-uii-mr/runs/train_nafnet/dists_loss_5e-2_hq_lr_1e-4/checkpoints/model_ema.pt --data-path /public_bme2/bme-dgshen/ZhongjianJiang/projects/Data/aca_test/artifacts --save /public_bme2/bme-dgshen/ZhongjianJiang/projects/shanghaitech-uii-mr/runs --save-dicom True --display-image False
```

- Inference BBDM for MR image denoising using `bbdm_sample.py`.
```bash
python bbdm_sample.py --model-type 'bbdm_unet_256c' --split-batch 4 --ckpt /public_bme2/bme-dgshen/ZhongjianJiang/projects/shanghaitech-uii-mr/runs/train_bbdm/unet_256c/checkpoints/model_ema.pt --data-path /public_bme2/bme-dgshen/ZhongjianJiang/projects/Data/aca_test/artifacts --save /public_bme2/bme-dgshen/ZhongjianJiang/projects/shanghaitech-uii-mr/runs --save-dicom True --display-image False
```

- Inference EDM for MR image denoising using `edm_sample.py`.
```bash
python edm_sample.py --model-type 'edm_unet_256c' --split-batch 4 --ckpt /public_bme2/bme-dgshen/ZhongjianJiang/projects/shanghaitech-uii-mr/runs/train_edm/unet_256c/checkpoints/model_ema.pt --data-path /public_bme2/bme-dgshen/ZhongjianJiang/projects/Data/aca_test/artifacts --save /public_bme2/bme-dgshen/ZhongjianJiang/projects/shanghaitech-uii-mr/runs --save-dicom True --display-image False
```

