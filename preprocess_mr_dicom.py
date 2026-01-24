# import json
import shutil
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
# import h5py
# import torch
from tqdm import tqdm

from src.data_utils.dicom_dataset import process_subject_dir


# from pyspark.sql import SparkSession
# import xml.etree.ElementTree as ET
# from src.data_utils.fastmri.data.transforms import tensor_to_complex_np, to_tensor
# from src.data_utils.fastmri import utils

def replace_specific_parent(path: Path, old_name: str, new_name: str) -> Path:
    parts = list(path.parts)
    parts = [new_name if part == old_name else part for part in parts]
    return Path(*parts)

def process(data_path):
    print("processing:", str(data_path))

    slice_files = sorted(list(data_path.glob('IM*')) + list(data_path.glob('*.dcm')))
    slice_list = []
    # gt_slice_list = []
    for slice_file in slice_files:
        try:
            slice_list.append(pydicom.dcmread(slice_file, force=True).pixel_array)
        except Exception as e:
            print(f"Error reading {slice_file}: {e}")
            return f"Error reading {slice_file}: {e}"
        # gt_slice_file = replace_fast_parent(slice_file)
        # gt_slice_list.append(pydicom.dcmread(gt_slice_file, force=True).pixel_array)
    try:
        slices = np.stack(slice_list)
        # gt_slices = np.stack(gt_slice_list)
        volume_mean = np.mean(slices)
        # gt_volume_mean = np.mean(gt_slices)
        volume_max = np.max(slices)
        volume_quantile_9995 = np.quantile(slices, 0.9995)
    except Exception as e:
        print(f"Error stacking slices in {data_path}: {e}")
        return f"Error stacking slices in {data_path}: {e}"

    # Add volume mean to each slice and save back
    for slice_file in slice_files:
        save_file_path = replace_specific_parent(slice_file, "training_data", "training_data_processed")
        save_file_path.parent.mkdir(parents=True, exist_ok=True)
        slice_ds = pydicom.dcmread(slice_file, force=True)
        # slice_ds.VolumeMeanValue = str(volume_mean)

        tag = (0x0019, 0x1001)
        data_element = pydicom.DataElement(tag, 'DS', str(volume_mean))
        slice_ds.add(data_element)

        tag = (0x0019, 0x1002)
        data_element = pydicom.DataElement(tag, 'DS', str(volume_max))
        slice_ds.add(data_element)

        tag = (0x0019, 0x1003)
        data_element = pydicom.DataElement(tag, 'DS', str(volume_quantile_9995))
        slice_ds.add(data_element)

        slice_ds.save_as(save_file_path)
        # print(pydicom.dcmread(slice_file, force=True)[(0x0019, 0x1001)])

        # gt_slice_file = replace_fast_parent(slice_file)
        # gt_slice_ds = pydicom.dcmread(gt_slice_file, force=True)
        # gt_slice_ds.VolumeMeanValue = float(gt_volume_mean)
        # gt_slice_ds.save_as(gt_slice_file)

    return f'{str(data_path)} process successfully!'

def multi_process(root_path, split='training', num_processes=10):
    import multiprocessing as mp
    from functools import partial
    root_path = Path(root_path)
    data_path = root_path / split
    if split in ["GE", "Philip", "Siemens", "DeepRecon"]:
        subject_dirs = sorted(list(data_path.glob('*/*/*/*/')))
    else:
        subject_dirs = sorted(list(data_path.glob('*/*/*/*/*/')))

    processpool = mp.Pool(processes=num_processes)
    main_func = partial(
        process,
    )
    processes = processpool.imap_unordered(main_func, subject_dirs, chunksize=5)

    for stdout in tqdm(processes, total=len(subject_dirs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        print(stdout)
        pass
        # break

def normalize_torch(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= torch.amin(img, dim=(1, 2, 3), keepdim=True)  # np.min(img)
    img /= torch.amax(img, dim=(1, 2, 3), keepdim=True)  # np.max(img)
    return img

def plot_images():
    from src.data_utils.dicom_dataset import MRIVolumeDicomDataset
    import matplotlib.pyplot as plt
    dataset = MRIVolumeDicomDataset(
        root='/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/deformable_registration_splited_processed/testing',
    )
    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        input_images, output_images, input_normalize_value, output_normalize_value, fname_lq, _ = data
        fname = '_'.join(Path(fname_lq).parts[-5:])
        if Path(f"./check_images/test_png_images/{fname}.png").exists():
            continue
        input_images = input_images * input_normalize_value[:, None, None, None]
        input_images = normalize_torch(input_images)
        output_images = output_images * output_normalize_value[:, None, None, None]
        output_images = normalize_torch(output_images)
        mid_slice_idx = input_images.shape[0] // 2
        fig, axes = plt.subplots(1, 2, dpi=300)

        axes[0].imshow(input_images[mid_slice_idx, 0].cpu().numpy(), cmap='gray')
        axes[0].axis('off')
        axes[0].set_title("Low Quality", fontsize=16)

        axes[1].imshow(output_images[mid_slice_idx, 0].cpu().numpy(), cmap='gray')
        axes[1].axis('off')
        axes[1].set_title("High Quality", fontsize=16)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
        # plt.show()
        plt.savefig(f"./check_images/test_png_images/{fname}.png", bbox_inches='tight', pad_inches=0.1)

        # # TODO: delete after debugging
        # if i > 0:
        #     break

class ImageDataset(Dataset):
    """
            A concrete class for handling image datasets, inherits from DiffusionData.

            This class is responsible for loading images from a specified directory,
            applying transformations to center crop the squared images of given resolution.

            Supported extension : ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
            Output data range   : [-1, 1]
        """

    def __init__(self, root, transform=None, resolution=256, resize_image=False, start_id=None, end_id=None):
        # Define the file extensions to search for
        extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        self.data = [file for ext in extensions for file in Path(root).rglob(ext)]
        self.data = sorted(set(self.data))

        # Subset the dataset
        self.data = self.data[start_id: end_id]
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution) if resize_image else nn.Identity(),
            transforms.CenterCrop(resolution),
            transforms.Normalize((0.5), (0.5)),
        ]) if transform is None else transform
        self.res = resolution

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        file = self.data[i]
        fname = Path(file).stem
        img = self.trans(Image.open(file).convert('RGB'))  # Convert to RGB and apply transformations
        if img.shape[0] == 1:
            img = torch.cat([img] * 3, dim=0)
        return img, fname

def create_dataloader(data_type='mri'):
    from src.data_utils.dicom_dataset import MRIDicomDataset
    if data_type == 'mri':
        dataset = MRIDicomDataset(
            root='/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/deformable_registration_splited_processed/validation',
            resolution=512,
            normalize_type='minmax',
        )
    else:
        dataset = ImageDataset(root="/mnt/e/deeplearning/data/computer_vision/Flickr/Flickr2K", resolution=768, resize_image=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    return dataloader


def check_images():
    from src.data_utils.dicom_dataset import MRIDicomDataset
    from torch.utils.data import DataLoader
    import warnings
    warnings.filterwarnings("error", category=RuntimeWarning)
    dataset = MRIDicomDataset(
        root='/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/deformable_registration_splited_processed/training',
        normalize_type='minmax',
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8)
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_images, output_images, input_normalize_value, output_normalize_value = data
        # print(input_normalize_value, output_normalize_value)
        pass
        # break

def read_dicom():
    dicom_path_in = Path("/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/ACA_data_transfer_organized/CSPINE/DICOHuangXiaoYan/FAT_Sag_T2_Flex_FAST/IM0")
    dicom_path_out = Path("/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/ACA_data_transfer_organized/CSPINE/DICOHuangXiaoYan/FAT_Sag_T2_Flex/IM0")
    dicom_path_ai = Path("/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/ACA_data_transfer_organized/CSPINE/DICOHuangXiaoYan/AI_FAT_Sag_T2_Flex_FAST/IM0")
    ds_in = pydicom.dcmread(dicom_path_in, force=True)
    ds_out = pydicom.dcmread(dicom_path_out, force=True)
    ds_ai = pydicom.dcmread(dicom_path_ai, force=True)
    print(ds_in)
    # print("VolumeMeanValue:", ds[(0x0019, 0x1001)].value)
    # print("VolumeMaxValue:", ds[(0x0019, 0x1002)].value)
    # print("VolumeQuantile9995Value:", ds[(0x0019, 0x1003)].value)

def test_vae():
    from diffusers import AutoencoderKLFlux2, AutoencoderKL
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_type = 'image'  # 'mri' or 'image'
    vae_type = 'flux1'  # 'flux2' or 'flux1'
    if vae_type == 'flux2':
        vae = AutoencoderKLFlux2.from_pretrained(pretrained_model_name_or_path=f'./model_weight/FLUX.2-dev', subfolder="vae").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=f'./model_weight/FLUX.1-dev', subfolder="vae").to(device)
    dataloader = create_dataloader(data_type=data_type)
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        if data_type == 'mri':
            input_images, output_images, input_normalize_value, output_normalize_value = data
            input_images, output_images = input_images.to(device), output_images.to(device)
            output_images_3c = output_images.repeat(1, 3, 1, 1)
        else:
            output_images = data[0]
            output_images_3c = output_images.to(device)

        latent = vae.encode(output_images_3c).latent_dist.mean # mean sample()
        recon_ouput_images = vae.decode(latent).sample
        recon_ouput_images = torch.clip(recon_ouput_images, -1, 1)
        save_image(torch.cat([output_images_3c, recon_ouput_images], dim=0), f'./vae_{vae_type}_recon_{data_type}_{i}.png', nrow=2, normalize=True, value_range=(-1, 1))

        if i >= 0:
            break

def filter_low_quality_siemens_data():
    excel_filepath = r"./extra_cfg/Siemens.xlsx"
    data_path = r"/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/deformable_registration_splited_processed/training"
    data_type = '15T_Siemens_MR'
    data_path = Path(data_path)

    columns = ['Vendor', 'Date', 'Organ', 'Sequence', 'Patient']
    high_quality_subjects = pd.DataFrame(columns=columns)
    high_quality_output_file = './extra_cfg/high_quality_siemens.csv'
    if not Path(high_quality_output_file).exists():
        high_quality_subjects.to_csv(high_quality_output_file, index=False)

    df_data = pd.read_excel(excel_filepath)
    result = df_data[df_data['Qualified'] == '√']
    subject_list = []
    for item in result.itertuples(index=False):
        filename, body_part, seq, _, _ = item
        filename = filename.split('_')[0]
        search_pattern = f'{data_type}/{filename}*/{body_part}/aca*{seq}/*/'
        subject_path = sorted(list(data_path.glob(search_pattern)))
        subject_list += subject_path

    subject_list = list(sorted(set(subject_list), key=lambda path: list(path.parts)[-4]))
    for subject in subject_list:
        parts = list(subject.parts)
        vendor = parts[-5]
        date = parts[-4]
        organ = parts[-3]
        sequence = parts[-2]
        patient = parts[-1]
        new_row = pd.DataFrame([[vendor, date, organ, sequence, patient]], columns=columns)
        new_row.to_csv(high_quality_output_file, mode='a', header=False, index=False)

def filter_low_quality_philip_data():
    excel_filepath = r"./extra_cfg/Philip.xlsx"
    data_path = r"/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/deformable_registration_splited_processed/training"
    data_type = '15T_Philip_MR' # 15T_Philip_MR
    data_path = Path(data_path)

    columns = ['Vendor', 'Date', 'Patient', 'Organ', 'Sequence']
    high_quality_subjects = pd.DataFrame(columns=columns)
    high_quality_output_file = './extra_cfg/high_quality_philip_ge.csv'
    if not Path(high_quality_output_file).exists():
        high_quality_subjects.to_csv(high_quality_output_file, index=False)

    df_data = pd.read_excel(excel_filepath)
    result = df_data[df_data['Qualified'] == '√']
    subject_list = []
    for item in result.itertuples(index=False):
        filename, id, body_part, seq, _, _, _ = item
        seq = re.sub(r'_\d+$', '', seq)
        search_pattern = f'{data_type}/{filename}/*/{body_part}/{seq}*FAST*/'
        subject_path = sorted(list(data_path.glob(search_pattern)))
        subject_list += subject_path

    subject_list = list(sorted(set(subject_list), key=lambda path: list(path.parts)[-4]))
    for subject in subject_list:
        parts = list(subject.parts)
        vendor = parts[-5]
        date = parts[-4]
        patient = parts[-3]
        organ = parts[-2]
        sequence = parts[-1]
        new_row = pd.DataFrame([[vendor, date, patient, organ, sequence]], columns=columns)
        new_row.to_csv(high_quality_output_file, mode='a', header=False, index=False)
    pass

def filter_low_quality_ge_data():
    excel_filepath = r"./extra_cfg/GE.xlsx"
    data_path = r"/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/deformable_registration_splited_processed/training"
    data_type = '15T_GE_MR' # 15T_GE_MR
    data_path = Path(data_path)

    columns = ['Vendor', 'Date', 'Patient', 'Organ', 'Sequence']
    high_quality_subjects = pd.DataFrame(columns=columns)
    high_quality_output_file = './extra_cfg/high_quality_philip_ge.csv'
    if not Path(high_quality_output_file).exists():
        high_quality_subjects.to_csv(high_quality_output_file, index=False)

    df_data = pd.read_excel(excel_filepath)
    result = df_data[df_data['FAST'] == '√'][df_data['FAST2'] == '√']
    subject_list = []
    for item in result.itertuples(index=False):
        filename, id, body_part, seq = item[:4]
        seq = re.sub(r'_\d+$', '', seq)
        search_pattern = f'{data_type}/{filename}/*/{body_part}/{seq}*FAST*/'
        subject_path = sorted(list(data_path.glob(search_pattern)))
        subject_list += subject_path

    subject_list = list(sorted(set(subject_list), key=lambda path: list(path.parts)[-4]))
    for subject in subject_list:
        parts = list(subject.parts)
        vendor = parts[-5]
        date = parts[-4]
        patient = parts[-3]
        organ = parts[-2]
        sequence = parts[-1]
        new_row = pd.DataFrame([[vendor, date, patient, organ, sequence]], columns=columns)
        new_row.to_csv(high_quality_output_file, mode='a', header=False, index=False)
    pass

def test_read_csv(root, is_siemens=False, duplicate_check=False):
    subject_dirs = []
    if is_siemens:
        df_filter_files = pd.read_csv('./extra_cfg/high_quality_siemens.csv')
        for item in df_filter_files.itertuples(index=False):
            vendor, date, organ, sequence, patient = item
            subject_dir = Path(root, vendor, str(date), organ, sequence, patient)
            subject_dirs.append(subject_dir)
    else:
        df_filter_files = pd.read_csv('./extra_cfg/high_quality_philip_ge.csv')
        for item in df_filter_files.itertuples(index=False):
            vendor, date, patient, organ, sequence = item
            subject_dir = Path(root, vendor, str(date), patient, organ, sequence)
            subject_dirs.append(subject_dir)
    return subject_dirs

def reorganize_directory_structure():
    root_path = r"/mnt/e/deeplearning/data/mri_reconstruction/normal/" # badGT # normal
    root_path = Path(root_path)
    new_root_path = r"/mnt/e/deeplearning/data/mri_reconstruction/normal_reorganized/" # badGT # normal
    new_root_path = Path(new_root_path)
    subject_dirs = sorted(list(root_path.glob('*/*/*/*/*/*/')))
    for subject_dir in subject_dirs:
        print(subject_dir)
    for subject in tqdm(subject_dirs, total=len(subject_dirs)):
        parts = list(subject.parts)
        vendor = parts[-6]
        date = parts[-5]
        patient = parts[-4]
        organ = parts[-3]
        sequence = parts[-2]
        acceleration = parts[-1]

        if '_ACA' in acceleration:
            acceleration = acceleration.replace('_ACA', '')
            new_sequence = f"AI_{sequence}_{acceleration}"
        elif acceleration == 'GT':
            new_sequence = sequence
        else:
            new_sequence = f"{sequence}_{acceleration}"
        new_subject_path = new_root_path / vendor / date / patient / organ / new_sequence
        new_subject_path.mkdir(parents=True, exist_ok=True)
        files = subject.glob('*')
        for file in files:
            shutil.copy(file, new_subject_path)

    pass

def move_ACA_results():
    ACA_data_path = r"/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/normal/" # badGT # normal
    ACA_data_path = Path(ACA_data_path)
    data_path = r"/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/normal(1)/ACA_test1/" # badGT(1)/ACA_test_badGT # normal(1)/ACA_test1
    data_path = Path(data_path)
    # ACA_subject_dirs = sorted(list(ACA_data_path.glob('*/*/*/*/*/FAST*_ACA/')))
    is_siemens = False # True for Siemens, False for GE, Philip
    if is_siemens:
        # For Siemens
        subject_dirs = sorted(list(data_path.glob('*/*/*/aca*/*/')))
    else:
        # For GE, Philip
        subject_dirs = sorted(list(data_path.glob('*/*/*/*/*FAST*/')))

    for subject in tqdm(subject_dirs, total=len(subject_dirs)):
        if is_siemens:
            parts = list(subject.parts)
            vendor = parts[-5]
            date = parts[-4]
            organ = parts[-3]
            sequence = parts[-2]
            patient = parts[-1]
            if 'AI_' in sequence:
                continue
            subject, gt_subject = process_subject_dir(subject, is_siemens=is_siemens)[0]
            ACA_subject_dir = ACA_data_path / vendor / date / patient / organ / '_'.join(gt_subject.parent.name.split('_')[1:])
            fast_subject_dir = ACA_subject_dir.glob('FAST*_ACA/')
            for fast_subject in fast_subject_dir:
                if fast_subject.name[4:-4] in sequence.split('_')[0]:
                    new_subject_path = subject.parent.parent / f"AI_{sequence}" / patient
                    new_subject_path.mkdir(parents=True, exist_ok=True)
                    files = fast_subject.glob('*')
                    for file in files:
                        shutil.copy(file, new_subject_path)
                    break
                else:
                    continue
        else:# For GE, Philip
            parts = list(subject.parts)
            vendor = parts[-5]
            date = parts[-4]
            patient = parts[-3]
            organ = parts[-2]
            sequence = parts[-1]
            if 'AI_' in sequence:
                continue
            subject, gt_subject = process_subject_dir(subject, is_siemens=is_siemens)[0]
            ACA_subject_dir = ACA_data_path / vendor / date / patient / organ / gt_subject.name
            fast_subject_dir = ACA_subject_dir.glob('FAST*_ACA/')
            for fast_subject in fast_subject_dir:
                if fast_subject.name[:-4] in sequence:
                    new_subject_path = subject.parent / f"AI_{sequence}"
                    new_subject_path.mkdir(parents=True, exist_ok=True)
                    files = fast_subject.glob('*')
                    for file in files:
                        shutil.copy(file, new_subject_path)
                    break
                else:
                    continue

    pass

def replace_ACA_with_AI():
    # 获取当前工作目录
    ACA_data_path = r"/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/badGT/test_badGT"  # badGT # normal # test_badGT # normal/ACA_test2
    ACA_data_path = Path(ACA_data_path)
    suffix = "_ACA"
    prefix = "AI_"

    print(f"当前操作目录: {ACA_data_path}")
    print("-" * 40)

    # 使用 rglob("*_ACA") 可以递归搜索，这里用 glob 只搜索当前层级
    subject_dirs = sorted(list(ACA_data_path.glob('*/*/*/*/*_ACA/')))
    # iterdir() 遍历当前目录下的所有项
    for subject_dir in tqdm(subject_dirs, total=len(subject_dirs)):
        # 确保是文件夹且以 _ACA 结尾
        if subject_dir.is_dir() and subject_dir.name.endswith(suffix):
            old_name = subject_dir.name

            # 逻辑：去掉末尾的 _ACA，并在开头加上 AI_
            # stem 在这里指文件夹名，pathlib 处理字符串非常方便
            new_name_str = prefix + old_name[:-len(suffix)]

            # 创建新的路径对象
            new_path = subject_dir.with_name(new_name_str)

            try:
                if not new_path.exists():
                    subject_dir.rename(new_path)
                    print(f"已重命名: {old_name}  =>  {new_path.name}")
                else:
                    print(f"跳过: {new_path.name} 已存在")
            except Exception as e:
                print(f"重命名 {old_name} 失败: {e}")

    print("-" * 40)
    print("处理完毕！")
    pass


def split_deeprecon_test(root_path, split='DeepRecon', dst_path="/data/yuning/zhongjian/Data/DeepRecon_test/"):
    root_path = Path(root_path)
    dst_path = Path(dst_path)
    data_path = root_path / split
    subject_dirs = sorted(list(data_path.glob('*/*/')))
    for i, subject in tqdm(enumerate(subject_dirs), total=len(subject_dirs)):
        if (i + 1) % 5 == 0:
            dst_file_path = dst_path / subject.parent.name
            dst_file_path.mkdir(parents=True, exist_ok=True)
            shutil.move(subject, dst_file_path)


if __name__ == '__main__':
    root_path = r"../Data/training_data/"
    # multi_process(root_path, split='Siemens', num_processes=20) # shanghaitech: training, validation, testing # uii: GE, Philip, Siemens, DeepRecon
    # plot_images()
    # check_images()
    # read_dicom()
    # test_vae()
    # filter_low_quality_siemens_data()
    # filter_low_quality_philip_data()
    # filter_low_quality_ge_data()
    # data_path = r"/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/deformable_registration_splited_processed/training"
    # test_read_csv(root=data_path, is_siemens=False)
    # reorganize_directory_structure()
    # move_ACA_results()
    # replace_ACA_with_AI()
    split_deeprecon_test(root_path)
    pass