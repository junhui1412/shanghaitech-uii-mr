import re
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pydicom
from torch.utils.data import Dataset
from torchvision import transforms
# from torchvision.utils import save_image
from tqdm.auto import tqdm

def replace_specific_parent(path: Path, old_name: str, new_name: str) -> Path:
    parts = list(path.parts)
    parts = [new_name if part == old_name else part for part in parts]
    return Path(*parts)

def replace_fast_parent(path: Path) -> Path:
    """
    Replace parent folder name containing _FAST / _FAST2 / ... with cleaned version,
    and return the updated full path.
    """
    parts = list(path.parts)
    new_parts = []

    for part in parts:
        # Match prefix_FASTdigits_suffix
        # Case 1: prefix_FASTdigits_suffix  -> prefix_suffix
        match = re.match(r"^(.+?)_FAST\d*(?:_(.+))?$", part)
        if match:
            prefix, suffix = match.groups()
            cleaned = prefix if suffix is None else f"{prefix}_{suffix}"
            new_parts.append(cleaned)
        else:
            new_parts.append(part)

    return Path(*new_parts)

def auto_replace_scan_folder(path: Path) -> Path:
    """
    Convert ACA-like scan folder path into the corresponding GT scan folder path.

    Logic matches the original code:
    - If date_folder contains '_': use GT2_xxx or GT1_xxx.
    - Else: remove components containing 'aca', 'ppa', or 'av'.
    Returns the full updated path.
    """

    # Extract full path components
    parts = list(path.parts)

    # We assume structure: data_root / date_folder / organ_folder / seq_folder / patient_folder / file
    # so we locate the folder before patient (seq_folder)
    # Example:
    # [..., date_folder, organ_folder, seq_folder, patient_folder, filename]
    if len(parts) < 5:
        return path  # insufficient structure

    # Identify critical components
    data_root = Path(parts[0])
    for i in range(1, len(parts)):
        data_root = data_root / parts[i]
        # We just want correct slicing later

    # Slice out required folders
    data_root = Path(parts[0])
    date_folder = parts[-5]
    organ_folder = parts[-4]
    seq_folder = parts[-3]
    patient_folder = parts[-2]
    filename = parts[-1]

    # Case 1: date folder contains '_'
    if "_" in date_folder:
        # seq_folder[5:] same as original code
        suffix = seq_folder[5:]

        seq_gt_folder = f"GT2_{suffix}"
        path_gt = path.parents[2] / seq_gt_folder / patient_folder

        # If GT2 path doesn't exist → fallback to GT1
        if not path_gt.exists():
            seq_gt_folder = f"GT1_{suffix}"
            path_gt = path.parents[2] / seq_gt_folder / patient_folder

    else:
        # Case 2: remove aca / ppa / av components
        comps = seq_folder.split("_")
        comps_new = [
            c for c in comps
            if ("aca" not in c and "ppa" not in c and "av" not in c)
        ]

        seq_gt_folder = "_".join(comps_new)
        path_gt = path.parents[2] / seq_gt_folder / patient_folder

    # Reconstruct the final file path
    new_full_path = path_gt / filename
    return new_full_path

def auto_replace_scan_folder_test(path: Path) -> Path:
    """
    Convert ACA-like scan folder path into the corresponding GT scan folder path.

    Logic matches the original code:
    - If date_folder contains '_': use GT2_xxx or GT1_xxx.
    - Else: remove components containing 'aca', 'ppa', or 'av'.
    Returns the full updated path.
    """

    # Extract full path components
    parts = list(path.parts)

    # We assume structure: data_root / date_folder / organ_folder / seq_folder / patient_folder / file
    # so we locate the folder before patient (seq_folder)
    # Example:
    # [..., date_folder, organ_folder, seq_folder, patient_folder, filename]
    if len(parts) < 3:
        return path  # insufficient structure

    # Identify critical components
    data_root = Path(parts[0])
    for i in range(1, len(parts)):
        data_root = data_root / parts[i]
        # We just want correct slicing later

    # Slice out required folders
    data_root = parts[-5]
    organ_folder = parts[-4]
    patient_folder = parts[-3]
    seq_folder = parts[-2]
    filename = parts[-1]

    # Case 1: date folder contains '_'
    if "MR" in patient_folder:
        # seq_folder[5:] same as original code
        suffix = seq_folder[5:]

        seq_gt_folder = f"GT2_{suffix}"
        path_gt = path.parents[2] / patient_folder / seq_gt_folder

        # If GT2 path doesn't exist → fallback to GT1
        if not path_gt.exists():
            seq_gt_folder = f"GT1_{suffix}"
            path_gt = path.parents[2] / patient_folder / seq_gt_folder

    else:
        # Case 2: remove aca / ppa / av components
        comps = seq_folder.split("_")
        comps_new = [
            c for c in comps
            if ("aca" not in c and "ppa" not in c and "av" not in c)
        ]

        seq_gt_folder = "_".join(comps_new)
        path_gt = path.parents[2] / patient_folder / seq_gt_folder

    # Reconstruct the final file path
    new_full_path = path_gt / filename
    return new_full_path

def process_subject(subject_dir, flag="GE_Philip"):
    # Process a single subject directory to get slice file pairs
    file_paths = []
    if flag == "Siemens":
        slice_files = list(subject_dir.glob('*.dcm'))
        for slice_file in slice_files:
            gt_slice_file = auto_replace_scan_folder(slice_file)
            if gt_slice_file.exists():  # check if GT file exists
                file_paths.append([slice_file, gt_slice_file])
            else:
                print(f"Skipping {slice_file}")
    elif flag == "GE_Philip":
        slice_files = subject_dir.glob('IM*')
        for slice_file in slice_files:
            gt_slice_file = replace_fast_parent(slice_file)
            if gt_slice_file.exists():  # check if GT file exists
                file_paths.append([slice_file, gt_slice_file])
            else:
                print(f"Skipping {slice_file}")
    elif flag == "DeepRecon":
        slice_files = list(subject_dir.glob('UID*.dcm'))
        for slice_file in slice_files:
            gt_slice_file = replace_specific_parent(slice_file, subject_dir.name, "GT")
            if gt_slice_file.exists():
                file_paths.append([slice_file, gt_slice_file])
            else:
                print(f"Skipping {slice_file}")
    # else:
    #     raise NotImplementedError(f"Unknown flag {flag}")

    return file_paths

def process_subject_dir(subject_dir, is_siemens=False, aca_type=None):
    # Only process the subject directory to get the pair of directories
    match = re.findall('AI_|UII_|_ACA', subject_dir.name)
    if match:
        return []

    file_paths = []
    if is_siemens:
        slice_files = list(subject_dir.glob('*.dcm')) + list(subject_dir.glob('IM*.DCM'))
        gt_slice_file = auto_replace_scan_folder(slice_files[0])
        if gt_slice_file.exists() and gt_slice_file != slice_files:  # check if GT file exists
            gt_subject_dir = gt_slice_file.parents[0]
            file_paths.append([subject_dir, gt_subject_dir])
        else:
            print(f"Skipping {subject_dir}")
            return []
    else:
        slice_files = list(subject_dir.glob('IM*'))
        gt_slice_file = replace_fast_parent(slice_files[0])
        if gt_slice_file.exists() and gt_slice_file != slice_files:  # check if GT file exists
            gt_subject_dir = gt_slice_file.parents[0]
            file_paths.append([subject_dir, gt_subject_dir])
        else:
            print(f"Skipping {subject_dir}")
            return []

    return file_paths

def process_test_subject_dir(subject_dir, is_siemens=False, aca_type=False):
    # Only process the subject directory to get the pair of directories
    match = re.findall('AI_|UII_|_ACA', subject_dir.name)
    if match:
        return []

    file_paths = []
    if is_siemens:
        slice_files = list(subject_dir.glob('*.DCM'))
        if aca_type:
            gt_slice_file = slice_files[0].parents[1] / 'GT' / slice_files[0].name
        else:
            gt_slice_file = auto_replace_scan_folder_test(slice_files[0])
        if gt_slice_file.exists() and gt_slice_file != slice_files:  # check if GT file exists
            gt_subject_dir = gt_slice_file.parents[0]
            file_paths.append([subject_dir, gt_subject_dir])
        else:
            print(f"Skipping {subject_dir}")
            return []
    else:
        slice_files = list(subject_dir.glob('IM*'))
        if aca_type:
            gt_slice_file = slice_files[0].parents[1] / 'GT' / slice_files[0].name
        else:
            gt_slice_file = replace_fast_parent(slice_files[0])
        if gt_slice_file.exists() and gt_slice_file != slice_files:  # check if GT file exists
            gt_subject_dir = gt_slice_file.parents[0]
            file_paths.append([subject_dir, gt_subject_dir])
        else:
            print(f"Skipping {subject_dir}")
            return []

    return file_paths

# def process_test_subject_dir2(subject_dir, is_siemens=False):
#     # Only process the subject directory to get the pair of directories
#     match = re.findall('AI_', subject_dir.name)
#     if match:
#         return []
#
#     file_paths = []
#     slice_files = list(subject_dir.glob('*'))
#     gt_slice_file = replace_fast_parent(slice_files[0])
#     if gt_slice_file.exists() and gt_slice_file != slice_files:  # check if GT file exists
#         gt_subject_dir = gt_slice_file.parents[0]
#         file_paths.append([subject_dir, gt_subject_dir])
#     else:
#         print(f"Skipping {subject_dir}")
#         return []
#
#     return file_paths

def pad_to_multiple_centered(img, multiple=16, mode="constant", return_pad_info=False):
    """
    img: Tensor of shape [B, C, H, W]
    multiple: pad height/width to the nearest upper multiple
    mode: padding mode, e.g., "reflect", "constant", "replicate", "circular"

    Returns:
        padded_img: centered padded tensor [B, C, H', W']
        pad_info: (pad_top, pad_bottom, pad_left, pad_right)
    """

    if img.ndim != 4:
        raise ValueError("Input must be NCHW (4D tensor).")

    _, _, h, w = img.shape

    # Target H and W: ceil to nearest multiple
    new_h = (h + multiple - 1) // multiple * multiple
    new_w = (w + multiple - 1) // multiple * multiple

    pad_h = new_h - h
    pad_w = new_w - w

    # Split padding evenly (centered image)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # PyTorch pad format: (left, right, top, bottom)
    pad = (pad_left, pad_right, pad_top, pad_bottom)

    padded_img = F.pad(img, pad, mode=mode)
    if not return_pad_info:
        return padded_img
    else:
        return padded_img, (pad_top, pad_bottom, pad_left, pad_right)

def pad_to_target_size_centered(img, target_h_w, mode="constant", return_pad_info=False):
    """
    img: Tensor of shape [B, C, H, W]
    target_h: target height
    target_w: target width
    mode: padding mode, e.g., "reflect", "constant", "replicate", "circular"

    Returns:
        padded_img: centered padded tensor [B, C, target_h, target_w]
        pad_info: (pad_top, pad_bottom, pad_left, pad_right)
    """

    if img.ndim != 4:
        raise ValueError("Input must be NCHW (4D tensor).")

    _, _, h, w = img.shape
    target_h, target_w = target_h_w

    pad_h = target_h - h
    pad_w = target_w - w

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Target size must be greater than or equal to image size.")

    # Split padding evenly (centered image)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # PyTorch pad format: (left, right, top, bottom)
    pad = (pad_left, pad_right, pad_top, pad_bottom)

    padded_img = F.pad(img, pad, mode=mode)
    if not return_pad_info:
        return padded_img
    else:
        return padded_img, (pad_top, pad_bottom, pad_left, pad_right)

def read_filter_csv_file(root, is_siemens=False):
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

class MRIDicomDataset(Dataset):
    def __init__(self, root, resolution=256, normalize_type='mean', use_csv=False):
        data_path = Path(root)
        self.normalize_type = normalize_type # 'mean' or 'minmax'
        self.file_paths = []
        # For GE, Philip
        subject_dirs = list(data_path.glob('*/*/*/*/*FAST*/')) if not use_csv else read_filter_csv_file(root, is_siemens=False)
        self.file_paths += self.load_slices_with_threadpool(subject_dirs, flag="GE_Philip", max_workers=16) # flag: "GE_Philip", "Siemens", "DeepRecon"

        # For Siemens
        subject_dirs = list(data_path.glob('*/*/*/aca*/*/')) if not use_csv else read_filter_csv_file(root, is_siemens=True)
        self.file_paths += self.load_slices_with_threadpool(subject_dirs, flag="Siemens", max_workers=16)

        # For DeepRecon
        subject_dirs = list(data_path.glob('*/*/*/UID*/*/')) if not use_csv else read_filter_csv_file(root, is_siemens=True)
        self.file_paths += self.load_slices_with_threadpool(subject_dirs, flag="DeepRecon", max_workers=16)

        # sorted file_paths for reproducibility
        self.file_paths = sorted(self.file_paths)

        if self.normalize_type == 'mean':
            self.trans = transforms.Compose([
                transforms.RandomCrop(resolution, pad_if_needed=True, padding_mode='reflect'),
            ])
        elif self.normalize_type in ['minmax', 'mean_minmax']:
            self.trans = transforms.Compose([
                transforms.RandomCrop(resolution, pad_if_needed=True, padding_mode='reflect'),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            raise NotImplementedError(f"normalize_type {self.normalize_type} not implemented.")

    def load_slices_with_threadpool(self, subject_dirs, flag="GE_Philip", max_workers=8):
        """read all slice files using ThreadPoolExecutor"""
        file_paths = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_subject, subject, flag=flag)
                for subject in subject_dirs
            ]

            for future in tqdm(as_completed(futures), total=len(futures)):
                file_paths += future.result()
        return file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        if self.normalize_type == 'mean':
            path_mov, path_fix = self.file_paths[index]
            mov_ds = pydicom.dcmread(path_mov, force=True)
            img_mov = mov_ds.pixel_array.astype(np.float32)
            mov_normalize_value = float(mov_ds[(0x0019, 0x1001)].value)
            img_mov = img_mov / mov_normalize_value

            fix_ds = pydicom.dcmread(path_fix, force=True)
            img_fix = fix_ds.pixel_array.astype(np.float32)
            fix_normalize_value = float(fix_ds[(0x0019, 0x1001)].value)
            img_fix = img_fix / fix_normalize_value
        elif self.normalize_type == 'minmax':
            path_mov, path_fix = self.file_paths[index]
            mov_ds = pydicom.dcmread(path_mov, force=True)
            img_mov = mov_ds.pixel_array.astype(np.float32)
            max_m = float(mov_ds[(0x0019, 0x1003)].value) # max_m = np.quantile(img_mov, 0.9995)
            # try:
            #     img_mov = img_mov / max_m
            # except RuntimeWarning as e:
            #     print(f"Error normalizing {path_mov}: {e}, max_m={max_m}")
            #     raise e
            img_mov = img_mov / max_m
            mov_normalize_value = max_m

            fix_ds = pydicom.dcmread(path_fix, force=True)
            img_fix = fix_ds.pixel_array.astype(np.float32)
            max_f = float(fix_ds[(0x0019, 0x1003)].value) # max_f = np.quantile(img_fix, 0.9995)
            img_fix = img_fix / max_f
            fix_normalize_value = max_f
        elif self.normalize_type == 'mean_minmax':
            path_mov, path_fix = self.file_paths[index]
            mov_ds = pydicom.dcmread(path_mov, force=True)
            img_mov = mov_ds.pixel_array.astype(np.float32)
            mean_m = float(mov_ds[(0x0019, 0x1001)].value)
            max_m = float(mov_ds[(0x0019, 0x1003)].value)
            img_mov = img_mov / max_m
            mov_normalize_value = max_m

            fix_ds = pydicom.dcmread(path_fix, force=True)
            img_fix = fix_ds.pixel_array.astype(np.float32)
            mean_f = float(fix_ds[(0x0019, 0x1001)].value)
            fix_normalize_value = mean_f / mean_m * max_m
            img_fix = img_fix / fix_normalize_value
        else:
            raise NotImplementedError(f"normalize_type {self.normalize_type} not implemented.")

        ts_mov, ts_fix = torch.from_numpy(img_mov), torch.from_numpy(img_fix)
        ts_mov, ts_fix = torch.chunk(self.trans(torch.stack([ts_mov, ts_fix])), 2, dim=0)
        # save_image(ts_mov, 'test_mov.png', normalize=True, value_range=(0, 1))
        # save_image(ts_fix, 'test_fix.png', normalize=True, value_range=(0, 1))
        return ts_mov, ts_fix, torch.tensor(mov_normalize_value), torch.tensor(fix_normalize_value)


class MRIVolumeDicomDataset(Dataset):
    def __init__(self, root, normalize_type='mean'):
        data_path = Path(root)
        self.normalize_type = normalize_type # 'mean' or 'minmax'
        self.file_paths = []
        # For GE, Philip
        subject_dirs = sorted(list(data_path.glob('*/*/*/*/*FAST*/')))
        self.file_paths += self.load_slices_with_threadpool(subject_dirs, is_siemens=False, max_workers=16)

        # For Siemens
        subject_dirs = sorted(list(data_path.glob('*/*/*/aca*/*/')))
        self.file_paths += self.load_slices_with_threadpool(subject_dirs, is_siemens=True, max_workers=16)
        # sorted file_paths for reproducibility
        self.file_paths = sorted(self.file_paths)

    def load_slices_with_threadpool(self, subject_dirs, is_siemens=False, max_workers=8):
        """read all slice files using ThreadPoolExecutor"""
        file_paths = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_subject_dir, subject, is_siemens=is_siemens)
                for subject in subject_dirs
            ]

            for future in tqdm(as_completed(futures), total=len(futures)):
                file_paths += future.result()
        return file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        subject_mov, subject_fix = self.file_paths[index]
        paths_mov, paths_fix = sorted(list(subject_mov.glob('IM*')), key=lambda s: int(s.stem[2:])) + sorted(list(subject_mov.glob('*.dcm'))), sorted(list(subject_fix.glob('IM*')), key=lambda s: int(s.stem[2:])) + sorted(list(subject_fix.glob('*.dcm')))

        img_mov_list, img_fix_list, mov_normalize_value_list, fix_normalize_value_list = [], [], [], []
        if self.normalize_type == 'mean':
            for path_mov in paths_mov:
                mov_ds = pydicom.dcmread(path_mov, force=True)
                img_mov = mov_ds.pixel_array.astype(np.float32)
                mov_normalize_value = float(mov_ds[(0x0019, 0x1001)].value)
                img_mov = img_mov / mov_normalize_value
                img_mov_list.append(img_mov)
                mov_normalize_value_list.append(mov_normalize_value)

            for path_fix in paths_fix:
                fix_ds = pydicom.dcmread(path_fix, force=True)
                img_fix = fix_ds.pixel_array.astype(np.float32)
                fix_normalize_value = float(fix_ds[(0x0019, 0x1001)].value)
                img_fix = img_fix / fix_normalize_value
                img_fix_list.append(img_fix)
                fix_normalize_value_list.append(fix_normalize_value)
            try:
                img_mov, img_fix = np.stack(img_mov_list, axis=0), np.stack(img_fix_list, axis=0)
                mov_normalize_value, fix_normalize_value = np.array(mov_normalize_value_list, dtype=np.float32), np.array(fix_normalize_value_list, dtype=np.float32)
            except Exception as e:
                print(f"Error stacking images for {subject_mov}: {e}")
        elif self.normalize_type == 'minmax':
            for path_mov in paths_mov:
                mov_ds = pydicom.dcmread(path_mov, force=True)
                img_mov = mov_ds.pixel_array.astype(np.float32)
                max_m = float(mov_ds[(0x0019, 0x1003)].value) # max_m = np.quantile(img_mov, 0.9995)
                img_mov = img_mov / max_m
                mov_normalize_value = max_m
                img_mov_list.append(img_mov)
                mov_normalize_value_list.append(mov_normalize_value)

            for path_fix in paths_fix:
                fix_ds = pydicom.dcmread(path_fix, force=True)
                img_fix = fix_ds.pixel_array.astype(np.float32)
                max_f = float(fix_ds[(0x0019, 0x1003)].value) # max_f = np.quantile(img_fix, 0.9995)
                img_fix = img_fix / max_f
                fix_normalize_value = max_f
                img_fix_list.append(img_fix)
                fix_normalize_value_list.append(fix_normalize_value)
            try:
                img_mov, img_fix = np.stack(img_mov_list, axis=0), np.stack(img_fix_list, axis=0)
                mov_normalize_value, fix_normalize_value = np.array(mov_normalize_value_list, dtype=np.float32), np.array(fix_normalize_value_list, dtype=np.float32)
            except Exception as e:
                print(f"Error stacking images for {subject_mov}: {e}")
                raise e
        else:
            raise NotImplementedError(f"normalize_type {self.normalize_type} not implemented.")

        ts_mov, ts_fix = torch.from_numpy(img_mov).unsqueeze(dim=1), torch.from_numpy(img_fix).unsqueeze(dim=1)
        ts_mov_normalize_value, ts_fix_normalize_value = torch.from_numpy(mov_normalize_value), torch.from_numpy(fix_normalize_value)
        # ts_mov, ts_fix = pad_to_multiple_centered(ts_mov), pad_to_multiple_centered(ts_fix)
        if self.normalize_type == 'minmax':
            ts_mov, ts_fix = ts_mov * 2.0 - 1.0, ts_fix * 2.0 - 1.0
        return ts_mov, ts_fix, ts_mov_normalize_value, ts_fix_normalize_value, str(subject_mov), str(subject_fix)


class MRIVolumeTestDicomDataset(Dataset):
    def __init__(self, root: str | list[str], normalize_type='mean'):
        if isinstance(root, str):
            data_path_list = [Path(root)]
        elif isinstance(root, list):
            data_path_list = [Path(p) for p in root]
        else:
            data_path_list = [root]
        self.normalize_type = normalize_type # 'mean' or 'minmax'
        self.file_paths = []

        for data_path in data_path_list:
            self.flag = False
            self.aca_type = False
            split = data_path.name
            if split in ["ACA_data_transfer_organized_test", "aca_test_20260202"]:
                if split == "aca_test_20260202":
                    self.aca_type = True
                # For GE, Philip
                subject_dirs = sorted(list(data_path.glob('*/*/*FAST*/')))
                self.file_paths += self.load_slices_with_threadpool(subject_dirs, is_siemens=False, max_workers=16)

                # For Siemens
                subject_dirs = sorted(list(data_path.glob('*/*/aca*/')))
                self.file_paths += self.load_slices_with_threadpool(subject_dirs, is_siemens=True, max_workers=16)
            else:
                self.flag = True
                # For GE, Philip
                subject_dirs = sorted(list(data_path.glob('*/*/*/*/*FAST*/')))
                self.file_paths += self.load_slices_with_threadpool(subject_dirs, is_siemens=False, max_workers=16)

                # For Siemens
                subject_dirs = sorted(list(data_path.glob('*/*/*/aca*/*/')))
                self.file_paths += self.load_slices_with_threadpool(subject_dirs, is_siemens=True, max_workers=16)
        # sorted file_paths for reproducibility
        self.file_paths = sorted(self.file_paths)

        # define center crop transform
        self.trans = transforms.Compose([
            transforms.CenterCrop(512),
        ])

    def load_slices_with_threadpool(self, subject_dirs, is_siemens=False, max_workers=8):
        """read all slice files using ThreadPoolExecutor"""
        file_paths = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_test_subject_dir if not self.flag else process_subject_dir, subject, is_siemens=is_siemens, aca_type=self.aca_type)
                for subject in subject_dirs
            ]

            for future in tqdm(as_completed(futures), total=len(futures)):
                file_paths += future.result()
        return file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        subject_mov, subject_fix = self.file_paths[index]
        paths_mov, paths_fix = sorted(list(subject_mov.glob('IM*')), key=lambda s: int(s.stem[2:])) + sorted(list(subject_mov.glob('*.dcm'))), sorted(list(subject_fix.glob('IM*')), key=lambda s: int(s.stem[2:])) + sorted(list(subject_fix.glob('*.dcm')))

        img_mov_list, img_fix_list, mov_normalize_value_list, fix_normalize_value_list = [], [], [], []
        if self.normalize_type == 'mean':
            for path_mov in paths_mov:
                mov_ds = pydicom.dcmread(path_mov, force=True)
                img_mov = mov_ds.pixel_array.astype(np.float32)
                img_mov_list.append(img_mov)

            for path_fix in paths_fix:
                fix_ds = pydicom.dcmread(path_fix, force=True)
                img_fix = fix_ds.pixel_array.astype(np.float32)
                img_fix_list.append(img_fix)
            try:
                img_mov, img_fix = np.stack(img_mov_list, axis=0), np.stack(img_fix_list, axis=0)
                img_fix = img_fix[: img_mov.shape[0]]
                mov_normalize_value_list, fix_normalize_value_list = [np.mean(img_mov)] * img_mov.shape[0], [np.mean(img_fix)] * img_fix.shape[0]
                mov_normalize_value, fix_normalize_value = np.array(mov_normalize_value_list, dtype=np.float32), np.array(fix_normalize_value_list, dtype=np.float32)
            except Exception as e:
                print(f"Error stacking images for {subject_mov}: {e}")
        elif self.normalize_type == 'minmax':
            for path_mov in paths_mov:
                mov_ds = pydicom.dcmread(path_mov, force=True)
                img_mov = mov_ds.pixel_array.astype(np.float32)
                img_mov_list.append(img_mov)

            for path_fix in paths_fix:
                fix_ds = pydicom.dcmread(path_fix, force=True)
                img_fix = fix_ds.pixel_array.astype(np.float32)
                img_fix_list.append(img_fix)
            try:
                img_mov, img_fix = np.stack(img_mov_list, axis=0), np.stack(img_fix_list, axis=0)
                img_fix = img_fix[: img_mov.shape[0]]
                mov_normalize_value_list, fix_normalize_value_list = [np.quantile(img_mov, 0.9995)] * img_mov.shape[0], [np.quantile(img_fix, 0.9995)] * img_fix.shape[0]
                mov_normalize_value, fix_normalize_value = np.array(mov_normalize_value_list, dtype=np.float32), np.array(fix_normalize_value_list, dtype=np.float32)
            except Exception as e:
                print(f"Error stacking images for {subject_mov}: {e}")
                raise e
        elif self.normalize_type == 'mean_minmax':
            for path_mov in paths_mov:
                mov_ds = pydicom.dcmread(path_mov, force=True)
                img_mov = mov_ds.pixel_array.astype(np.float32)
                img_mov_list.append(img_mov)

            for path_fix in paths_fix:
                fix_ds = pydicom.dcmread(path_fix, force=True)
                img_fix = fix_ds.pixel_array.astype(np.float32)
                img_fix_list.append(img_fix)
            try:
                img_mov, img_fix = np.stack(img_mov_list, axis=0), np.stack(img_fix_list, axis=0)
                img_fix = img_fix[: img_mov.shape[0]]
                mean_m = np.mean(img_mov)
                mean_f = np.mean(img_fix)
                max_m = np.quantile(img_mov, 0.9995)
                mov_normalize_value_list = [max_m] * img_mov.shape[0]
                fix_normalize_value_list = [mean_f / mean_m * max_m] * img_fix.shape[0]
                mov_normalize_value, fix_normalize_value = np.array(mov_normalize_value_list, dtype=np.float32), np.array(fix_normalize_value_list, dtype=np.float32)
            except Exception as e:
                print(f"Error stacking images for {subject_mov}: {e}")
                raise e
        else:
            raise NotImplementedError(f"normalize_type {self.normalize_type} not implemented.")

        ts_mov, ts_fix = torch.from_numpy(img_mov).unsqueeze(dim=1), torch.from_numpy(img_fix).unsqueeze(dim=1)
        ts_mov_normalize_value, ts_fix_normalize_value = torch.from_numpy(mov_normalize_value), torch.from_numpy(fix_normalize_value)
        ts_mov, ts_fix = ts_mov / ts_mov_normalize_value.view(-1, 1, 1, 1), ts_fix / ts_fix_normalize_value.view(-1, 1, 1, 1)
        if self.normalize_type in ['minmax', 'mean_minmax']:
            ts_mov, ts_fix = ts_mov * 2.0 - 1.0, ts_fix * 2.0 - 1.0
        # h, w = ts_mov.shape[-2:]
        # if h > 512 and w > 512:
        #     ts_mov, ts_fix = self.trans(ts_mov), self.trans(ts_fix)
        return ts_mov, ts_fix, ts_mov_normalize_value, ts_fix_normalize_value, str(subject_mov), str(subject_fix)

    # @staticmethod
    # def collate_fn(batch):
    #     ts_mov, ts_fix, mov_normalize_value, fix_normalize_value, lq_path, hq_path = zip(*batch)


class SingleMRIVolumeTestDicomDataset(Dataset):
    """Single input volume for test dataset"""
    def __init__(self, root, normalize_type='mean'):
        data_path = Path(root)
        self.normalize_type = normalize_type # 'mean' or 'minmax'
        self.file_paths = []

        subject_dirs = sorted(list(data_path.glob('*/*/'))) # /patient/sequence/
        self.file_paths += subject_dirs
        # sorted file_paths for reproducibility
        self.file_paths = sorted(self.file_paths)

        # define center crop transform
        self.trans = transforms.Compose([
            transforms.CenterCrop(512),
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        subject_mov = self.file_paths[index]
        paths_mov = sorted(list(subject_mov.glob('I*')))

        img_mov_list, mov_normalize_value_list = [], []
        if self.normalize_type == 'mean':
            for path_mov in paths_mov:
                mov_ds = pydicom.dcmread(path_mov, force=True)
                img_mov = mov_ds.pixel_array.astype(np.float32)
                img_mov_list.append(img_mov)

            try:
                img_mov = np.stack(img_mov_list, axis=0)
                mov_normalize_value_list = [np.mean(img_mov)] * img_mov.shape[0]
                mov_normalize_value = np.array(mov_normalize_value_list, dtype=np.float32)
            except Exception as e:
                print(f"Error stacking images for {subject_mov}: {e}")
        elif self.normalize_type == 'minmax':
            for path_mov in paths_mov:
                mov_ds = pydicom.dcmread(path_mov, force=True)
                img_mov = mov_ds.pixel_array.astype(np.float32)
                img_mov_list.append(img_mov)

            try:
                img_mov = np.stack(img_mov_list, axis=0)
                mov_normalize_value_list = [np.quantile(img_mov, 0.9995)] * img_mov.shape[0]
                mov_normalize_value = np.array(mov_normalize_value_list, dtype=np.float32)
            except Exception as e:
                print(f"Error stacking images for {subject_mov}: {e}")
                raise e
        else:
            raise NotImplementedError(f"normalize_type {self.normalize_type} not implemented.")

        ts_mov = torch.from_numpy(img_mov).unsqueeze(dim=1)
        ts_mov_normalize_value = torch.from_numpy(mov_normalize_value)
        ts_mov = ts_mov / ts_mov_normalize_value.view(-1, 1, 1, 1)
        if self.normalize_type == 'minmax':
            ts_mov = ts_mov * 2.0 - 1.0
        h, w = ts_mov.shape[-2:]
        # if h > 512 and w > 512:
        #     ts_mov, ts_fix = self.trans(ts_mov), self.trans(ts_fix)
        return ts_mov, ts_mov_normalize_value, str(subject_mov)

    # @staticmethod
    # def collate_fn(batch):
    #     ts_mov, ts_fix, mov_normalize_value, fix_normalize_value, lq_path, hq_path = zip(*batch)

if __name__ == '__main__':
    # dataset = MriTrainConDataset(data_path=r'/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/deformable_registration_splited/training')
    # dataset = MRIDicomDataset(root=r'/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/deformable_registration_splited_processed/training')
    dataset = MRIVolumeTestDicomDataset(root=r'/mnt/e/deeplearning/data/mri_reconstruction/shanghaitech_uii_mr/ACA_data_transfer_organized')
    # data = dataset[0]
    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        pass

    # from PIL import Image
    # # file = r"/mnt/e/deeplearning/data/computer_vision/FFHQ/00000.png"
    # file = r"/mnt/e/deeplearning/data/computer_vision/natural_image_test_dataset/imagenet/ILSVRC2012_test_00000003.JPEG"
    # image = Image.open(file).convert('RGB')
    #
    # ts_image = transforms.ToTensor()(image)
    #
    # trans = transforms.Compose([
    #     transforms.RandomCrop(512, pad_if_needed=True),
    #     # transforms.Normalize([0.5], [0.5]),
    # ])
    # ts_image1, ts_image2 = torch.chunk(trans(torch.stack([ts_image, ts_image])), 2, dim=0)
    # save_image(ts_image1, 'ts_image1.png', normalize=True, value_range=(0, 1))
    # save_image(ts_image2, 'ts_image2.png', normalize=True, value_range=(0, 1))
    pass