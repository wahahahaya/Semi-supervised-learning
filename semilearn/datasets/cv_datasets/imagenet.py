# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import gc
import copy
import json
import random
import numpy as np
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import math
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset

mean, std = {}, {}
mean['imagenet'] = [0.485, 0.456, 0.406]
std['imagenet'] = [0.229, 0.224, 0.225]

def get_imagenet(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_medium = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    # 設定 .npy 檔案的路徑
    # 根據你的描述，檔案在 /ephemeral/imagenet_1k/
    base_npy_path = data_dir
    
    train_data_path = os.path.join(base_npy_path, 'imagenet200_id_images.npy')
    train_label_path = os.path.join(base_npy_path, 'imagenet200_id_labels.npy')
    
    val_data_path = os.path.join(base_npy_path, 'imagenet200_val_images.npy')
    val_label_path = os.path.join(base_npy_path, 'imagenet200_val_labels.npy')

    # 計算每類要取多少張 labeled data
    # 這裡需要預先載入一次 label 來計算總類別數，或者直接用傳入的 num_classes
    # 為了安全起見，這裡先計算 label_perclass
    # 注意：這裡假設 targets 是 0-indexed 的整數
    
    # 1. 建立 Labeled Dataset
    # lb_list_txt 如果有指定 index 檔案，會從裡面讀取 index
    lb_dset = ImagenetNpyDataset(
        data_path=train_data_path,
        label_path=train_label_path,
        transform=transform_weak,
        ulb=False,
        alg=alg,
        label_perclass=num_labels // num_classes # 這裡簡化計算，假設數據分佈均勻
    )

    # 2. 建立 Unlabeled Dataset
    # 這裡將整個 training set 作為 unlabeled set
    ulb_dset = ImagenetNpyDataset(
        data_path=train_data_path,
        label_path=train_label_path,
        transform=transform_weak,
        alg=alg,
        ulb=True,
        medium_transform=transform_medium,
        strong_transform=transform_strong,
        include_lb_to_ulb=include_lb_to_ulb,
        # 如果需要排除已選的 labeled index，可傳入 lb_dset.selected_indices
        lb_index=lb_dset.selected_indices 
    )

    # 3. 建立 Validation Dataset
    eval_dset = ImagenetNpyDataset(
        data_path=val_data_path,
        label_path=val_label_path,
        transform=transform_val,
        alg=alg,
        ulb=False
    )

    # 統計數據 (Logging)
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    
    for lb in lb_dset.targets:
        lb_count[lb] += 1
        
    # 注意：如果是 Unlabeled，targets 可能還是原始 label (用於評估) 或者被設為 -1
    for ulb in ulb_dset.targets:
        ulb_count[ulb] += 1

    save_dir = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'Note.txt'), 'w') as f:
        f.write("lb_count: {}\n".format(lb_count))
        f.write("ulb_count: {}\n".format(ulb_count))
        f.close()

    return lb_dset, ulb_dset, eval_dset


class ImagenetNpyDataset(BasicDataset):
    def __init__(self, data_path, label_path, transform, ulb, alg, 
                 lb_list_txt=None, medium_transform=None, strong_transform=None, 
                 label_perclass=-1, include_lb_to_ulb=True, lb_index=None):
        
        self.alg = alg
        self.is_ulb = ulb
        self.label_perclass = label_perclass
        self.transform = transform
        self.include_lb_to_ulb = include_lb_to_ulb
        self.lb_list_txt = lb_list_txt
        
        # 載入數據
        # mmap_mode='r' 非常重要，避免一次將數 GB 的圖片載入 RAM
        try:
            self.data_mmap = np.load(data_path, mmap_mode='r')
            self.targets_all = np.load(label_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load npy files: {data_path} or {label_path}. Error: {e}")

        self.selected_indices = [] # 用來存儲被選中的 index

        # 處理 Labeled / Unlabeled 的數據分割
        self._init_data_selection(lb_index)

        # 設定 Augmentation
        self.medium_transform = medium_transform
        if self.medium_transform is None and self.is_ulb:
             assert self.alg not in ['sequencematch'], f"alg {self.alg} requires strong augmentation"
        
        self.strong_transform = strong_transform
        if self.strong_transform is None and self.is_ulb:
            assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch', 'refixmatch'], f"alg {self.alg} requires strong augmentation"

    def _init_data_selection(self, excluded_indices=None):
        """
        根據 label_perclass 或 lb_list_txt 決定要使用哪些數據索引。
        """
        num_samples = len(self.targets_all)
        all_indices = np.arange(num_samples)
        
        # === 分支 A: 如果是 Unlabeled Dataset ===
        if self.is_ulb:
            # 通常 Unlabeled set 包含所有數據
            # 如果不想包含已標註的數據 (include_lb_to_ulb=False)，則排除之
            if not self.include_lb_to_ulb and excluded_indices is not None:
                # 排除 excluded_indices
                mask = np.ones(num_samples, dtype=bool)
                mask[excluded_indices] = False
                self.indices = all_indices[mask]
            else:
                self.indices = all_indices
            
            # Unlabeled set 的 targets 通常還是保留真實 label (為了計算 accuracy)，
            # 但在訓練 loop 中是否使用取決於算法
            self.targets = self.targets_all[self.indices]
            return

        # === 分支 B: 如果有指定 index 的 txt 檔案 (Loading Mode) ===
        if self.lb_list_txt is not None and os.path.exists(self.lb_list_txt):
            print(f"Loading specific labeled indices from: {self.lb_list_txt}")
            with open(self.lb_list_txt, 'r') as f:
                # 假設 txt 每一行是一個整數 index
                loaded_indices = [int(line.strip()) for line in f if line.strip().isdigit()]
            
            self.indices = np.array(loaded_indices)
            self.targets = self.targets_all[self.indices]
            self.selected_indices = loaded_indices
            return

        # === 分支 C: 隨機取樣並存檔 (Sampling Mode) ===
        if self.label_perclass > 0:
            print(f"Randomly sampling {self.label_perclass} per class from .npy data...")
            
            indices_per_class = {}
            for idx, target in enumerate(self.targets_all):
                if target not in indices_per_class:
                    indices_per_class[target] = []
                indices_per_class[target].append(idx)
            
            sampled_indices = []
            
            # 排序 key 確保順序固定 (雖然 dict 在新版 python 有序，但安全起見)
            for cls in sorted(indices_per_class.keys()):
                indices = indices_per_class[cls]
                # 如果樣本數不夠，就全取
                k = min(self.label_perclass, len(indices))
                # 隨機選取
                selected = random.sample(indices, k)
                sampled_indices.extend(selected)
            
            self.indices = np.array(sampled_indices)
            self.targets = self.targets_all[self.indices]
            self.selected_indices = sampled_indices

            # 存檔邏輯 (存 index)
            save_name = 'lb_labels_sampled_idx.txt'
            print(f"Saving sampled indices to {save_name} ...")
            with open(save_name, 'w') as f:
                for idx in sorted(sampled_indices):
                    f.write(f"{idx}\n")
            return

        # === 分支 D: 預設全選 (例如 Validation Set) ===
        self.indices = all_indices
        self.targets = self.targets_all

    def __sample__(self, index):
        """
        根據內部的 self.indices 映射到原始 .npy 的真實位置
        """
        real_index = self.indices[index]
        
        # 從 mmap 中讀取圖片數據
        # 假設 .npy 形狀是 (N, H, W, C) 或 (N, C, H, W)
        # PIL Image.fromarray 需要 (H, W, C) 且 dtype 為 uint8
        img_array = self.data_mmap[real_index]
        
        # 如果是 (C, H, W) 需要轉置為 (H, W, C)，視你存檔時的格式而定
        # ImageNet 存成 npy 通常已經是 uint8，如果是 float 0-1 則需要轉換
        if img_array.shape[0] == 3: # 猜測是 (3, H, W)
             img_array = img_array.transpose(1, 2, 0)
        
        img = Image.fromarray(img_array)
        target = self.targets[index] # 這裡是已經篩選過的 targets
        
        return img, target

    def __len__(self):
        return len(self.indices)