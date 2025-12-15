# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math
import torch

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data


mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]


def mix_data_ressl_compliant(id_data, ood_data, target_ratio):
    """
    Mix ID and OOD data based on a specific target_ratio.
    Constraint: ID data count is fixed based on a hardcoded max_ratio of 0.8.
    """
    max_ratio = 0.8  # Hardcoded as requested
    max_ood_avail = len(ood_data)
    len_original_id = len(id_data)

    limit_id_by_ood = int(max_ood_avail * (1 - max_ratio) / max_ratio)
    
    n_id_fixed = min(len_original_id, limit_id_by_ood)
    
    print(f"\n=== RE-SSL Fairness Adjustment ===")
    print(f"Max Ratio Constraint: {max_ratio}")
    print(f"Max OOD Available:    {max_ood_avail}")
    print(f"Original ID Count:    {len_original_id}")
    print(f"Bottleneck ID Limit:  {limit_id_by_ood} (Calculated to fit r=0.8)")
    print(f"FINAL FIXED ID Count: {n_id_fixed}")

    np.random.seed(0)
    fixed_id_indices = np.random.choice(len_original_id, n_id_fixed, replace=False)
    fixed_id_data = id_data[fixed_id_indices]

    if target_ratio == 0.0:
        n_ood_needed = 0
        current_ood_data = np.empty((0, *ood_data.shape[1:]), dtype=ood_data.dtype)
    else:
        # Calculate needed OOD
        n_ood_needed = int(n_id_fixed * target_ratio / (1 - target_ratio))
    
    # Safety Check: strict logic should prevent this, but good for debugging
    if n_ood_needed > max_ood_avail:
        print(f"[Warning] Mathematical impossibility! Needed {n_ood_needed} OOD but only have {max_ood_avail}.")
        n_ood_needed = max_ood_avail 

    if n_ood_needed > 0:
        # Randomly sample the required OOD data
        ood_indices = np.random.choice(max_ood_avail, n_ood_needed, replace=False)
        current_ood_data = ood_data[ood_indices]

    mixed_data = np.concatenate([fixed_id_data, current_ood_data])
    
    print(f"[Current Experiment] Ratio: {target_ratio} | ID: {n_id_fixed} | OOD: {n_ood_needed} | Total: {len(mixed_data)}")
        
    # Return mixed data, indices of ID (to retrieve correct GT), and OOD data
    return mixed_data, fixed_id_indices, current_ood_data




def get_cifar(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=True, download=True)
    data, targets = dset.data, dset.targets
    
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)

    if hasattr(args, 'ood_data_path') and args.ood_data_path is not None:
        print(f"Loading OOD data from: {args.ood_data_path}")
        ood_data = np.load(args.ood_data_path)

        if ood_data.shape[1:] != ulb_data.shape[1:]:
            print(f"[Warning] OOD shape {ood_data.shape} differs from ID shape {ulb_data.shape}")

        mixed_data, id_indices, ood_content = mix_data_ressl_compliant(
            ulb_data, 
            ood_data, 
            args.ood_ratio
        )

        ulb_data = mixed_data

        id_targets_kept = ulb_targets[id_indices]
        ood_targets = -1 * np.ones(len(ood_content), dtype=int)
        ulb_targets = np.concatenate([id_targets_kept, ood_targets])
        print(f"Applied OOD Ratio {args.ood_ratio}. Final Unlabeled Size: {len(ulb_data)}")

        if hasattr(args, 'proxy_checkpoint') and args.proxy_checkpoint:
            from vra_lib.transforms import get_dataset_stats
            from vra_lib.features import load_model, compute_prototypes, extract_features, calculate_distance
            from vra_lib.algorithm import generate_vrm_distribution, solve_optimal_threshold
            from vra_lib.utils import set_seed, plot_cdf, analyze_and_save_results
            print("\n" + "="*40)
            print(" [VRA] Initializing Vicinal Reference Alignment...")
            print("="*40)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
            # 1. 取得 VRA 專用的 dataset stats (與訓練 Proxy model 時一致)
            # 注意：這裡使用 args.dataset (或手動指定 'cifar100')，視你的 args 結構而定
            vra_dataset_name = args.dataset if hasattr(args, 'dataset') else 'cifar100'
            img_size_vra, mean_vra, std_vra = get_dataset_stats(vra_dataset_name)
            
            # 2. 載入 Proxy Model
            # 假設 args 裡有 model_arch，如果沒有，請在這裡指定預設值，例如 'vit_small_patch2_32'
            arch = args.model_arch if hasattr(args, 'model_arch') else 'vit_small_patch2_32'
            print(f" [VRA] Loading Proxy Model: {arch} from {args.proxy_checkpoint}")
            
            proxy_model = load_model(
                arch, 
                num_classes=num_classes, 
                checkpoint_path=args.proxy_checkpoint, 
                device=device
            )

            # 3. 計算 Prototypes (利用 Labeled Data)
            print(" [VRA] Computing Prototypes from Labeled Data...")
            prototypes = compute_prototypes(
                proxy_model, 
                lb_data,      # 這是原始影像 (N, H, W, C)
                lb_targets,   # 對應標籤
                img_size_vra, 
                mean_vra, 
                std_vra, 
                num_classes, 
                device
            )

            # 4. 生成 VRM Reference Distribution (In-Distribution 參考曲線)
            print(" [VRA] Generating VRM Reference Distribution...")
            dist_ref = generate_vrm_distribution(
                proxy_model, 
                lb_data, 
                prototypes, 
                img_size_vra, 
                mean_vra, 
                std_vra, 
                k=50,             # 可視情況調整，或從 args 傳入
                device=device
            )

            # 5. 計算 Unlabeled Data 的距離 (包含 ID 和 OOD)
            print(f" [VRA] Extracting features for {len(ulb_data)} unlabeled samples...")
            feats_unlabeled = extract_features(
                proxy_model, 
                ulb_data, 
                img_size_vra, 
                mean_vra, 
                std_vra, 
                device=device
            )
            dist_unlabeled = calculate_distance(feats_unlabeled, prototypes, device=device)

            # 6. 求解最佳閾值 (Wasserstein Distance)
            best_tau, min_wd, _ = solve_optimal_threshold(dist_ref, dist_unlabeled)
            
            # 7. 執行篩選
            kept_indices = dist_unlabeled <= best_tau
            num_kept = np.sum(kept_indices)
            
            print(f" [VRA] Result: Threshold={best_tau:.4f}, MinWD={min_wd:.4f}")
            print(f" [VRA] Filtering: Kept {num_kept} / {len(ulb_data)} samples ({num_kept/len(ulb_data):.2%})")

            analysis_save_path = os.path.join(args.save_dir, args.save_name, 'vra_analysis.txt')
            analyze_and_save_results(kept_indices, ulb_targets, analysis_save_path, best_tau, min_wd)
            save_path = os.path.join(args.save_dir, args.save_name, 'vra_cdf.png')
            plot_cdf({'Reference': dist_ref, 'Unlabeled': dist_unlabeled, 'Filtered': dist_unlabeled[kept_indices]}, best_tau, save_path)

            # 更新 Unlabeled Data
            ulb_data = ulb_data[kept_indices]
            ulb_targets = ulb_targets[kept_indices]
            
            print("="*40 + "\n")
        
        else:
            print("[Warning] No checkpoint provided in args. VRA skipped.")
            


    
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))

    save_dir = os.path.join(args.save_dir, args.save_name)
    with open(os.path.join(save_dir, f'Note.txt'), 'w') as f:
        f.write("lb_count: {}\n".format(lb_count))
        f.write("ulb_count: {}\n".format(ulb_count + [(ulb_targets == -1).sum()]))
        f.write("OOD unlabeled images: {}\n".format((ulb_targets == -1).sum()))
        f.close()
    # lb_count = lb_count / lb_count.sum()
    # ulb_count = ulb_count / ulb_count.sum()
    # args.lb_class_dist = lb_count
    # args.ulb_class_dist = ulb_count

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets
        # if len(ulb_data) == len(data):
        #     lb_data = ulb_data 
        #     lb_targets = ulb_targets
        # else:
        #     lb_data = np.concatenate([lb_data, ulb_data], axis=0)
        #     lb_targets = np.concatenate([lb_targets, ulb_targets], axis=0)
    
    # output the distribution of labeled data for remixmatch
    # count = [0 for _ in range(num_classes)]
    # for c in lb_targets:
    #     count[c] += 1
    # dist = np.array(count, dtype=float)
    # dist = dist / dist.sum()
    # dist = dist.tolist()
    # out = {"distribution": dist}
    # output_file = r"./data_statistics/"
    # output_path = output_file + str(name) + '_' + str(num_labels) + '.json'
    # if not os.path.exists(output_file):
    #     os.makedirs(output_file, exist_ok=True)
    # with open(output_path, 'w') as w:
    #     json.dump(out, w)

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, transform_strong, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False)
    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False)

    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=False, download=True)
    test_data, test_targets = dset.data, dset.targets
    eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, None, False)

    return lb_dset, ulb_dset, eval_dset
