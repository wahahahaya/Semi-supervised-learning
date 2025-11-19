# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# fixmatch, flexmatch, pseudolabel, vat

import os
import gc
import copy
import json
import random
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import math
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset


mean, std = {}, {}
mean['imagenet'] = [0.485, 0.456, 0.406]
std['imagenet'] = [0.229, 0.224, 0.225]


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


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

    data_dir = os.path.join(data_dir, name.lower())

    train_imglist = '/home/ubuntu/SSL_data_filtering/SSL_data_filtering/OpenOOD/data/benchmark_imglist/imagenet200/train_imagenet200.txt'
    val_imglist = '/home/ubuntu/SSL_data_filtering/SSL_data_filtering/OpenOOD/data/benchmark_imglist/imagenet200/val_imagenet200.txt'
    imgpath = '/ephemeral/'
    dataset = ImagenetDataset(root=imgpath, transform=transform_weak, ulb=False, alg=alg, imglist_pth=train_imglist)
    label_perclass = num_labels // (max(dataset.targets)+1)

    lb_dset = ImagenetDataset(root=imgpath, transform=transform_weak, ulb=False, alg=alg, imglist_pth=train_imglist, label_perclass=label_perclass)

    ulb_dset = ImagenetDataset(root=imgpath, transform=transform_weak, alg=alg, imglist_pth=train_imglist, ulb=True, medium_transform=transform_medium, strong_transform=transform_strong, include_lb_to_ulb=include_lb_to_ulb, lb_index=lb_dset.lb_idx)

    eval_dset = ImagenetDataset(root=imgpath, transform=transform_val, alg=alg, imglist_pth=val_imglist, ulb=False)

    if args.use_noise:
        noise_path = args.noise_path
        ulb_dset = ImagenetDataset(root="", transform=transform_weak, alg=alg, imglist_pth=noise_path, ulb=True, medium_transform=transform_medium, strong_transform=transform_strong, include_lb_to_ulb=include_lb_to_ulb, lb_index=lb_dset.lb_idx)

    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    ood_count = 0
    for lb in lb_dset.targets:
        lb_count[lb] += 1
    for ulb in ulb_dset.targets:
        if ulb >= 0:
            ulb_count[ulb] += 1
        if ulb == -1:
            ood_count += 1
    save_dir = os.path.join(args.save_dir, args.save_name)
    noise_name = "None"
    with open(os.path.join(save_dir, f'{noise_name}.txt'), 'w') as f:
        f.write("Dataset: {}\n".format(noise_name))
        f.write("lb_count: {}\n".format(lb_count))
        f.write("ulb_count: {}\n".format(ulb_count + [ood_count]))
        f.write("OOD unlabeled images: {}\n".format(ood_count))
        f.close()

    return lb_dset, ulb_dset, eval_dset
    


class ImagenetDataset(BasicDataset, ImageFolder):
    def __init__(self, root, transform, ulb, alg, imglist_pth=None, medium_transform=None, strong_transform=None, label_perclass=-1, include_lb_to_ulb=True, lb_index=None):
        self.alg = alg
        self.is_ulb = ulb
        self.label_perclass = label_perclass
        self.transform = transform
        self.root = root
        self.include_lb_to_ulb = include_lb_to_ulb
        self.lb_index = lb_index

        if imglist_pth is not None:
            samples = self._make_dataset_from_list(imglist_pth)
        else:
            raise ValueError("You must provide imglist_pth for ImagenetDataset")

        if len(samples) == 0:
            raise RuntimeError(f"Found 0 samples in {imglist_pth}")
        
        self.data = [s[0] for s in samples]
        self.targets = [s[1] for s in samples]

        self.loader = default_loader

        # classes, class_to_idx = self.find_classes(self.root)
        # self.classes = classes
        # self.class_to_idx = class_to_idx

        unique_targets = sorted(set(self.targets))
        self.classes = [str(c) for c in unique_targets]
        self.class_to_idx = {str(c): c for c in unique_targets}


        self.medium_transform = medium_transform
        if self.medium_transform is None:
            if self.is_ulb:
                assert self.alg not in ['sequencematch'], f"alg {self.alg} requires strong augmentation"
        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch', 'refixmatch'], f"alg {self.alg} requires strong augmentation"


    def __sample__(self, index):
        path = self.data[index]
        sample = self.loader(path)
        target = self.targets[index]
        return sample, target

    
    def _make_dataset_from_list(self, imglist_pth):
        """
        txt file may like:
        'imagenet_1k/train/n04372370/n04372370_9138.JPEG 844\n'
        """
        instances = []
        buckets = {} 

        with open(imglist_pth, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for line in lines:
            path, target = line.split()
            target = int(target)
            full_path = os.path.join(self.root, path)
            if os.path.isfile(full_path):
                buckets.setdefault(target, []).append((full_path, target))

        lb_idx = {}
        if self.label_perclass > 0 and not self.is_ulb:
            for cls, items in buckets.items():
                k = min(self.label_perclass, len(items))
                chosen = random.sample(items, k)
                instances.extend(chosen)
                lb_idx[cls] = [os.path.basename(p) for p, _ in chosen]
        else:
            for cls, items in buckets.items():
                instances.extend(items)
            lb_idx = {}

        gc.collect()
        self.lb_idx = lb_idx
        return instances

