import os
import random
import numbers
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader


# 图片扩展名白名单（大小写不敏感）
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def _build_index(dir_path):
    """扫描目录，按 stem 建 {stem: filename} 索引，扩展名自动探测"""
    index = {}
    for f in os.listdir(dir_path):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXTS:
            index[stem] = f
    return index


def _read_split(split_file):
    """读 train.txt / test.txt，每行一个 stem"""
    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def _pair_files(images_dir, labels_dir, split_file):
    """按 split_file 的 stem 去 images/ 和 labels/ 配对，扩展名可以不同"""
    stems = _read_split(split_file)
    img_idx = _build_index(images_dir)
    lbl_idx = _build_index(labels_dir)

    images, labels = [], []
    missing_img, missing_lbl = [], []
    for s in stems:
        if s not in img_idx:
            missing_img.append(s)
            continue
        if s not in lbl_idx:
            missing_lbl.append(s)
            continue
        images.append(os.path.join(images_dir, img_idx[s]))
        labels.append(os.path.join(labels_dir, lbl_idx[s]))

    if missing_img:
        print('[WARN] {} stems missing in images, e.g. {}'.format(len(missing_img), missing_img[:5]))
    if missing_lbl:
        print('[WARN] {} stems missing in labels, e.g. {}'.format(len(missing_lbl), missing_lbl[:5]))
    if len(images) == 0:
        raise RuntimeError('No paired files found. Check: {}\n{}\n{}'.format(
            images_dir, labels_dir, split_file))
    print('[INFO] {} pairs loaded from {}'.format(len(images), split_file))
    return images, labels


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = F.vflip(img)
            mask = F.vflip(mask)
        return img, mask


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask


class RandomRotate(object):
    def __init__(self, degrees, p=0.5, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
        self.p = p

    @staticmethod
    def get_params(degrees):
        return random.uniform(degrees[0], degrees[1])

    def __call__(self, img, mask):
        if random.random() < self.p:
            angle = self.get_params(self.degrees)
            img_rot = F.rotate(img, angle, self.resample, self.expand, self.center)
            # mask 必须最近邻，否则二值标签被插值成 0~255 中间值
            mask_rot = F.rotate(mask, angle, Image.NEAREST, self.expand, self.center)
            return img_rot, mask_rot
        return img, mask


class CrackDataset(Dataset):
    """Crack500 等裂缝数据集：读 txt 划分，images/labels 为扁平目录"""

    def __init__(self, images_dir, labels_dir, split_file, trainsize, augmentations=True):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.images, self.labels = _pair_files(images_dir, labels_dir, split_file)
        self.size = len(self.images)

        self.resize = transforms.Resize((trainsize, trainsize))
        self.rotate = RandomRotate(degrees=(0, 360))
        self.vf = RandomVerticalFlip(p=0.5)
        self.hf = RandomHorizontalFlip(p=0.5)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.labels[index])
        name = os.path.basename(self.images[index])

        image = self.resize(image)
        gt = self.resize(gt)

        if self.augmentations:
            image, gt = self.rotate(image, gt)
            image, gt = self.vf(image, gt)
            image, gt = self.hf(image, gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size


# 保留旧类名，兼容 import
PolypDataset = CrackDataset


def get_loader(images_dir, labels_dir, split_file, batchsize, trainsize,
               shuffle=True, num_workers=4, pin_memory=False, augmentation=True):
    dataset = CrackDataset(images_dir, labels_dir, split_file, trainsize, augmentation)
    return DataLoader(dataset=dataset,
                      batch_size=batchsize,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory)


class val_dataset(object):
    """验证集：逐张 load_data()，返回 (image, gt, name, image_for_post)"""

    def __init__(self, images_dir, labels_dir, split_file, testsize):
        self.testsize = testsize
        self.images, self.labels = _pair_files(images_dir, labels_dir, split_file)
        self.size = len(self.images)

        self.transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.labels[self.index])
        name = os.path.basename(self.images[self.index])

        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        if name.lower().endswith(('.jpg', '.jpeg')):
            name = os.path.splitext(name)[0] + '.png'

        self.index += 1
        self.index = self.index % self.size
        return image, gt, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size


class test_dataset(object):
    """测试集：逐张 load_data()，返回 (image, gt, name)"""

    def __init__(self, images_dir, labels_dir, split_file, testsize):
        self.testsize = testsize
        self.images, self.labels = _pair_files(images_dir, labels_dir, split_file)
        self.size = len(self.images)

        self.transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.labels[self.index])
        name = os.path.basename(self.images[self.index])
        if name.lower().endswith(('.jpg', '.jpeg')):
            name = os.path.splitext(name)[0] + '.png'

        self.index += 1
        self.index = self.index % self.size
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size