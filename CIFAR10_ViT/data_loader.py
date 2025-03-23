# -*-coding: utf-8 -*-
# @Time    : 2025/2/23 13:40
# @Author  : XCC
# @File    : data_loader.py
# @Software: PyCharm

import os
from torchvision import transforms
import pickle
import numpy as np
from torch.utils.data import Dataset

# 图像预处理
transform_train_1 = transforms.Compose([
    transforms.ToTensor(),  # 转为tensor (H, W, C) -> (C, H, W)
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]  # 标准化
)

transform_train_2 = transforms.Compose([
    transforms.ToTensor(),  # 转为tensor (H, W, C) -> (C, H, W) 并归一化
    transforms.Resize(256),
    transforms.RandomCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]  # 标准化
)

transform_train_3 = transforms.Compose([
    transforms.ToTensor(),  # 转为tensor (H, W, C) -> (C, H, W) 并归一化
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]  # 标准化
)

transform_eval1 = transforms.Compose([
    transforms.ToTensor(),  # 转为tensor (H, W, C) -> (C, H, W) 并归一化
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]  # 标准化
)

transform_eval2 = transforms.Compose([
    transforms.ToTensor(),  # 转为tensor (H, W, C) -> (C, H, W) 并归一化
    transforms.Resize(227),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]  # 标准化
)

transform_eval3 = transforms.Compose([
    transforms.ToTensor(),  # 转为tensor (H, W, C) -> (C, H, W) 并归一化
    transforms.Resize(224),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]  # 标准化
)


def load_cifar10_batch(fold_path, batch_id=1, mode='train'):
    if mode == 'test':
        file_path = os.path.join(fold_path, 'test_batch')
    else:
        file_path = os.path.join(fold_path, 'data_batch_' + str(batch_id))

    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')

    imgs = batch['data'].reshape((len(batch['data']), 3, 32, 32)) / 255.  # (C, H, W)
    labels = batch['labels']

    return np.array(imgs, dtype='float32'), np.array(labels)

class CIFAR10_Dataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.folder_path = data_dir + '/cifar-10-batches-py'
        self.transform = transform
        self.mode = mode
        if mode == 'train':
            # 加载batch1-batch4作为训练集
            self.imgs, self.labels = load_cifar10_batch(self.folder_path, mode='train')
            for i in range(2, 5):
                imgs_batch, labels_batch = load_cifar10_batch(folder_path=self.folder_path, batch_id=i, mode='train')
                self.imgs, self.labels = np.concatenate([self.imgs, imgs_batch]), np.concatenate([self.labels, labels_batch])
        elif mode == 'val':
            # 加载batch5作为验证集
            self.imgs, self.labels = load_cifar10_batch(self.folder_path, batch_id=5, mode='val')
        elif mode == 'test':
            # 加载test_batch作为测试集
            self.imgs, self.labels = load_cifar10_batch(self.folder_path, mode='test')

    def __getitem__(self, idx):
        img, label = self.imgs[idx], self.labels[idx]
        img = self.transform(img.transpose(1, 2, 0))  # (C, H, W) -> (H, W, C)
        return img, label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    file_path = 'datasets/cifar-10-batches-py/data_batch_1'
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
    print(batch.keys())