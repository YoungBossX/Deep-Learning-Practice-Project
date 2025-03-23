# -*-coding: utf-8 -*-_
# @Time    : 2025/2/22 14:30
# @Author  : XCC
# @File    : data_loader.py
# @Software: PyCharm

import os
import argparse
import torch
import  torchvision
import matplotlib.pyplot as plt
from argparse import Namespace
from torchvision import  transforms
from utils import init_logger

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])

def load_dataset(args, mode):
    logger = init_logger(args, __name__) # __name__ 的作用：为日志器提供唯一的模块名称标识，便于区分日志来源。
    dataset = None

    # 加载数据集
    if mode == 'train':
        logger.info("Start loading the Train_Dataset!")
        train_dataset = torchvision.datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
        logger.info("Num of train_dataset examples: {}".format(len(train_dataset)))
        dataset = train_dataset
    elif mode == 'test':
        logger.info("Start loading the Test_Dataset!")
        test_dataset = torchvision.datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
        logger.info("Num of test_dataset examples: {}".format(len(test_dataset)))
        dataset = test_dataset
    return dataset

if __name__ == '__main__':
    args = Namespace(
        logs_dir=os.path.join(os.getcwd(), "logs"),  # 指定日志存放目录（这里用当前工作目录下的 logs 文件夹）
        data_path="./datasets"
    )
    # 创建数据集
    train_dataset = load_dataset(args, mode="train")
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
    )
    # 可视化数据
    images, labels = next(iter(train_loader))  # 获取一个batch的数据
    plt.figure(figsize=(10, 6)) # 设置画布大小
    for j in range(32):
        plt.subplot(8, 4, j + 1)
        plt.imshow(images[j].squeeze(), cmap='gray')
        plt.title(f"Label: {labels[j].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()