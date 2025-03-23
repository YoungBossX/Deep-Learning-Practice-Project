# -*-coding: utf-8 -*-
# @Time    : 2025/2/23 13:54
# @Author  : XCC
# @File    : cal_mean _std.py
# @Software: PyCharm

import torch
from torchvision import datasets, transforms

# 加载数据集
# 定义数据转换（仅转换为Tensor）
transform = transforms.Compose([
    transforms.ToTensor()  # 将图像转换为Tensor，并归一化到 [0,1]
])

train_dataset = datasets.CIFAR10(root='./datasets', train=True, transform=transform)
print("train_dataset:{}".format(train_dataset[0]))

num_images = len(train_dataset)
print("num_images:{}".format(num_images))

# 计算均值
mean = torch.zeros(3)
for images, _ in train_dataset:
    mean += images.mean(dim=(1, 2))  # 每个通道的均值
mean /= num_images
print(f"MeanShape: {mean.shape}")
print(f"Mean: {mean.tolist()}") # [0.49139961528778076, 0.48215857124328613, 0.44653093814849854]

# 计算标准差
std = torch.zeros(3)
for images, _ in train_dataset:
    std += (images - mean.reshape(3, 1, 1)).pow(2).mean(dim=(1, 2))
std = (std / num_images).sqrt()
print(f"Std: {std.tolist()}") # [0.24703224003314972, 0.24348513782024384, 0.2615878584384918]