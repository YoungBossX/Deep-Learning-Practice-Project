# -*-coding: utf-8 -*-
# @Time    : 2025/3/20 20:30
# @Author  : XCC
# @File    : AlexNet.py
# @Software: PyCharm

import torch
from torch import nn

# AlexNet在2012年提出，是第一个现代深度卷积网络模型
# 使用很多现代卷积神经网络技术：
# （1）使用GPU进行训练；
# （2）采用ReLU作为激活函数；
# （3）使用Dropout防止过拟合；
# （4）使用数据增强来提高模型准确度等

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=11,
                stride=4,
                padding=0
            ), # 输出图像大小=((H-F+2P)/S)+1 -> (batch, 96, 55, 55)
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )  # 输出图像大小 -> (batch, 96, 27, 27)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=5,
                stride=1,
                padding=2
            ), # 输出图像大小=((H-F+2P)/S)+1 -> (batch, 256, 27, 27)
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )  # 输出图像大小 -> (batch, 256, 13, 13)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1
            ), # 输出图像大小=((H-F+2P)/S)+1 -> (batch, 384, 13, 13)
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ), # 输出图像大小=((H-F+2P)/S)+1 -> (batch, 256, 13, 13)
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )  # 输出图像大小 -> (batch, 256, 6, 6)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(
                256*6*6,
                4096),
            nn.ReLU(),
            nn.Dropout(0.5) # 防止过拟合
        )

        self.fc2 = nn.Sequential(
            nn.Linear(
                4096,
                1024),
            nn.ReLU(),
            nn.Dropout(0.5)
            )

        self.fc3 = nn.Sequential(
            nn.Linear(
                1024,
                10),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.conv1(x) # (batch, 3, 227, 227) -> (batch, 96, 27, 27)
        x = self.conv2(x) # (batch, 96, 27, 27) -> (batch, 256, 13, 13)
        x = self.conv3(x) # (batch, 256, 13, 13) -> (batch, 384, 13, 13)
        x = self.conv4(x) # (batch, 384, 13, 13) -> (batch, 384, 13, 13)
        x = self.conv5(x) # (batch, 384, 13, 13) -> (batch, 256, 6, 6)
        x = x.view(x.size(0), -1) # (batch, 256, 6, 6) -> (batch, 256*6*6 = 9216)
        x = self.fc1(x) # (batch, 256*6*6) -> (batch, 4096)
        x = self.fc2(x) # (batch, 4096) -> (batch, 1024)
        x = self.fc3(x) # (batch, 1024) -> (batch, 10)
        return x

if __name__ == '__main__':
    x = torch.randn(1, 3, 227, 227)
    model = AlexNet()
    print(model)
    print(model(x).shape)
