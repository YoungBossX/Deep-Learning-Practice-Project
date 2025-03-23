# -*-coding: utf-8 -*-
# @Time    : 2025/3/20 20:14
# @Author  : XCC
# @File    : CNN.py
# @Software: PyCharm

import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ) # 输出图像大小 -> (batch, 16, 16, 16)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ) # 输出图像大小 -> (batch, 64, 8, 8)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 8 * 8,
                      out_features=1024
                      ),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1024,
                      out_features=128
                      ),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=128,
                      out_features=10
                      ),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # 拉成一维 (batch, 64, 8, 8) -> (batch, 64*8*8)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    x = torch.randn(3, 3, 32, 32)
    model = CNN()
    print(model)
    print(model(x).shape)