# -*-coding: utf-8 -*-
# @Time    : 2025/3/20 23:41
# @Author  : XCC
# @File    : VGG16.py
# @Software: PyCharm
import torch
from torch import nn

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),# (batch, 3, 224, 224) -> (batch, 64, 224, 224)
            nn.SELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ), # (batch, 64, 224, 224) -> (batch, 64, 224, 224)
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ), # (batch, 64, 224, 224) -> (batch, 64, 112, 112)

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ), # (batch, 64, 112, 112) -> (batch, 128, 112, 112)
            nn.SELU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ), # (batch, 128, 112, 112) -> (batch, 128, 112, 112)
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ), # (batch, 128, 112, 112) -> (batch, 128, 56, 56)

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ), # (batch, 128, 56, 56) -> (batch, 256, 56, 56)
            nn.SELU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ), # (batch, 256, 56, 56) -> (batch, 256, 56, 56)
            nn.SELU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ), # (batch, 256, 56, 56) -> (batch, 256, 56, 56)
            nn.SELU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ), # (batch, 256, 56, 56) -> (batch, 256, 28, 28)

            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ), # (batch, 256, 28, 28) -> (batch, 512, 28, 28)
            nn.SELU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ), # (batch, 512, 28, 28) -> (batch, 512, 28, 28)
            nn.SELU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ), # (batch, 512, 28, 28) -> (batch, 512, 28, 28)
            nn.SELU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ), # (batch, 512, 28, 28) -> (batch, 512, 14, 14)

            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ), # (batch, 512, 14, 14) -> (batch, 512, 14, 14)
            nn.SELU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ), # (batch, 512, 14, 14) -> (batch, 512, 14, 14)
            nn.SELU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ), # (batch, 512, 14, 14) -> (batch, 512, 14, 14)
            nn.SELU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ) # (batch, 512, 14, 14) -> (batch, 512, 7, 7)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096), # (batch, 512*7*7) -> (batch, 4096)
            nn.SELU(),
            nn.Linear(4096, 1024), # (batch, 4096) -> (batch, 1024)
            nn.SELU(),
            nn.Linear(1024, 10) # (batch, 1024) -> (batch, 10)
        )

    def forward(self, x):
        x = self.conv(x) # (batch, 3, 224, 224) -> (batch, 512, 7, 7)
        x = x.view(x.size(0), -1) # (batch, 512, 7, 7) -> (batch, 512*7*7)
        x = self.classifier(x) # (batch, 512*7*7) -> (batch, 10)
        return x

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = VGG16()
    print(model)
    print(model(x).shape)