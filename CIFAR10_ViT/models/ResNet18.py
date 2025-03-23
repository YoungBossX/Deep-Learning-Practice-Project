# -*-coding: utf-8 -*-
# @Time    : 2025/3/21 22:53
# @Author  : XCC
# @File    : ResNet18.py
# @Software: PyCharm

import torch
from torch import nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1], padding=1):
        super(BasicBlock, self).__init__()
        self.reslayer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.reslayer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, BasicBlock, num_classes=10):
        super(ResNet18, self).__init__()

        self.inchannels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = self._make_layer(BasicBlock, 64, [[1, 1], [1, 1]])
        self.conv3 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]])
        self.conv4 = self._make_layer(BasicBlock, 256, [[2, 1], [1, 1]])
        self.conv5 = self._make_layer(BasicBlock, 512, [[2, 1], [1, 1]])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, strides):
        layer = []
        for stride in strides:
            layer.append(block(self.inchannels, out_channels, stride))
            self.inchannels = out_channels
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x) # (batch_size, 3, 224, 224) -> (batch_size, 64, 56, 56)
        x = self.conv2(x) # (batch_size, 64, 56, 56) -> (batch_size, 64, 56, 56)
        x = self.conv3(x) # (batch_size, 64, 56, 56) -> (batch_size, 128, 28, 28)
        x = self.conv4(x) # (batch_size, 128, 28, 28) -> (batch_size, 256, 14, 14)
        x = self.conv5(x) # (batch_size, 256, 14, 14) -> (batch_size, 512, 7, 7)

        x = self.avgpool(x) # (batch_size, 512, 7, 7) -> (batch_size, 512, 1, 1)
        x = x.view(x.size(0), -1) # (batch_size, 512, 1, 1) -> (batch_size, 512)
        x = self.fc(x) # (batch_size, 512) -> (batch_size, num_classes)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = ResNet18(BasicBlock)
    print(model)
    print(model(x).shape)