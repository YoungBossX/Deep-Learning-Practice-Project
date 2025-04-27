# -*-coding: utf-8 -*-
# @Time    : 2025/4/26 23:21
# @Author  : XCC
# @File    : img_encoder.py.py
# @Software: PyCharm
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResidualBlock, self).__init__()
        self.cov1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        out = self.relu(self.bn1(self.cov1(x)))
        out = self.bn2(self.conv2(out))
        out += self.conv3(x)
        out = self.relu(out)
        return out

class ImgEncoder(nn.Module):
    def __init__(self):
        super(ImgEncoder, self).__init__()
        self.res_block1 = ResidualBlock(in_channel=1, out_channel=16, stride=2) # (batch, 16, 14, 14)
        self.res_block2 = ResidualBlock(in_channel=16, out_channel=4, stride=2) # (batch, 4, 7, 7)
        self.res_block3 = ResidualBlock(in_channel=4, out_channel=1, stride=2) # (batch, 1, 4, 4)
        self.linear = nn.Linear(in_features=1*4*4, out_features=8) # (batch, 8)
        self.layer_norm = nn.LayerNorm(8) # (batch, 8)

    def forward(self, x):
        x = self.res_block1(x) # (batch, 16, 14, 14)
        x = self.res_block2(x) # (batch, 4, 7, 7)
        x = self.res_block3(x) # (batch, 1, 4, 4)
        x = x.view(x.size(0), -1) # (batch, 1*4*4)
        x = self.linear(x) # (batch, 8)
        x = self.layer_norm(x) # (batch, 8)
        return x


if __name__ == '__main__':
    img_encoder = ImgEncoder()
    out = img_encoder(torch.randn(1, 1, 28, 28))
    print(out.shape)