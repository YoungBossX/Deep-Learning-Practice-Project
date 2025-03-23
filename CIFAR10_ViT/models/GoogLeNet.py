# -*-coding: utf-8 -*-
# @Time    : 2025/3/21 22:53
# @Author  : XCC
# @File    : GoogLeNet.py
# @Software: PyCharm

from torch import nn
import torch

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv_block, self).__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, in_3x3, out_3x3, in_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()

        self.branch1 = Conv_block(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            Conv_block(in_channels, in_3x3, kernel_size=1),
            Conv_block(in_3x3, out_3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            Conv_block(in_channels, in_5x5, kernel_size=1),
            Conv_block(in_5x5, out_5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv_block(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

class InceptionAux_block(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux_block, self).__init__()

        self.averagepool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = Conv_block(in_channels, 128, kernel_size=1) # output[batch, 128, 4, 4]
        self.drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagepool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class GoogLeNet(nn.Module):
    def __init__(self, in_channels, num_classes, aux_logits=True):
        super(GoogLeNet, self).__init__()

        self.aux_logits = aux_logits

        self.conv1 = Conv_block(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = Conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux_block(512, num_classes)
            self.aux2 = InceptionAux_block(528, num_classes)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1) # 224/2/2/2/2/2 = 7 32/2/2/2/2/2 = 1
        self.dropout = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x) # (batch, 3, 224, 224) -> (batch, 64, 112, 112)
        x = self.maxpool1(x) # (batch, 64, 112, 112) -> (batch, 64, 56, 56)

        x = self.conv2(x) # (batch, 64, 56, 56) -> (batch, 192, 56, 56)
        x = self.maxpool2(x) # (batch, 192, 56, 56) -> (batch, 192, 28, 28)

        x = self.inception3a(x) # (batch, 192, 28, 28) -> (batch, 256, 28, 28)
        x = self.inception3b(x) # (batch, 256, 28, 28) -> (batch, 480, 28, 28)
        x = self.maxpool3(x) # (batch, 480, 28, 28) -> (batch, 480, 14, 14)

        x = self.inception4a(x) # (batch, 480, 14, 14) -> (batch, 512, 14, 14)
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        x = self.inception4b(x) #  (batch, 512, 14, 14) -> (batch, 512, 14, 14)
        x = self.inception4c(x) # (batch, 512, 14, 14) -> (batch, 512, 14, 14)
        x = self.inception4d(x) # (batch, 512, 14, 14) -> (batch, 528, 14, 14)
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        x = self.inception4e(x) # (batch, 528, 14, 14) -> (batch, 832, 14, 14)
        x = self.maxpool4(x) # (batch, 832, 14, 14) -> (batch, 832, 7, 7)

        x = self.inception5a(x) # (batch, 832, 7, 7) -> (batch, 832, 7, 7)
        x = self.inception5b(x) # (batch, 832, 7, 7) -> (batch, 1024, 7, 7)

        x = self.avgpool(x) # (batch, 1024, 7, 7) -> (batch, 1024, 1, 1)
        x = x.view(x.size(0), -1) # (batch, 1024, 1, 1) -> (batch, 1024)
        x = self.dropout(x) # (batch, 1024) -> (batch, 1024)
        x = self.fc1(x) # (batch, 1024) -> (batch, num_classes)

        if self.aux_logits and self.training:
            return x, aux1, aux2
        return x

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = GoogLeNet(3, 10)
    print(model)

    model.train()
    output = model(x)
    output, aux1, aux2 = model(x)
    print(f"主输出尺寸: {output.shape}")
    print(f"辅助输出1尺寸: {aux1.shape}")
    print(f"辅助输出2尺寸: {aux2.shape}")

    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params:,}")