# -*-coding: utf-8 -*-
# @Time    : 2025/2/22 14:29
# @Author  : XCC
# @File    : model.py
# @Software: PyCharm
import torch
from torch import nn

# 模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # 建立第一个卷积(Conv2d) -> 激活函数(Relu) -> 池化层(MaxPooling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),  # 输入通道数为1，输出通道数为16，卷积核大小为5，步长为1，填充为2
            nn.ReLU(),
            nn.MaxPool2d(2)  # 池化核大小为2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1,2), # 输入通道数为16，输出通道数为32，卷积核大小为5，步长为1，填充为2
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32*7*7, 128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    input = torch.randn(32, 1, 28, 28)
    model = Net()
    output = model(input)
    print(output.size())