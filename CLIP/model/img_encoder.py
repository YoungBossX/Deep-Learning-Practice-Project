# -*-coding: utf-8 -*-
# @Time    : 2025/4/26 23:21
# @Author  : XCC
# @File    : img_encoder.py.py
# @Software: PyCharm
from torch import nn
import torch
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResidualBlock, self).__init__()
        

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()

if __name__ == '__main__':
    img_encoder = ImgEncoder()
    out = img_encoder(torch.randn(1, 1, 28, 28))
    print(out.shape)