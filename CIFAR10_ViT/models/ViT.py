# -*-coding: utf-8 -*-
# @Time    : 2025/3/21 22:54
# @Author  : XCC
# @File    : ViT.py
# @Software: PyCharm

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

# 判断t是否是元组，如果是，直接返回t；如果不是，则将t复制为元组(t, t)再返回。
# 用来处理当给出的图像尺寸或块尺寸是int类型（如224）时，直接返回为同值元组（如(224, 224)）。
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

