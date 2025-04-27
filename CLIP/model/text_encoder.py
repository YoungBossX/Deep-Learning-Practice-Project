# -*-coding: utf-8 -*-
# @Time    : 2025/4/26 23:21
# @Author  : XCC
# @File    : text_encoder.py
# @Software: PyCharm
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import embedding


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=16)
        self.dense1 = nn.Linear(in_features=16, out_features=64)
        self.dense2 = nn.Linear(in_features=64, out_features=16)
        self.linear = nn.Linear(in_features=16, out_features=8)
        self.layer_norm = nn.LayerNorm(8)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.linear(x)
        x = self.layer_norm(x)
        return x

if __name__ == '__main__':
    text_encoder = TextEncoder()
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    y = text_encoder(x)
    print(y.shape)