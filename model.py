from typing import Tuple, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, channels: int, features: int, kernel: int, stride: int, padding: int,
                 pool: Optional[Tuple[int, int]] = None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(channels, features, kernel_size=kernel, stride=stride, padding=padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(features)
        self.pool = nn.MaxPool2d(kernel_size=pool[0], stride=pool[1]) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        if self.pool:
            x = self.pool(x)
        return x


class FullyConnectedBlock(nn.Module):

    def __init__(self, input_size: int, output_size: int, norm: bool = True, drop: float = None):
        super(FullyConnectedBlock, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm1d(output_size) if norm else None
        self.drop = nn.Dropout(drop) if drop else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        if self.norm:
            x = self.norm(x)
        if self.drop:
            x = self.drop(x)
        return x


class AudioCommandClassifier(nn.Module):
    def __init__(self, num_classes: int = 30):
        super(AudioCommandClassifier, self).__init__()
        self.features = nn.Sequential(
            # for every layer input(W,H)-->output(W,H)-->pooling(W,H)
            # (161,101) --> (81,51) --> (40,25)
            ConvBlock(1, 16, 5, 2, 2, (2, 2)),
            # (40,25) --> (40,25) --> (20,12)
            ConvBlock(16, 32, 3, 1, 1, (2, 2)),
            # (20,12) --> (20,12) --> (10,6)
            ConvBlock(32, 64, 3, 1, 1, (2, 2))
        )
        self.classifier = nn.Sequential(
            FullyConnectedBlock(64 * 10 * 6, 1024),
            FullyConnectedBlock(1024, 128),
            FullyConnectedBlock(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

