import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Tuple


# ===================== Utility Functions =====================
class UpsampleBlock(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor):
        return F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)


class DownsampleBlock(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor):
        return F.interpolate(x, scale_factor=1.0 / self.scale_factor, mode="bilinear", align_corners=False)


def conv1x1(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)


def conv3x3(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
