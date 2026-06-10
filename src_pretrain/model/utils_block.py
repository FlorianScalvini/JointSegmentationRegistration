import torch
import torch.nn as nn
from typing import Sequence

class Conv3dReLU(nn.Sequential):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Sequence[int] | int, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv3d( in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)
        self.nm = nn.InstanceNorm3d(out_channels)

    def forward(self, x) -> torch.Tensor:
        out = self.conv(x)
        out = self.nm(out)
        out = self.relu(out)
        return out
