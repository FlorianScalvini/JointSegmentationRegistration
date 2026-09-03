from torch import nn
import torch.nn.functional as F
import torch
from typing import Sequence


class Unet(nn.Module):
    def __init__(self, in_channels: int, channels: Sequence[int], out_channels: int, t_dim: int | None = None, final_activation: nn.Module | None = None) -> None:
        super().__init__()
        self.encoder = EncoderUnet(in_channels, channels, t_dim)
        self.decoder = DecoderUnet(channels, out_channels, t_dim)
        self.final_activation = final_activation

    def forward(self, x, t=None) -> torch.Tensor:
        feats = self.encoder(x, t)
        out = self.decoder(feats, t)
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out

class DecoderUnet(nn.Module):
    def __init__(self, channels: Sequence[int], out_channels: int, t_dim: int | None = None) -> None:
        super().__init__()
        ch = list(channels)
        self.decoder = nn.ModuleList([
            UnetUpBlock(ch[i], ch[i-1], kernel_size=3, t_dim=t_dim)
            for i in range(len(channels) - 1, 0, -1)
        ])
        self.final_conv = nn.Conv3d(ch[0], out_channels, kernel_size=1)

    def forward(self, feats: Sequence[torch.Tensor], t=None) -> torch.Tensor:
        x = feats[-1]
        for i in range(len(self.decoder)):
            x_skip = feats[-(i + 2)]
            x = self.decoder[i](x, x_skip, t)
        out = self.final_conv(x)
        return out
    
class EncoderUnet(nn.Module):
    def __init__(self, in_channels: int, channels: Sequence[int], t_dim: int | None) -> None:
        super().__init__()
        ch = [in_channels] + list(channels)
        self.encoder = nn.ModuleList([
            UnetBlock(ch[i], ch[i+1], ch[i+1],
                      kernel_size=3, padding=1,
                      stride=1 if i == 0 else 2,
                      t_dim=t_dim)
            for i in range(len(channels))
        ])

    def forward(self, x: torch.Tensor, t=None) -> Sequence[torch.Tensor]:
        feats = []
        for stage in self.encoder:
            x = stage(x, t)
            feats.append(x)
        return feats


class UnetBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int,  t_dim: int | None, kernel_size: Sequence[int] | int, stride=1, padding=1, bias=False):
        super(UnetBlock, self).__init__()
        self.cvbact_1 = Conv3dReLU(in_channels, mid_channels, kernel_size, padding=padding, stride=stride)
        self.cvbact_2 = Conv3dReLU(mid_channels, out_channels, kernel_size, padding=padding, stride=1)
        if t_dim is not None:
            self.time_mlp = nn.Linear(t_dim, out_channels * 2, bias=True)
            self.temporal_modulation = True
        else:
            self.temporal_modulation = False
        self.spatial_dims = 3

    def forward(self, x, t=None) -> torch.Tensor:
        out = self.cvbact_1(x)
        out = self.cvbact_2(out)
        if self.temporal_modulation:
            t_embed = self.time_mlp(t)
            spatial_shape = [1] * self.spatial_dims
            gamma, beta = t_embed.chunk(2, dim=-1)
            gamma = gamma.view(*gamma.shape, *spatial_shape)
            beta = beta.view(*beta.shape, *spatial_shape)
            out = out * gamma + beta
        return out

class UnetUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Sequence[int] | int, t_dim: int | None):
        super().__init__()
        self.upsample = nn.Sequential(
            Conv3dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]),
            nn.Upsample(scale_factor=2.0, mode='trilinear', align_corners=True)
        )
        self.conv_block = Conv3dReLU(out_channels * 2, out_channels, kernel_size=1)

        self.spatial_dims = 3
        if t_dim is not None:
            self.time_mlp = nn.Linear(t_dim, out_channels * 2, bias=True)
            self.temporal_modulation = True
        else:
            self.temporal_modulation = False

    def forward(self, x, x_skip, t=None):
        out = self.upsample(x)
        out = torch.cat((out, x_skip), dim=1)
        out = self.conv_block(out)
        if self.temporal_modulation:
            t_embed = self.time_mlp(t)
            spatial_shape = [1] * self.spatial_dims
            gamma, beta = t_embed.chunk(2, dim=-1)
            gamma = gamma.view(*gamma.shape, *spatial_shape)
            beta = beta.view(*beta.shape, *spatial_shape)
            out = out * gamma + beta
        return out

    
class Conv3dReLU(nn.Sequential):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Sequence[int] | int, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.relu = nn.LeakyReLU(inplace=False)
        self.nm = nn.InstanceNorm3d(out_channels)

    def forward(self, x) -> torch.Tensor:
        out = self.conv(x)
        out = self.nm(out)
        out = self.relu(out)
        return out
