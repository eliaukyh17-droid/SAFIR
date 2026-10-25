"""
Fusion Decoder for TIFDFusion.
Multi-scale feature aggregation with skip connections, adapted from CDTFusion res_decoder.py.
Takes 4-level fused features and produces the final fusion image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ResidualBlock(nn.Module):
    """Residual block with LeakyReLU, adapted from CDTFusion."""
    def __init__(self, in_channels, out_channels, stride=1, norm_type='bn'):
        super().__init__()
        norm_layer = nn.InstanceNorm2d if norm_type == 'in' else nn.BatchNorm2d
        norm_kwargs = {'affine': True} if norm_type == 'in' else {}
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.bn1 = norm_layer(out_channels, **norm_kwargs)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = norm_layer(out_channels, **norm_kwargs)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                norm_layer(out_channels, **norm_kwargs)
            )
        else:
            self.downsample = None

        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class UpBlock(nn.Module):
    """Upsample + skip connection + residual block."""
    def __init__(self, in_ch, skip_ch, out_ch, norm_type='bn'):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1),
        )
        self.conv = ResidualBlock(in_ch // 2 + skip_ch, out_ch, norm_type=norm_type)
        init.xavier_uniform_(self.up[1].weight)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # Align spatial sizes
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FusionDecoder(nn.Module):
    """
    U-Net style decoder that takes 4-level fused features and produces fusion output.

    Feature pyramid (input):
      f1: (B, C0, H, W)      - full resolution
      f2: (B, C1, H/2, W/2)  - 1/2 resolution
      f3: (B, C2, H/4, W/4)  - 1/4 resolution
      f4: (B, C3, H/8, W/8)  - 1/8 resolution (deepest)

    Decoder path:
      f4 -> up + f3 -> up + f2 -> up + f1 -> output
    """
    def __init__(self, channels=(64, 128, 256, 512), out_channels=1, use_skip=True, decoder_norm='bn'):
        super().__init__()
        self.use_skip = use_skip
        c0, c1, c2, c3 = channels

        # Determine norm_type for each UpBlock based on decoder_norm setting
        if decoder_norm == 'selective_in':
            # v14c: shallowest block uses InstanceNorm, deeper blocks use BatchNorm
            up1_norm, up2_norm, up3_norm = 'in', 'bn', 'bn'
        elif decoder_norm == 'in':
            # v14b: all InstanceNorm
            up1_norm, up2_norm, up3_norm = 'in', 'in', 'in'
        else:
            # Default (v13a): all BatchNorm
            up1_norm, up2_norm, up3_norm = 'bn', 'bn', 'bn'

        # Bottom-up path
        self.up3 = UpBlock(c3, c2 if use_skip else 0, c2, norm_type=up3_norm)
        self.up2 = UpBlock(c2, c1 if use_skip else 0, c1, norm_type=up2_norm)
        self.up1 = UpBlock(c1, c0 if use_skip else 0, c0, norm_type=up1_norm)

        # Output head
        self.out_conv = nn.Sequential(
            nn.Conv2d(c0, c0 // 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c0 // 2, out_channels, 1),
            nn.Sigmoid(),
        )
        init.xavier_uniform_(self.out_conv[0].weight)
        init.xavier_uniform_(self.out_conv[2].weight)

    def forward(self, fused_features):
        """
        Args:
            fused_features: list of 4 feature maps [f1, f2, f3, f4]
                f1: (B, C0, H, W), f2: (B, C1, H/2, W/2),
                f3: (B, C2, H/4, W/4), f4: (B, C3, H/8, W/8)
        Returns:
            out: (B, out_channels, H, W)
        """
        f1, f2, f3, f4 = fused_features

        # Decode bottom-up
        x = self.up3(f4, f3 if self.use_skip else None)   # -> (B, C2, H/4, W/4)
        x = self.up2(x, f2 if self.use_skip else None)    # -> (B, C1, H/2, W/2)
        x = self.up1(x, f1 if self.use_skip else None)    # -> (B, C0, H, W)

        out = self.out_conv(x)
        return out
