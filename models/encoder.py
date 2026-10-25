"""
Dual-Stream Hybrid Encoder for TIFDFusion.
- Stage 1-2 (shallow): CNN ResidualBlocks for local texture extraction
- Stage 3-4 (deep): Mamba VSSBlocks for global semantic modeling
Adapted from CDTFusion (encoder.py) and S4Fusion (block.py, layers.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from einops import repeat
from timm.models.layers import DropPath, trunc_normal_

from .selective_scan import selective_scan_fn


# ========================= CNN Components (adapted from CDTFusion) =========================

class ResidualBlock(nn.Module):
    """Residual block with optional downsampling."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class CNNStage(nn.Module):
    """CNN encoder stage: N residual blocks + stride-2 conv downsampling."""
    def __init__(self, in_ch, out_ch, num_blocks=2):
        super().__init__()
        layers = [ResidualBlock(in_ch, out_ch)]
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_ch, out_ch))
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        feat = self.blocks(x)
        down = self.pool(feat)
        return feat, down


# ========================= Mamba Components (adapted from S4Fusion) =========================

class SS2D(nn.Module):
    """Selective Scan 2D: 4-direction selective scan for 2D feature maps."""
    def __init__(self, d_model, d_state=16, d_conv=3, expand=1.5, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False,
                 device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        # This is a runtime option, not a learned value or checkpoint key.
        self.scan_backend = "auto"
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, groups=self.d_inner,
                                bias=conv_bias, kernel_size=d_conv,
                                padding=(d_conv - 1) // 2, **factory_kwargs)
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001,
                 dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                   "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L),
                               torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                              dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False,
            backend=self.scan_backend,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    """Vision State Space Block: LayerNorm + SS2D + residual."""
    def __init__(self, hidden_dim, drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop_rate=0., d_state=16, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.self_attention(self.ln_1(x)))


class PatchMerging2D(nn.Module):
    """Downsample by merging 2x2 patches (H,W -> H/2,W/2, C -> 2C)."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class MambaStage(nn.Module):
    """Mamba encoder stage: N VSSBlocks + PatchMerging downsampling.
    Input/Output in BHWC format for Mamba compatibility."""
    def __init__(self, dim, depth=2, d_state=16, drop_path=0., drop_rate=0., downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim, drop_path=drop_path, d_state=d_state, drop_rate=drop_rate)
            for _ in range(depth)
        ])
        self.downsample = PatchMerging2D(dim) if downsample else None

    def forward(self, x):
        """x: (B, H, W, C)"""
        for blk in self.blocks:
            x = blk(x)
        feat = x
        if self.downsample is not None:
            down = self.downsample(x)
        else:
            down = x
        return feat, down


class CNN2Mamba(nn.Module):
    """Bridge layer: converts CNN output (B,C,H,W) to Mamba input (B,H,W,C)
    with channel projection if needed."""
    def __init__(self, cnn_ch, mamba_ch):
        super().__init__()
        self.proj = nn.Conv2d(cnn_ch, mamba_ch, 1) if cnn_ch != mamba_ch else nn.Identity()
        self.norm = nn.LayerNorm(mamba_ch)

    def forward(self, x):
        """x: (B, C, H, W) -> (B, H, W, C)"""
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x


class Mamba2CNN(nn.Module):
    """Bridge layer: converts Mamba output (B,H,W,C) back to CNN format (B,C,H,W)."""
    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous()


# ========================= Main Dual-Stream Encoder =========================

class SingleStreamEncoder(nn.Module):
    """
    Single-stream hybrid encoder:
      Stage 1 (CNN): in_ch -> channels[0], /2
      Stage 2 (CNN): channels[0] -> channels[1], /4
      Stage 3 (Mamba): channels[1] -> channels[2], /8
      Stage 4 (Mamba): channels[2] -> channels[3], /16
    Returns 4 multi-scale features in (B,C,H,W) format.
    """
    def __init__(self, in_channels, channels=(64, 128, 256, 512),
                 cnn_depths=(2, 2), mamba_depths=(2, 2), d_state=16, drop_path_rate=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1-2: CNN
        self.stage1 = CNNStage(channels[0], channels[0], cnn_depths[0])
        self.stage2 = CNNStage(channels[0], channels[1], cnn_depths[1])

        # Bridge CNN -> Mamba
        self.bridge = CNN2Mamba(channels[1], channels[2])

        # Stage 3-4: Mamba
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(mamba_depths))]
        self.stage3 = MambaStage(channels[2], mamba_depths[0], d_state, drop_path=dp_rates[0], downsample=True)
        self.stage4 = MambaStage(channels[3], mamba_depths[1], d_state, drop_path=dp_rates[-1], downsample=False)

        # Bridge Mamba -> CNN for output
        self.mamba2cnn = Mamba2CNN()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            features: list of 4 feature maps [(B,C0,H,W), (B,C1,H/2,W/2),
                       (B,C2,H/4,W/4), (B,C3,H/8,W/8)]
        """
        x = self.stem(x)              # (B, C0, H, W)

        f1, x = self.stage1(x)        # f1: (B,C0,H,W), x: (B,C0,H/2,W/2)
        f2, x = self.stage2(x)        # f2: (B,C1,H/2,W/2), x: (B,C1,H/4,W/4)

        x = self.bridge(x)            # (B, H/4, W/4, C2)
        f3, x = self.stage3(x)        # f3: (B,H/4,W/4,C2), x: (B,H/8,W/8,C3)
        f4, _ = self.stage4(x)        # f4: (B,H/8,W/8,C3)

        # Convert Mamba outputs to CNN format
        f3 = self.mamba2cnn(f3)        # (B, C2, H/4, W/4)
        f4 = self.mamba2cnn(f4)        # (B, C3, H/8, W/8)

        return [f1, f2, f3, f4]


class DualStreamEncoder(nn.Module):
    """
    Dual-stream encoder: separate vis and ir streams with independent weights.
    Returns vis_features and ir_features, each containing 4 multi-scale feature maps.
    """
    def __init__(self, vis_in_channels=3, ir_in_channels=1,
                 channels=(64, 128, 256, 512),
                 cnn_depths=(2, 2), mamba_depths=(2, 2),
                 d_state=16, drop_path_rate=0.1):
        super().__init__()
        self.vis_encoder = SingleStreamEncoder(
            vis_in_channels, channels, cnn_depths, mamba_depths, d_state, drop_path_rate)
        self.ir_encoder = SingleStreamEncoder(
            ir_in_channels, channels, cnn_depths, mamba_depths, d_state, drop_path_rate)

    def forward(self, vis, ir):
        """
        Args:
            vis: (B, 3, H, W)
            ir: (B, 1, H, W)
        Returns:
            vis_features: list of 4 feature maps
            ir_features: list of 4 feature maps
        """
        vis_features = self.vis_encoder(vis)
        ir_features = self.ir_encoder(ir)
        return vis_features, ir_features
