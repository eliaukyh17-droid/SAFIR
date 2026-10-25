"""
TIFM: Task-Interactive Fusion Module (Core Innovation)
- Wavelet decomposition (DWT/IDWT) for frequency-domain separation
- Segmentation prior guides low-frequency fusion (scene structure)
- Detection prior guides high-frequency fusion (edge details)
- Cross-attention for task prior injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward, DWTInverse
    HAS_WAVELETS = True
except ImportError:
    HAS_WAVELETS = False
    print("Warning: pytorch_wavelets not installed. Using simple frequency decomposition fallback.")


# ========================= Wavelet Transform Wrappers =========================

class WaveletDecompose(nn.Module):
    """Discrete Wavelet Transform: decompose into low-freq (LL) and high-freq (LH,HL,HH)."""
    def __init__(self, wavelet='haar'):
        super().__init__()
        if HAS_WAVELETS:
            self.dwt = DWTForward(J=1, wave=wavelet, mode='zero')
        else:
            self.dwt = None

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            ll: (B, C, H/2, W/2) low-frequency approximation
            hf: (B, C*3, H/2, W/2) high-frequency details (LH, HL, HH concatenated)
        """
        if self.dwt is not None:
            ll, yh = self.dwt(x)
            # yh is list of length J, each (B, C, 3, H/2, W/2)
            hf = yh[0]  # (B, C, 3, H/2, W/2)
            B, C, _, Hh, Wh = hf.shape
            hf = hf.view(B, C * 3, Hh, Wh)  # (B, C*3, H/2, W/2)
            return ll, hf
        else:
            # Fallback: simple average pooling for LL, difference for HF
            ll = F.avg_pool2d(x, 2)
            x_up = F.interpolate(ll, size=x.shape[-2:], mode='bilinear', align_corners=False)
            hf_full = x - x_up
            hf = F.avg_pool2d(hf_full, 2)
            hf = hf.repeat(1, 3, 1, 1)  # approximate 3 subbands
            return ll, hf


class WaveletReconstruct(nn.Module):
    """Inverse Discrete Wavelet Transform: reconstruct from LL and HF subbands."""
    def __init__(self, wavelet='haar'):
        super().__init__()
        if HAS_WAVELETS:
            self.idwt = DWTInverse(wave=wavelet, mode='zero')
        else:
            self.idwt = None

    def forward(self, ll, hf, target_size=None):
        """
        Args:
            ll: (B, C, H/2, W/2)
            hf: (B, C*3, H/2, W/2)
            target_size: optional (H, W) for output alignment
        Returns:
            out: (B, C, H, W)
        """
        if self.idwt is not None:
            B, C3, Hh, Wh = hf.shape
            C = C3 // 3
            hf = hf.view(B, C, 3, Hh, Wh)
            out = self.idwt((ll, [hf]))
            if target_size is not None and out.shape[-2:] != target_size:
                out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            return out
        else:
            # Fallback: simple upsample + add
            C = ll.shape[1]
            hf_single = hf[:, :C, :, :]
            out = F.interpolate(ll, scale_factor=2, mode='bilinear', align_corners=False)
            hf_up = F.interpolate(hf_single, size=out.shape[-2:], mode='bilinear', align_corners=False)
            out = out + hf_up
            if target_size is not None and out.shape[-2:] != target_size:
                out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            return out


# ========================= Cross-Attention for Task Prior Injection =========================

class CrossAttention(nn.Module):
    """
    Cross-attention with Spatial Reduction (SRA, like SegFormer/PVT).
    Query from fusion features, key/value from task prior features.
    K/V are spatially reduced by sr_ratio to avoid O(N^2) memory explosion.
    """
    def __init__(self, q_dim, kv_dim, num_heads=8, sr_ratio=1,
                 qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q_proj = nn.Conv2d(q_dim, q_dim, 1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(kv_dim, q_dim, 1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(kv_dim, q_dim, 1, bias=qkv_bias)
        self.out_proj = nn.Conv2d(q_dim, q_dim, 1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)

        # Spatial reduction for K/V (like SegFormer's Efficient Self-Attention)
        if sr_ratio > 1:
            self.sr_norm = nn.LayerNorm(kv_dim)

    def forward(self, query, key_value):
        """
        Args:
            query: (B, C_q, H, W) - fusion features
            key_value: (B, C_kv, H_kv, W_kv) - task prior features
        Returns:
            out: (B, C_q, H, W)
        """
        B, C, H, W = query.shape

        # Align spatial dimensions if different
        if key_value.shape[-2:] != (H, W):
            key_value = F.interpolate(key_value, size=(H, W), mode='bilinear', align_corners=False)

        # LayerNorm for query
        q = query.flatten(2).transpose(1, 2)
        q = self.norm_q(q)
        q = q.transpose(1, 2).reshape(B, C, H, W)

        # Spatial reduction for K/V to reduce attention matrix size
        if self.sr_ratio > 1:
            kv_reduced = F.adaptive_avg_pool2d(key_value, (
                max(1, H // self.sr_ratio), max(1, W // self.sr_ratio)))
            Hk, Wk = kv_reduced.shape[-2:]
            kv = kv_reduced.flatten(2).transpose(1, 2)
            kv = self.sr_norm(kv)
            kv = kv.transpose(1, 2).reshape(B, -1, Hk, Wk)
        else:
            Hk, Wk = H, W
            kv = key_value.flatten(2).transpose(1, 2)
            kv = self.norm_kv(kv)
            kv = kv.transpose(1, 2).reshape(B, -1, Hk, Wk)

        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        # Reshape for multi-head attention
        def split_heads(x):
            batch, _, height, width = x.shape
            return (x.reshape(batch, self.num_heads, self.head_dim, height, width)
                    .permute(0, 1, 3, 4, 2)
                    .reshape(batch, self.num_heads, height * width, self.head_dim))

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # Attention: (B,nh,H*W,hd) @ (B,nh,hd,Hk*Wk) -> (B,nh,H*W,Hk*Wk)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B,nh,H*W,hd)
        out = (out.reshape(B, self.num_heads, H, W, self.head_dim)
               .permute(0, 1, 4, 2, 3)
               .reshape(B, C, H, W))
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


# ========================= TIFM: Task-Interactive Fusion Module =========================

class SmokeReliabilityEstimator(nn.Module):
    """Infer spatial VI/IR reliability from features only.

    The estimator deliberately does not accept smoke masks, transmission maps,
    severity labels, or synthesis parameters.  Those fields remain audit-only
    dataset metadata, so inference cannot take a synthetic-data shortcut.
    """
    def __init__(self, dim):
        super().__init__()
        hidden = max(16, dim // 2)
        self.net = nn.Sequential(
            nn.Conv2d(dim * 3, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 2, 1),
        )

    def forward(self, vis_ll, ir_ll):
        evidence = torch.cat([vis_ll, ir_ll, torch.abs(vis_ll - ir_ll)], dim=1)
        weights = self.net(evidence).softmax(dim=1)
        return weights[:, 0:1], weights[:, 1:2]


class DualDomainDegradationEncoder(nn.Module):
    """Infer reliability from complementary spatial and DWT evidence.

    This compact visual-only encoder follows the degradation-adapter pattern
    used by ControlFusion and the spatial/frequency collaboration principle of
    GD2Fusion.  Synthetic transmission or smoke metadata are never inputs;
    they remain optional training targets for the predicted reliability maps.
    """
    def __init__(self, dim):
        super().__init__()
        hidden = max(16, dim // 2)

        self.spatial_branch = nn.Sequential(
            nn.Conv2d(dim * 3, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
        )
        # Each modality contributes one energy map for every DWT direction;
        # their absolute difference contributes another three maps.
        self.frequency_branch = nn.Sequential(
            nn.Conv2d(9, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
        )
        self.reliability_head = nn.Conv2d(hidden, 2, 1)

    @staticmethod
    def _directional_energy(hf, channels):
        batch, _, height, width = hf.shape
        subbands = hf.reshape(batch, channels, 3, height, width)
        return subbands.abs().mean(dim=1)

    def forward(self, vis_ll, ir_ll, vis_hf, ir_hf):
        spatial_evidence = torch.cat(
            [vis_ll, ir_ll, torch.abs(vis_ll - ir_ll)], dim=1)
        spatial_features = self.spatial_branch(spatial_evidence)

        channels = vis_ll.shape[1]
        vis_energy = self._directional_energy(vis_hf, channels)
        ir_energy = self._directional_energy(ir_hf, channels)
        frequency_evidence = torch.cat(
            [vis_energy, ir_energy, torch.abs(vis_energy - ir_energy)], dim=1)
        frequency_features = self.frequency_branch(frequency_evidence)

        degradation_features = self.fuse(
            torch.cat([spatial_features, frequency_features], dim=1))
        weights = self.reliability_head(degradation_features).softmax(dim=1)
        return weights[:, 0:1], weights[:, 1:2], degradation_features


class FrequencySelectiveRouter(nn.Module):
    """Predict a local max-detail/reliability-blend routing coefficient."""
    def __init__(self, dim):
        super().__init__()
        hidden = max(16, dim // 2)
        self.net = nn.Sequential(
            nn.Conv2d(dim * 3 + 2, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, vis_ll, ir_ll, w_vis, w_ir):
        evidence = torch.cat(
            [vis_ll, ir_ll, torch.abs(vis_ll - ir_ll), w_vis, w_ir], dim=1)
        return self.net(evidence)


class MultiScaleGradientSemanticGate(nn.Module):
    """Detection-semantic boundary gate derived from PG-Fusion's MGRB.

    PG-Fusion aggregates Laplacian responses at several spatial scales to
    preserve target contours.  Here the same idea is adapted to the existing
    wavelet high-frequency path: fixed multi-scale Laplacian evidence predicts
    a zero-centred gate for the already available detection-prior guide.  The
    module therefore adds no second task network and is exactly neutral at
    initialization.
    """

    def __init__(self, channels, hidden_channels=8):
        super().__init__()
        laplacian = torch.tensor(
            [[-1.0, -1.0, -1.0],
             [-1.0, 8.0, -1.0],
             [-1.0, -1.0, -1.0]])
        self.register_buffer(
            "laplacian", laplacian.view(1, 1, 3, 3), persistent=False)
        hidden_channels = max(4, int(hidden_channels))
        self.gate = nn.Sequential(
            nn.Conv2d(4, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        # A zero final projection makes the new branch an identity-preserving
        # residual at initialization while retaining non-zero gradients.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.channels = int(channels)

    @staticmethod
    def _normalise_energy(energy):
        denominator = energy.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return energy / denominator

    def _gradient_energy(self, features, scale):
        height, width = features.shape[-2:]
        pool = min(int(scale), height, width)
        if pool > 1:
            reduced = F.avg_pool2d(features, kernel_size=pool, stride=pool)
        else:
            reduced = features
        channels = reduced.shape[1]
        kernel = self.laplacian.to(dtype=reduced.dtype).repeat(
            channels, 1, 1, 1)
        response = F.conv2d(reduced, kernel, padding=1, groups=channels)
        energy = response.abs().mean(dim=1, keepdim=True)
        if energy.shape[-2:] != (height, width):
            energy = F.interpolate(
                energy, size=(height, width), mode="bilinear",
                align_corners=False)
        return self._normalise_energy(energy)

    def forward(self, high_frequency, detection_guide):
        if detection_guide is None:
            return torch.zeros_like(high_frequency), None
        gradient_evidence = [
            self._gradient_energy(high_frequency, scale)
            for scale in (1, 2, 4)
        ]
        semantic_energy = self._normalise_energy(
            detection_guide.abs().mean(dim=1, keepdim=True))
        gate = torch.tanh(self.gate(torch.cat(
            gradient_evidence + [semantic_energy], dim=1)))
        correction = (
            torch.tanh(self.residual_scale)
            * gate
            * torch.tanh(detection_guide)
        )
        return correction, gate


class TIFMBlock(nn.Module):
    """
    Single-level Task-Interactive Fusion Module.
    For one level of the feature pyramid:
      1. DWT decompose vis_feat and ir_feat
      2. Seg prior -> cross-attention -> guide low-freq fusion
      3. Det prior -> cross-attention -> guide high-freq fusion
      4. IDWT reconstruct fused feature
      5. Output refinement
    """
    def __init__(self, dim, seg_prior_dim, det_prior_dim, num_heads=8,
                 wavelet='haar', use_smoke_reliability=False,
                 use_frequency_router=False, use_prior_conditioning=False,
                 use_dual_domain_degradation=False,
                 external_reliability=False,
                 preserve_high_frequency=False,
                 residual_reliability_modulation=False,
                 use_multiscale_gradient_semantic_gate=False):
        super().__init__()

        self.use_smoke_reliability = bool(use_smoke_reliability)
        self.use_frequency_router = bool(use_frequency_router)
        self.use_prior_conditioning = bool(use_prior_conditioning)
        self.use_dual_domain_degradation = bool(use_dual_domain_degradation)
        self.external_reliability = bool(external_reliability)
        self.preserve_high_frequency = bool(preserve_high_frequency)
        self.residual_reliability_modulation = bool(
            residual_reliability_modulation)
        self.use_multiscale_gradient_semantic_gate = bool(
            use_multiscale_gradient_semantic_gate)
        # The router depends on locally inferred reliability, but its ablation
        # may be enabled independently from reliability-weighted LF fusion.
        needs_reliability = self.use_smoke_reliability or self.use_frequency_router \
            or self.use_prior_conditioning
        if not needs_reliability or self.external_reliability:
            self.reliability_estimator = None
        elif self.use_dual_domain_degradation:
            self.reliability_estimator = DualDomainDegradationEncoder(dim)
        else:
            self.reliability_estimator = SmokeReliabilityEstimator(dim)
        self.frequency_router = FrequencySelectiveRouter(dim) if self.use_frequency_router else None
        self.reliability_strength_logit = nn.Parameter(torch.tensor(0.0)) \
            if self.residual_reliability_modulation else None

        # Wavelet decomposition
        self.decompose = WaveletDecompose(wavelet)
        self.reconstruct = WaveletReconstruct(wavelet)

        # Low-frequency fusion (guided by segmentation prior)
        # sr_ratio: spatially reduce K/V to avoid OOM on high-res features
        sr_ratio = max(1, 128 // 16)  # target ~16x16 for K/V tokens
        if dim <= 64:
            sr_ratio = 8   # level0: 128x128 -> 16x16
        elif dim <= 128:
            sr_ratio = 4   # level1: 64x64 -> 16x16
        elif dim <= 256:
            sr_ratio = 2   # level2: 32x32 -> 16x16
        else:
            sr_ratio = 1   # level3: 16x16 -> no reduction
        self.seg_cross_attn = CrossAttention(q_dim=dim, kv_dim=seg_prior_dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.lf_fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # DW-Conv
            nn.Conv2d(dim, dim, 1),
        )
        self.lf_gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid(),
        )

        # High-frequency fusion (guided by detection prior)
        hf_dim = dim * 3  # 3 subbands concatenated
        self.det_cross_attn = CrossAttention(q_dim=hf_dim, kv_dim=det_prior_dim, num_heads=num_heads, sr_ratio=sr_ratio)
        # Learnable scale for detection prior modulation (replaces hardcoded 0.1)
        self.det_scale = nn.Parameter(torch.tensor(0.5))
        # Residual refinement for max-based high-frequency fusion
        self.hf_refine = nn.Sequential(
            nn.Conv2d(hf_dim * 2, hf_dim, 1),
            nn.GELU(),
            nn.Conv2d(hf_dim, hf_dim, 3, padding=1, groups=min(hf_dim, dim)),
            nn.Conv2d(hf_dim, hf_dim, 1),
            nn.Tanh(),
        )
        # Learnable residual scale (initialized small to avoid disrupting max baseline)
        self.hf_res_scale = nn.Parameter(torch.tensor(0.1))
        self.gradient_semantic_gate = (
            MultiScaleGradientSemanticGate(hf_dim)
            if self.use_multiscale_gradient_semantic_gate else None
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

        # Diagnostics are opt-in and never participate in the forward graph.
        # Figure-export scripts may enable this on one scale to capture honest
        # learned tensors without changing the model outputs or checkpoint.
        self.capture_debug = False
        self.debug_tensors = {}

    def forward(self, vis_feat, ir_feat, seg_prior=None, det_prior=None,
                return_aux=False, reliability_override=None):
        """
        Args:
            vis_feat: (B, C, H, W)
            ir_feat: (B, C, H, W)
            seg_prior: (B, C_seg, H_s, W_s) or None
            det_prior: (B, C_det, H_d, W_d) or None
        Returns:
            fused: (B, C, H, W)
        """
        target_size = vis_feat.shape[-2:]

        # 1. DWT decomposition
        vis_ll, vis_hf = self.decompose(vis_feat)
        ir_ll, ir_hf = self.decompose(ir_feat)

        # Infer reliability from features only.  No synthesis metadata enters
        # the model graph at training or inference time.
        w_vis = None
        w_ir = None
        degradation_features = None
        if reliability_override is not None:
            w_vis, w_ir = reliability_override
        elif self.reliability_estimator is not None:
            if self.use_dual_domain_degradation:
                w_vis, w_ir, degradation_features = self.reliability_estimator(
                    vis_ll, ir_ll, vis_hf, ir_hf)
            else:
                w_vis, w_ir = self.reliability_estimator(vis_ll, ir_ll)

        # 2. Low-frequency fusion (scene structure, seg prior guided)
        if self.use_smoke_reliability:
            if w_vis is None or w_ir is None:
                raise RuntimeError(
                    "Smoke reliability is enabled but no reliability maps "
                    "were produced")
            if self.residual_reliability_modulation:
                # Bounded residual gains retain both modalities instead of
                # replacing them with a potentially over-suppressed mixture.
                strength = 0.5 * torch.sigmoid(
                    self.reliability_strength_logit)
                vis_gain = 1.0 + strength * (2.0 * w_vis - 1.0)
                ir_gain = 1.0 + strength * (2.0 * w_ir - 1.0)
                lf_inputs = [vis_gain * vis_ll, ir_gain * ir_ll]
            else:
                lf_inputs = [w_vis * vis_ll, w_ir * ir_ll]
        else:
            lf_inputs = [vis_ll, ir_ll]
        lf_cat = torch.cat(lf_inputs, dim=1)  # (B, 2C, H/2, W/2)
        lf_base = self.lf_fuse(lf_cat)  # (B, C, H/2, W/2)

        seg_guide = None
        gate = None
        if seg_prior is not None:
            seg_guide = self.seg_cross_attn(lf_base, seg_prior)
            if self.use_prior_conditioning:
                # Preserve a non-zero prior path while reducing semantic
                # injection where the visible branch is locally unreliable.
                seg_guide = seg_guide * (0.5 + 0.5 * w_vis)
            gate = self.lf_gate(seg_guide)
            lf_fused = lf_base + gate * seg_guide
        else:
            lf_fused = lf_base

        # 3. High-frequency fusion (edge details, det prior guided)
        # Max-based fusion: preserve the stronger gradient from either source.
        hf_max = torch.where(torch.abs(vis_hf) >= torch.abs(ir_hf), vis_hf, ir_hf)

        hf_reliable = None
        router_alpha = None
        hf_base = hf_max
        if self.reliability_estimator is not None:
            w_vis_hf = w_vis.repeat(1, 3 * vis_ll.shape[1], 1, 1)
            w_ir_hf = w_ir.repeat(1, 3 * ir_ll.shape[1], 1, 1)
            hf_reliable = w_vis_hf * vis_hf + w_ir_hf * ir_hf
            if self.use_smoke_reliability and not self.preserve_high_frequency:
                hf_base = hf_reliable
        if self.frequency_router is not None:
            router_alpha = self.frequency_router(vis_ll, ir_ll, w_vis, w_ir)
            hf_base = router_alpha * hf_max + (1.0 - router_alpha) * hf_reliable

        # Learnable residual correction on top of max baseline
        hf_cat = torch.cat([vis_hf, ir_hf], dim=1)  # (B, 6C, H/2, W/2)
        hf_residual_base = self.hf_refine(hf_cat)  # (B, 3C, H/2, W/2), Tanh output [-1,1]
        hf_residual = hf_residual_base

        det_guide = None
        gradient_semantic_gate = None
        if det_prior is not None:
            det_guide = self.det_cross_attn(hf_base, det_prior)
            if self.use_prior_conditioning:
                det_weight = (0.5 + 0.5 * w_ir).repeat(1, 3 * vis_ll.shape[1], 1, 1)
                det_guide = det_guide * det_weight
            # Detection prior further refines the residual with learnable scale
            hf_residual = hf_residual + self.det_scale * torch.tanh(det_guide)
            if self.gradient_semantic_gate is not None:
                boundary_correction, gradient_semantic_gate = (
                    self.gradient_semantic_gate(hf_base, det_guide))
                hf_residual = hf_residual + boundary_correction

        hf_fused = hf_base + self.hf_res_scale * hf_residual

        # 4. IDWT reconstruction
        fused = self.reconstruct(lf_fused, hf_fused, target_size=target_size)

        # 5. Final refinement with residual
        fused = fused + self.refine(fused)

        if self.capture_debug:
            def snapshot(tensor):
                return None if tensor is None else tensor.detach().float().cpu()

            self.debug_tensors = {
                'vis_ll': snapshot(vis_ll),
                'ir_ll': snapshot(ir_ll),
                'vis_hf': snapshot(vis_hf),
                'ir_hf': snapshot(ir_hf),
                'lf_base': snapshot(lf_base),
                'seg_guide': snapshot(seg_guide),
                'lf_gate': snapshot(gate),
                'lf_fused': snapshot(lf_fused),
                'hf_max': snapshot(hf_max),
                'hf_reliable': snapshot(hf_reliable),
                'router_alpha': snapshot(router_alpha),
                'reliability_vis': snapshot(w_vis),
                'reliability_ir': snapshot(w_ir),
                'degradation_features': snapshot(degradation_features),
                'hf_residual_base': snapshot(hf_residual_base),
                'det_guide': snapshot(det_guide),
                'gradient_semantic_gate': snapshot(gradient_semantic_gate),
                'hf_residual_corrected': snapshot(hf_residual),
                'hf_fused': snapshot(hf_fused),
                'fused_feature': snapshot(fused),
                'reliability_strength': None if self.reliability_strength_logit is None
                    else float((0.5 * torch.sigmoid(
                        self.reliability_strength_logit)).detach().cpu()),
                'det_scale': float(self.det_scale.detach().cpu()),
                'hf_res_scale': float(self.hf_res_scale.detach().cpu()),
            }

        if return_aux:
            return fused, {
                'reliability_vis': w_vis,
                'reliability_ir': w_ir,
                'router_alpha': router_alpha,
                'degradation_features': degradation_features,
            }
        return fused


class TIFM(nn.Module):
    """
    Multi-level Task-Interactive Fusion Module.
    Applies TIFMBlock at each level of the 4-level feature pyramid.
    """
    def __init__(self, channels=(64, 128, 256, 512),
                 seg_prior_channels=(64, 128, 256, 512),
                 det_prior_channels=(64, 128, 256, 512),
                 num_heads=8, wavelet='haar', use_smoke_reliability=False,
                 use_frequency_router=False, use_prior_conditioning=False,
                 use_dual_domain_degradation=False,
                 use_hierarchical_degradation=False,
                 preserve_high_frequency=False,
                 residual_reliability_modulation=False,
                 use_multiscale_gradient_semantic_gate=False):
        super().__init__()
        self.num_levels = len(channels)
        self.use_hierarchical_degradation = bool(
            use_hierarchical_degradation)
        if self.use_hierarchical_degradation and not use_dual_domain_degradation:
            raise ValueError(
                "Hierarchical degradation requires dual-domain degradation")
        self.hierarchical_degradation = (
            self._build_hierarchical_degradation(channels, wavelet)
            if self.use_hierarchical_degradation else None
        )
        self.tifm_blocks = nn.ModuleList([
            TIFMBlock(
                dim=channels[i],
                seg_prior_dim=seg_prior_channels[i],
                det_prior_dim=det_prior_channels[i],
                num_heads=max(1, num_heads // (2 ** (self.num_levels - 1 - i))),
                wavelet=wavelet,
                use_smoke_reliability=use_smoke_reliability,
                use_frequency_router=use_frequency_router,
                use_prior_conditioning=use_prior_conditioning,
                use_dual_domain_degradation=(
                    use_dual_domain_degradation
                    and not self.use_hierarchical_degradation),
                external_reliability=self.use_hierarchical_degradation,
                preserve_high_frequency=preserve_high_frequency,
                residual_reliability_modulation=(
                    residual_reliability_modulation),
                use_multiscale_gradient_semantic_gate=(
                    use_multiscale_gradient_semantic_gate),
            )
            for i in range(self.num_levels)
        ])

    @staticmethod
    def _build_hierarchical_degradation(channels, wavelet):
        # Lazy import avoids a module cycle: hdde reuses the established DWT
        # and DDE implementations from this file.
        from .hdde import HierarchicalDegradationPyramid
        return HierarchicalDegradationPyramid(channels, wavelet=wavelet)

    def enable_debug_capture(self, levels=(0,)):
        """Capture intermediate tensors for selected pyramid levels only."""
        selected = set(levels)
        for index, block in enumerate(self.tifm_blocks):
            block.capture_debug = index in selected
            block.debug_tensors = {}

    def forward(self, vis_features, ir_features, seg_priors=None, det_priors=None,
                return_aux=False):
        """
        Args:
            vis_features: list of 4 feature maps [(B,C0,H0,W0), ...]
            ir_features: list of 4 feature maps
            seg_priors: list of 4 seg prior features or None
            det_priors: list of 4 det prior features or None
        Returns:
            fused_features: list of 4 fused feature maps
        """
        fused_features = []
        aux = {
            'reliability_vis': [],
            'reliability_ir': [],
            'router_alpha': [],
            'degradation_features': [],
        }
        hierarchical_vis = None
        hierarchical_ir = None
        hierarchical_features = None
        if self.hierarchical_degradation is not None:
            hierarchical_vis, hierarchical_ir, hierarchical_features = (
                self.hierarchical_degradation(vis_features, ir_features))
        for i in range(self.num_levels):
            seg_p = seg_priors[i] if seg_priors is not None else None
            det_p = det_priors[i] if det_priors is not None else None
            reliability_override = None
            if hierarchical_vis is not None:
                reliability_override = (
                    hierarchical_vis[i], hierarchical_ir[i])
            block_output = self.tifm_blocks[i](
                vis_features[i], ir_features[i], seg_p, det_p,
                return_aux=return_aux,
                reliability_override=reliability_override,
            )
            if return_aux:
                fused, block_aux = block_output
                for key in aux:
                    if key == 'degradation_features' and \
                            hierarchical_features is not None:
                        aux[key].append(hierarchical_features[i])
                    else:
                        aux[key].append(block_aux[key])
            else:
                fused = block_output
            fused_features.append(fused)
        if return_aux:
            return fused_features, aux
        return fused_features
