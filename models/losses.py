"""
Loss functions for TIFDFusion.
- L_pixel: L1 loss
- L_SSIM: Structural Similarity loss (adapted from CDTFusion loss_ssim.py)
- L_gradient: Gradient preservation loss
- L_seg_q: Segmentation quality loss (frozen SegFormer evaluation)
- L_det_q: Detection quality loss (frozen YOLOv8 evaluation)
"""

import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


# ========================= SSIM Loss (adapted from CDTFusion) =========================

def _gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size, channel, device=None):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    if device is not None:
        window = window.to(device)
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    """SSIM loss: 1 - SSIM(pred, target)."""
    def __init__(self, window_size=7):
        super().__init__()
        self.window_size = window_size
        self.window = None
        self.channel = None

    def forward(self, img1, img2):
        channel = img1.size(1)
        if self.window is None or self.channel != channel:
            self.window = _create_window(self.window_size, channel, device=img1.device)
            self.channel = channel
        else:
            self.window = self.window.to(img1.device)
        return 1.0 - _ssim(img1, img2, self.window, self.window_size, channel)


# ========================= Gradient Loss =========================

class GradientLoss(nn.Module):
    """Direction-separated gradient preservation loss using Sobel filters."""
    def __init__(self):
        super().__init__()
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0))

    def _sobel(self, img):
        """Compute directional Sobel gradients (gx, gy) for single-channel input."""
        sobel_x = self.sobel_x.to(img.device)
        sobel_y = self.sobel_y.to(img.device)
        gx = F.conv2d(img, sobel_x, padding=1)
        gy = F.conv2d(img, sobel_y, padding=1)
        return gx, gy

    def forward(self, fused, vis, ir):
        """
        Direction-separated gradient loss: preserve the stronger directional
        gradient from either source, matching TDFusion's formulation.
        Loss = L1(gx_fused, max_abs_gx) + L1(gy_fused, max_abs_gy)
        """
        # Handle channel mismatch: convert to grayscale
        if vis.shape[1] == 3:
            vis_gray = 0.299 * vis[:, 0:1] + 0.587 * vis[:, 1:2] + 0.114 * vis[:, 2:3]
        else:
            vis_gray = vis

        if ir.shape[1] != 1:
            ir_gray = ir[:, 0:1]
        else:
            ir_gray = ir

        if fused.shape[1] == 3:
            fused_gray = 0.299 * fused[:, 0:1] + 0.587 * fused[:, 1:2] + 0.114 * fused[:, 2:3]
        else:
            fused_gray = fused

        # Directional Sobel gradients
        gx_fused, gy_fused = self._sobel(fused_gray)
        gx_vis, gy_vis = self._sobel(vis_gray)
        gx_ir, gy_ir = self._sobel(ir_gray)

        # Max-abs selection per direction (preserves sign/direction)
        gx_max = torch.where(torch.abs(gx_vis) >= torch.abs(gx_ir), gx_vis, gx_ir)
        gy_max = torch.where(torch.abs(gy_vis) >= torch.abs(gy_ir), gy_vis, gy_ir)

        return F.l1_loss(gx_fused, gx_max) + F.l1_loss(gy_fused, gy_max)


# ========================= Laplacian Loss =========================

class LaplacianLoss(nn.Module):
    """Second-order gradient preservation loss using Laplacian filter.
    Encourages fused image to preserve fine textures and edge curvature
    from the source with stronger second-order response.
    Loss = L1(laplacian(fused), max_abs_laplacian(vis, ir))
    """
    def __init__(self):
        super().__init__()
        # Standard 3x3 Laplacian kernel (8-connected)
        laplacian = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.unsqueeze(0).unsqueeze(0))

    def _lap(self, img):
        """Compute Laplacian for single-channel input."""
        kernel = self.laplacian.to(img.device)
        return F.conv2d(img, kernel, padding=1)

    def forward(self, fused, vis, ir):
        # Handle channel mismatch: convert to grayscale
        if vis.shape[1] == 3:
            vis_gray = 0.299 * vis[:, 0:1] + 0.587 * vis[:, 1:2] + 0.114 * vis[:, 2:3]
        else:
            vis_gray = vis

        if ir.shape[1] != 1:
            ir_gray = ir[:, 0:1]
        else:
            ir_gray = ir

        if fused.shape[1] == 3:
            fused_gray = 0.299 * fused[:, 0:1] + 0.587 * fused[:, 1:2] + 0.114 * fused[:, 2:3]
        else:
            fused_gray = fused

        lap_fused = self._lap(fused_gray)
        lap_vis = self._lap(vis_gray)
        lap_ir = self._lap(ir_gray)

        # Max-abs selection: preserve the stronger second-order response
        lap_max = torch.where(torch.abs(lap_vis) >= torch.abs(lap_ir), lap_vis, lap_ir)

        return F.l1_loss(lap_fused, lap_max)


# ========================= Task Quality Losses =========================

class SegQualityLoss(nn.Module):
    """
    Segmentation quality loss: evaluates how well the fused image supports segmentation.
    Uses a frozen SegFormer to predict segmentation on the fused image,
    then computes CE loss against ground truth labels.
    """
    def __init__(self, seg_model_name='./pretrained/segformer-b2',
                 num_classes=9, class_mapping=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_mapping = tuple(class_mapping) if class_mapping is not None else None
        self._warned_class_mismatch = False
        try:
            from transformers import SegformerForSemanticSegmentation
            self.seg_model = SegformerForSemanticSegmentation.from_pretrained(seg_model_name, local_files_only=True)
            for param in self.seg_model.parameters():
                param.requires_grad = False
            self.seg_model.eval()
            self.available = True
        except Exception as e:
            print(f"Warning: SegQualityLoss unavailable: {e}")
            self.seg_model = None
            self.available = False

    def _predict(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        outputs = self.seg_model(pixel_values=x)
        logits = outputs.logits  # (B, num_classes, H/4, W/4)
        return logits

    def forward(self, fused, seg_labels):
        """
        Args:
            fused: (B, C, H, W) fused image
            seg_labels: (B, H, W) ground truth segmentation labels
        Returns:
            loss: scalar
        """
        if not self.available or seg_labels is None:
            return torch.tensor(0.0, device=fused.device, requires_grad=True)

        logits = self._predict(fused)
        # Upsample logits to label size
        logits = F.interpolate(logits, size=seg_labels.shape[-2:], mode='bilinear', align_corners=False)

        # Never truncate Cityscapes logits to MSRS classes: channel indices do
        # not imply semantic correspondence.  Use an explicit audited mapping,
        # or disable this optional CE path while retaining frozen priors.
        if logits.shape[1] != self.num_classes:
            mapping = self.class_mapping
            valid = (mapping is not None and len(mapping) == self.num_classes
                     and min(mapping) >= 0 and max(mapping) < logits.shape[1])
            if valid:
                logits = logits[:, list(mapping)]
            else:
                if not self._warned_class_mismatch:
                    warnings.warn(
                        f"SegQualityLoss disabled: frozen model outputs {logits.shape[1]} "
                        f"classes but labels contain {self.num_classes}; provide an audited "
                        "segformer_class_mapping to enable CE.", RuntimeWarning)
                    self._warned_class_mismatch = True
                return fused.sum() * 0.0

        loss = F.cross_entropy(logits, seg_labels.long(), ignore_index=255)
        return loss


class DetQualityLoss(nn.Module):
    """
    Detection quality loss: evaluates how well the fused image supports detection.
    Uses feature-level comparison: the fused image's features from frozen YOLO backbone
    should be close to the combined vis+ir features.
    This is a perceptual-style loss rather than direct detection loss.
    """
    def __init__(self, det_model_path='./pretrained/yolov8s.pt'):
        super().__init__()
        try:
            from ultralytics import YOLO
            model = YOLO(det_model_path)
            self.backbone = model.model.model
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            self.hook_layers = [2, 4, 6]
            self.available = True
        except Exception as e:
            print(f"Warning: DetQualityLoss unavailable: {e}")
            self.backbone = None
            self.available = False

    def _extract(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.hook_layers:
                features.append(x.clone())
            if len(features) == len(self.hook_layers):
                break
        return features

    def forward(self, fused, vis, ir):
        """
        Perceptual detection loss: features of fused should be close to
        features of (vis+ir)/2 in the frozen YOLO backbone.
        """
        if not self.available:
            return torch.tensor(0.0, device=fused.device, requires_grad=True)

        if fused.shape[1] == 1:
            fused_3ch = fused.repeat(1, 3, 1, 1)
        else:
            fused_3ch = fused

        ir_3ch = ir.repeat(1, 3, 1, 1) if ir.shape[1] == 1 else ir
        ref = 0.5 * vis + 0.5 * ir_3ch

        fused_feats = self._extract(fused_3ch)
        ref_feats = self._extract(ref)

        loss = 0.0
        for ff, rf in zip(fused_feats, ref_feats):
            loss = loss + F.l1_loss(ff, rf)
        return loss / len(fused_feats)


# ========================= SCD Loss =========================

class SCDLoss(nn.Module):
    """
    Sum of Correlations of Differences (SCD) loss.
    SCD = corr(fused - ir, vis) + corr(fused - vis, ir)
    Higher SCD means better fusion. We minimize -SCD.
    """
    def __init__(self):
        super().__init__()

    def _correlation(self, a, b):
        """Pearson correlation between two tensors (per sample, then mean)."""
        a_flat = a.view(a.size(0), -1)
        b_flat = b.view(b.size(0), -1)
        a_mean = a_flat.mean(dim=1, keepdim=True)
        b_mean = b_flat.mean(dim=1, keepdim=True)
        a_centered = a_flat - a_mean
        b_centered = b_flat - b_mean
        numerator = (a_centered * b_centered).sum(dim=1)
        denominator = torch.sqrt((a_centered ** 2).sum(dim=1) * (b_centered ** 2).sum(dim=1) + 1e-8)
        return (numerator / denominator).mean()

    def forward(self, fused, vis_gray, ir):
        """Returns -SCD (to minimize = maximize SCD)."""
        r1 = self._correlation(fused - ir, vis_gray)
        r2 = self._correlation(fused - vis_gray, ir)
        scd = r1 + r2
        return -scd  # minimize negative SCD


# ========================= Combined Loss =========================

class TIFDLoss(nn.Module):
    """
    Combined loss for TIFDFusion with phase-aware weighting.
    """
    def __init__(self, weights=None, seg_model_name=None, det_model_path=None, num_classes=9,
                 pixel_adaptive_temperature=5.0, seg_class_mapping=None):
        super().__init__()
        self.pixel_loss = nn.L1Loss(reduction='none')
        self.ssim_loss = SSIMLoss(window_size=7)
        self.grad_loss = GradientLoss()
        self.lap_loss = LaplacianLoss()

        # Adaptive pixel loss: Sobel filters for gradient-based source weighting
        self.pixel_adaptive_temperature = pixel_adaptive_temperature
        _sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        _sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('_pixel_sobel_x', _sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('_pixel_sobel_y', _sobel_y.unsqueeze(0).unsqueeze(0))

        # Task quality losses (lazy init)
        self.seg_quality_loss = None
        self.det_quality_loss = None
        if seg_model_name:
            self.seg_quality_loss = SegQualityLoss(
                seg_model_name, num_classes, class_mapping=seg_class_mapping)
        if det_model_path:
            self.det_quality_loss = DetQualityLoss(det_model_path)

        # SCD loss
        self.scd_loss = SCDLoss()

        # Default weights
        self.weights = weights or {
            'pixel': 1.0,
            'ssim': 1.0,
            'gradient': 1.0,
            'laplacian': 0.0,
            'scd': 0.0,
            'seg_quality': 0.0,
            'det_quality': 0.0,
            'smoke_consistency': 0.0,
            'smoke_edge_consistency': 0.0,
            'reliability': 0.0,
        }

    def update_weights(self, new_weights):
        """Update loss weights (for phase transitions)."""
        self.weights.update(new_weights)

    def forward(self, fused, vis, ir, seg_labels=None, clean_vis=None,
                transmission=None, smoke_mask=None, reliability_maps=None,
                clean_fused=None):
        """
        Args:
            fused: (B, C, H, W) fused image
            vis: (B, 3, H, W) visible image
            ir: (B, 1, H, W) infrared image
            seg_labels: (B, H, W) segmentation labels or None
            clean_vis: clean VI reference paired with a possibly smoky input
            transmission: synthetic VI transmission map, used only as training supervision
            smoke_mask: 1 - transmission
            reliability_maps: predicted VI reliability maps from TIFM levels
            clean_fused: detached fusion output for the paired clean VI/IR input
        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses
        """
        loss_dict = {}

        # Synthetic smoke is an input corruption, not a desired fusion target.
        # When available, the paired clean VI therefore supervises all standard
        # fusion losses.  Real/inference data remain backward compatible.
        vis_reference = clean_vis if clean_vis is not None else vis
        if vis_reference.shape[1] == 3:
            vis_gray = (0.299 * vis_reference[:, 0:1] +
                        0.587 * vis_reference[:, 1:2] +
                        0.114 * vis_reference[:, 2:3])
        else:
            vis_gray = vis_reference
        fused_cmp = fused if fused.shape[1] == 1 else \
            (0.299 * fused[:, 0:1] + 0.587 * fused[:, 1:2] + 0.114 * fused[:, 2:3])

        # Adaptive-weight dual-source pixel loss
        # Compute per-image gradient magnitude for vis and ir
        _sx = self._pixel_sobel_x.to(fused.device)
        _sy = self._pixel_sobel_y.to(fused.device)
        grad_vis = torch.sqrt(
            F.conv2d(vis_gray, _sx, padding=1).pow(2) +
            F.conv2d(vis_gray, _sy, padding=1).pow(2) + 1e-6
        ).mean(dim=[1, 2, 3])  # (B,)
        grad_ir = torch.sqrt(
            F.conv2d(ir, _sx, padding=1).pow(2) +
            F.conv2d(ir, _sy, padding=1).pow(2) + 1e-6
        ).mean(dim=[1, 2, 3])  # (B,)
        # Softmax weighting: higher gradient source gets more weight
        grad_stack = torch.stack([grad_vis, grad_ir], dim=1)  # (B, 2)
        w = F.softmax(grad_stack / self.pixel_adaptive_temperature, dim=1)  # (B, 2)
        pixel_loss_vis = self.pixel_loss(fused_cmp, vis_gray).mean(dim=[1, 2, 3])  # (B,)
        pixel_loss_ir = self.pixel_loss(fused_cmp, ir).mean(dim=[1, 2, 3])  # (B,)
        loss_dict['pixel'] = (w[:, 0] * pixel_loss_vis + w[:, 1] * pixel_loss_ir).mean()

        # Dual-source SSIM loss: structural similarity to both sources
        loss_dict['ssim'] = 0.5 * (self.ssim_loss(fused_cmp, vis_gray) +
                                    self.ssim_loss(fused_cmp, ir))

        # Gradient loss: fused gradient >= max(vis_grad, ir_grad)
        loss_dict['gradient'] = self.grad_loss(fused, vis_reference, ir)

        # Laplacian loss
        if self.weights.get('laplacian', 0) > 0:
            loss_dict['laplacian'] = self.lap_loss(fused, vis_reference, ir)
        else:
            loss_dict['laplacian'] = torch.tensor(0.0, device=fused.device)

        # SCD loss
        if self.weights.get('scd', 0) > 0:
            loss_dict['scd'] = self.scd_loss(fused_cmp, vis_gray, ir)
        else:
            loss_dict['scd'] = torch.tensor(0.0, device=fused.device)

        # Seg quality loss
        if self.weights.get('seg_quality', 0) > 0 and self.seg_quality_loss is not None:
            loss_dict['seg_quality'] = self.seg_quality_loss(fused, seg_labels)
        else:
            loss_dict['seg_quality'] = torch.tensor(0.0, device=fused.device)

        # Det quality loss
        if self.weights.get('det_quality', 0) > 0 and self.det_quality_loss is not None:
            loss_dict['det_quality'] = self.det_quality_loss(
                fused, vis_reference, ir)
        else:
            loss_dict['det_quality'] = torch.tensor(0.0, device=fused.device)

        # Paired clean-view consistency.  The clean branch is detached by the
        # caller, so only the smoky-input branch is optimized.  A small global
        # floor also discourages smoke from perturbing nominally clear regions.
        if (self.weights.get('smoke_consistency', 0) > 0 and
                clean_fused is not None and smoke_mask is not None):
            clean_cmp = clean_fused if clean_fused.shape[1] == 1 else \
                (0.299 * clean_fused[:, 0:1] +
                 0.587 * clean_fused[:, 1:2] +
                 0.114 * clean_fused[:, 2:3])
            mask = F.interpolate(
                smoke_mask, size=fused_cmp.shape[-2:], mode='bilinear',
                align_corners=False).clamp(0.0, 1.0)
            active = (mask.flatten(1).amax(dim=1) > 1e-6).to(fused.dtype)
            spatial_weight = 0.25 + 0.75 * mask
            per_sample = (
                (torch.abs(fused_cmp - clean_cmp.detach()) * spatial_weight)
                .flatten(1).sum(dim=1) /
                spatial_weight.flatten(1).sum(dim=1).clamp_min(1e-6)
            )
            loss_dict['smoke_consistency'] = (
                (per_sample * active).sum() / active.sum().clamp_min(1.0)
            )
        else:
            loss_dict['smoke_consistency'] = fused.sum() * 0.0

        # Smoke-aware boundary consistency.  Unlike the source Laplacian
        # objective, this compares the smoky-input output with the detached
        # clean-teacher output, so synthetic smoke contours are never treated
        # as details that should be preserved.  The same smoke weighting as
        # the intensity consistency term focuses supervision on corrupted
        # regions while retaining a small global stability floor.
        if (self.weights.get('smoke_edge_consistency', 0) > 0 and
                clean_fused is not None and smoke_mask is not None):
            clean_cmp = clean_fused if clean_fused.shape[1] == 1 else \
                (0.299 * clean_fused[:, 0:1] +
                 0.587 * clean_fused[:, 1:2] +
                 0.114 * clean_fused[:, 2:3])
            mask = F.interpolate(
                smoke_mask, size=fused_cmp.shape[-2:], mode='bilinear',
                align_corners=False).clamp(0.0, 1.0)
            active = (mask.flatten(1).amax(dim=1) > 1e-6).to(fused.dtype)
            spatial_weight = 0.25 + 0.75 * mask
            smoky_edges = self.lap_loss._lap(fused_cmp)
            clean_edges = self.lap_loss._lap(clean_cmp.detach())
            per_sample = (
                (torch.abs(smoky_edges - clean_edges) * spatial_weight)
                .flatten(1).sum(dim=1) /
                spatial_weight.flatten(1).sum(dim=1).clamp_min(1e-6)
            )
            loss_dict['smoke_edge_consistency'] = (
                (per_sample * active).sum() / active.sum().clamp_min(1.0)
            )
        else:
            loss_dict['smoke_edge_consistency'] = fused.sum() * 0.0

        # Reliability supervision is available only for synthetic training.
        # Treat IR reliability as one and VI reliability as transmission, then
        # normalize the pair to match the estimator's two-way softmax:
        # target_w_vis = t / (1 + t).  Clean pixels therefore target 0.5,
        # while increasingly opaque smoke progressively suppresses VI.
        valid_maps = [value for value in (reliability_maps or [])
                      if value is not None]
        if (self.weights.get('reliability', 0) > 0 and
                transmission is not None and valid_maps):
            reliability_losses = []
            for predicted_vis in valid_maps:
                local_t = F.interpolate(
                    transmission, size=predicted_vis.shape[-2:],
                    mode='bilinear', align_corners=False).clamp(0.0, 1.0)
                target_vis = local_t / (1.0 + local_t)
                local_smoke = 1.0 - local_t
                local_weight = 0.25 + 0.75 * local_smoke
                bce = F.binary_cross_entropy(
                    predicted_vis.clamp(1e-6, 1.0 - 1e-6),
                    target_vis,
                    reduction='none',
                )
                reliability_losses.append(
                    (bce * local_weight).sum() /
                    local_weight.sum().clamp_min(1e-6)
                )
            loss_dict['reliability'] = torch.stack(reliability_losses).mean()
        else:
            loss_dict['reliability'] = fused.sum() * 0.0

        # Weighted sum (only tensor losses, skip tracking keys)
        total_loss = sum(self.weights.get(k, 0) * v
                         for k, v in loss_dict.items()
                         if isinstance(v, torch.Tensor))
        loss_dict['total'] = total_loss

        return total_loss, loss_dict
