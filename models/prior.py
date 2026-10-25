"""
Frozen Task Prior Extractors for TIFDFusion.
- SegFormer: extracts multi-scale semantic features (frozen)
- YOLOv8: extracts multi-scale detection features (frozen)
Both outputs are projected to match fusion encoder channel dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerPriorExtractor(nn.Module):
    """
    Extract multi-scale semantic prior features from a frozen SegFormer.
    Uses HuggingFace transformers SegformerModel to get 4-stage hidden states.
    All SegFormer weights are frozen; only the channel alignment projections are trainable.
    """
    def __init__(self, pretrained_name='./pretrained/segformer-b2',
                 target_channels=(64, 128, 256, 512)):
        super().__init__()
        self.target_channels = target_channels

        try:
            from transformers import SegformerModel
            self.backbone = SegformerModel.from_pretrained(pretrained_name, local_files_only=True)
            # Freeze all parameters
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

            # SegFormer-B2 hidden sizes: [64, 128, 320, 512]
            seg_channels = [64, 128, 320, 512]
            self.available = True
        except Exception as e:
            print(f"Warning: Could not load SegFormer: {e}")
            print("SegFormer prior will output zeros.")
            seg_channels = target_channels
            self.backbone = None
            self.available = False

        # Channel alignment projections (trainable)
        self.align_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(seg_channels[i], target_channels[i], 1, bias=False),
                nn.BatchNorm2d(target_channels[i]),
            )
            for i in range(4)
        ])

    @torch.no_grad()
    def _extract_features(self, x):
        """Extract 4-stage features from SegFormer. Input must be 3-channel RGB."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of 4 tensors
        return list(hidden_states)

    def forward(self, vis, ir):
        """
        Args:
            vis: (B, 3, H, W) visible image
            ir: (B, 1, H, W) infrared image
        Returns:
            priors: list of 4 feature maps [(B, C0, H0, W0), ...]
        """
        if not self.available:
            # Return zero tensors as placeholder
            B = vis.shape[0]
            priors = []
            h, w = vis.shape[-2:]
            for i, c in enumerate(self.target_channels):
                scale = 2 ** (i + 1)
                priors.append(torch.zeros(B, c, h // scale, w // scale, device=vis.device))
            return priors

        # Use visible image for semantic understanding
        with torch.no_grad():
            features = self._extract_features(vis)

        # Align channels (projections are trainable)
        priors = []
        for i in range(4):
            feat = features[i]
            feat = self.align_projs[i](feat)
            priors.append(feat)

        return priors


class YOLOv8PriorExtractor(nn.Module):
    """
    Extract multi-scale detection prior features from a frozen YOLOv8 backbone.
    Uses ultralytics YOLO to get backbone C2/C3/C4/C5 features via forward hooks.
    All YOLO weights are frozen; only channel alignment projections are trainable.
    """
    def __init__(self, pretrained_path='./pretrained/yolov8s.pt',
                 target_channels=(64, 128, 256, 512)):
        super().__init__()
        self.target_channels = target_channels
        self._hook_features = {}

        try:
            from ultralytics import YOLO
            model = YOLO(pretrained_path)
            self.backbone = model.model.model  # nn.Sequential of layers
            # Freeze all parameters
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

            # YOLOv8s backbone layer indices and output channels
            # Layer 2: (B, 64, H/4, W/4)
            # Layer 4: (B, 128, H/8, W/8)
            # Layer 6: (B, 256, H/16, W/16)
            # Layer 8: (B, 512, H/32, W/32)  -- after SPPF
            self.hook_layers = [2, 4, 6, 8]
            yolo_channels = [64, 128, 256, 512]
            self.available = True
        except Exception as e:
            print(f"Warning: Could not load YOLOv8: {e}")
            print("YOLOv8 prior will output zeros.")
            yolo_channels = target_channels
            self.backbone = None
            self.available = False

        # Channel alignment projections (trainable)
        self.align_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(yolo_channels[i], target_channels[i], 1, bias=False),
                nn.BatchNorm2d(target_channels[i]),
            )
            for i in range(4)
        ])

    @torch.no_grad()
    def _extract_features(self, x):
        """Run YOLOv8 backbone and collect intermediate features."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.hook_layers:
                features.append(x.clone())
            if len(features) == 4:
                break

        return features

    def forward(self, vis, ir):
        """
        Args:
            vis: (B, 3, H, W) visible image
            ir: (B, 1, H, W) infrared image
        Returns:
            priors: list of 4 feature maps [(B, C0, H0, W0), ...]
        """
        if not self.available:
            B = vis.shape[0]
            priors = []
            h, w = vis.shape[-2:]
            for i, c in enumerate(self.target_channels):
                scale = 2 ** (i + 2)
                priors.append(torch.zeros(B, c, h // scale, w // scale, device=vis.device))
            return priors

        # Use both vis and ir: concatenate for richer detection cues
        # Convert ir to 3-channel and average with vis for detection
        ir_3ch = ir.repeat(1, 3, 1, 1)
        det_input = 0.5 * vis + 0.5 * ir_3ch

        with torch.no_grad():
            features = self._extract_features(det_input)

        # Align channels
        priors = []
        for i in range(min(4, len(features))):
            feat = features[i]
            feat = self.align_projs[i](feat)
            priors.append(feat)

        # Pad if fewer than 4 levels
        while len(priors) < 4:
            priors.append(torch.zeros_like(priors[-1]))

        return priors


class TaskPriorExtractor(nn.Module):
    """
    Combined task prior extractor: SegFormer + YOLOv8.
    Manages both frozen backbones and their trainable alignment projections.
    """
    def __init__(self, seg_model_name='./pretrained/segformer-b2',
                 det_model_path='./pretrained/yolov8s.pt',
                 target_channels=(64, 128, 256, 512)):
        super().__init__()
        self.seg_extractor = SegFormerPriorExtractor(seg_model_name, target_channels)
        self.det_extractor = YOLOv8PriorExtractor(det_model_path, target_channels)

    def forward(self, vis, ir):
        """
        Args:
            vis: (B, 3, H, W)
            ir: (B, 1, H, W)
        Returns:
            seg_priors: list of 4 segmentation prior features
            det_priors: list of 4 detection prior features
        """
        seg_priors = self.seg_extractor(vis, ir)
        det_priors = self.det_extractor(vis, ir)
        return seg_priors, det_priors

    def train(self, mode=True):
        """Override train to keep frozen backbones in eval mode."""
        super().train(mode)
        # Always keep frozen backbones in eval
        if self.seg_extractor.available and self.seg_extractor.backbone is not None:
            self.seg_extractor.backbone.eval()
        if self.det_extractor.available and self.det_extractor.backbone is not None:
            self.det_extractor.backbone.eval()
        return self
