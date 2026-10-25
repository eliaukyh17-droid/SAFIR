"""
TIFDFusion: Task-Interactive Frequency-Division Fusion Network
Main model assembly that connects all components:
  - DualStreamEncoder (CNN + Mamba hybrid)
  - TIFM (Task-Interactive Fusion Module)
  - FusionDecoder
  - TaskPriorExtractor (frozen SegFormer + YOLOv8)
"""

import torch
import torch.nn as nn

from .encoder import DualStreamEncoder
from .tifm import TIFM
from .decoder import FusionDecoder
from .prior import TaskPriorExtractor


class TIFDFusion(nn.Module):
    """
    Task-Interactive Frequency-Division Fusion Network.

    Training phases:
      Phase 1: Encoder + Decoder only (no task priors)
      Phase 2: + TIFM with segmentation prior
      Phase 3: + TIFM with segmentation + detection priors

    Inference: full pipeline with all priors enabled.
    """
    def __init__(self,
                 vis_in_channels=3,
                 ir_in_channels=1,
                 channels=(64, 128, 256, 512),
                 cnn_depths=(2, 2),
                 mamba_depths=(2, 2),
                 d_state=16,
                 drop_path_rate=0.1,
                 out_channels=1,
                 num_heads=8,
                 wavelet='haar',
                 seg_model_name=None,
                 det_model_path=None,
                 use_skip=True,
                 decoder_norm='bn',
                 no_tifm=False,
                 swap_priors=False,
                 disable_seg_prior=False,
                 disable_det_prior=False,
                 use_smoke_reliability=False,
                 use_frequency_router=False,
                 use_prior_conditioning=False,
                 use_dual_domain_degradation=False,
                 use_hierarchical_degradation=False,
                 preserve_high_frequency=False,
                 residual_reliability_modulation=False,
                 use_multiscale_gradient_semantic_gate=False):
        super().__init__()

        # Dual-stream encoder
        self.encoder = DualStreamEncoder(
            vis_in_channels=vis_in_channels,
            ir_in_channels=ir_in_channels,
            channels=channels,
            cnn_depths=cnn_depths,
            mamba_depths=mamba_depths,
            d_state=d_state,
            drop_path_rate=drop_path_rate,
        )

        # Task-Interactive Fusion Module
        self.tifm = TIFM(
            channels=channels,
            seg_prior_channels=channels,
            det_prior_channels=channels,
            num_heads=num_heads,
            wavelet=wavelet,
            use_smoke_reliability=use_smoke_reliability,
            use_frequency_router=use_frequency_router,
            use_prior_conditioning=use_prior_conditioning,
            use_dual_domain_degradation=use_dual_domain_degradation,
            use_hierarchical_degradation=use_hierarchical_degradation,
            preserve_high_frequency=preserve_high_frequency,
            residual_reliability_modulation=(
                residual_reliability_modulation),
            use_multiscale_gradient_semantic_gate=(
                use_multiscale_gradient_semantic_gate),
        )

        # Fusion decoder
        self.decoder = FusionDecoder(
            channels=channels,
            out_channels=out_channels,
            use_skip=use_skip,
            decoder_norm=decoder_norm,
        )

        # Task prior extractor (frozen backbones)
        self.prior_extractor = None
        if seg_model_name or det_model_path:
            self.prior_extractor = TaskPriorExtractor(
                seg_model_name=seg_model_name or '',
                det_model_path=det_model_path or '',
                target_channels=channels,
            )

        # ---- Ablation flags ----
        self.no_tifm = no_tifm
        self.swap_priors = swap_priors
        self.disable_seg_prior = disable_seg_prior
        self.disable_det_prior = disable_det_prior
        # w/o-TIFM ablation: replace frequency-division fusion with a plain conv fusion
        if no_tifm:
            self.simple_fuse = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c * 2, c, 3, padding=1), nn.GELU(),
                    nn.Conv2d(c, c, 3, padding=1),
                ) for c in channels
            ])

        # Phase control
        self._phase = 1
        self._use_seg_prior = False
        self._use_det_prior = False

    def set_phase(self, phase):
        """
        Set training phase:
          1: Encoder + Decoder only (no TIFM priors)
          2: + TIFM with segmentation prior
          3: + TIFM with both segmentation and detection priors
        """
        self._phase = phase
        self._use_seg_prior = (phase >= 2) and not self.disable_seg_prior
        self._use_det_prior = (phase >= 3) and not self.disable_det_prior
        print(f"[TIFDFusion] Phase {phase}: seg_prior={self._use_seg_prior}, det_prior={self._use_det_prior}"
              f"{' | no_tifm' if self.no_tifm else ''}{' | swap_priors' if self.swap_priors else ''}")

    def freeze_encoder(self, freeze=True):
        """Freeze/unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def get_trainable_params(self):
        """Get parameters that should be trained in current phase."""
        params = []
        # Encoder and decoder are always trainable
        params.extend(self.encoder.parameters())
        params.extend(self.decoder.parameters())
        # TIFM is always trainable (priors are controlled by phase)
        params.extend(self.tifm.parameters())
        # Prior alignment projections are trainable
        if self.prior_extractor is not None:
            for proj in self.prior_extractor.seg_extractor.align_projs:
                params.extend(proj.parameters())
            for proj in self.prior_extractor.det_extractor.align_projs:
                params.extend(proj.parameters())
        return params

    def forward(self, vis, ir, return_aux=False):
        """
        Args:
            vis: (B, 3, H, W) visible image
            ir: (B, 1, H, W) infrared image
        Returns:
            fused: (B, out_channels, H, W) fused image
        """
        # 1. Dual-stream encoding
        vis_features, ir_features = self.encoder(vis, ir)

        # 2. Extract task priors (if available and phase allows)
        seg_priors = None
        det_priors = None
        if self.prior_extractor is not None:
            if self._use_seg_prior or self._use_det_prior:
                seg_p, det_p = self.prior_extractor(vis, ir)
                if self._use_seg_prior:
                    seg_priors = seg_p
                if self._use_det_prior:
                    det_priors = det_p

        # 3. Task-interactive fusion (ablation-aware)
        if self.no_tifm:
            # w/o-TIFM: plain conv fusion, no frequency division, no prior injection
            fused_features = [
                self.simple_fuse[i](torch.cat([vis_features[i], ir_features[i]], dim=1))
                for i in range(len(vis_features))
            ]
            aux = {
                'reliability_vis': [],
                'reliability_ir': [],
                'router_alpha': [],
                'degradation_features': [],
            }
        else:
            if self.swap_priors:
                # swap-prior ablation: seg prior guides high-freq, det prior guides low-freq
                seg_priors, det_priors = det_priors, seg_priors
            tifm_output = self.tifm(
                vis_features, ir_features, seg_priors, det_priors,
                return_aux=return_aux,
            )
            if return_aux:
                fused_features, aux = tifm_output
            else:
                fused_features = tifm_output

        # 4. Decode to fusion image
        fused = self.decoder(fused_features)

        if return_aux:
            return fused, aux
        return fused


def build_model(cfg):
    """Build TIFDFusion model from config dict."""
    model_cfg = cfg.get('model', cfg)
    enc_cfg = model_cfg.get('encoder', {})
    tifm_cfg = model_cfg.get('tifm', {})
    dec_cfg = model_cfg.get('decoder', {})
    prior_cfg = model_cfg.get('prior', {})
    abl_cfg = cfg.get('ablation', {})  # ablation toggles (empty for the main model)

    model = TIFDFusion(
        vis_in_channels=enc_cfg.get('vis_in_channels', 3),
        ir_in_channels=enc_cfg.get('ir_in_channels', 1),
        channels=tuple(enc_cfg.get('channels', [64, 128, 256, 512])),
        cnn_depths=tuple(enc_cfg.get('cnn_depths', [2, 2])),
        mamba_depths=tuple(enc_cfg.get('mamba_depths', [2, 2])),
        d_state=enc_cfg.get('d_state', 16),
        drop_path_rate=enc_cfg.get('drop_path_rate', 0.1),
        out_channels=dec_cfg.get('out_channels', 1),
        num_heads=tifm_cfg.get('num_heads', 8),
        wavelet=tifm_cfg.get('wavelet', 'haar'),
        seg_model_name=prior_cfg.get('seg_model', None),
        det_model_path=prior_cfg.get('det_model', None),
        use_skip=dec_cfg.get('use_skip', True),
        decoder_norm=dec_cfg.get('decoder_norm', 'bn'),
        no_tifm=abl_cfg.get('no_tifm', False),
        swap_priors=abl_cfg.get('swap_priors', False),
        disable_seg_prior=abl_cfg.get('disable_seg_prior', False),
        disable_det_prior=abl_cfg.get('disable_det_prior', False),
        use_smoke_reliability=tifm_cfg.get('use_smoke_reliability', False),
        use_frequency_router=tifm_cfg.get('use_frequency_router', False),
        use_prior_conditioning=tifm_cfg.get('use_prior_conditioning', False),
        use_dual_domain_degradation=tifm_cfg.get(
            'use_dual_domain_degradation', False),
        use_hierarchical_degradation=tifm_cfg.get(
            'use_hierarchical_degradation', False),
        preserve_high_frequency=tifm_cfg.get(
            'preserve_high_frequency', False),
        residual_reliability_modulation=tifm_cfg.get(
            'residual_reliability_modulation', False),
        use_multiscale_gradient_semantic_gate=tifm_cfg.get(
            'use_multiscale_gradient_semantic_gate', False),
    )
    return model
