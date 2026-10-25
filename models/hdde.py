"""Target-preserving hierarchical degradation guidance for SAFIR-Fuse."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tifm import DualDomainDegradationEncoder, WaveletDecompose


class HierarchicalDegradationPyramid(nn.Module):
    """Propagate reliable deep degradation evidence toward shallow levels.

    Only the two deepest pyramid levels own DDE encoders. Their reliability
    logits are combined with bounded residual guidance and then propagated to
    shallower resolutions with deliberately weak strength. This follows the
    top-down lateral pattern used by feature pyramids while avoiding a second
    independently learned shallow degradation estimator, which can confuse
    smoke texture with target edges.
    """

    def __init__(self, channels, source_levels=(2, 3), wavelet="haar"):
        super().__init__()
        if len(channels) != 4 or tuple(source_levels) != (2, 3):
            raise ValueError(
                "HierarchicalDegradationPyramid currently requires four "
                "levels with source_levels=(2, 3)"
            )

        self.num_levels = len(channels)
        self.source_levels = tuple(source_levels)
        self.decompose = nn.ModuleList([
            WaveletDecompose(wavelet) for _ in channels
        ])
        self.encoders = nn.ModuleDict({
            str(level): DualDomainDegradationEncoder(channels[level])
            for level in self.source_levels
        })

        # All strengths are bounded by 0.5. The negative initialization gives
        # a conservative ~0.125 residual contribution at training start.
        initial_logit = -1.1
        self.deep_residual_logit = nn.Parameter(torch.tensor(initial_logit))
        self.shallow_strength_logits = nn.ParameterList([
            nn.Parameter(torch.tensor(initial_logit))
            for _ in range(self.source_levels[0])
        ])

    @staticmethod
    def _probability_logit(probability):
        probability = probability.clamp(1e-4, 1.0 - 1e-4)
        return torch.log(probability) - torch.log1p(-probability)

    @staticmethod
    def _normalized_weights(logit):
        w_vis = torch.sigmoid(logit)
        return w_vis, 1.0 - w_vis

    def forward(self, vis_features, ir_features):
        reliability_vis = [None] * self.num_levels
        reliability_ir = [None] * self.num_levels
        degradation_features = [None] * self.num_levels

        local_logits = {}
        for level in self.source_levels:
            vis_ll, vis_hf = self.decompose[level](vis_features[level])
            ir_ll, ir_hf = self.decompose[level](ir_features[level])
            w_vis, _, degradation = self.encoders[str(level)](
                vis_ll, ir_ll, vis_hf, ir_hf
            )
            local_logits[level] = self._probability_logit(w_vis)
            degradation_features[level] = degradation

        middle_level, deepest_level = self.source_levels
        deepest_logit = local_logits[deepest_level]
        reliability_vis[deepest_level], reliability_ir[deepest_level] = (
            self._normalized_weights(deepest_logit)
        )

        deep_guidance = F.interpolate(
            deepest_logit,
            size=local_logits[middle_level].shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).tanh()
        deep_strength = 0.5 * torch.sigmoid(self.deep_residual_logit)
        middle_logit = local_logits[middle_level] + deep_strength * deep_guidance
        reliability_vis[middle_level], reliability_ir[middle_level] = (
            self._normalized_weights(middle_logit)
        )

        # Shallow estimates are weak top-down projections of deep evidence.
        # They start close to the neutral clean value of 0.5.
        for level in range(middle_level - 1, -1, -1):
            target_size = (
                vis_features[level].shape[-2] // 2,
                vis_features[level].shape[-1] // 2,
            )
            propagated = F.interpolate(
                middle_logit,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            strength = 0.5 * torch.sigmoid(
                self.shallow_strength_logits[level]
            )
            shallow_logit = strength * propagated
            reliability_vis[level], reliability_ir[level] = (
                self._normalized_weights(shallow_logit)
            )

        return reliability_vis, reliability_ir, degradation_features
