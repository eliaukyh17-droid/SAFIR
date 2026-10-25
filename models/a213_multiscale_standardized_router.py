"""A2.13 frozen multi-scale statistical smoke router.

The A2.11 router backbone and local gates remain unchanged.  This adapter
summarizes its three feature maps together with the fixed seven-channel input
evidence, standardizes those summaries with training-only moments, and learns
one conservative binary smoke score.  Rejected samples still use the exact
canonical A2.11 path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SUMMARY_NAMES = ("mean", "std", "maximum", "top_tail_mean")
TAIL_DIVISOR = 16
STANDARD_DEVIATION_FLOOR = 1e-6
STANDARDIZED_CLIP = 8.0


def _channel_statistics(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError("A2.13 statistics require a BCHW tensor")
    flattened = value.flatten(2)
    tail_count = max(1, flattened.shape[-1] // TAIL_DIVISOR)
    top_tail = flattened.topk(tail_count, dim=-1, sorted=False).values.mean(dim=-1)
    return torch.cat(
        (
            flattened.mean(dim=-1),
            flattened.std(dim=-1, unbiased=False),
            flattened.amax(dim=-1),
            top_tail,
        ),
        dim=1,
    )


def statistical_feature_dimension(
    router_widths: Sequence[int] = (16, 24, 32),
    evidence_channels: int = 7,
) -> int:
    widths = tuple(int(value) for value in router_widths)
    if len(widths) != 3 or min(widths) <= 0 or int(evidence_channels) <= 0:
        raise ValueError("A2.13 feature dimensions are invalid")
    return len(SUMMARY_NAMES) * (sum(widths) + int(evidence_channels))


def multiscale_statistical_features(
    source_router: nn.Module,
    router_output: dict,
    visible: torch.Tensor,
    infrared: torch.Tensor,
) -> torch.Tensor:
    features = tuple(router_output["features"])
    if len(features) != 3:
        raise ValueError("A2.13 requires exactly three frozen router stages")
    evidence = source_router.evidence(visible, infrared)
    summary = torch.cat(
        tuple(_channel_statistics(value) for value in (*features, evidence)),
        dim=1,
    )
    expected = statistical_feature_dimension(
        tuple(int(value.shape[1]) for value in features),
        int(evidence.shape[1]),
    )
    if summary.shape != (visible.shape[0], expected):
        raise RuntimeError("A2.13 statistical feature shape changed")
    return summary


class FrozenMultiScaleStandardizedRouter(nn.Module):
    """Preserve an A2.11 router while replacing only its global decision."""

    def __init__(self, source_router: nn.Module):
        super().__init__()
        self.source_router = source_router
        widths = tuple(int(value) for value in source_router.widths)
        dimension = statistical_feature_dimension(widths)
        self.feature_dimension = dimension
        self.register_buffer("standardizer_mean", torch.zeros(dimension))
        self.register_buffer("standardizer_scale", torch.ones(dimension))
        self.register_buffer("standardizer_fitted", torch.tensor(False))
        self.score_head = nn.Linear(dimension, 1)
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    @torch.no_grad()
    def set_standardizer(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        mean = torch.as_tensor(
            mean, device=self.standardizer_mean.device,
            dtype=self.standardizer_mean.dtype,
        )
        scale = torch.as_tensor(
            scale, device=self.standardizer_scale.device,
            dtype=self.standardizer_scale.dtype,
        )
        expected = (self.feature_dimension,)
        if mean.shape != expected or scale.shape != expected:
            raise ValueError("A2.13 standardizer shape changed")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all()):
            raise ValueError("A2.13 standardizer contains non-finite values")
        if bool((scale < 0).any()):
            raise ValueError("A2.13 standardizer scale must be non-negative")
        self.standardizer_mean.copy_(mean)
        self.standardizer_scale.copy_(scale.clamp_min(STANDARD_DEVIATION_FLOOR))
        self.standardizer_fitted.fill_(True)

    def standardized_features(
        self,
        router_output: dict,
        visible: torch.Tensor,
        infrared: torch.Tensor,
    ) -> torch.Tensor:
        summary = multiscale_statistical_features(
            self.source_router, router_output, visible, infrared,
        )
        standardized = (
            summary - self.standardizer_mean
        ) / self.standardizer_scale
        return standardized.clamp(-STANDARDIZED_CLIP, STANDARDIZED_CLIP)

    def forward(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        source = self.source_router(visible, infrared)
        statistics = self.standardized_features(
            source, visible, infrared,
        )
        smoke_score = self.score_head(statistics).squeeze(1)
        class_logits = torch.stack((-smoke_score, smoke_score, torch.zeros_like(smoke_score)), dim=1)
        probabilities = torch.softmax(class_logits, dim=1)
        entropy = -(
            probabilities.clamp_min(1e-8)
            * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1) / math.log(3.0)
        output = dict(source)
        output.update({
            "statistical_features": statistics,
            "smoke_score": smoke_score,
            "class_logits": class_logits,
            "class_probabilities": probabilities,
            "p_clean": probabilities[:, 0],
            "p_smoke": probabilities[:, 1],
            "p_unknown": probabilities[:, 2],
            "uncertainty": entropy,
        })
        return output


def install_a213_router(model: nn.Module) -> FrozenMultiScaleStandardizedRouter:
    if isinstance(model.router, FrozenMultiScaleStandardizedRouter):
        raise RuntimeError("A2.13 router is already installed")
    source_router = model.router
    reference = next(source_router.parameters())
    wrapper = FrozenMultiScaleStandardizedRouter(source_router).to(
        device=reference.device,
        dtype=reference.dtype,
    )
    model.router = wrapper
    return wrapper


def freeze_except_score_head(model: nn.Module) -> tuple[nn.Parameter, ...]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.router.score_head.parameters():
        parameter.requires_grad_(True)
    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    expected = tuple(model.router.score_head.parameters())
    if {id(value) for value in trainable} != {id(value) for value in expected}:
        raise RuntimeError("A2.13 trainable set escaped the score head")
    return trainable


def paired_binary_ranking_loss(
    smoke_score: torch.Tensor,
    clean_score: torch.Tensor,
    *,
    binary_weight: float = 1.0,
    anchor_weight: float = 1.0,
    pair_weight: float = 1.0,
    anchor_margin: float = 4.0,
    pair_margin: float = 8.0,
) -> dict[str, torch.Tensor]:
    if smoke_score.shape != clean_score.shape or smoke_score.ndim != 1:
        raise ValueError("A2.13 paired scores must be equal one-dimensional tensors")
    binary = 0.5 * (
        F.binary_cross_entropy_with_logits(smoke_score, torch.ones_like(smoke_score))
        + F.binary_cross_entropy_with_logits(clean_score, torch.zeros_like(clean_score))
    )
    anchor = 0.5 * (
        F.softplus(float(anchor_margin) - smoke_score).mean()
        + F.softplus(float(anchor_margin) + clean_score).mean()
    )
    pair = F.softplus(
        float(pair_margin) - (smoke_score - clean_score)
    ).mean()
    total = (
        float(binary_weight) * binary
        + float(anchor_weight) * anchor
        + float(pair_weight) * pair
    )
    return {"binary": binary, "anchor": anchor, "pair": pair, "total": total}
