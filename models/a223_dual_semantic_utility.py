"""Dual-semantic localization and gain-weighted utility fusion for A2.23."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.a219_dual_support_benefit_rescue import boundary_map


def depthwise_refine(channels: int, groups: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        nn.Conv2d(channels, channels, 1, bias=False),
        nn.GroupNorm(groups, channels),
        nn.SiLU(inplace=True),
    )


class DualSemanticUtilityPredictor(nn.Module):
    """Share spatial evidence while keeping localization and utility distinct."""

    def __init__(
        self,
        *,
        feature_channels: Sequence[int] = (16, 24, 32),
        lateral_channels: int = 12,
        evidence_channels: int = 7,
        fine_channels: int = 16,
        fusion_channels: int = 32,
        localization_logit_delta_cap: float = 4.0,
    ):
        super().__init__()
        if tuple(int(value) for value in feature_channels) != (16, 24, 32):
            raise ValueError("A2.23 frozen feature channels changed")
        if (
            int(lateral_channels),
            int(evidence_channels),
            int(fine_channels),
            int(fusion_channels),
        ) != (12, 7, 16, 32):
            raise ValueError("A2.23 predictor architecture changed")
        if float(localization_logit_delta_cap) != 4.0:
            raise ValueError("A2.23 localization logit cap changed")
        self.localization_logit_delta_cap = float(localization_logit_delta_cap)
        self.laterals = nn.ModuleList(
            nn.Conv2d(int(channels), 12, 1) for channels in feature_channels
        )
        self.pyramid_refine = nn.ModuleList(
            depthwise_refine(12, 3) for _ in feature_channels
        )
        self.fine_path = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(53, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )
        self.fusion_refine = depthwise_refine(32, 8)
        self.localization_delta_head = nn.Conv2d(32, 1, 1)
        self.utility_head = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(self.localization_delta_head.weight)
        nn.init.zeros_(self.localization_delta_head.bias)
        nn.init.zeros_(self.utility_head.weight)
        nn.init.zeros_(self.utility_head.bias)

    def forward(
        self,
        evidence: torch.Tensor,
        features: Sequence[torch.Tensor],
        frozen_localization_logits: torch.Tensor,
        frozen_localization: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if evidence.ndim != 4 or evidence.shape[1] != 7:
            raise ValueError("A2.23 evidence must be Bx7xHxW")
        if len(features) != 3:
            raise ValueError("A2.23 requires three frozen feature maps")
        if frozen_localization.shape != frozen_localization_logits.shape:
            raise ValueError("A2.23 frozen localization/logit shape mismatch")
        if frozen_localization.shape[-2:] != evidence.shape[-2:]:
            raise ValueError("A2.23 frozen localization resolution changed")

        lateral = [layer(value) for layer, value in zip(self.laterals, features)]
        p3 = self.pyramid_refine[2](lateral[2])
        p2 = self.pyramid_refine[1](
            lateral[1]
            + F.interpolate(
                p3, size=lateral[1].shape[-2:], mode="bilinear", align_corners=False
            )
        )
        p1 = self.pyramid_refine[0](
            lateral[0]
            + F.interpolate(
                p2, size=lateral[0].shape[-2:], mode="bilinear", align_corners=False
            )
        )
        target_size = evidence.shape[-2:]
        fine = self.fine_path(torch.cat((evidence, frozen_localization), dim=1))
        pyramid = [
            F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
            for value in (p1, p2, p3)
        ]
        fused = self.fusion(
            torch.cat((fine, *pyramid, frozen_localization), dim=1)
        )
        fused = fused + self.fusion_refine(fused)
        localization_delta = self.localization_logit_delta_cap * torch.tanh(
            self.localization_delta_head(fused)
        )
        localization_logits = frozen_localization_logits + localization_delta
        utility_logits = self.utility_head(fused)
        return {
            "localization_logit_delta": localization_delta,
            "localization_logits": localization_logits,
            "localization_support": torch.sigmoid(localization_logits),
            "utility_logits": utility_logits,
            "utility_multiplier": torch.sigmoid(utility_logits),
        }


class A223DualSemanticUtilityFusion(nn.Module):
    """Frozen A2.19 route/residual with separate localization and utility."""

    VALID_ROUTE_MODES = frozenset(("canonical", "smoke", "auto"))

    def __init__(self, source_model: nn.Module, predictor: nn.Module | None = None):
        super().__init__()
        self.source_model = source_model
        self.predictor = predictor or DualSemanticUtilityPredictor()
        self.freeze_source()

    def freeze_source(self) -> None:
        self.source_model.eval()
        for parameter in self.source_model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.source_model.eval()
        self.predictor.train(mode)
        return self

    def semantic_prediction(
        self, visible: torch.Tensor, infrared: torch.Tensor
    ) -> dict:
        source_prediction = self.source_model.controller_prediction(visible, infrared)
        frozen = source_prediction["frozen_auxiliary"]
        router = self.source_model.source_model.source_model.source_model.router
        evidence = router.source_router.evidence(visible, infrared)
        prediction = self.predictor(
            evidence,
            frozen["features"],
            frozen["support_logits"],
            frozen["support"],
        )
        return {
            **prediction,
            "frozen_localization_logits": frozen["support_logits"],
            "frozen_localization_support": frozen["support"],
            "old_smoke_decision": self.source_model._old_decision(frozen),
            "rescue_probability": source_prediction["rescue_probability"],
            "frozen_auxiliary": frozen,
        }

    def forward(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        *,
        route_mode: str = "auto",
        return_aux: bool = False,
    ):
        if route_mode not in self.VALID_ROUTE_MODES:
            raise ValueError(f"unsupported A2.23 route mode: {route_mode}")
        if route_mode == "canonical":
            canonical = self.source_model(visible, infrared, route_mode="canonical")
            if return_aux:
                return canonical, {
                    "route_mode": "canonical",
                    "smoke_decision": torch.zeros(
                        visible.shape[0], dtype=torch.bool, device=visible.device
                    ),
                    "predictor_evaluated": False,
                    "canonical": canonical,
                }
            return canonical

        prediction = self.semantic_prediction(visible, infrared)
        frozen = prediction["frozen_auxiliary"]
        canonical = frozen["canonical"]
        proposal = frozen["proposal"]
        _, refiner_auxiliary = self.source_model.source_model.source_model.refiner(
            visible,
            infrared,
            canonical,
            proposal,
            prediction["localization_support"],
        )
        pre_utility = (
            refiner_auxiliary["low_residual"] + refiner_auxiliary["high_residual"]
        )
        bounded = prediction["utility_multiplier"] * pre_utility
        refined = (canonical + bounded).clamp(0.0, 1.0)
        old_decision = prediction["old_smoke_decision"]
        rescue_decision = prediction["rescue_probability"] >= 0.5
        decision = (
            torch.ones_like(old_decision)
            if route_mode == "smoke"
            else old_decision | rescue_decision
        )
        output = torch.where(decision.view(-1, 1, 1, 1), refined, canonical)
        if return_aux:
            auxiliary = dict(frozen)
            auxiliary.update(prediction)
            auxiliary.update(refiner_auxiliary)
            auxiliary.update(
                {
                    "route_mode": route_mode,
                    "predictor_evaluated": True,
                    "old_smoke_decision": old_decision,
                    "rescue_decision": rescue_decision,
                    "smoke_decision": decision,
                    "canonical": canonical,
                    "proposal": proposal,
                    "pre_utility_residual": pre_utility,
                    "bounded_residual": bounded,
                    "output_delta": refined - canonical,
                    "refined_smoke_output": refined,
                }
            )
            return output, auxiliary
        return output

    def trainable_parameters(self):
        yield from self.predictor.parameters()


def exact_scalar_l1_multiplier(
    canonical: torch.Tensor,
    residual: torch.Tensor,
    teacher: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the smallest channel-shared scalar minimizer and pixel L1."""
    if canonical.shape != residual.shape or canonical.shape != teacher.shape:
        raise ValueError("A2.23 oracle tensors must have identical shapes")
    if canonical.ndim != 4:
        raise ValueError("A2.23 oracle tensors must be BCHW")
    numerator = teacher.float() - canonical.float()
    denominator = residual.float()
    ratios = torch.where(
        denominator.abs() > 1.0e-12,
        numerator / denominator,
        torch.zeros_like(denominator),
    ).clamp(0.0, 1.0)
    zeros = torch.zeros_like(ratios[:, :1])
    ones = torch.ones_like(zeros)
    candidates = torch.cat((zeros, ones, ratios), dim=1)
    candidates, _ = candidates.sort(dim=1)
    outputs = (
        canonical.float().unsqueeze(1)
        + candidates.unsqueeze(2) * residual.float().unsqueeze(1)
    ).clamp(0.0, 1.0)
    losses = (outputs - teacher.float().unsqueeze(1)).abs().mean(dim=2)
    indices = losses.argmin(dim=1, keepdim=True)
    multiplier = candidates.gather(1, indices)
    minimum = losses.gather(1, indices).squeeze(1)
    return multiplier, minimum


def balanced_binary_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    target = target.float().clamp(0.0, 1.0)
    positive_fraction = target.mean(dim=(1, 2, 3), keepdim=True).clamp(
        1.0e-4, 1.0 - 1.0e-4
    )
    positive = -(1.0 - positive_fraction) * target * F.logsigmoid(logits)
    negative = -positive_fraction * (1.0 - target) * F.logsigmoid(-logits)
    return (positive + negative).mean()


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    false_positive_weight: float,
    false_negative_weight: float,
    exponent: float,
) -> torch.Tensor:
    probability = torch.sigmoid(logits).float()
    target = target.float().clamp(0.0, 1.0)
    dimensions = (1, 2, 3)
    true_positive = (probability * target).sum(dim=dimensions)
    false_positive = (probability * (1.0 - target)).sum(dim=dimensions)
    false_negative = ((1.0 - probability) * target).sum(dim=dimensions)
    index = (true_positive + 1.0) / (
        true_positive
        + float(false_positive_weight) * false_positive
        + float(false_negative_weight) * false_negative
        + 1.0
    )
    return (1.0 - index).pow(float(exponent)).mean()


def a223_phase0_loss(
    smoke: dict,
    clean: dict,
    mask: torch.Tensor,
    forced: torch.Tensor,
    canonical: torch.Tensor,
    teacher: torch.Tensor,
    task_multiplier: torch.Tensor,
    attainable_gain: torch.Tensor,
    contract: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep mask supervision off utility and task supervision off localization."""
    mask = mask.float().clamp(0.0, 1.0)
    localization_logits = smoke["localization_logits"]
    localization = smoke["localization_support"]
    localization_bce = balanced_binary_cross_entropy(localization_logits, mask)
    localization_tversky = focal_tversky_loss(
        localization_logits,
        mask,
        contract["tversky_false_positive_weight"],
        contract["tversky_false_negative_weight"],
        contract["focal_tversky_exponent"],
    )
    localization_coverage = F.l1_loss(
        localization.mean(dim=(1, 2, 3)), mask.mean(dim=(1, 2, 3))
    )
    localization_boundary = F.l1_loss(
        boundary_map(localization), boundary_map(mask)
    )
    clean_localization = F.binary_cross_entropy_with_logits(
        clean["localization_logits"], torch.zeros_like(clean["localization_logits"])
    )

    predicted_utility = smoke["utility_multiplier"]
    gain = attainable_gain.detach().float().clamp_min(0.0)
    pointwise_utility_l1 = (predicted_utility - task_multiplier.detach()).abs().squeeze(1)
    gain_weighted_utility = (pointwise_utility_l1 * gain).sum() / (
        gain.sum() + float(contract["gain_denominator_epsilon"])
    )
    clean_utility = F.binary_cross_entropy_with_logits(
        clean["utility_logits"], torch.zeros_like(clean["utility_logits"])
    )
    dimensions = (1, 2, 3)
    forced_mae = (forced.float() - teacher.float()).abs().mean(dim=dimensions)
    canonical_mae = (
        (canonical.float() - teacher.float()).abs().mean(dim=dimensions).detach()
    )
    relative_teacher = (forced_mae / canonical_mae.clamp_min(1.0e-6)).mean()
    terms = {
        "localization_balanced_smoke_bce": localization_bce,
        "localization_focal_tversky": localization_tversky,
        "localization_coverage_l1": localization_coverage,
        "localization_boundary_l1": localization_boundary,
        "clean_localization_bce": clean_localization,
        "gain_weighted_utility_l1": gain_weighted_utility,
        "paired_teacher_relative_mae": relative_teacher,
        "clean_utility_bce": clean_utility,
    }
    total = sum(
        float(contract["weights"][name]) * value for name, value in terms.items()
    )
    return total, {**terms, "total": total}
