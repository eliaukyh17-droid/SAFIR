"""Mask-supervised multiscale local-support predictor for A2.17."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskSupervisedMultiscaleSupportPredictor(nn.Module):
    """Decode frozen router evidence/features into one full-resolution support."""

    def __init__(
        self,
        *,
        feature_channels: Sequence[int] = (16, 24, 32),
        lateral_channels: int = 8,
        evidence_channels: int = 7,
        fusion_channels: int = 16,
    ):
        super().__init__()
        if tuple(int(value) for value in feature_channels) != (16, 24, 32):
            raise ValueError("A2.17 frozen router feature channels changed")
        if (int(lateral_channels), int(evidence_channels), int(fusion_channels)) != (
            8,
            7,
            16,
        ):
            raise ValueError("A2.17 support architecture lock changed")
        self.laterals = nn.ModuleList(
            nn.Conv2d(int(channels), 8, 1)
            for channels in feature_channels
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(31, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
            nn.Conv2d(16, 16, 1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        evidence: torch.Tensor,
        features: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if evidence.ndim != 4 or evidence.shape[1] != 7:
            raise ValueError("A2.17 evidence must be Bx7xHxW")
        if len(features) != 3:
            raise ValueError("A2.17 requires three frozen feature maps")
        target_size = evidence.shape[-2:]
        projected = []
        for lateral, feature in zip(self.laterals, features):
            value = lateral(feature)
            value = F.interpolate(
                value, size=target_size, mode="bilinear", align_corners=False,
            )
            projected.append(value)
        fused = self.fusion(torch.cat((evidence, *projected), dim=1))
        refined = fused + self.refine(fused)
        logits = self.head(refined)
        support = torch.sigmoid(logits)
        return {"support_logits": logits, "support": support}


class A217MaskSupervisedSupportFusion(nn.Module):
    """Frozen A2.15 final model with a new bounded local-support predictor."""

    VALID_ROUTE_MODES = frozenset(("canonical", "smoke", "auto"))

    def __init__(self, source_model: nn.Module, predictor: nn.Module | None = None):
        super().__init__()
        self.source_model = source_model
        self.predictor = predictor or MaskSupervisedMultiscaleSupportPredictor()
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

    def _predict_support(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        source_auxiliary: dict,
    ) -> dict[str, torch.Tensor]:
        router = self.source_model.source_model.router
        source_router = getattr(router, "source_router", None)
        if source_router is None:
            raise RuntimeError("A2.17 frozen source router is unavailable")
        evidence = source_router.evidence(visible, infrared)
        return self.predictor(evidence, source_auxiliary["features"])

    def forward(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        *,
        route_mode: str = "auto",
        return_aux: bool = False,
    ):
        if route_mode not in self.VALID_ROUTE_MODES:
            raise ValueError(f"unsupported A2.17 route mode: {route_mode}")
        if route_mode == "canonical":
            canonical = self.source_model(
                visible, infrared, route_mode="canonical",
            )
            if return_aux:
                return canonical, {
                    "route_mode": "canonical",
                    "smoke_decision": torch.zeros(
                        visible.shape[0], dtype=torch.bool, device=visible.device,
                    ),
                    "predictor_evaluated": False,
                    "canonical": canonical,
                }
            return canonical

        base = self.source_model.source_model
        source_output, source_auxiliary = base(
            visible, infrared, route_mode=route_mode, return_aux=True,
        )
        decision = source_auxiliary["smoke_decision"]
        canonical = source_auxiliary.get("canonical", source_output)
        if not bool(decision.any()):
            if not torch.equal(source_output, canonical):
                raise RuntimeError("A2.17 rejected output is not exact canonical")
            if return_aux:
                auxiliary = dict(source_auxiliary)
                auxiliary.update({
                    "route_mode": route_mode,
                    "predictor_evaluated": False,
                    "canonical": canonical,
                })
                return canonical, auxiliary
            return canonical

        prediction = self._predict_support(
            visible, infrared, source_auxiliary,
        )
        proposal = source_auxiliary["smoke_output"]
        refined, refiner_auxiliary = self.source_model.refiner(
            visible,
            infrared,
            canonical,
            proposal,
            prediction["support"],
        )
        selector = decision.view(-1, 1, 1, 1)
        output = torch.where(selector, refined, canonical)
        if return_aux:
            auxiliary = dict(source_auxiliary)
            auxiliary.update(prediction)
            auxiliary.update(refiner_auxiliary)
            auxiliary.update({
                "route_mode": route_mode,
                "predictor_evaluated": True,
                "canonical": canonical,
                "proposal": proposal,
                "refined_smoke_output": refined,
            })
            return output, auxiliary
        return output

    def trainable_parameters(self):
        yield from self.predictor.parameters()


def soft_dice_loss(support: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    support = support.float()
    target = target.float()
    intersection = (support * target).sum(dim=(1, 2, 3))
    denominator = support.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def support_total_variation(support: torch.Tensor) -> torch.Tensor:
    horizontal = (support[..., 1:] - support[..., :-1]).abs().mean()
    vertical = (support[..., 1:, :] - support[..., :-1, :]).abs().mean()
    return horizontal + vertical


def mask_supervision_loss(
    smoke_prediction: dict[str, torch.Tensor],
    clean_prediction: dict[str, torch.Tensor],
    target: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = target.to(
        device=smoke_prediction["support"].device,
        dtype=smoke_prediction["support"].dtype,
    ).clamp(0.0, 1.0)
    smoke_bce = F.binary_cross_entropy_with_logits(
        smoke_prediction["support_logits"], target,
    )
    dice = soft_dice_loss(smoke_prediction["support"], target)
    coverage = (
        smoke_prediction["support"].mean(dim=(1, 2, 3))
        - target.mean(dim=(1, 2, 3))
    ).abs().mean()
    clean_bce = F.binary_cross_entropy_with_logits(
        clean_prediction["support_logits"],
        torch.zeros_like(clean_prediction["support_logits"]),
    )
    variation = support_total_variation(smoke_prediction["support"])
    terms = {
        "smoke_bce": smoke_bce,
        "soft_dice": dice,
        "coverage": coverage,
        "clean_bce": clean_bce,
        "total_variation": variation,
    }
    total = sum(float(weights[name]) * value for name, value in terms.items())
    return total, {**terms, "total": total}
