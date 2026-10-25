"""Zero-initialized dual-support and benefit-rescue controller for A2.19."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseRefine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(4, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class DualSupportBenefitRescueController(nn.Module):
    """Refine frozen support in two roles and predict beneficial rescue."""

    def __init__(
        self,
        *,
        feature_channels: Sequence[int] = (16, 24, 32),
        support_logit_delta_cap: float = 4.0,
        rescue_initial_bias: float = -4.0,
    ):
        super().__init__()
        if tuple(int(value) for value in feature_channels) != (16, 24, 32):
            raise ValueError("A2.19 frozen feature channels changed")
        if float(support_logit_delta_cap) != 4.0:
            raise ValueError("A2.19 support logit delta cap changed")
        if float(rescue_initial_bias) != -4.0:
            raise ValueError("A2.19 rescue initial bias changed")
        self.support_logit_delta_cap = float(support_logit_delta_cap)
        self.laterals = nn.ModuleList(
            nn.Conv2d(channels, 16, 1) for channels in feature_channels
        )
        self.pyramid_refine = nn.ModuleList(DepthwiseRefine(16) for _ in range(3))
        self.evidence_stem = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 24, 3, padding=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.SiLU(inplace=True),
        )
        self.fusion_refine = DepthwiseRefine(24)
        self.conditioning_delta_head = nn.Conv2d(24, 1, 1)
        self.multiplier_delta_head = nn.Conv2d(24, 1, 1)
        self.side_delta_heads = nn.ModuleList(nn.Conv2d(16, 1, 1) for _ in range(3))
        self.rescue_head = nn.Sequential(
            nn.Linear(71, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 1),
        )
        for head in (
            self.conditioning_delta_head,
            self.multiplier_delta_head,
            *self.side_delta_heads,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.rescue_head[-1].weight)
        nn.init.constant_(self.rescue_head[-1].bias, rescue_initial_bias)

    @staticmethod
    def _support_statistics(value: torch.Tensor) -> torch.Tensor:
        flat = value.flatten(1)
        return torch.stack(
            (flat.mean(1), flat.std(1, unbiased=False), flat.amax(1)), dim=1
        )

    def forward(
        self,
        evidence: torch.Tensor,
        features: Sequence[torch.Tensor],
        frozen_support_logits: torch.Tensor,
        frozen_support: torch.Tensor,
        smoke_score: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if evidence.ndim != 4 or evidence.shape[1] != 7:
            raise ValueError("A2.19 evidence must be Bx7xHxW")
        if len(features) != 3:
            raise ValueError("A2.19 requires three frozen feature maps")
        target_size = evidence.shape[-2:]
        if frozen_support.shape != frozen_support_logits.shape:
            raise ValueError("A2.19 frozen support/logit shape mismatch")
        if frozen_support.shape[-2:] != target_size:
            raise ValueError("A2.19 frozen support resolution changed")
        if smoke_score.shape != (evidence.shape[0],):
            raise ValueError("A2.19 smoke score shape changed")

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
        pyramid = (p1, p2, p3)
        evidence_features = self.evidence_stem(
            torch.cat((evidence, frozen_support), dim=1)
        )
        resized = [
            F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
            for value in pyramid
        ]
        fused = self.fusion(torch.cat((evidence_features, *resized), dim=1))
        fused = fused + self.fusion_refine(fused)
        cap = self.support_logit_delta_cap
        conditioning_logits = frozen_support_logits + cap * torch.tanh(
            self.conditioning_delta_head(fused)
        )
        multiplier_logits = frozen_support_logits + cap * torch.tanh(
            self.multiplier_delta_head(fused)
        )
        conditioning_support = torch.sigmoid(conditioning_logits)
        multiplier_support = torch.sigmoid(multiplier_logits)
        side_logits = [
            F.interpolate(
                frozen_support_logits,
                size=value.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            + cap * torch.tanh(head(value))
            for head, value in zip(self.side_delta_heads, pyramid)
        ]
        pooled = [
            F.adaptive_avg_pool2d(value, 1).flatten(1)
            for value in (*pyramid, evidence_features)
        ]
        controller_input = torch.cat(
            (
                *pooled,
                self._support_statistics(conditioning_support),
                self._support_statistics(multiplier_support),
                smoke_score.unsqueeze(1),
            ),
            dim=1,
        )
        rescue_logit = self.rescue_head(controller_input).squeeze(1)
        return {
            "conditioning_support_logits": conditioning_logits,
            "conditioning_support": conditioning_support,
            "multiplier_support_logits": multiplier_logits,
            "multiplier_support": multiplier_support,
            "side_support_logits": side_logits,
            "rescue_logit": rescue_logit,
            "rescue_probability": torch.sigmoid(rescue_logit),
        }


class A219DualSupportBenefitRescueFusion(nn.Module):
    """Frozen A2.17 plus a conservative rescue-only controller."""

    VALID_ROUTE_MODES = frozenset(("canonical", "smoke", "auto"))

    def __init__(self, source_model: nn.Module, controller: nn.Module | None = None):
        super().__init__()
        self.source_model = source_model
        self.controller = controller or DualSupportBenefitRescueController()
        self.freeze_source()

    def freeze_source(self) -> None:
        self.source_model.eval()
        for parameter in self.source_model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.source_model.eval()
        self.controller.train(mode)
        return self

    def _old_decision(self, auxiliary: dict) -> torch.Tensor:
        base = self.source_model.source_model.source_model
        return base.automatic_decision(auxiliary)

    def _controller_prediction(
        self, visible: torch.Tensor, infrared: torch.Tensor, auxiliary: dict
    ) -> dict:
        router = self.source_model.source_model.source_model.router
        evidence = router.source_router.evidence(visible, infrared)
        return self.controller(
            evidence,
            auxiliary["features"],
            auxiliary["support_logits"],
            auxiliary["support"],
            auxiliary["smoke_score"],
        )

    def controller_prediction(
        self, visible: torch.Tensor, infrared: torch.Tensor
    ) -> dict:
        _, auxiliary = self.source_model(
            visible, infrared, route_mode="smoke", return_aux=True
        )
        prediction = self._controller_prediction(visible, infrared, auxiliary)
        return {**prediction, "frozen_auxiliary": auxiliary}

    def forward(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        *,
        route_mode: str = "auto",
        return_aux: bool = False,
    ):
        if route_mode not in self.VALID_ROUTE_MODES:
            raise ValueError(f"unsupported A2.19 route mode: {route_mode}")
        if route_mode == "canonical":
            canonical = self.source_model(visible, infrared, route_mode="canonical")
            if return_aux:
                return canonical, {
                    "route_mode": "canonical",
                    "smoke_decision": torch.zeros(
                        visible.shape[0], dtype=torch.bool, device=visible.device
                    ),
                    "controller_evaluated": False,
                    "canonical": canonical,
                }
            return canonical

        frozen_forced, frozen_auxiliary = self.source_model(
            visible, infrared, route_mode="smoke", return_aux=True
        )
        prediction = self._controller_prediction(visible, infrared, frozen_auxiliary)
        canonical = frozen_auxiliary["canonical"]
        proposal = frozen_auxiliary["proposal"]
        conditioned, refiner_auxiliary = self.source_model.source_model.refiner(
            visible,
            infrared,
            canonical,
            proposal,
            prediction["conditioning_support"],
        )
        del conditioned
        pre_support = (
            refiner_auxiliary["low_residual"] + refiner_auxiliary["high_residual"]
        )
        bounded = prediction["multiplier_support"] * pre_support
        refined = (canonical + bounded).clamp(0.0, 1.0)
        old_decision = self._old_decision(frozen_auxiliary)
        rescue_decision = prediction["rescue_probability"] >= 0.5
        decision = (
            torch.ones_like(old_decision)
            if route_mode == "smoke"
            else old_decision | rescue_decision
        )
        output = torch.where(decision.view(-1, 1, 1, 1), refined, canonical)
        if return_aux:
            auxiliary = dict(frozen_auxiliary)
            auxiliary.update(prediction)
            auxiliary.update(refiner_auxiliary)
            auxiliary.update(
                {
                    "route_mode": route_mode,
                    "controller_evaluated": True,
                    "old_smoke_decision": old_decision,
                    "rescue_decision": rescue_decision,
                    "smoke_decision": decision,
                    "canonical": canonical,
                    "proposal": proposal,
                    "bounded_residual": bounded,
                    "output_delta": refined - canonical,
                    "refined_smoke_output": refined,
                    "frozen_a217_forced": frozen_forced,
                }
            )
            return output, auxiliary
        return output

    def trainable_parameters(self):
        yield from self.controller.parameters()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    support = torch.sigmoid(logits).float()
    target = target.float()
    intersection = (support * target).sum(dim=(1, 2, 3))
    denominator = support.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def boundary_map(value: torch.Tensor) -> torch.Tensor:
    horizontal = F.pad((value[..., 1:] - value[..., :-1]).abs(), (0, 1, 0, 0))
    vertical = F.pad((value[..., 1:, :] - value[..., :-1, :]).abs(), (0, 0, 0, 1))
    return (horizontal + vertical).clamp(0.0, 1.0)


def a219_phase0_loss(
    smoke: dict,
    clean: dict,
    mask: torch.Tensor,
    benefit_target: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = mask.float().clamp(0.0, 1.0)
    smoke_logits = (
        smoke["conditioning_support_logits"],
        smoke["multiplier_support_logits"],
    )
    clean_logits = (
        clean["conditioning_support_logits"],
        clean["multiplier_support_logits"],
    )
    smoke_bce = sum(
        F.binary_cross_entropy_with_logits(value, mask) for value in smoke_logits
    )
    dice = sum(soft_dice_loss(value, mask) for value in smoke_logits)
    clean_bce = sum(
        F.binary_cross_entropy_with_logits(value, torch.zeros_like(value))
        for value in clean_logits
    )
    boundary = sum(
        F.l1_loss(boundary_map(torch.sigmoid(value)), boundary_map(mask))
        for value in smoke_logits
    )
    side_bce = sum(
        F.binary_cross_entropy_with_logits(
            value,
            F.interpolate(mask, size=value.shape[-2:], mode="area"),
        )
        for value in smoke["side_support_logits"]
    )
    rescue_bce = 0.5 * (
        F.binary_cross_entropy_with_logits(
            smoke["rescue_logit"], benefit_target.float()
        )
        + F.binary_cross_entropy_with_logits(
            clean["rescue_logit"], torch.zeros_like(clean["rescue_logit"])
        )
    )
    terms = {
        "smoke_bce": smoke_bce,
        "soft_dice": dice,
        "clean_bce": clean_bce,
        "boundary_l1": boundary,
        "side_bce": side_bce,
        "rescue_bce": rescue_bce,
    }
    total = sum(float(weights[name]) * value for name, value in terms.items())
    return total, {**terms, "total": total}
