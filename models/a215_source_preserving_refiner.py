"""Canonical-anchored bounded residual refiner for A2.15."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def luminance(visible: torch.Tensor) -> torch.Tensor:
    if visible.ndim != 4 or visible.shape[1] not in (1, 3):
        raise ValueError("visible must be Bx1/3xHxW")
    if visible.shape[1] == 1:
        return visible
    weights = visible.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (visible[:, :3] * weights).sum(dim=1, keepdim=True)


def box_blur(value: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = int(kernel)
    if kernel <= 0 or kernel % 2 != 1:
        raise ValueError("box blur kernel must be a positive odd integer")
    return F.avg_pool2d(value, kernel, stride=1, padding=kernel // 2)


def signed_gradients(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    horizontal = F.pad(value[..., 1:] - value[..., :-1], (0, 1, 0, 0))
    vertical = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    return horizontal, vertical


def gradient_magnitude(value: torch.Tensor) -> torch.Tensor:
    horizontal, vertical = signed_gradients(value)
    return torch.sqrt(horizontal.square() + vertical.square() + 1.0e-8)


class DilatedResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                width, width, 3, padding=dilation, dilation=dilation,
                groups=width, bias=False,
            ),
            nn.Conv2d(width, width, 1, bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.body(value)


class CanonicalAnchoredResidualRefiner(nn.Module):
    """Predict a spatially supported residual with hard frequency caps."""

    def __init__(
        self,
        *,
        evidence_channels: int = 12,
        width: int = 32,
        dilations: Sequence[int] = (1, 2, 4),
        lowpass_kernel: int = 7,
        highpass_mean_kernel: int = 3,
        low_residual_cap: float = 0.12,
        high_residual_cap: float = 0.03,
    ):
        super().__init__()
        if int(evidence_channels) != 12 or int(width) != 32:
            raise ValueError("A2.15 evidence/width lock changed")
        if tuple(int(value) for value in dilations) != (1, 2, 4):
            raise ValueError("A2.15 dilation lock changed")
        self.evidence_channels = int(evidence_channels)
        self.width = int(width)
        self.lowpass_kernel = int(lowpass_kernel)
        self.highpass_mean_kernel = int(highpass_mean_kernel)
        self.low_residual_cap = float(low_residual_cap)
        self.high_residual_cap = float(high_residual_cap)
        self.total_residual_cap = self.low_residual_cap + self.high_residual_cap
        self.stem = nn.Sequential(
            nn.Conv2d(self.evidence_channels, self.width, 3, padding=1, bias=False),
            nn.GroupNorm(8, self.width),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*(
            DilatedResidualBlock(self.width, int(dilation))
            for dilation in dilations
        ))
        self.head = nn.Conv2d(self.width, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def evidence(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        canonical: torch.Tensor,
        proposal: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        visible_luma = luminance(visible)
        tensors = (infrared, canonical, proposal, support)
        if any(value.ndim != 4 or value.shape[1] != 1 for value in tensors):
            raise ValueError("A2.15 scalar inputs must be Bx1xHxW")
        if any(value.shape != canonical.shape for value in tensors):
            raise ValueError("A2.15 scalar input shapes differ")
        if visible_luma.shape != canonical.shape:
            raise ValueError("A2.15 visible/canonical shapes differ")
        residual = proposal - canonical
        return torch.cat((
            visible_luma,
            infrared,
            canonical,
            proposal,
            residual,
            box_blur(canonical, 5),
            box_blur(proposal, 5),
            gradient_magnitude(visible_luma),
            gradient_magnitude(infrared),
            gradient_magnitude(canonical),
            gradient_magnitude(proposal),
            support.clamp(0.0, 1.0),
        ), dim=1)

    def forward(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        canonical: torch.Tensor,
        proposal: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        support = support.clamp(0.0, 1.0)
        features = self.blocks(self.stem(
            self.evidence(visible, infrared, canonical, proposal, support)
        ))
        raw = self.head(features)
        low = self.low_residual_cap * torch.tanh(
            box_blur(raw[:, :1], self.lowpass_kernel)
        )
        high_raw = raw[:, 1:2]
        high = self.high_residual_cap * torch.tanh(
            high_raw - box_blur(high_raw, self.highpass_mean_kernel)
        )
        delta = support * (low + high)
        output = (canonical + delta).clamp(0.0, 1.0)
        return output, {
            "support": support,
            "raw_residual": raw,
            "low_residual": low,
            "high_residual": high,
            "bounded_residual": delta,
            "output_delta": output - canonical,
        }


class A215CanonicalAnchoredFusion(nn.Module):
    """Frozen A2.13 source plus an exact-bypass bounded residual refiner."""

    VALID_ROUTE_MODES = frozenset(("canonical", "smoke", "auto"))

    def __init__(self, source_model: nn.Module, refiner: nn.Module | None = None):
        super().__init__()
        self.source_model = source_model
        self.refiner = refiner or CanonicalAnchoredResidualRefiner()
        self.freeze_source()

    def freeze_source(self) -> None:
        self.source_model.eval()
        for parameter in self.source_model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.source_model.eval()
        self.refiner.train(mode)
        return self

    @staticmethod
    def _support(auxiliary: dict, target: torch.Tensor) -> torch.Tensor:
        gates = auxiliary.get("local_gates_used", auxiliary.get("local_gates"))
        if not gates:
            raise RuntimeError("A2.15 frozen local support is missing")
        resized = [
            F.interpolate(gate, size=target.shape[-2:], mode="bilinear", align_corners=False)
            for gate in gates
        ]
        return torch.stack(resized, dim=0).amax(dim=0).clamp(0.0, 1.0)

    def forward(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        *,
        route_mode: str = "auto",
        support_override: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        if route_mode not in self.VALID_ROUTE_MODES:
            raise ValueError(f"unsupported A2.15 route mode: {route_mode}")
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
                    "refiner_evaluated": False,
                    "canonical": canonical,
                }
            return canonical

        source_output, source_auxiliary = self.source_model(
            visible, infrared, route_mode=route_mode, return_aux=True,
        )
        decision = source_auxiliary["smoke_decision"]
        canonical = source_auxiliary.get("canonical")
        if canonical is None:
            canonical = source_output
        if not bool(decision.any()):
            if not torch.equal(source_output, canonical):
                raise RuntimeError("A2.15 rejected source is not exact canonical")
            if return_aux:
                auxiliary = dict(source_auxiliary)
                auxiliary.update({
                    "refiner_evaluated": False,
                    "canonical": canonical,
                })
                return canonical, auxiliary
            return canonical

        proposal = source_auxiliary["smoke_output"]
        if support_override is None:
            support = self._support(source_auxiliary, canonical)
        else:
            if support_override.ndim != 4 or support_override.shape[1] != 1:
                raise ValueError("support_override must be Bx1xHxW")
            support = F.interpolate(
                support_override, size=canonical.shape[-2:], mode="area",
            ).clamp(0.0, 1.0)
        refined, refiner_auxiliary = self.refiner(
            visible, infrared, canonical, proposal, support,
        )
        selector = decision.view(-1, 1, 1, 1)
        output = torch.where(selector, refined, canonical)
        if return_aux:
            auxiliary = dict(source_auxiliary)
            auxiliary.update(refiner_auxiliary)
            auxiliary.update({
                "route_mode": route_mode,
                "refiner_evaluated": True,
                "canonical": canonical,
                "proposal": proposal,
                "refined_smoke_output": refined,
            })
            return output, auxiliary
        return output

    def trainable_parameters(self):
        yield from self.refiner.parameters()


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[-2:] != value.shape[-2:]:
        mask = F.interpolate(mask, size=value.shape[-2:], mode="area")
    weight = mask.expand_as(value)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    return masked_mean((prediction - target).abs(), mask)


def masked_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weight = mask.expand_as(prediction)
    count = weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
    p_mean = (prediction * weight).sum(dim=(1, 2, 3)) / count
    t_mean = (target * weight).sum(dim=(1, 2, 3)) / count
    p = (prediction - p_mean[:, None, None, None]) * weight
    t = (target - t_mean[:, None, None, None]) * weight
    numerator = (p * t).sum(dim=(1, 2, 3))
    denominator = torch.sqrt(
        p.square().sum(dim=(1, 2, 3))
        * t.square().sum(dim=(1, 2, 3)) + 1.0e-8
    )
    return (1.0 - numerator / denominator).mean()


def source_preserving_loss(
    output: torch.Tensor,
    canonical: torch.Tensor,
    teacher: torch.Tensor,
    visible: torch.Tensor,
    infrared: torch.Tensor,
    smoke_mask: torch.Tensor,
    auxiliary: dict,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = smoke_mask.clamp(0.0, 1.0)
    outside = 1.0 - mask
    visible_luma = luminance(visible)
    out_x, out_y = signed_gradients(output)
    vis_x, vis_y = signed_gradients(visible_luma)
    ir_x, ir_y = signed_gradients(infrared)
    can_x, can_y = signed_gradients(canonical)

    choose_x = vis_x.abs() >= ir_x.abs()
    choose_y = vis_y.abs() >= ir_y.abs()
    source_x = torch.where(choose_x, vis_x, ir_x)
    source_y = torch.where(choose_y, vis_y, ir_y)
    output_gradient = gradient_magnitude(output)
    source_envelope = torch.maximum(
        torch.maximum(gradient_magnitude(visible_luma), gradient_magnitude(infrared)),
        gradient_magnitude(canonical),
    )
    canonical_gradient = gradient_magnitude(canonical)
    local_mean_output = box_blur(output, 7)
    local_mean_canonical = box_blur(canonical, 7)
    local_std_output = torch.sqrt(
        (box_blur(output.square(), 7) - local_mean_output.square()).clamp_min(0.0)
        + 1.0e-8
    )
    local_std_canonical = torch.sqrt(
        (box_blur(canonical.square(), 7) - local_mean_canonical.square()).clamp_min(0.0)
        + 1.0e-8
    )
    delta = auxiliary["output_delta"]
    delta_high = delta - box_blur(delta, 7)
    terms = {
        "teacher_smoke_region": masked_l1(output, teacher, mask),
        "teacher_low_frequency": masked_l1(
            box_blur(output, 7), box_blur(teacher, 7), mask,
        ),
        "canonical_clear_region": masked_l1(output, canonical, outside),
        "source_signed_gradient": 0.5 * (
            masked_l1(out_x, source_x, mask) + masked_l1(out_y, source_y, mask)
        ),
        "unsupported_edge": masked_mean(
            F.relu(output_gradient - source_envelope), mask,
        ),
        "canonical_gradient_floor": masked_mean(
            F.relu(canonical_gradient - output_gradient), mask,
        ),
        "local_variance_floor": masked_mean(
            F.relu(local_std_canonical - local_std_output), mask,
        ),
        "canonical_correlation": masked_correlation_loss(output, canonical, mask),
        "residual_budget": masked_mean(delta.abs(), mask),
        "residual_high_frequency": masked_mean(delta_high.abs(), mask),
    }
    if set(terms) != set(weights):
        raise ValueError("A2.15 loss keys changed")
    total = sum(float(weights[name]) * terms[name] for name in weights)
    terms["total"] = total
    return total, terms
