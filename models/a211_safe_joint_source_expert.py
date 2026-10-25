"""A2.11 safe conditional joint-source fusion expert.

The canonical TIFDFusion model is an immutable safety path.  Clean, rejected,
or uncertain inputs return the canonical tensor directly.  A separate smoke
expert is evaluated only for an explicitly forced or high-confidence smoke
route.  The expert receives the still-separated visible/infrared encoder
features in addition to the canonical TIFM features, so it does not have to
recover source information from an already fused output.

Smoke masks are optional training supervision/route overrides.  They are not
required by the automatic inference interface.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _luminance(visible: torch.Tensor) -> torch.Tensor:
    if visible.ndim != 4 or visible.shape[1] not in (1, 3):
        raise ValueError(f"visible must be Bx1/3xHxW, got {visible.shape}")
    if visible.shape[1] == 1:
        return visible
    weights = visible.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (visible[:, :3] * weights).sum(dim=1, keepdim=True)


def _gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    horizontal = F.pad(image[..., 1:] - image[..., :-1], (0, 1, 0, 0))
    vertical = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(horizontal.square() + vertical.square() + 1e-8)


def _validate_pair(visible: torch.Tensor, infrared: torch.Tensor) -> None:
    if visible.ndim != 4 or visible.shape[1] not in (1, 3):
        raise ValueError(f"visible must be Bx1/3xHxW, got {visible.shape}")
    if infrared.ndim != 4 or infrared.shape[1] != 1:
        raise ValueError(f"infrared must be Bx1xHxW, got {infrared.shape}")
    if visible.shape[0] != infrared.shape[0]:
        raise ValueError("visible and infrared batch sizes differ")
    if visible.shape[-2:] != infrared.shape[-2:]:
        raise ValueError("visible and infrared spatial sizes differ")


class _RouterStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, stride=stride, padding=1,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, 3, padding=1,
                groups=out_channels, bias=False,
            ),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ConservativeSmokeRouter(nn.Module):
    """Visual-only clean/smoke/unknown router with spatial smoke maps.

    The classifier is deliberately initialized toward ``clean`` and all local
    mask heads toward a small probability.  Automatic routing therefore starts
    as an exact canonical bypass rather than an untrained smoke intervention.
    """

    def __init__(
        self,
        widths: Sequence[int] = (16, 24, 32),
        *,
        initial_local_gate: float = 0.01,
        clean_logit: float = 4.0,
        smoke_logit: float = -4.0,
        unknown_logit: float = 0.0,
    ):
        super().__init__()
        widths = tuple(int(value) for value in widths)
        if len(widths) != 3 or min(widths) <= 0:
            raise ValueError("router widths must contain three positive integers")
        if not 0.0 < initial_local_gate < 1.0:
            raise ValueError("initial_local_gate must lie strictly between zero and one")
        self.widths = widths

        stages = []
        current = 7
        for index, width in enumerate(widths):
            stages.append(_RouterStage(current, width, 1 if index == 0 else 2))
            current = width
        self.stages = nn.ModuleList(stages)
        self.local_heads = nn.ModuleList(nn.Conv2d(width, 1, 1) for width in widths)
        self.global_head = nn.Linear(widths[-1], 3)

        local_bias = math.log(initial_local_gate / (1.0 - initial_local_gate))
        for head in self.local_heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, local_bias)
        nn.init.zeros_(self.global_head.weight)
        with torch.no_grad():
            self.global_head.bias.copy_(
                self.global_head.bias.new_tensor(
                    (clean_logit, smoke_logit, unknown_logit)
                )
            )

    @staticmethod
    def evidence(
        visible: torch.Tensor,
        infrared: torch.Tensor,
    ) -> torch.Tensor:
        _validate_pair(visible, infrared)
        visible_luma = _luminance(visible)
        visible_low = F.avg_pool2d(visible_luma, 5, stride=1, padding=2)
        infrared_low = F.avg_pool2d(infrared, 5, stride=1, padding=2)
        return torch.cat(
            (
                visible_luma,
                infrared,
                (visible_luma - infrared).abs(),
                _gradient_magnitude(visible_luma),
                _gradient_magnitude(infrared),
                (visible_luma - visible_low).abs(),
                (infrared - infrared_low).abs(),
            ),
            dim=1,
        )

    def forward(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        features = []
        value = self.evidence(visible, infrared)
        for stage in self.stages:
            value = stage(value)
            features.append(value)

        local_logits = [head(feature) for head, feature in zip(self.local_heads, features)]
        local_gates = [torch.sigmoid(logits) for logits in local_logits]
        pooled = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
        class_logits = self.global_head(pooled)
        probabilities = torch.softmax(class_logits, dim=1)
        entropy = -(
            probabilities.clamp_min(1e-8)
            * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1) / math.log(3.0)
        return {
            "features": features,
            "local_logits": local_logits,
            "local_gates": local_gates,
            "class_logits": class_logits,
            "class_probabilities": probabilities,
            "p_clean": probabilities[:, 0],
            "p_smoke": probabilities[:, 1],
            "p_unknown": probabilities[:, 2],
            "uncertainty": entropy,
            "mask_coverage": local_gates[0].mean(dim=(1, 2, 3)),
        }


class BidirectionalChannelInteraction(nn.Module):
    """Efficient two-way channel attention between modality-specific features."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        heads = min(int(num_heads), int(channels))
        while channels % heads != 0:
            heads -= 1
        self.channels = int(channels)
        self.num_heads = max(1, heads)
        self.visible_qkv = nn.Sequential(
            nn.Conv2d(channels, 3 * channels, 1, bias=False),
            nn.Conv2d(
                3 * channels, 3 * channels, 3, padding=1,
                groups=3 * channels, bias=False,
            ),
        )
        self.infrared_qkv = copy.deepcopy(self.visible_qkv)
        self.visible_out = nn.Conv2d(channels, channels, 1)
        self.infrared_out = nn.Conv2d(channels, channels, 1)
        self.visible_temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.infrared_temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

    def _reshape(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        head_channels = channels // self.num_heads
        return value.reshape(
            batch, self.num_heads, head_channels, height * width,
        )

    @staticmethod
    def _attend(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        attention = (query @ key.transpose(-2, -1)) * temperature
        return attention.softmax(dim=-1) @ value

    def forward(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if visible.shape != infrared.shape:
            raise ValueError("visible and infrared features must have equal shapes")
        visible_q, visible_k, visible_v = self.visible_qkv(visible).chunk(3, dim=1)
        infrared_q, infrared_k, infrared_v = self.infrared_qkv(infrared).chunk(3, dim=1)
        visible_context = self._attend(
            self._reshape(visible_q), self._reshape(infrared_k),
            self._reshape(infrared_v), self.visible_temperature,
        )
        infrared_context = self._attend(
            self._reshape(infrared_q), self._reshape(visible_k),
            self._reshape(visible_v), self.infrared_temperature,
        )
        batch, _, height, width = visible.shape
        visible_context = visible_context.reshape(batch, self.channels, height, width)
        infrared_context = infrared_context.reshape(batch, self.channels, height, width)
        return (
            visible + self.visible_out(visible_context),
            infrared + self.infrared_out(infrared_context),
        )


class JointSourceLevelAdapter(nn.Module):
    """Inject still-separated source information into one canonical level."""

    def __init__(self, channels: int, condition_channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = int(channels)
        self.interaction = BidirectionalChannelInteraction(channels, num_heads)
        self.condition_projection = nn.Sequential(
            nn.Conv2d(condition_channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
        )
        self.delta_body = nn.Sequential(
            nn.Conv2d(4 * channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.zeros_(self.delta_body[-1].weight)
        nn.init.zeros_(self.delta_body[-1].bias)

    def forward(
        self,
        canonical: torch.Tensor,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        condition: torch.Tensor,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if canonical.shape != visible.shape or canonical.shape != infrared.shape:
            raise ValueError("canonical/visible/infrared feature shapes must match")
        target_size = canonical.shape[-2:]
        condition = F.interpolate(
            condition, size=target_size, mode="bilinear", align_corners=False,
        )
        gate = F.interpolate(
            gate, size=target_size, mode="bilinear", align_corners=False,
        ).clamp(0.0, 1.0)
        visible_context, infrared_context = self.interaction(visible, infrared)
        condition = self.condition_projection(condition)
        raw_delta = self.delta_body(
            torch.cat(
                (canonical, visible_context, infrared_context, condition), dim=1,
            )
        )
        delta = gate * raw_delta
        return canonical + delta, {
            "visible_context": visible_context,
            "infrared_context": infrared_context,
            "raw_delta": raw_delta,
            "delta": delta,
            "gate": gate,
        }


class JointSourceSmokeExpert(nn.Module):
    """Three-level joint-source expert with a trainable canonical decoder copy."""

    def __init__(
        self,
        canonical_decoder: nn.Module,
        channels: Sequence[int] = (64, 128, 256, 512),
        router_widths: Sequence[int] = (16, 24, 32),
        num_heads: int = 4,
    ):
        super().__init__()
        channels = tuple(int(value) for value in channels)
        router_widths = tuple(int(value) for value in router_widths)
        if len(channels) != 4 or len(router_widths) != 3:
            raise ValueError("A2.11 requires four base levels and three expert levels")
        self.channels = channels
        self.levels = nn.ModuleList(
            JointSourceLevelAdapter(channels[index], router_widths[index], num_heads)
            for index in range(3)
        )
        self.decoder = copy.deepcopy(canonical_decoder)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(True)
        self.visible_reconstruction = nn.Sequential(
            nn.Conv2d(channels[0], max(8, channels[0] // 2), 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, channels[0] // 2), 1, 1),
            nn.Sigmoid(),
        )
        self.infrared_reconstruction = copy.deepcopy(self.visible_reconstruction)
        self._freeze_decoder_statistics()

    def _freeze_decoder_statistics(self) -> None:
        # The copied decoder remains trainable, but its BN running statistics
        # stay fixed so initial smoke output exactly matches canonical eval.
        self.decoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_decoder_statistics()
        return self

    def forward(
        self,
        visible_features: Sequence[torch.Tensor],
        infrared_features: Sequence[torch.Tensor],
        canonical_features: Sequence[torch.Tensor],
        router_features: Sequence[torch.Tensor],
        local_gates: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | list[torch.Tensor]]]:
        if not (
            len(visible_features) == len(infrared_features)
            == len(canonical_features) == 4
        ):
            raise ValueError("base feature pyramids must contain four levels")
        if len(router_features) != 3 or len(local_gates) != 3:
            raise ValueError("router pyramids must contain three levels")

        adapted = []
        level_auxiliary = []
        for index, level in enumerate(self.levels):
            output, auxiliary = level(
                canonical_features[index], visible_features[index],
                infrared_features[index], router_features[index],
                local_gates[index],
            )
            adapted.append(output)
            level_auxiliary.append(auxiliary)
        adapted.append(canonical_features[3])
        fused = self.decoder(adapted)
        return fused, {
            "adapted_features": adapted,
            "level_auxiliary": level_auxiliary,
            "visible_reconstruction": self.visible_reconstruction(adapted[0]),
            "infrared_reconstruction": self.infrared_reconstruction(adapted[0]),
        }


@dataclass(frozen=True)
class RouteThresholds:
    smoke_probability_min: float = 0.95
    smoke_clean_margin_min: float = 0.80
    uncertainty_max: float = 0.20
    mask_coverage_min: float = 0.01

    def __post_init__(self):
        if not 0.0 <= self.smoke_probability_min <= 1.0:
            raise ValueError("smoke_probability_min must lie in [0, 1]")
        if not -1.0 <= self.smoke_clean_margin_min <= 1.0:
            raise ValueError("smoke_clean_margin_min must lie in [-1, 1]")
        if not 0.0 <= self.uncertainty_max <= 1.0:
            raise ValueError("uncertainty_max must lie in [0, 1]")
        if not 0.0 <= self.mask_coverage_min <= 1.0:
            raise ValueError("mask_coverage_min must lie in [0, 1]")


class SafeConditionalJointSourceFusion(nn.Module):
    """Immutable canonical safety path plus a separately trainable smoke path."""

    VALID_ROUTE_MODES = frozenset(("canonical", "smoke", "auto", "external"))

    def __init__(
        self,
        base_model: nn.Module,
        *,
        channels: Sequence[int] = (64, 128, 256, 512),
        router_widths: Sequence[int] = (16, 24, 32),
        num_heads: int = 4,
        thresholds: RouteThresholds | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.router = ConservativeSmokeRouter(router_widths)
        self.smoke_expert = JointSourceSmokeExpert(
            base_model.decoder, channels, router_widths, num_heads,
        )
        self.thresholds = thresholds or RouteThresholds()
        self.freeze_base()

    def freeze_base(self) -> None:
        self.base_model.eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        self.router.train(mode)
        self.smoke_expert.train(mode)
        return self

    def baseline(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
    ) -> torch.Tensor:
        self.base_model.eval()
        with torch.no_grad():
            return self.base_model(visible, infrared).detach()

    def _canonical_context(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        self.base_model.eval()
        with torch.no_grad():
            visible_features, infrared_features = self.base_model.encoder(
                visible, infrared,
            )
            segmentation_priors = None
            detection_priors = None
            if self.base_model.prior_extractor is not None and (
                self.base_model._use_seg_prior or self.base_model._use_det_prior
            ):
                segmentation, detection = self.base_model.prior_extractor(
                    visible, infrared,
                )
                if self.base_model._use_seg_prior:
                    segmentation_priors = segmentation
                if self.base_model._use_det_prior:
                    detection_priors = detection
            if self.base_model.no_tifm:
                canonical_features = [
                    self.base_model.simple_fuse[index](
                        torch.cat(
                            (visible_features[index], infrared_features[index]),
                            dim=1,
                        )
                    )
                    for index in range(len(visible_features))
                ]
            else:
                if self.base_model.swap_priors:
                    segmentation_priors, detection_priors = (
                        detection_priors, segmentation_priors,
                    )
                canonical_features = self.base_model.tifm(
                    visible_features, infrared_features,
                    segmentation_priors, detection_priors,
                )
            canonical = self.base_model.decoder(canonical_features).detach()
        return visible_features, infrared_features, canonical_features, canonical

    def automatic_decision(self, router_output: dict) -> torch.Tensor:
        thresholds = self.thresholds
        return (
            (router_output["p_smoke"] >= thresholds.smoke_probability_min)
            & (
                router_output["p_smoke"] - router_output["p_clean"]
                >= thresholds.smoke_clean_margin_min
            )
            & (router_output["uncertainty"] <= thresholds.uncertainty_max)
            & (router_output["mask_coverage"] >= thresholds.mask_coverage_min)
        )

    @staticmethod
    def _external_decision(
        route_override: torch.Tensor | bool | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if route_override is None:
            raise ValueError("external routing requires route_override")
        decision = torch.as_tensor(route_override, device=device, dtype=torch.bool)
        if decision.ndim == 0:
            decision = decision.expand(batch_size)
        if decision.shape != (batch_size,):
            raise ValueError("route_override must be scalar or shape [batch]")
        return decision

    @staticmethod
    def _override_gates(
        gate_override: torch.Tensor | None,
        router_output: dict,
    ) -> list[torch.Tensor]:
        if gate_override is None:
            return router_output["local_gates"]
        if gate_override.ndim != 4 or gate_override.shape[1] != 1:
            raise ValueError("gate_override must be Bx1xHxW")
        return [
            F.interpolate(
                gate_override, size=feature.shape[-2:], mode="area",
            ).clamp(0.0, 1.0)
            for feature in router_output["features"]
        ]

    def forward(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        *,
        route_mode: str = "auto",
        route_override: torch.Tensor | bool | None = None,
        gate_override: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        _validate_pair(visible, infrared)
        if route_mode not in self.VALID_ROUTE_MODES:
            raise ValueError(f"unsupported route_mode: {route_mode}")

        # This is the formal clean-performance lock: no router or smoke module
        # participates in the canonical branch.
        if route_mode == "canonical":
            canonical = self.baseline(visible, infrared)
            if return_aux:
                return canonical, {
                    "route_mode": route_mode,
                    "smoke_decision": torch.zeros(
                        visible.shape[0], dtype=torch.bool, device=visible.device,
                    ),
                }
            return canonical

        router_output = self.router(visible, infrared)
        if route_mode == "smoke":
            decision = torch.ones(
                visible.shape[0], dtype=torch.bool, device=visible.device,
            )
        elif route_mode == "external":
            decision = self._external_decision(
                route_override, visible.shape[0], visible.device,
            )
        else:
            if route_override is not None:
                raise ValueError("route_override is valid only for external routing")
            decision = self.automatic_decision(router_output)

        # A rejected all-clean batch returns the canonical model directly.
        if not bool(decision.any()):
            canonical = self.baseline(visible, infrared)
            if return_aux:
                auxiliary = dict(router_output)
                auxiliary.update({
                    "route_mode": route_mode,
                    "smoke_decision": decision,
                    "expert_evaluated": False,
                })
                return canonical, auxiliary
            return canonical

        visible_features, infrared_features, canonical_features, canonical = (
            self._canonical_context(visible, infrared)
        )
        local_gates = self._override_gates(gate_override, router_output)
        smoke, expert_auxiliary = self.smoke_expert(
            visible_features, infrared_features, canonical_features,
            router_output["features"], local_gates,
        )
        selector = decision.view(-1, 1, 1, 1)
        fused = torch.where(selector, smoke, canonical)
        if return_aux:
            auxiliary = dict(router_output)
            auxiliary.update(expert_auxiliary)
            auxiliary.update({
                "route_mode": route_mode,
                "smoke_decision": decision,
                "expert_evaluated": True,
                "canonical": canonical,
                "smoke_output": smoke,
                "local_gates_used": local_gates,
            })
            return fused, auxiliary
        return fused

    def trainable_parameters(self):
        yield from self.router.parameters()
        yield from self.smoke_expert.parameters()
