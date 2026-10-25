"""Explicit-condition safe wrapper around a frozen A2.23 smoke expert."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn


class A32PromptSelectedSafeResidualFusion(nn.Module):
    """Select canonical or smoke-expert inference from an explicit condition.

    ``normal`` is a hard short circuit to the canonical TIFDFusion path.  The
    smoke expert is evaluated only for the exact ``smoke`` condition.  There
    is intentionally no learned prompt encoder or automatic degradation
    classifier in this wrapper.
    """

    VALID_CONDITIONS = frozenset(("normal", "smoke"))

    def __init__(self, source_model: nn.Module):
        super().__init__()
        self.source_model = source_model
        self.expert_evaluations = 0
        self.freeze_source()

    def freeze_source(self) -> None:
        self.source_model.eval()
        for parameter in self.source_model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.source_model.eval()
        return self

    @classmethod
    def validate_condition(cls, condition: str) -> str:
        if not isinstance(condition, str) or condition not in cls.VALID_CONDITIONS:
            raise ValueError(
                "A3.2 degradation_condition must be the exact string "
                "'normal' or 'smoke'"
            )
        return condition

    def reset_expert_evaluations(self) -> None:
        self.expert_evaluations = 0

    def forward(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        *,
        degradation_condition: str,
        return_aux: bool = False,
    ):
        condition = self.validate_condition(degradation_condition)
        if condition == "normal":
            canonical = self.source_model(
                visible, infrared, route_mode="canonical"
            )
            if return_aux:
                return canonical, {
                    "degradation_condition": condition,
                    "expert_evaluated": False,
                    "canonical": canonical,
                    "output": canonical,
                    "bounded_residual": torch.zeros_like(canonical),
                }
            return canonical

        self.expert_evaluations += 1
        output, source_auxiliary = self.source_model(
            visible,
            infrared,
            route_mode="smoke",
            return_aux=True,
        )
        canonical = source_auxiliary["canonical"]
        bounded_residual = output - canonical
        if return_aux:
            return output, {
                **source_auxiliary,
                "degradation_condition": condition,
                "expert_evaluated": True,
                "canonical": canonical,
                "output": output,
                "bounded_residual": bounded_residual,
            }
        return output

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        yield from (parameter for parameter in self.parameters() if parameter.requires_grad)
