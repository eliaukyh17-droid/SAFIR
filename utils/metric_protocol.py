"""Statistical helpers for paired, direction-aware fusion evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def directed_difference(candidate, reference, direction: str):
    """Return an advantage where positive always means that candidate is better."""
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference shapes differ")
    if direction == "higher":
        return candidate - reference
    if direction == "lower":
        return reference - candidate
    raise ValueError(f"unknown metric direction: {direction}")


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(tuple(values), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite vector")
    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    # Bounded chunks avoid a large resamples-by-samples allocation.
    for start in range(0, resamples, 512):
        stop = min(start + 512, resamples)
        indices = generator.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, (alpha, 1.0 - alpha))
    return float(low), float(high)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Holm family-wise adjusted p-values, returned in original order."""
    values = np.asarray(tuple(p_values), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty vector")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(values.size, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = (values.size - rank) * values[index]
        running = max(running, adjusted)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]


def paired_wilcoxon_p(values: Iterable[float]) -> float:
    values = np.asarray(tuple(values), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite vector")
    if np.all(values == 0.0):
        return 1.0
    from scipy.stats import wilcoxon

    result = wilcoxon(values, alternative="two-sided", zero_method="pratt")
    p_value = float(result.pvalue)
    if not math.isfinite(p_value):
        raise FloatingPointError("non-finite Wilcoxon p-value")
    return p_value
