"""Pre-registered A3.7 output-only supplementary fusion metrics.

This module does not modify or recompute the nine A3.3 metrics.  It adds a
fixed twelve-metric panel to be evaluated on the already frozen A3.3 images.
All inputs are two-dimensional grayscale arrays represented on [0, 255].

The three source-error metrics are intentionally reported together and are
marked as one redundant family.  They must not be counted as three independent
scientific wins.  Their definitions follow the common two-source convention:
the two source-to-fused errors (or PSNR values) are averaged.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import convolve

from utils.metrics_extended import compute_supplemental_metrics, to_gray_255


PROTOCOL = "A37_FROZEN_OUTPUT_SUPPLEMENTAL_METRIC_AUDIT_V1"
AUDITED_EXISTING_METRICS = (
    "CC", "Nabf", "MS_SSIM", "NMI", "QNCIE", "QY", "QCB",
)
NEW_A37_METRICS = (
    "Source_MSE", "Source_NRMSE", "Source_PSNR", "EI", "CE",
)
SUPPLEMENTAL_METRICS = (*AUDITED_EXISTING_METRICS, *NEW_A37_METRICS)
METRIC_DIRECTIONS = {
    metric: (
        "lower"
        if metric in {"Nabf", "Source_MSE", "Source_NRMSE", "CE"}
        else "higher"
    )
    for metric in SUPPLEMENTAL_METRICS
}
METRIC_FAMILIES = {
    "CC": "source_structure",
    "Nabf": "edge_artifact",
    "MS_SSIM": "source_structure",
    "NMI": "information_transfer",
    "QNCIE": "information_transfer",
    "QY": "source_structure",
    "QCB": "perceptual_quality",
    "Source_MSE": "source_error_redundant",
    "Source_NRMSE": "source_error_redundant",
    "Source_PSNR": "source_error_redundant",
    "EI": "detail_contrast",
    "CE": "distribution_divergence",
}


def _validate_shapes(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> None:
    if fused.shape != visible.shape or fused.shape != infrared.shape:
        raise ValueError(
            "fused, visible and infrared shapes must match: "
            f"{fused.shape}, {visible.shape}, {infrared.shape}"
        )


def _pair_mse(source: np.ndarray, fused: np.ndarray) -> float:
    return float(np.mean(np.square(source - fused), dtype=np.float64))


def source_mse(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    """Mean two-source MSE on the fixed [0, 255] grayscale scale (lower)."""
    return 0.5 * (
        _pair_mse(visible, fused) + _pair_mse(infrared, fused)
    )


def _pair_nrmse(source: np.ndarray, fused: np.ndarray) -> float:
    numerator = math.sqrt(_pair_mse(source, fused))
    denominator = math.sqrt(float(np.mean(np.square(source), dtype=np.float64)))
    if denominator <= 1e-12:
        # A zero-energy source has zero normalized error only for an identical
        # output.  Otherwise the fixed epsilon keeps the result finite.
        return 0.0 if numerator <= 1e-12 else numerator / 1e-12
    return numerator / denominator


def source_nrmse(
    fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray,
) -> float:
    """Mean Euclidean-normalized two-source RMSE (lower is better)."""
    return 0.5 * (
        _pair_nrmse(visible, fused) + _pair_nrmse(infrared, fused)
    )


def source_psnr(
    fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray,
) -> float:
    """PSNR from the mean two-source MSE and fixed data range 255 (higher)."""
    # This follows the common VIFB-style two-source convention: average the
    # two MSE terms first, then convert the result to PSNR.  The floor yields a
    # deterministic 120 dB cap for exact identity.
    mse = max(source_mse(fused, visible, infrared), 255.0**2 * 1e-12)
    return float(10.0 * math.log10(255.0**2 / mse))


def edge_intensity(fused: np.ndarray) -> float:
    """Mean Sobel gradient magnitude of the fused image (higher is better).

    The unscaled 3x3 Sobel kernels and symmetric boundary handling are fixed
    here so the metric is independent of OpenCV/PyTorch padding defaults.
    """
    kernel_x = np.asarray(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    kernel_y = kernel_x.T
    gradient_x = convolve(fused, kernel_x, mode="reflect")
    gradient_y = convolve(fused, kernel_y, mode="reflect")
    return float(np.mean(np.hypot(gradient_x, gradient_y)))


def _histogram_probabilities(image: np.ndarray) -> np.ndarray:
    quantized = np.rint(np.clip(image, 0.0, 255.0)).astype(np.uint8)
    counts = np.bincount(quantized.ravel(), minlength=256).astype(np.float64)
    return counts / float(counts.sum())


def _directed_histogram_cross_entropy(
    source: np.ndarray, fused: np.ndarray,
) -> float:
    """VIFB-style source-to-fused histogram divergence in bits.

    Only common non-zero bins contribute, matching the commonly distributed
    IVIF implementation.  This convention is locked before evaluation because
    alternative epsilon-smoothed definitions are not numerically equivalent.
    """
    source_probability = _histogram_probabilities(source)
    fused_probability = _histogram_probabilities(fused)
    valid = (source_probability > 0.0) & (fused_probability > 0.0)
    return float(np.sum(
        source_probability[valid]
        * np.log2(source_probability[valid] / fused_probability[valid])
    ))


def cross_entropy(
    fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray,
) -> float:
    """Mean VIFB-style source-to-fused histogram divergence (lower)."""
    return 0.5 * (
        _directed_histogram_cross_entropy(visible, fused)
        + _directed_histogram_cross_entropy(infrared, fused)
    )


def compute_a37_supplemental_metrics(
    fused, visible, infrared,
) -> dict[str, float]:
    """Compute all twelve pre-registered A3.7 supplementary metrics."""
    fused = to_gray_255(fused)
    visible = to_gray_255(visible)
    infrared = to_gray_255(infrared)
    _validate_shapes(fused, visible, infrared)

    values = compute_supplemental_metrics(fused, visible, infrared)
    values.update({
        "Source_MSE": source_mse(fused, visible, infrared),
        "Source_NRMSE": source_nrmse(fused, visible, infrared),
        "Source_PSNR": source_psnr(fused, visible, infrared),
        "EI": edge_intensity(fused),
        "CE": cross_entropy(fused, visible, infrared),
    })
    if tuple(values) != SUPPLEMENTAL_METRICS:
        raise RuntimeError(f"A3.7 metric order changed: {tuple(values)}")
    nonfinite = [name for name, value in values.items() if not math.isfinite(value)]
    if nonfinite:
        raise FloatingPointError(f"non-finite A3.7 metrics: {nonfinite}")
    return {name: float(values[name]) for name in SUPPLEMENTAL_METRICS}
