"""Audited supplementary metrics for infrared-visible image fusion.

The historical TIFDFusion benchmark uses nine metrics from ``utils.metrics``.
This module leaves those definitions untouched and adds seven complementary
metrics collected from the local IVIF-ZOO/UniFusion/LENFusion evaluators.  The
implementations here deliberately remove repository-specific paths, make
constant-image behaviour finite and document the optimization direction.

All public functions consume two-dimensional grayscale arrays in [0, 255].
``compute_extended_fusion_metrics`` also accepts tensors/arrays in the same
ranges supported by the historical evaluator.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d


LEGACY_METRICS = (
    "EN", "SD", "SF", "AG", "MI", "SCD", "VIF", "Qabf", "SSIM",
)
SUPPLEMENTAL_METRICS = (
    "CC", "Nabf", "MS_SSIM", "NMI", "QNCIE", "QY", "QCB",
)
# SSIM_mean replaces the legacy two-source SSIM sum in aggregate rankings so
# the same information is not counted twice.  It is a positive rescaling and
# therefore has exactly the same ordering as the historical SSIM column.
RANKED_METRICS = (
    "EN", "SD", "SF", "AG", "MI", "SCD", "VIF", "Qabf", "SSIM_mean",
    *SUPPLEMENTAL_METRICS,
)
REPORT_METRICS = (*LEGACY_METRICS, "SSIM_mean", *SUPPLEMENTAL_METRICS)
METRIC_DIRECTIONS = {
    metric: ("lower" if metric == "Nabf" else "higher")
    for metric in RANKED_METRICS
}
METRIC_FAMILIES = {
    "EN": "information",
    "MI": "information",
    "NMI": "information",
    "QNCIE": "information",
    "SD": "detail",
    "SF": "detail",
    "AG": "detail",
    "SCD": "structure",
    "VIF": "structure",
    "SSIM_mean": "structure",
    "MS_SSIM": "structure",
    "CC": "structure",
    "QY": "structure",
    "QCB": "structure",
    "Qabf": "edge_artifact",
    "Nabf": "edge_artifact",
}


def to_gray_255(image) -> np.ndarray:
    """Convert a tensor-like object or ndarray to float64 grayscale [0, 255]."""
    if hasattr(image, "detach"):
        image = image.detach()
    if hasattr(image, "cpu"):
        image = image.cpu()
    if hasattr(image, "numpy"):
        image = image.numpy()
    value = np.asarray(image)
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"expected one image, got shape {value.shape}")
        value = value[0]
    if value.ndim == 3 and value.shape[0] in (1, 3):
        if value.shape[0] == 1:
            value = value[0]
        else:
            value = (
                0.299 * value[0] + 0.587 * value[1] + 0.114 * value[2]
            )
    elif value.ndim == 3 and value.shape[-1] in (1, 3):
        if value.shape[-1] == 1:
            value = value[..., 0]
        else:
            value = (
                0.299 * value[..., 0]
                + 0.587 * value[..., 1]
                + 0.114 * value[..., 2]
            )
    if value.ndim != 2:
        raise ValueError(f"expected a grayscale image, got shape {value.shape}")
    value = value.astype(np.float64, copy=False)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError("image must be non-empty and finite")
    if float(value.min()) < -0.5:
        value = (value + 1.0) * 127.5
    elif float(value.max()) <= 1.0:
        value = value * 255.0
    return np.clip(value, 0.0, 255.0)


def _quantize(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0.0, 255.0)).astype(np.uint8)


def _entropy_bits(image: np.ndarray) -> float:
    counts = np.bincount(_quantize(image).ravel(), minlength=256).astype(float)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _mutual_information_bits(left: np.ndarray, right: np.ndarray) -> float:
    histogram = np.histogram2d(
        _quantize(left).ravel(), _quantize(right).ravel(),
        bins=256, range=((0, 256), (0, 256)),
    )[0]
    joint = histogram / histogram.sum()
    marginal_left = joint.sum(axis=1, keepdims=True)
    marginal_right = joint.sum(axis=0, keepdims=True)
    independent = marginal_left * marginal_right
    valid = joint > 0
    return float(np.sum(joint[valid] * np.log2(joint[valid] / independent[valid])))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = math.sqrt(
        float(np.sum(np.square(left_centered)))
        * float(np.sum(np.square(right_centered)))
    )
    if denominator <= 1e-12:
        return 1.0 if np.allclose(left, right, atol=1e-12) else 0.0
    return float(np.sum(left_centered * right_centered) / denominator)


def CC(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    """Mean Pearson correlation with the two source images (higher is better)."""
    return 0.5 * (
        _correlation(fused, visible) + _correlation(fused, infrared)
    )


def NMI(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    """IVIF-ZOO normalized mutual information convention (higher is better)."""
    numerator = 2.0 * (
        _mutual_information_bits(visible, fused)
        + _mutual_information_bits(infrared, fused)
    )
    denominator = _entropy_bits(visible) + _entropy_bits(infrared)
    return numerator / denominator if denominator > 1e-12 else 0.0


def QNCIE(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    """Nonlinear-correlation information entropy quality (higher is better)."""
    correlations = np.array(
        [
            [1.0, _correlation(visible, infrared), _correlation(visible, fused)],
            [_correlation(infrared, visible), 1.0, _correlation(infrared, fused)],
            [_correlation(fused, visible), _correlation(fused, infrared), 1.0],
        ],
        dtype=np.float64,
    )
    eigenvalues = np.clip(np.linalg.eigvalsh(correlations), 0.0, None)
    probabilities = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
    nonzero = probabilities > 0
    entropy = -float(np.sum(
        probabilities[nonzero] * np.log2(probabilities[nonzero])
    ))
    return 1.0 - entropy / math.log2(256.0)


def _sobel_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertical_kernel = np.array(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
    ) / 8.0
    horizontal_kernel = np.array(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
    ) / 8.0
    vertical = convolve2d(
        image, vertical_kernel, mode="same", boundary="symm",
    )
    horizontal = convolve2d(
        image, horizontal_kernel, mode="same", boundary="symm",
    )
    return vertical, horizontal


def Nabf(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    """Noise/artifact introduced by fusion (lower is better).

    This is a guarded port of the IVIF-ZOO implementation.  It preserves its
    constants while replacing undefined zero divisions with explicit masks.
    """
    gv_a, gh_a = _sobel_gradients(visible)
    gv_b, gh_b = _sobel_gradients(infrared)
    gv_f, gh_f = _sobel_gradients(fused)
    g_a = np.hypot(gh_a, gv_a)
    g_b = np.hypot(gh_b, gv_b)
    g_f = np.hypot(gh_f, gv_f)

    def quality(source_g, source_v, source_h):
        ratio = np.zeros_like(source_g)
        valid = (source_g > 1e-12) & (g_f > 1e-12)
        ratio[valid] = (
            np.minimum(source_g[valid], g_f[valid])
            / np.maximum(source_g[valid], g_f[valid])
        )
        source_angle = np.arctan2(source_v, source_h)
        fused_angle = np.arctan2(gv_f, gh_f)
        angle_similarity = (
            np.abs(np.abs(source_angle - fused_angle) - np.pi / 2.0)
            * 2.0 / np.pi
        )
        gradient_quality = 0.9999 / (
            1.0 + np.exp(-19.0 * (ratio - 0.5))
        )
        angle_quality = 0.9995 / (
            1.0 + np.exp(-22.0 * (angle_similarity - 0.5))
        )
        return np.sqrt(gradient_quality * angle_quality)

    quality_a = quality(g_a, gv_a, gh_a)
    quality_b = quality(g_b, gv_b, gh_b)
    weight_a = np.where(g_a >= 2.0, g_a**1.5, 0.0)
    weight_b = np.where(g_b >= 2.0, g_b**1.5, 0.0)
    denominator = float(np.sum(weight_a + weight_b))
    if denominator <= 1e-12:
        return 0.0
    artifact = (g_f > g_a) & (g_f > g_b)
    value = np.sum(
        artifact
        * ((1.0 - quality_a) * weight_a + (1.0 - quality_b) * weight_b)
    ) / denominator
    return float(np.clip(value, 0.0, 1.0))


def _ssim_components(
    left: np.ndarray,
    right: np.ndarray,
    sigma: float = 1.5,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    left = left / 255.0
    right = right / 255.0
    mean_left = gaussian_filter(left, sigma=sigma, mode="reflect", truncate=2.0)
    mean_right = gaussian_filter(right, sigma=sigma, mode="reflect", truncate=2.0)
    variance_left = np.maximum(
        gaussian_filter(left * left, sigma=sigma, mode="reflect", truncate=2.0)
        - mean_left * mean_left,
        0.0,
    )
    variance_right = np.maximum(
        gaussian_filter(right * right, sigma=sigma, mode="reflect", truncate=2.0)
        - mean_right * mean_right,
        0.0,
    )
    covariance = (
        gaussian_filter(left * right, sigma=sigma, mode="reflect", truncate=2.0)
        - mean_left * mean_right
    )
    c1 = 0.01**2
    c2 = 0.03**2
    luminance = (
        (2.0 * mean_left * mean_right + c1)
        / (np.square(mean_left) + np.square(mean_right) + c1)
    )
    contrast_structure = (
        (2.0 * covariance + c2)
        / (variance_left + variance_right + c2)
    )
    ssim_map = luminance * contrast_structure
    return (
        float(np.mean(ssim_map)),
        float(np.mean(contrast_structure)),
        ssim_map,
        variance_left,
        variance_right,
    )


def _downsample(image: np.ndarray) -> np.ndarray:
    if image.shape[0] % 2:
        image = image[:-1, :]
    if image.shape[1] % 2:
        image = image[:, :-1]
    return 0.25 * (
        image[0::2, 0::2] + image[1::2, 0::2]
        + image[0::2, 1::2] + image[1::2, 1::2]
    )


def _pair_ms_ssim(left: np.ndarray, right: np.ndarray) -> float:
    weights = np.array((0.0448, 0.2856, 0.3001, 0.2363, 0.1333))
    available_levels = min(5, max(1, int(math.log2(min(left.shape))) - 3))
    weights = weights[:available_levels]
    weights = weights / weights.sum()
    contrast_values = []
    ssim_value = 1.0
    for level in range(available_levels):
        ssim_value, contrast, *_ = _ssim_components(left, right)
        contrast_values.append(max(contrast, 1e-12))
        if level + 1 < available_levels:
            left = _downsample(left)
            right = _downsample(right)
    if available_levels == 1:
        return float(np.clip(ssim_value, -1.0, 1.0))
    value = np.prod(
        np.power(contrast_values[:-1], weights[:-1])
    ) * max(ssim_value, 1e-12) ** weights[-1]
    return float(np.clip(value, 0.0, 1.0))


def MS_SSIM(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    """Mean two-source multi-scale SSIM (higher is better)."""
    return 0.5 * (
        _pair_ms_ssim(visible, fused) + _pair_ms_ssim(infrared, fused)
    )


def QY(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    """Yang local structural fusion quality (higher is better)."""
    _, _, source_map, variance_visible, variance_infrared = _ssim_components(
        visible, infrared,
    )
    _, _, visible_map, _, _ = _ssim_components(visible, fused)
    _, _, infrared_map, _, _ = _ssim_components(infrared, fused)
    related = source_map >= 0.75
    weight = variance_visible / (
        variance_visible + variance_infrared + 1e-12
    )
    quality = np.where(
        related,
        weight * visible_map + (1.0 - weight) * infrared_map,
        np.maximum(visible_map, infrared_map),
    )
    return float(np.mean(quality))


def _normalize_unit(image: np.ndarray) -> np.ndarray:
    minimum = float(np.min(image))
    span = float(np.max(image)) - minimum
    return (image - minimum) / span if span > 1e-12 else np.zeros_like(image)


def QCB(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    """Chen-Blum perceptual fusion quality (higher is better)."""
    images = [_normalize_unit(value) for value in (visible, infrared, fused)]
    height, width = fused.shape
    fy = np.fft.fftfreq(height) * (height / 30.0)
    fx = np.fft.fftfreq(width) * (width / 30.0)
    vv, uu = np.meshgrid(fx, fy)
    radius = np.sqrt(uu * uu + vv * vv)
    response = (
        np.exp(-(radius / 15.3870) ** 2)
        - 0.7622 * np.exp(-(radius / 1.3456) ** 2)
    )
    filtered = [
        np.fft.ifft2(np.fft.fft2(image) * response).real
        for image in images
    ]
    def perceptual_contrast(image):
        # These truncate values reproduce a 31x31 kernel (radius 15) while
        # scipy's separable Gaussian implementation keeps the full diagnostic
        # practical on 108x3 image panels.
        local_2 = gaussian_filter(
            image, sigma=2.0, mode="constant", truncate=7.5,
        )
        local_4 = gaussian_filter(
            image, sigma=4.0, mode="constant", truncate=3.75,
        )
        denominator = np.where(
            np.abs(local_4) > 1e-12,
            local_4,
            np.where(local_4 < 0.0, -1e-12, 1e-12),
        )
        contrast = np.divide(local_2, denominator) - 1.0
        magnitude = np.abs(contrast)
        return magnitude**3 / (magnitude**2 + 1e-4)

    source_a, source_b, fused_response = [
        perceptual_contrast(image) for image in filtered
    ]

    def similarity(source):
        small = np.minimum(source, fused_response)
        large = np.maximum(source, fused_response)
        result = np.ones_like(large)
        np.divide(small, large, out=result, where=large > 1e-12)
        return result

    quality_a = similarity(source_a)
    quality_b = similarity(source_b)
    energy_a = np.square(source_a)
    energy_b = np.square(source_b)
    denominator = energy_a + energy_b
    valid = denominator > 1e-12
    weight_a = np.where(valid, energy_a / np.maximum(denominator, 1e-12), 0.5)
    weight_b = np.where(valid, energy_b / np.maximum(denominator, 1e-12), 0.5)
    quality = weight_a * quality_a + weight_b * quality_b
    return float(np.clip(np.mean(quality), 0.0, 1.0))


def compute_supplemental_metrics(fused, visible, infrared) -> dict[str, float]:
    fused = to_gray_255(fused)
    visible = to_gray_255(visible)
    infrared = to_gray_255(infrared)
    if fused.shape != visible.shape or fused.shape != infrared.shape:
        raise ValueError("fused, visible and infrared shapes must match")
    return {
        "CC": CC(fused, visible, infrared),
        "Nabf": Nabf(fused, visible, infrared),
        "MS_SSIM": MS_SSIM(fused, visible, infrared),
        "NMI": NMI(fused, visible, infrared),
        "QNCIE": QNCIE(fused, visible, infrared),
        "QY": QY(fused, visible, infrared),
        "QCB": QCB(fused, visible, infrared),
    }


def compute_extended_fusion_metrics(fused, visible, infrared) -> dict[str, float]:
    """Return legacy values plus audited supplementary metrics.

    ``SSIM`` retains the historical sum convention.  ``SSIM_mean`` is emitted
    for cross-paper reporting and is the only SSIM variant used in rankings.
    """
    from utils.metrics import compute_fusion_metrics

    legacy = compute_fusion_metrics(fused, visible, infrared)
    result = dict(legacy)
    result["SSIM_mean"] = 0.5 * float(legacy["SSIM"])
    result.update(compute_supplemental_metrics(fused, visible, infrared))
    nonfinite = [key for key, value in result.items() if not math.isfinite(float(value))]
    if nonfinite:
        raise FloatingPointError(f"non-finite fusion metrics: {nonfinite}")
    return result
