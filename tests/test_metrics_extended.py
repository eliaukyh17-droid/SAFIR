import math

import numpy as np

from utils.metrics_extended import (
    METRIC_DIRECTIONS,
    RANKED_METRICS,
    compute_supplemental_metrics,
    to_gray_255,
)


def _images(size=192):
    yy, xx = np.mgrid[:size, :size]
    visible = np.clip(30 + 0.85 * xx + 20 * np.sin(yy / 11), 0, 255)
    infrared = np.clip(40 + 0.75 * yy + 30 * (xx > size // 2), 0, 255)
    fused = 0.55 * visible + 0.45 * infrared
    return fused, visible, infrared


def test_protocol_contains_sixteen_nonduplicated_ranked_metrics():
    assert len(RANKED_METRICS) == 16
    assert len(set(RANKED_METRICS)) == 16
    assert set(METRIC_DIRECTIONS) == set(RANKED_METRICS)
    assert METRIC_DIRECTIONS["Nabf"] == "lower"
    assert all(
        direction == "higher"
        for metric, direction in METRIC_DIRECTIONS.items()
        if metric != "Nabf"
    )


def test_tensor_style_ranges_and_channel_layouts_normalize_consistently():
    base = np.linspace(0.0, 1.0, 35).reshape(5, 7)
    expected = base * 255.0
    assert np.allclose(to_gray_255(base), expected)
    assert np.allclose(to_gray_255(base[None, None]), expected)
    rgb = np.stack((base, base, base), axis=0)
    assert np.allclose(to_gray_255(rgb), expected)
    tanh = 2.0 * base - 1.0
    assert np.allclose(to_gray_255(tanh), expected)


def test_supplemental_metrics_are_source_symmetric_and_finite():
    fused, visible, infrared = _images()
    forward = compute_supplemental_metrics(fused, visible, infrared)
    reverse = compute_supplemental_metrics(fused, infrared, visible)
    assert set(forward) == {"CC", "Nabf", "MS_SSIM", "NMI", "QNCIE", "QY", "QCB"}
    for metric, value in forward.items():
        assert math.isfinite(value), (metric, value)
        assert math.isclose(value, reverse[metric], rel_tol=1e-9, abs_tol=1e-9)


def test_identity_case_has_expected_bounded_quality():
    _, visible, _ = _images()
    values = compute_supplemental_metrics(visible, visible, visible)
    assert math.isclose(values["CC"], 1.0, abs_tol=1e-12)
    assert math.isclose(values["MS_SSIM"], 1.0, abs_tol=1e-10)
    assert math.isclose(values["QY"], 1.0, abs_tol=1e-10)
    assert 0.0 <= values["Nabf"] <= 1.0
    assert 0.0 <= values["QCB"] <= 1.0


def test_constant_images_never_produce_nan_or_infinity():
    fused = np.full((64, 64), 100.0)
    visible = np.full((64, 64), 100.0)
    infrared = np.full((64, 64), 150.0)
    with np.errstate(divide="raise", invalid="raise"):
        values = compute_supplemental_metrics(fused, visible, infrared)
    assert all(math.isfinite(value) for value in values.values())
