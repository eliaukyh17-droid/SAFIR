import math

import numpy as np

from utils.metrics_a37_supplemental import (
    METRIC_DIRECTIONS,
    METRIC_FAMILIES,
    SUPPLEMENTAL_METRICS,
    compute_a37_supplemental_metrics,
    cross_entropy,
    edge_intensity,
    source_mse,
    source_nrmse,
    source_psnr,
)


def _images(size=96):
    yy, xx = np.mgrid[:size, :size]
    visible = np.clip(20.0 + 1.4 * xx + 18.0 * np.sin(yy / 8.0), 0, 255)
    infrared = np.clip(35.0 + 1.1 * yy + 25.0 * (xx > size // 2), 0, 255)
    fused = 0.6 * visible + 0.4 * infrared
    return fused, visible, infrared


def test_protocol_has_twelve_unique_supplemental_metrics_and_fixed_directions():
    assert len(SUPPLEMENTAL_METRICS) == 12
    assert len(set(SUPPLEMENTAL_METRICS)) == 12
    assert set(METRIC_DIRECTIONS) == set(SUPPLEMENTAL_METRICS)
    assert set(METRIC_FAMILIES) == set(SUPPLEMENTAL_METRICS)
    assert {
        metric for metric, direction in METRIC_DIRECTIONS.items()
        if direction == "lower"
    } == {"Nabf", "Source_MSE", "Source_NRMSE", "CE"}


def test_all_metrics_are_finite_and_source_symmetric():
    fused, visible, infrared = _images()
    forward = compute_a37_supplemental_metrics(fused, visible, infrared)
    reverse = compute_a37_supplemental_metrics(fused, infrared, visible)
    assert tuple(forward) == SUPPLEMENTAL_METRICS
    for metric, value in forward.items():
        assert math.isfinite(value), (metric, value)
        assert math.isclose(value, reverse[metric], rel_tol=1e-9, abs_tol=1e-9)


def test_source_error_metrics_use_locked_two_source_convention():
    fused = np.full((16, 16), 10.0)
    visible = np.full((16, 16), 10.0)
    infrared = np.full((16, 16), 20.0)
    assert math.isclose(source_mse(fused, visible, infrared), 50.0)
    assert math.isclose(source_nrmse(fused, visible, infrared), 0.25)
    expected_psnr = 10.0 * math.log10(255.0**2 / 50.0)
    assert math.isclose(source_psnr(fused, visible, infrared), expected_psnr)


def test_edge_intensity_is_zero_for_constant_and_positive_for_edge():
    constant = np.full((32, 32), 100.0)
    edge = constant.copy()
    edge[:, 16:] = 200.0
    assert math.isclose(edge_intensity(constant), 0.0, abs_tol=1e-12)
    assert edge_intensity(edge) > 0.0


def test_cross_entropy_identity_is_zero_and_constant_cases_are_finite():
    image = np.tile(np.arange(32, dtype=np.float64), (32, 1))
    assert math.isclose(cross_entropy(image, image, image), 0.0, abs_tol=1e-12)
    fused = np.full((32, 32), 100.0)
    visible = np.full((32, 32), 100.0)
    infrared = np.full((32, 32), 150.0)
    values = compute_a37_supplemental_metrics(fused, visible, infrared)
    assert all(math.isfinite(value) for value in values.values())
