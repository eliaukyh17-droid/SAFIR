import numpy as np
import pytest

from utils.metric_protocol import bootstrap_mean_ci, directed_difference, holm_adjust


def test_directed_difference_normalizes_metric_direction():
    assert np.allclose(directed_difference([2, 4], [1, 3], "higher"), [1, 1])
    assert np.allclose(directed_difference([1, 3], [2, 4], "lower"), [1, 1])
    with pytest.raises(ValueError):
        directed_difference([1], [1], "sideways")


def test_holm_adjust_matches_known_example_and_is_order_invariant():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])
    permuted = holm_adjust([0.03, 0.01, 0.04])
    assert np.allclose(permuted, [0.06, 0.03, 0.06])


def test_bootstrap_ci_is_deterministic_and_contains_constant_mean():
    first = bootstrap_mean_ci([2.0] * 10, resamples=100, seed=7)
    second = bootstrap_mean_ci([2.0] * 10, resamples=100, seed=7)
    assert first == second == (2.0, 2.0)
