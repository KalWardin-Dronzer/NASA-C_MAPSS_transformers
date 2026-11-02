import math

import pytest

import numpy as np

from cmapss.metrics import conformal_scale, interval_coverage, phm08_score, rmse


def test_phm08_penalises_late_predictions_more():
    early = phm08_score([50], [40])     # predicted 10 cycles early
    late = phm08_score([50], [60])      # predicted 10 cycles late
    assert early == pytest.approx(math.exp(10 / 13) - 1)
    assert late == pytest.approx(math.exp(1) - 1)
    assert late > early


def test_rmse_and_coverage():
    assert rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))
    assert interval_coverage([0, 10], [1, 0], [1, 1], k=2) == 0.5


def test_conformal_scale_covers_fresh_points_at_the_target_rate():
    rng = np.random.default_rng(0)
    std = rng.uniform(1, 5, size=4000)          # the model's spread varies by point
    errors = rng.normal(scale=std)              # true errors scale with it
    q = conformal_scale(errors[:2000], np.zeros(2000), std[:2000], alpha=0.05)
    assert q == pytest.approx(1.96, abs=0.15)
    assert interval_coverage(errors[2000:], np.zeros(2000), std[2000:], k=q) == pytest.approx(0.95, abs=0.015)


def test_conformal_scale_takes_the_largest_score_when_the_set_is_small():
    # n = 19 at alpha 0.05 needs the 20/19 quantile, capped at the maximum.
    assert conformal_scale(np.arange(19.0), np.zeros(19), np.ones(19), alpha=0.05) == 18.0
