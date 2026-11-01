import math

import pytest

from cmapss.metrics import interval_coverage, phm08_score, rmse


def test_phm08_penalises_late_predictions_more():
    early = phm08_score([50], [40])     # predicted 10 cycles early
    late = phm08_score([50], [60])      # predicted 10 cycles late
    assert early == pytest.approx(math.exp(10 / 13) - 1)
    assert late == pytest.approx(math.exp(1) - 1)
    assert late > early


def test_rmse_and_coverage():
    assert rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))
    assert interval_coverage([0, 10], [1, 0], [1, 1], k=2) == 0.5
