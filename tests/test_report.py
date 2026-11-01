import pytest

from cmapss.report import phm08_capped_truth, select


def _run(val_rmse, test_rmse):
    return {"training": {"val_rmse": val_rmse}, "test": {"hybrid": {"rmse": test_rmse}}}


def test_learning_rate_is_chosen_by_validation_never_by_test():
    by_lr = {1e-3: [_run(15.0, 30.0), _run(16.0, 30.0)],    # better on validation, worse on test
             1e-4: [_run(17.0, 20.0), _run(18.0, 20.0)]}
    assert select(by_lr) == 1e-3


def test_phm08_capped_truth_caps_the_official_life_at_125():
    result = {"per_engine": {"true": [200.0, 50.0], "pred": [125.0, 50.0]}}
    assert phm08_capped_truth(result) == pytest.approx(0.0)
