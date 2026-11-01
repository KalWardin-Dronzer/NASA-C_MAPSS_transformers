"""Scoring as in the C-MAPSS literature."""

import numpy as np


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def phm08_score(y_true, y_pred) -> float:
    """The PHM08 challenge score (Saxena et al., 2008): asymmetric, so a late
    prediction (predicted RUL above the truth) costs more than an early one."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def interval_coverage(y_true, mean, std, k: float = 2.0) -> float:
    """Share of true values inside mean ± k·std."""
    y_true, mean, std = (np.asarray(a, float) for a in (y_true, mean, std))
    return float(np.mean(np.abs(y_true - mean) <= k * std))
