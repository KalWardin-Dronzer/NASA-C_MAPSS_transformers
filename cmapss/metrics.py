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


def conformal_scale(y_true, pred, std, alpha: float = 0.05) -> float:
    """Split-conformal multiplier q from a calibration set: pred ± q·std then covers about 1 - alpha of new points
    drawn like the calibration ones. The score is the error in units of the model's own spread, so intervals stay
    wider where the model is less sure."""
    scores = np.abs(np.asarray(y_true, float) - np.asarray(pred, float)) / np.maximum(np.asarray(std, float), 1e-8)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def interval_coverage(y_true, mean, std, k: float = 2.0) -> float:
    """Share of true values inside mean ± k·std."""
    y_true, mean, std = (np.asarray(a, float) for a in (y_true, mean, std))
    return float(np.mean(np.abs(y_true - mean) <= k * std))
