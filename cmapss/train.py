"""
Train and evaluate one model on one C-MAPSS subset.

    python -m cmapss.train --subset FD002 --seed 1

Validation engines are held out whole, and the normaliser is fitted on the
training engines only. The official test set is scored once, on each engine's
last window, against the official remaining life (uncapped). Results go to
results/runs/<subset>_lr<rate>_seed<seed>.json.

Uncertainty is calibrated by split conformal prediction on the validation
engines: each validation window's score is |target - prediction| / MC-dropout
std, q is the conformal 95% quantile of those scores, and each test interval is
prediction ± q·std. Validation targets are capped at RUL_CAP, so the intervals
aim to cover the capped remaining life. This method was fixed, and committed,
before any calibrated test coverage was computed.
"""

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from cmapss.data import (RUL_CAP, WINDOW, Normaliser, last_windows, load_subset, split_engines,
                         training_windows)
from cmapss.metrics import conformal_scale, interval_coverage, mae, phm08_score, rmse

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CMAPSS_DATA", ROOT / "data" / "CMAPSSData"))
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "artifacts"
VAL_FRACTION = 0.2
CONFORMAL_ALPHA = 0.05


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def window_features(X: np.ndarray) -> np.ndarray:
    """Per-feature mean, last value and least-squares slope over each window, for the baseline."""
    t = np.arange(X.shape[1], dtype=np.float32)
    t = (t - t.mean()) / (np.square(t - t.mean()).sum())
    slope = np.einsum("t,ntf->nf", t, X)
    return np.concatenate([X.mean(axis=1), X[:, -1, :], slope], axis=1)


def scores(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "rmse_capped_truth": rmse(np.minimum(y_true, RUL_CAP), y_pred),
        "mae": mae(y_true, y_pred),
        "phm08_score": phm08_score(y_true, y_pred),
    }


def result_path(subset: str, learning_rate: float, seed: int) -> Path:
    return RESULTS_DIR / "runs" / f"{subset}_lr{learning_rate:g}_seed{seed}.json"


def run(subset: str, seed: int, epochs: int = 80, batch_size: int = 128, learning_rate: float = 1e-3,
        data_dir: Path = DATA_DIR, mc_passes: int = 50, verbose: int = 2) -> dict:
    import keras
    import tensorflow as tf
    from sklearn.linear_model import Ridge

    from cmapss.model import build_hybrid, mc_dropout_predict

    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    train_raw, test_raw = load_subset(data_dir, subset)
    fit_engines, val_engines = split_engines(train_raw["engine"], VAL_FRACTION, seed)
    normaliser = Normaliser.fit(train_raw[train_raw["engine"].isin(fit_engines)])
    features = normaliser.features
    train = normaliser.transform(train_raw)
    test = normaliser.transform(test_raw)

    X, y, engines = training_windows(train, features)
    in_val = np.isin(engines, list(val_engines))
    X_fit, y_fit, X_val, y_val = X[~in_val], y[~in_val], X[in_val], y[in_val]
    X_test, y_test, test_engines = last_windows(test, features)

    model = build_hybrid(WINDOW, len(features), learning_rate=learning_rate)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]
    started = time.time()
    history = model.fit(X_fit, y_fit / RUL_CAP, validation_data=(X_val, y_val / RUL_CAP), epochs=epochs,
                        batch_size=batch_size, callbacks=callbacks, verbose=verbose)
    train_seconds = time.time() - started

    def predict(X_):
        return np.clip(model.predict(X_, batch_size=1024, verbose=0).ravel() * RUL_CAP, 0, None)

    pred_test = predict(X_test)
    mc_mean, mc_std = mc_dropout_predict(model, X_test, passes=mc_passes, scale=RUL_CAP)
    # After the test passes, so the test draws match runs made before calibration existed.
    pred_val = predict(X_val)
    _, val_std = mc_dropout_predict(model, X_val, passes=mc_passes, scale=RUL_CAP, batch_size=2048, compiled=True)
    q = conformal_scale(y_val, pred_val, val_std, CONFORMAL_ALPHA)

    baseline = Ridge(alpha=1.0).fit(window_features(X_fit), y_fit)
    baseline_test = np.clip(baseline.predict(window_features(X_test)), 0, None)
    baseline_val = baseline.predict(window_features(X_val))

    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / f"{subset}_lr{learning_rate:g}_seed{seed}.keras")

    val_loss = history.history["val_loss"]
    best_epoch = int(np.argmin(val_loss)) + 1
    result = {
        "subset": subset,
        "seed": seed,
        "config": {"window": WINDOW, "rul_cap": RUL_CAP, "val_fraction": VAL_FRACTION, "batch_size": batch_size,
                   "max_epochs": epochs, "learning_rate": learning_rate, "mc_passes": mc_passes, "features": features,
                   "conditions": normaliser.conditions},
        "data": {"train_engines": len(fit_engines), "val_engines": len(val_engines),
                 "test_engines": int(len(test_engines)), "train_windows": int(len(X_fit)),
                 "val_windows": int(len(X_val)),
                 "test_engines_shorter_than_window": int((test_raw.groupby("engine")["cycle"].max() < WINDOW).sum())},
        "training": {"epochs_run": len(val_loss), "best_epoch": best_epoch, "seconds": round(train_seconds, 1),
                     "val_rmse": float(np.sqrt(val_loss[best_epoch - 1])) * RUL_CAP},
        "test": {
            "hybrid": scores(y_test, pred_test),
            "hybrid_mc_mean": scores(y_test, mc_mean),
            "ridge_baseline": scores(y_test, baseline_test),
            "mc_coverage_2sd": interval_coverage(y_test, mc_mean, mc_std, 2.0),
            "mc_coverage_2sd_capped_truth": interval_coverage(np.minimum(y_test, RUL_CAP), mc_mean, mc_std, 2.0),
            "mc_mean_std": float(mc_std.mean()),
            "conformal_coverage": interval_coverage(y_test, pred_test, mc_std, q),
            "conformal_coverage_capped_truth": interval_coverage(np.minimum(y_test, RUL_CAP), pred_test, mc_std, q),
            "conformal_mean_halfwidth": float(q * mc_std.mean()),
        },
        "calibration": {"method": "split conformal on validation windows, score |target - prediction| / mc_std",
                        "alpha": CONFORMAL_ALPHA, "scale": q, "windows": int(len(X_val)),
                        "validation_coverage": interval_coverage(y_val, pred_val, val_std, q)},
        "validation": {"ridge_baseline_rmse": rmse(y_val, baseline_val)},
        "per_engine": {"engine": test_engines.tolist(), "true": y_test.tolist(), "pred": pred_test.round(2).tolist(),
                       "mc_std": mc_std.round(2).tolist()},
        "environment": {"python": platform.python_version(), "tensorflow": tf.__version__,
                        "keras": keras.__version__, "commit": _commit()},
    }
    path = result_path(subset, learning_rate, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", required=True, choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    result = run(args.subset, args.seed, epochs=args.epochs, batch_size=args.batch_size,
                 learning_rate=args.learning_rate)
    test = result["test"]
    print(f"{args.subset} seed {args.seed}: test RMSE {test['hybrid']['rmse']:.2f} "
          f"(capped truth {test['hybrid']['rmse_capped_truth']:.2f}), PHM08 {test['hybrid']['phm08_score']:.0f}; "
          f"ridge baseline RMSE {test['ridge_baseline']['rmse']:.2f}; MC ±2σ coverage {test['mc_coverage_2sd']:.0%}; "
          f"conformal 95% coverage {test['conformal_coverage_capped_truth']:.0%} (capped truth); "
          f"{result['training']['epochs_run']} epochs in {result['training']['seconds']:.0f}s")


if __name__ == "__main__":
    main()
