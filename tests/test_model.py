import numpy as np
import pytest

keras = pytest.importorskip("keras")

from cmapss.model import build_hybrid, mc_dropout_predict  # noqa: E402


def test_model_maps_windows_to_one_rul_each_and_mc_dropout_varies():
    keras.utils.set_random_seed(0)
    model = build_hybrid(window=30, n_features=15)
    X = np.random.default_rng(0).normal(size=(8, 30, 15)).astype("float32")
    assert model.predict(X, verbose=0).shape == (8, 1)
    mean, std = mc_dropout_predict(model, X, passes=5, scale=125.0)
    assert mean.shape == std.shape == (8,)
    assert (std > 0).all()   # dropout stays on, so passes differ
    _, compiled_std = mc_dropout_predict(model, X, passes=5, scale=125.0, compiled=True)
    assert (compiled_std > 0).all()   # the traced graph still draws a new mask each pass


def test_saved_model_reloads(tmp_path):
    model = build_hybrid(window=30, n_features=15)
    path = tmp_path / "m.keras"
    model.save(path)
    X = np.zeros((2, 30, 15), "float32")
    reloaded = keras.models.load_model(path)
    np.testing.assert_allclose(reloaded.predict(X, verbose=0), model.predict(X, verbose=0), rtol=1e-5)
