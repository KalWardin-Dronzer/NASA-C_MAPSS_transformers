"""
Each test pins one fix to the original notebook (legacy/), on small synthetic
tables so the suite runs without the dataset.
"""

import numpy as np
import pandas as pd
import pytest

from cmapss.data import (RUL_CAP, SENSORS, Normaliser, condition_of, last_windows, load_subset, split_engines,
                         training_windows)


def engine_rows(engine: str, cycles: int, sensor_value: float = 0.0, setting1: float = 0.0) -> pd.DataFrame:
    rows = pd.DataFrame({"engine": engine, "unit": int(engine.split("-")[1]), "cycle": np.arange(1, cycles + 1)})
    rows["setting1"], rows["setting2"], rows["setting3"] = setting1, 0.0, 100.0
    for i, s in enumerate(SENSORS):
        rows[s] = sensor_value + i + np.arange(cycles) * 0.01
    rows["rul"] = cycles - rows["cycle"]
    return rows


def write_subset(tmp_path, subset: str, train_units: dict, test_units: dict, final_rul: list) -> None:
    def write(name, units):
        rows = []
        for unit, cycles in units.items():
            for c in range(1, cycles + 1):
                rows.append([unit, c, 0.0, 0.0, 100.0] + [float(unit)] * len(SENSORS))
        np.savetxt(tmp_path / name, np.array(rows), fmt="%g")
    write(f"train_{subset}.txt", train_units)
    write(f"test_{subset}.txt", test_units)
    np.savetxt(tmp_path / f"RUL_{subset}.txt", np.array(final_rul), fmt="%g")


def test_engines_are_keyed_by_subset_so_unit_numbers_never_merge(tmp_path):
    """The notebook merged unit 1 of FD002, FD003 and FD004 into one engine."""
    write_subset(tmp_path, "FD002", {1: 40, 2: 40}, {1: 10}, [5])
    write_subset(tmp_path, "FD003", {1: 40}, {1: 10}, [7])
    a, _ = load_subset(tmp_path, "FD002")
    b, _ = load_subset(tmp_path, "FD003")
    both = pd.concat([a, b])
    assert both["engine"].nunique() == 3
    assert set(both["engine"]) == {"FD002-1", "FD002-2", "FD003-1"}


def test_windows_never_span_two_engines():
    frame = pd.concat([engine_rows("FD002-1", 40, 0.0), engine_rows("FD002-2", 40, 100.0)])
    X, y, engines = training_windows(frame, SENSORS, window=30)
    assert len(X) == 2 * (40 - 30 + 1)
    for window, engine in zip(X, engines):
        expected = 0.0 if engine == "FD002-1" else 100.0
        assert window[:, 0].min() >= expected and window[:, 0].max() < expected + 1


def test_training_targets_are_capped():
    X, y, _ = training_windows(engine_rows("FD001-1", 300), SENSORS, window=30)
    assert y.max() == RUL_CAP
    assert y.min() == 0


def test_split_is_by_engine():
    engines = [f"FD002-{i}" for i in range(1, 101) for _ in range(5)]
    fit, held = split_engines(engines, 0.2, seed=3)
    assert len(held) == 20 and not fit & held and fit | held == set(engines)


def test_every_test_engine_gets_one_window_even_when_short():
    """The notebook dropped test engines shorter than its window; 47 of them in FD002-FD004."""
    frame = pd.concat([engine_rows("FD002-1", 12), engine_rows("FD002-2", 80)])
    frame["rul"] = frame["rul"] + 20
    X, y, engines = last_windows(frame, SENSORS, window=30)
    assert list(engines) == ["FD002-1", "FD002-2"]
    assert X.shape == (2, 30, len(SENSORS))
    # The short engine is padded with its first reading and ends on its last one.
    assert X[0, 0, 0] == X[0, 18, 0] and X[0, -1, 0] == pytest.approx(0.11)
    assert list(y) == [20, 20]


def test_test_rul_adds_the_official_remaining_life(tmp_path):
    write_subset(tmp_path, "FD001", {1: 40}, {1: 10, 2: 15}, [30, 7])
    _, test = load_subset(tmp_path, "FD001")
    last = test.sort_values("cycle").groupby("engine").tail(1).set_index("engine")["rul"]
    assert last["FD001-1"] == 30 and last["FD001-2"] == 7
    first = test[test["cycle"] == 1].set_index("engine")["rul"]
    assert first["FD001-1"] == 30 + 9


def test_normaliser_uses_training_statistics_per_condition():
    train = pd.concat([engine_rows("FD002-1", 50, 0.0, setting1=0.0), engine_rows("FD002-2", 50, 500.0, setting1=42.0)])
    norm = Normaliser.fit(train)
    assert len(norm.conditions) == 2
    out = norm.transform(train)
    # Within each condition every sensor is centred on its own mean, so the big offset disappears.
    for cond in norm.conditions:
        rows = out[out[f"cond_{cond}"] == 1.0]
        assert abs(rows[norm.sensors].to_numpy().mean()) < 1e-6
    shifted = engine_rows("FD002-9", 10, 500.0, setting1=42.0)
    assert norm.transform(shifted)[norm.sensors].to_numpy().mean() < 0   # early cycles sit below the training mean


def test_unknown_condition_is_an_error():
    norm = Normaliser.fit(engine_rows("FD001-1", 50))
    with pytest.raises(ValueError):
        norm.transform(engine_rows("FD001-2", 10, setting1=35.0))


def test_noise_around_zero_is_one_condition():
    frame = engine_rows("FD001-1", 4)
    frame["setting2"] = [0.0004, -0.0003, 0.0, -0.0001]
    assert condition_of(frame).nunique() == 1
