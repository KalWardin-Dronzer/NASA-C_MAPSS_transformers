"""
Loading, normalising and windowing the NASA C-MAPSS turbofan data.

Each fix here answers a defect in the original notebook (now in legacy/):

- Engines are keyed by (subset, unit). Unit numbers restart in every subset, and
  the notebook concatenated FD002-FD004 before grouping by unit, so 609 engines
  became 260 and 13.5% of its training windows spliced two engines together.
- Training targets are capped at RUL_CAP cycles (a piecewise-linear target):
  early in life the sensors carry no information about how far away failure is.
- Sensors are standardised within each operating condition, with statistics
  from training engines only. FD002 and FD004 mix six conditions, and the
  condition shifts the raw readings far more than degradation does.
- Every test engine is scored on its last window. Engines shorter than the
  window are padded with their first reading instead of being dropped.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RUL_CAP = 125
WINDOW = 30
SETTINGS = ["setting1", "setting2", "setting3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def read_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, :len(COLUMNS)]
    frame.columns = COLUMNS
    return frame


def load_subset(data_dir: Path, subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and test tables for one subset, with engine keys and RUL targets.

    Training RUL counts down to the engine's last cycle. Test RUL adds the
    official remaining life after the last observed cycle (RUL_FD00x.txt).
    """
    data_dir = Path(data_dir)
    train = read_table(data_dir / f"train_{subset}.txt")
    test = read_table(data_dir / f"test_{subset}.txt")
    for frame in (train, test):
        frame.insert(0, "engine", subset + "-" + frame["unit"].astype(str))
    train["rul"] = train.groupby("engine")["cycle"].transform("max") - train["cycle"]
    final = np.loadtxt(data_dir / f"RUL_{subset}.txt", ndmin=1)
    units = np.sort(test["unit"].unique())
    if len(final) != len(units):
        raise ValueError(f"{subset}: {len(final)} RUL values for {len(units)} test engines")
    after_last = dict(zip(units, final))
    last_cycle = test.groupby("unit")["cycle"].transform("max")
    test["rul"] = test["unit"].map(after_last) + last_cycle - test["cycle"]
    return train, test


def condition_of(frame: pd.DataFrame) -> pd.Series:
    """Operating condition from the three settings, rounded to the levels the simulator used."""
    # abs() folds the -0.0 that noise around zero rounds to into 0.0.
    return (frame["setting1"].round(0).astype(int).astype(str) + "/"
            + frame["setting2"].round(2).abs().astype(str) + "/"
            + frame["setting3"].round(0).astype(int).astype(str))


@dataclass
class Normaliser:
    """Per-condition standardisation fitted on training engines only."""
    means: pd.DataFrame
    stds: pd.DataFrame
    sensors: list[str]
    conditions: list[str]

    @classmethod
    def fit(cls, train: pd.DataFrame, min_std: float = 1e-4) -> "Normaliser":
        keyed = train.assign(condition=condition_of(train))
        means = keyed.groupby("condition")[SENSORS].mean()
        stds = keyed.groupby("condition")[SENSORS].std(ddof=0)
        # A sensor that is constant within every condition carries no signal.
        sensors = [s for s in SENSORS if (stds[s] > min_std).any()]
        return cls(means, stds.replace(0, 1.0), sensors, sorted(means.index))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        condition = condition_of(frame)
        unknown = set(condition) - set(self.conditions)
        if unknown:
            raise ValueError(f"conditions not seen in training: {sorted(unknown)}")
        out = frame[["engine", "cycle", "rul"]].copy()
        out[self.sensors] = ((frame[self.sensors] - self.means.loc[condition, self.sensors].values)
                             / self.stds.loc[condition, self.sensors].values)
        if len(self.conditions) > 1:
            for name in self.conditions:
                out[f"cond_{name}"] = (condition == name).astype(float).values
        return out

    @property
    def features(self) -> list[str]:
        extra = [f"cond_{c}" for c in self.conditions] if len(self.conditions) > 1 else []
        return self.sensors + extra


def _pad(values: np.ndarray, window: int) -> np.ndarray:
    """Repeat the first reading so a short history still fills a window."""
    if len(values) >= window:
        return values
    return np.concatenate([np.repeat(values[:1], window - len(values), axis=0), values])


def training_windows(frame: pd.DataFrame, features: list[str], window: int = WINDOW,
                     cap: int = RUL_CAP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every window of every engine, built engine by engine so none spans two engines.

    Returns (X, y, engine of each window); y is the capped RUL at the window's last cycle.
    """
    xs, ys, engines = [], [], []
    for engine, rows in frame.sort_values(["engine", "cycle"]).groupby("engine", sort=True):
        values = _pad(rows[features].to_numpy(np.float32), window)
        rul = np.minimum(_pad(rows["rul"].to_numpy(np.float32)[:, None], window)[:, 0], cap)
        for end in range(window, len(values) + 1):
            xs.append(values[end - window:end])
            ys.append(rul[end - 1])
            engines.append(engine)
    return np.stack(xs), np.asarray(ys, np.float32), np.asarray(engines)


def last_windows(frame: pd.DataFrame, features: list[str], window: int = WINDOW) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One window per engine, ending at its last observed cycle, with the true (uncapped) RUL there."""
    xs, ys, engines = [], [], []
    for engine, rows in frame.sort_values(["engine", "cycle"]).groupby("engine", sort=True):
        values = _pad(rows[features].to_numpy(np.float32), window)
        xs.append(values[-window:])
        ys.append(float(rows["rul"].iloc[-1]))
        engines.append(engine)
    return np.stack(xs), np.asarray(ys, np.float32), np.asarray(engines)


def split_engines(engines, fraction: float, seed: int) -> tuple[set, set]:
    """Split engine keys, not windows, so no engine contributes to both sides."""
    unique = np.array(sorted(set(engines)))
    rng = np.random.default_rng(seed)
    held = set(rng.choice(unique, size=max(1, round(fraction * len(unique))), replace=False))
    return set(unique) - held, held
