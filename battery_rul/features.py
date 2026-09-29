"""Sliding windows, tree features and training sets.

Information parity is the design rule here. Every model family receives the same
``L`` past capacity values:

* DLinear / NLinear / Linear read the raw window.
* Tree models read the *same* window, summarised as features. No feature uses
  anything outside the window - in particular no cycle index, which would hand
  the trees the cell's age for free.

Targets are the next ``H`` values of the causal capacity series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LABELS, LINEAR_TRAIN, RANGE_RESTRICT_SOH
from .dataset import Cell


def make_windows(series: np.ndarray, input_len: int, horizon: int, stride: int = 1
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All (X, Y) pairs from one series.

    ``X[i] = series[t-L+1 : t+1]`` and ``Y[i] = series[t+1 : t+H+1]`` where
    ``t = ends[i]`` is the 0-based index of the last input value, so the last
    observed cycle is ``ends[i] + 1``.
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series) - input_len - horizon + 1
    if n <= 0:
        return (np.empty((0, input_len)), np.empty((0, horizon)), np.empty(0, dtype=int))
    ends = np.arange(input_len - 1, input_len - 1 + n, stride)
    X = np.stack([series[t - input_len + 1: t + 1] for t in ends])
    Y = np.stack([series[t + 1: t + horizon + 1] for t in ends])
    return X, Y, ends


def window_at(series: np.ndarray, t0: int, input_len: int) -> np.ndarray:
    """The input window for a forecast at cycle ``t0`` (1-based): cycles t0-L+1..t0."""
    if t0 < input_len:
        raise ValueError(f"origin t0={t0} has fewer than L={input_len} observed cycles")
    return np.asarray(series[t0 - input_len: t0], dtype=np.float64)


# --------------------------------------------------------------------------
# Tree features
# --------------------------------------------------------------------------

_LAGS = (2, 3, 5, 10, 20)
_ROLLS = (5, 10, 20, 40)


def _slope(block: np.ndarray) -> np.ndarray:
    w = block.shape[1]
    t = np.arange(w) - (w - 1) / 2
    return (block * t).sum(axis=1) / float((t ** 2).sum())


def window_features(X: np.ndarray, level_relative: bool = False) -> pd.DataFrame:
    """Describe each window with level and shape features.

    ``level_relative=False`` keeps level features in Ah (standard practice).
    ``level_relative=True`` expresses them relative to the last observed value,
    so the features carry no absolute level - the tree analogue of NLinear.
    Shape features (slopes, spread, differences) are level-free either way.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or len(X) == 0:
        return pd.DataFrame()
    L = X.shape[1]
    last = X[:, -1]
    d = np.diff(X, axis=1)

    level = {
        "cap_first": X[:, 0], "cap_mean": X.mean(axis=1), "cap_median": np.median(X, axis=1),
        "cap_min": X.min(axis=1), "cap_max": X.max(axis=1),
    }
    for lag in _LAGS:
        if lag <= L:
            level[f"lag_{lag}"] = X[:, -lag]
    for w in _ROLLS:
        if w <= L:
            level[f"roll_mean_{w}"] = X[:, -w:].mean(axis=1)

    if level_relative:
        level = {f"{k}_rel": v - last for k, v in level.items()}
        feats = dict(level)
    else:
        feats = {"cap_last": last, **level}

    t = np.arange(L, dtype=np.float64)
    feats.update({
        "slope": _slope(X),
        "curvature": np.polyfit(t, X.T, 2)[0] if L >= 3 else np.zeros(len(X)),
        "cap_std": X.std(axis=1),
        "cap_range": X.max(axis=1) - X.min(axis=1),
        "fade_total": X[:, 0] - last,
        "diff_mean": d.mean(axis=1), "diff_std": d.std(axis=1),
        "diff_min": d.min(axis=1), "diff_max": d.max(axis=1),
        "n_increases": (d > 0).sum(axis=1).astype(float),
    })
    for w in _ROLLS:
        if w <= L:
            feats[f"roll_std_{w}"] = X[:, -w:].std(axis=1)
            if w >= 3:
                feats[f"roll_slope_{w}"] = _slope(X[:, -w:])
    return pd.DataFrame(feats)


def make_targets(X: np.ndarray, Y: np.ndarray, mode: str) -> np.ndarray:
    """'absolute' -> future capacity in Ah; 'delta' -> change from the last value."""
    if mode == "absolute":
        return Y
    if mode == "delta":
        return Y - X[:, -1:]
    raise ValueError(f"unknown target mode {mode!r}")


def invert_targets(X: np.ndarray, T: np.ndarray, mode: str) -> np.ndarray:
    return T if mode == "absolute" else T + X[:, -1:]


# --------------------------------------------------------------------------
# Training sets
# --------------------------------------------------------------------------

@dataclass
class TrainingSet:
    """Windows from a fold's training cells, split chronologically per cell."""

    X_fit: np.ndarray
    Y_fit: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    fit_cells: np.ndarray           # cell name of each fit window
    val_cells: np.ndarray
    n_removed_by_range: int = 0

    @property
    def X_all(self) -> np.ndarray:
        return np.concatenate([self.X_fit, self.X_val])

    @property
    def Y_all(self) -> np.ndarray:
        return np.concatenate([self.Y_fit, self.Y_val])

    @property
    def n_windows(self) -> int:
        return len(self.X_fit) + len(self.X_val)

    def target_range(self) -> tuple[float, float]:
        y = self.Y_all
        return float(y.min()), float(y.max())


def build_training_set(train_cells: list[Cell], input_len: int, horizon: int,
                       val_frac: float = LINEAR_TRAIN.val_frac, stride: int = 1,
                       range_restrict_soh: float | None = None,
                       reference: str = LABELS.primary_reference) -> TrainingSet:
    """Stack windows from every training cell.

    The last ``val_frac`` of each cell's windows (chronologically) form the
    validation part, used by the linear models for early stopping only.

    ``range_restrict_soh`` (RQ2 extrapolation arm) removes every window whose
    targets dip below that state of health, so the end-of-life region never
    appears in training.
    """
    parts = {"X_fit": [], "Y_fit": [], "X_val": [], "Y_val": [], "fit_cells": [], "val_cells": []}
    removed = 0
    for cell in train_cells:
        X, Y, _ = make_windows(cell.capacity, input_len, horizon, stride)
        if len(X) == 0:
            continue
        if range_restrict_soh is not None:
            keep = Y.min(axis=1) >= range_restrict_soh * cell.references[reference]
            removed += int((~keep).sum())
            X, Y = X[keep], Y[keep]
            if len(X) == 0:
                continue
        n_val = int(round(val_frac * len(X)))
        n_val = min(max(n_val, 1), len(X) - 1) if len(X) > 1 else 0
        cut = len(X) - n_val
        parts["X_fit"].append(X[:cut]); parts["Y_fit"].append(Y[:cut])
        parts["X_val"].append(X[cut:]); parts["Y_val"].append(Y[cut:])
        parts["fit_cells"].append(np.full(cut, cell.name)); parts["val_cells"].append(np.full(n_val, cell.name))

    if not parts["X_fit"]:
        raise ValueError(f"no training windows for L={input_len}, H={horizon}")

    def cat(key, width):
        arrs = [a for a in parts[key] if len(a)]
        return np.concatenate(arrs) if arrs else np.empty((0, width))

    return TrainingSet(
        X_fit=cat("X_fit", input_len), Y_fit=cat("Y_fit", horizon),
        X_val=cat("X_val", input_len), Y_val=cat("Y_val", horizon),
        fit_cells=np.concatenate(parts["fit_cells"]),
        val_cells=np.concatenate([a for a in parts["val_cells"]]) if parts["val_cells"] else np.empty(0),
        n_removed_by_range=removed,
    )


class Standardizer:
    """Mean/std scaling fitted on training values only."""

    def __init__(self) -> None:
        self.mean_: float = 0.0
        self.std_: float = 1.0

    def fit(self, values: np.ndarray) -> "Standardizer":
        values = np.asarray(values, dtype=np.float64)
        self.mean_ = float(values.mean())
        std = float(values.std())
        self.std_ = std if std > 1e-12 else 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean_) / self.std_

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.std_ + self.mean_
