"""Naive baselines, forecasting the same horizon from the same window.

Every learned model must beat these to claim skill:

* **Persistence** repeats the last observed capacity. It never reaches the EOL
  threshold, so its RUL is always censored - which is itself informative about
  why RUL needs a trend model.
* **LinearTrend** fits a straight line to the window and extends it. This is the
  classical threshold-crossing baseline and the one that matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PersistenceForecaster:
    input_len: int
    horizon: int
    name: str = "Persistence"
    n_params: int = 0

    def predict_block(self, windows: np.ndarray) -> np.ndarray:
        windows = np.atleast_2d(windows)
        return np.repeat(windows[:, -1:], self.horizon, axis=1)


@dataclass
class LinearTrendForecaster:
    input_len: int
    horizon: int
    name: str = "LinearTrend"
    n_params: int = 0

    def predict_block(self, windows: np.ndarray) -> np.ndarray:
        windows = np.atleast_2d(np.asarray(windows, dtype=np.float64))
        L = windows.shape[1]
        t = np.arange(L, dtype=np.float64)
        tc = t - t.mean()
        slope = (windows * tc).sum(axis=1) / float((tc ** 2).sum())
        intercept = windows.mean(axis=1) - slope * t.mean()
        future = np.arange(L, L + self.horizon, dtype=np.float64)
        return intercept[:, None] + slope[:, None] * future[None, :]
