"""Roll-out forecasting and threshold-crossing RUL - identical for every model.

From a forecast origin ``t0`` a model sees the window of cycles ``t0-L+1..t0``,
predicts ``H`` cycles, appends them to its own history and repeats. The
predicted end of life is the first point where the forecast stays at or below
the EOL threshold for ``persist`` consecutive cycles (the same rule that defines
the true EOL). Predicted RUL = predicted EOL - t0.

Using one roll-out engine for all models is what guarantees horizon parity:
no model ever sees a capacity value after ``t0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .cleaning import first_sustained_crossing


class BlockForecaster(Protocol):
    input_len: int
    horizon: int

    def predict_block(self, windows: np.ndarray) -> np.ndarray: ...


@dataclass
class RolloutResult:
    trajectories: np.ndarray        # (n, steps) forecast capacity after t0, in Ah
    crossing: list[int | None]      # 1-based step of predicted EOL, None if censored
    cap: int                        # maximum roll-out length

    def predicted_rul(self, i: int) -> int:
        k = self.crossing[i]
        return self.cap if k is None else int(k)

    def censored(self, i: int) -> bool:
        return self.crossing[i] is None


def rollout(forecaster: BlockForecaster, windows: np.ndarray, thresholds: np.ndarray,
            cap: int, persist: int) -> RolloutResult:
    """Roll out ``n`` windows together until each crosses its threshold or ``cap``.

    ``windows`` is (n, L) in Ah; ``thresholds`` is (n,) in Ah.
    """
    windows = np.atleast_2d(np.asarray(windows, dtype=np.float64))
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=np.float64), (len(windows),))
    L, H = forecaster.input_len, forecaster.horizon
    if windows.shape[1] != L:
        raise ValueError(f"window length {windows.shape[1]} != model input length {L}")

    buffer = windows.copy()
    crossing: list[int | None] = [None] * len(windows)
    steps = 0
    while steps < cap:
        block = np.asarray(forecaster.predict_block(buffer[:, -L:]), dtype=np.float64)
        if block.shape != (len(windows), H):
            raise ValueError(f"forecaster returned {block.shape}, expected {(len(windows), H)}")
        if not np.all(np.isfinite(block)):
            raise FloatingPointError("non-finite forecast values")
        buffer = np.concatenate([buffer, block], axis=1)
        steps += H
        traj = buffer[:, L:]
        for i in range(len(windows)):
            if crossing[i] is None:
                crossing[i] = first_sustained_crossing(traj[i], thresholds[i], persist)
        if all(c is not None for c in crossing):
            break
    traj = buffer[:, L:L + cap]
    crossing = [c if (c is not None and c <= cap) else None for c in crossing]
    return RolloutResult(trajectories=traj, crossing=crossing, cap=cap)
