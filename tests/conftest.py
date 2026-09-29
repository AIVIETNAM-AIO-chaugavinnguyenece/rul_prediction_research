import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def fade_series():
    """A deterministic fading capacity series with noise and one regeneration jump."""
    rng = np.random.default_rng(0)
    n = 200
    t = np.arange(n)
    x = 1.9 - 0.0025 * t - 1e-6 * t ** 2 + rng.normal(0, 0.003, n)
    x[120:] += 0.03 * np.exp(-np.arange(n - 120) / 6)
    return x


@pytest.fixture
def make_cell():
    from battery_rul.dataset import Cell

    def _make(capacity, name="B0005", threshold=1.6, eol=None):
        capacity = np.asarray(capacity, dtype=float)
        return Cell(name=name, dataset="NASA", group="nasa_core", capacity=capacity, label=capacity.copy(),
                    thresholds={"rated": threshold, "initial": threshold}, eol={"rated": eol, "initial": eol},
                    references={"rated": 2.0, "initial": float(capacity[:5].mean())})
    return _make
