import numpy as np
import pytest
import torch

from battery_rul.features import Standardizer, build_training_set
from battery_rul.models.baselines import LinearTrendForecaster, PersistenceForecaster
from battery_rul.models.linear import (
    DLinear, MovingAvg, effective_map, moving_average_matrix, train_linear,
)


def test_moving_average_matches_replicate_padding():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    k = 3
    padded = np.r_[x[0], x, x[-1]]
    expected = np.convolve(padded, np.ones(k) / k, mode="valid")
    got = MovingAvg(k)(torch.tensor(x[None], dtype=torch.float32)).numpy()[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_moving_average_matrix_reproduces_layer():
    L, k = 12, 5
    A = moving_average_matrix(L, k)
    x = np.random.default_rng(1).normal(size=L)
    layer = MovingAvg(k)(torch.tensor(x[None], dtype=torch.float32)).numpy()[0]
    np.testing.assert_allclose(A @ x, layer, rtol=1e-5, atol=1e-6)


@pytest.fixture
def trained(fade_series, make_cell):
    ts = build_training_set([make_cell(fade_series)], 16, 4)
    return {name: train_linear(name, ts, 16, 4, 5, seed=0) for name in ("DLinear", "NLinear", "Linear")}


@pytest.mark.parametrize("name", ["DLinear", "NLinear", "Linear"])
def test_effective_map_reproduces_model(trained, name, fade_series):
    f = trained[name]
    windows = np.stack([fade_series[i:i + 16] for i in range(0, 60, 7)])
    z = f.scaler.transform(windows)
    em = effective_map(f)
    with torch.no_grad():
        direct = f.model(torch.tensor(z, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(z @ em["W"].T + em["b"], direct, rtol=1e-4, atol=1e-5)


def test_nlinear_is_level_preserving(trained):
    """NLinear's effective weights sum to one per step: shift in, same shift out."""
    np.testing.assert_allclose(effective_map(trained["NLinear"])["W"].sum(axis=1), 1.0, atol=1e-5)


def test_training_is_seed_deterministic(fade_series, make_cell):
    ts = build_training_set([make_cell(fade_series)], 16, 4)
    a = train_linear("DLinear", ts, 16, 4, 5, seed=3)
    b = train_linear("DLinear", ts, 16, 4, 5, seed=3)
    w = ts.X_val[:3]
    np.testing.assert_array_equal(a.predict_block(w), b.predict_block(w))


def test_baselines():
    w = np.array([[2.0, 1.9, 1.8, 1.7]])
    np.testing.assert_array_equal(PersistenceForecaster(4, 3).predict_block(w), [[1.7, 1.7, 1.7]])
    np.testing.assert_allclose(LinearTrendForecaster(4, 3).predict_block(w), [[1.6, 1.5, 1.4]])


def test_standardizer_round_trip():
    s = Standardizer().fit(np.array([1.0, 2.0, 3.0]))
    x = np.array([[0.5, 4.0]])
    np.testing.assert_allclose(s.inverse(s.transform(x)), x)
