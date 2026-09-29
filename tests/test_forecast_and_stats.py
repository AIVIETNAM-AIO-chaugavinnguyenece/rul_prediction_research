import numpy as np
import pytest

from battery_rul.dataset import forecast_origins
from battery_rul.features import window_at
from battery_rul.forecast import rollout
from battery_rul.models.baselines import LinearTrendForecaster, PersistenceForecaster
from battery_rul.stats import holm, rank_biserial, verdict, wilcoxon


def test_linear_trend_rollout_crosses_where_expected():
    # capacity 2.0 falling 0.01 per cycle: 1.6 is reached 40 cycles after 2.0
    series = 2.0 - 0.01 * np.arange(30)                      # last observed 1.71
    f = LinearTrendForecaster(input_len=10, horizon=4)
    res = rollout(f, series[None, -10:], np.array([1.605]), cap=200, persist=5)
    # forecast step k has value 1.71 - 0.01k; first k with value <= 1.605 is k = 11
    # (threshold off the exact grid so floating-point rounding cannot decide the tie)
    assert res.crossing[0] == 11
    assert not res.censored(0) and res.predicted_rul(0) == 11


def test_persistence_is_censored():
    res = rollout(PersistenceForecaster(8, 4), np.full((1, 8), 1.9), np.array([1.6]), cap=50, persist=5)
    assert res.censored(0) and res.predicted_rul(0) == 50
    assert res.trajectories.shape == (1, 50)


def test_rollout_rejects_wrong_window_length():
    with pytest.raises(ValueError):
        rollout(PersistenceForecaster(8, 4), np.ones((1, 6)), np.array([0.5]), cap=10, persist=2)


def test_forecast_uses_only_the_past(fade_series, make_cell):
    """Altering every cycle after the origin cannot change the forecast."""
    cell = make_cell(fade_series, eol=150)
    altered = make_cell(np.r_[fade_series[:75], fade_series[75:] - 0.4], eol=150)
    origin = [o for o in forecast_origins(cell, 16) if o.t0 == 75][0]
    f = LinearTrendForecaster(16, 4)
    a = rollout(f, window_at(cell.capacity, origin.t0, 16)[None], np.array([1.6]), 300, 5)
    b = rollout(f, window_at(altered.capacity, origin.t0, 16)[None], np.array([1.6]), 300, 5)
    np.testing.assert_array_equal(a.trajectories, b.trajectories)


def test_origins_respect_window_and_eol(fade_series, make_cell):
    cell = make_cell(fade_series, eol=40)
    origins = forecast_origins(cell, input_len=16, fracs=(0.25, 0.5, 0.75))
    assert [o.t0 for o in origins] == [20, 30]               # 10 < L is skipped
    assert all(o.true_rul == 40 - o.t0 for o in origins)


def test_holm_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02])


def test_rank_biserial_sign_and_bounds():
    a, b = np.array([3.0, 4, 5, 6]), np.array([1.0, 2, 3, 4])
    assert rank_biserial(a, b) == 1.0 and rank_biserial(b, a) == -1.0


def test_wilcoxon_all_ties():
    assert wilcoxon([1, 2, 3], [1, 2, 3])[1] == 1.0


def test_verdict_wording():
    assert verdict(0.01, True) == "supported"
    assert verdict(0.01, False) == "contradicted"
    assert verdict(0.2, True) == "not detected"
