import numpy as np
import pytest

from battery_rul.features import (
    build_training_set, invert_targets, make_targets, make_windows, window_at, window_features,
)


def test_window_alignment(fade_series):
    L, H = 12, 5
    X, Y, ends = make_windows(fade_series, L, H)
    assert X.shape[1] == L and Y.shape[1] == H
    for i in (0, len(X) // 2, len(X) - 1):
        t = ends[i]
        np.testing.assert_array_equal(X[i], fade_series[t - L + 1: t + 1])
        np.testing.assert_array_equal(Y[i], fade_series[t + 1: t + H + 1])
    assert ends[-1] + H == len(fade_series) - 1


def test_window_at_matches_origin(fade_series):
    w = window_at(fade_series, t0=40, input_len=16)          # cycles 25..40
    np.testing.assert_array_equal(w, fade_series[24:40])
    with pytest.raises(ValueError):
        window_at(fade_series, t0=10, input_len=16)


def test_features_contain_no_cycle_index(fade_series):
    X, _, _ = make_windows(fade_series, 16, 4)
    for relative in (False, True):
        cols = window_features(X, level_relative=relative).columns
        assert not any("cycle" in c or "end" in c for c in cols)


def test_level_relative_features_are_shift_invariant(fade_series):
    X, _, _ = make_windows(fade_series, 16, 4)
    a = window_features(X, level_relative=True)
    b = window_features(X + 0.5, level_relative=True)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-9)
    c = window_features(X + 0.5, level_relative=False)
    assert not np.allclose(window_features(X).to_numpy(), c.to_numpy())


@pytest.mark.parametrize("mode", ["absolute", "delta"])
def test_target_round_trip(fade_series, mode):
    X, Y, _ = make_windows(fade_series, 16, 4)
    np.testing.assert_allclose(invert_targets(X, make_targets(X, Y, mode), mode), Y)


def test_training_split_is_chronological_per_cell(fade_series, make_cell):
    cells = [make_cell(fade_series, "B0005"), make_cell(fade_series[:150], "B0006")]
    ts = build_training_set(cells, 16, 4, val_frac=0.2)
    for name, cell in (("B0005", cells[0]), ("B0006", cells[1])):
        fit = ts.X_fit[ts.fit_cells == name]
        val = ts.X_val[ts.val_cells == name]
        X, _, _ = make_windows(cell.capacity, 16, 4)
        np.testing.assert_array_equal(np.r_[fit, val], X)   # val = the last windows, in order


def test_range_restriction_removes_low_targets(fade_series, make_cell):
    cell = make_cell(fade_series)
    ts = build_training_set([cell], 16, 4, range_restrict_soh=0.85)
    assert ts.n_removed_by_range > 0
    assert ts.Y_all.min() >= 0.85 * cell.references["rated"]
