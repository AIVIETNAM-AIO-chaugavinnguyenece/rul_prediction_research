import numpy as np
import pandas as pd

from battery_rul.cleaning import clean_capacity, first_sustained_crossing, label_cell


def _raw(x):
    return pd.DataFrame({"cycle": np.arange(1, len(x) + 1), "capacity_Ah": x})


def test_causal_input_has_no_look_ahead(fade_series):
    """Changing the future must not change any past value of capacity_input."""
    t = 100
    a, _ = clean_capacity(_raw(fade_series), "B0005")
    altered = fade_series.copy()
    altered[t + 1:] -= 0.3                                   # a drastic change after t
    b, _ = clean_capacity(_raw(altered), "B0005")
    np.testing.assert_allclose(a["capacity_input"][: t + 1], b["capacity_input"][: t + 1])


def test_centred_label_does_look_ahead(fade_series):
    """The label series is allowed to (and does) use the future - never a model input."""
    t = 100
    a, _ = clean_capacity(_raw(fade_series), "B0005")
    altered = fade_series.copy()
    altered[t + 1:] -= 0.3
    b, _ = clean_capacity(_raw(altered), "B0005")
    assert not np.allclose(a["capacity_label"][: t + 1], b["capacity_label"][: t + 1])


def test_outlier_is_replaced_in_causal_series(fade_series):
    x = fade_series.copy()
    x[80] *= 0.75
    df, _ = clean_capacity(_raw(x), "B0005")
    assert bool(df.loc[80, "is_outlier_causal"])                 # row i = position i
    assert abs(df.loc[80, "capacity_input"] - fade_series[80]) < 0.02


def test_impossible_values_dropped_and_reported():
    x = np.r_[np.linspace(1.9, 1.5, 50), 0.0, 9.0, np.nan]
    df, report = clean_capacity(_raw(x), "B0005")
    assert report["n_below_min"] == 1 and report["n_above_max"] == 1 and report["n_nan"] == 1
    assert len(df) == 50 and df["cycle"].tolist() == list(range(1, 51))


def test_sustained_crossing_ignores_single_dip():
    s = np.r_[np.ones(10), 0.7, np.ones(10), np.full(8, 0.7)]
    assert first_sustained_crossing(s, 0.8, persist=1) == 11
    assert first_sustained_crossing(s, 0.8, persist=5) == 22
    assert first_sustained_crossing(s, 0.1, persist=5) is None


def test_label_cell_eol_and_rul(fade_series):
    df, _ = clean_capacity(_raw(fade_series), "B0005")
    df, info = label_cell(df, "B0005")
    eol = info["eol_rated"]
    assert eol is not None and 1 <= eol <= len(df)
    assert df.loc[eol - 1, "rul_rated"] == 0
    assert df.loc[0, "rul_rated"] == eol - 1
    assert info["threshold_rated_Ah"] == 1.6
