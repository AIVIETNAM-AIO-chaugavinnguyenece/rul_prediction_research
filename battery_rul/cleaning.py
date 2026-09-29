"""Cleaning, end-of-life detection and RUL labels.

The one rule this module exists to enforce: **model inputs are causal**.

* ``capacity_input`` uses a *trailing* Hampel filter, so the value at cycle t
  depends only on cycles <= t. A forecast made at cycle t0 can use it without
  seeing the future.
* ``capacity_label`` uses a *centred* Hampel filter plus a centred moving
  average. It looks ahead a few cycles, which is fine for deciding when a cell
  truly reached end of life, and wrong for anything a model consumes.

Test targets are never altered beyond this: the forecast targets are the causal
series, and the true EOL comes from the label series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLEANING, LABELS, RATED_AH, CleaningConfig, LabelConfig, family_of

_MAD_SCALE = 1.4826  # makes the MAD a consistent estimator of the standard deviation


def _hampel(x: pd.Series, window: int, n_mad: float, centred: bool) -> tuple[pd.Series, pd.Series]:
    """Return (cleaned series, outlier flags). Outliers become the local median."""
    roll = x.rolling(window, center=centred, min_periods=1)
    med = roll.median()
    dev = (x - med).abs()
    mad = dev.rolling(window, center=centred, min_periods=1).median()
    scale = _MAD_SCALE * mad.replace(0, np.nan)
    flags = (dev > n_mad * scale).fillna(False)
    return x.where(~flags, med), flags


def clean_capacity(raw: pd.DataFrame, cell: str, cfg: CleaningConfig = CLEANING) -> tuple[pd.DataFrame, dict]:
    """Drop impossible cycles, then build the causal input and centred label series.

    Returns the cleaned table (may be empty) and a report of what was dropped.
    """
    rated = RATED_AH[family_of(cell)]
    out = raw.copy()
    out["capacity_raw"] = pd.to_numeric(out["capacity_Ah"], errors="coerce")

    lo, hi = cfg.min_capacity_frac * rated, cfg.max_capacity_frac * rated
    is_nan = out["capacity_raw"].isna()
    below = ~is_nan & (out["capacity_raw"] < lo)
    above = ~is_nan & (out["capacity_raw"] > hi)
    report = {
        "n_read": int(len(out)),
        "n_nan": int(is_nan.sum()),
        "n_below_min": int(below.sum()),
        "n_above_max": int(above.sum()),
        "valid_range_Ah": [round(lo, 4), round(hi, 4)],
        "raw_min_Ah": None if is_nan.all() else float(out["capacity_raw"].min()),
        "raw_max_Ah": None if is_nan.all() else float(out["capacity_raw"].max()),
    }
    out = out[~(is_nan | below | above)].reset_index(drop=True)
    report["n_kept"] = int(len(out))
    if out.empty:
        return out, report

    out["cycle"] = np.arange(1, len(out) + 1)
    x = out["capacity_raw"]

    out["capacity_input"], out["is_outlier_causal"] = _hampel(
        x, cfg.hampel_window, cfg.hampel_n_mad, centred=False)
    label_clean, out["is_outlier_centred"] = _hampel(
        x, cfg.hampel_window, cfg.hampel_n_mad, centred=True)
    out["capacity_label"] = label_clean.rolling(cfg.smooth_window, center=True, min_periods=1).mean()

    # Capacity regeneration: a rise of more than regen_jump_frac x rated between
    # consecutive cycles in the causal series.
    out["is_regeneration"] = out["capacity_input"].diff() > cfg.regen_jump_frac * rated
    return out, report


def first_sustained_crossing(values: np.ndarray, threshold: float, persist: int) -> int | None:
    """1-based position of the first run of ``persist`` values at or below
    ``threshold``, or None. A single noisy dip does not count as failure."""
    below = np.asarray(values) <= threshold
    if not below.any():
        return None
    run = 0
    for i, flag in enumerate(below):
        run = run + 1 if flag else 0
        if run >= persist:
            return i - persist + 2
    return None


def label_cell(df: pd.DataFrame, cell: str, cfg: LabelConfig = LABELS) -> tuple[pd.DataFrame, dict]:
    """Attach SoH, EOL and RUL columns under both reference conventions."""
    if df.empty:
        raise ValueError(f"{cell}: no usable cycles after cleaning")
    rated = RATED_AH[family_of(cell)]
    n_init = min(cfg.initial_cycles, len(df))
    initial = float(df["capacity_label"].iloc[:n_init].mean())

    info: dict = {
        "cell": cell,
        "family": family_of(cell),
        "n_cycles": int(len(df)),
        "rated_Ah": rated,
        "initial_Ah": round(initial, 4),
        "initial_over_rated": round(initial / rated, 4),
        "final_Ah": round(float(df["capacity_input"].iloc[-1]), 4),
        "min_Ah": round(float(df["capacity_input"].min()), 4),
        "n_outliers_causal": int(df["is_outlier_causal"].sum()),
        "n_regeneration": int(df["is_regeneration"].sum()),
    }
    info["regen_per_100_cycles"] = round(100 * info["n_regeneration"] / len(df), 3)

    for ref_name, ref in (("rated", rated), ("initial", initial)):
        threshold = cfg.eol_frac * ref
        df[f"soh_{ref_name}"] = df["capacity_input"] / ref
        eol = first_sustained_crossing(df["capacity_label"].to_numpy(), threshold, cfg.eol_persist)
        naive = first_sustained_crossing(df["capacity_label"].to_numpy(), threshold, 1)
        info[f"threshold_{ref_name}_Ah"] = round(threshold, 4)
        info[f"eol_{ref_name}"] = eol
        info[f"eol_{ref_name}_first_dip"] = naive
        df[f"rul_{ref_name}"] = np.nan if eol is None else np.clip(eol - df["cycle"], 0, None)

    eol_primary = info[f"eol_{cfg.primary_reference}"]
    info["reaches_eol"] = eol_primary is not None
    if eol_primary is not None:
        fade = initial - float(df["capacity_label"].iloc[eol_primary - 1])
        info["fade_rate_mAh_per_cycle"] = round(1000 * fade / max(eol_primary, 1), 4)
    else:
        info["fade_rate_mAh_per_cycle"] = np.nan
    return df, info


#: Columns written to the processed per-cell files.
CELL_COLUMNS = [
    "cycle", "capacity_raw", "capacity_input", "capacity_label",
    "soh_rated", "soh_initial", "rul_rated", "rul_initial",
    "is_outlier_causal", "is_outlier_centred", "is_regeneration",
    "ambient_C", "mean_temp_C", "max_temp_C", "discharge_time_s",
    "v_min", "v_mean", "current_mean_A", "internal_resistance_ohm", "test_time_s", "source_file",
]
