"""Scoring forecasts and aggregating them to the unit of replication.

Metric definitions, stated once:

* ``rul_error``         predicted RUL - true RUL, in cycles (positive = late)
* ``abs_rul_error``     |rul_error|
* ``rel_abs_rul_error`` |rul_error| / true RUL  <- primary metric (pools datasets)
* ``alpha_lambda_hit``  1 if |rul_error| <= alpha x true RUL (Saxena et al., 2008)
* ``censored``          1 if the forecast never crossed the threshold within the cap
* ``block_rmse``        RMSE (Ah) of the first H forecast cycles vs the causal series
* ``traj_rmse``         RMSE (Ah) of the roll-out from t0 to the true EOL

The unit of replication is a (cell, origin) pair. Seeds are *not* units: seed
results are averaged within each unit before any test, so a model with five
seeds does not get five times the statistical weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ALPHA_LAMBDA
from .dataset import Cell, Fold, Origin
from .forecast import RolloutResult

UNIT_COLS = ["dataset", "cell", "frac"]
METRICS = ["rel_abs_rul_error", "abs_rul_error", "rul_error", "alpha_lambda_hit",
           "censored", "block_rmse", "traj_rmse"]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.asarray(a[:n]) - np.asarray(b[:n])) ** 2)))


def score_rollout(result: RolloutResult, origins: list[Origin], cell: Cell, fold: Fold,
                  model: str, seed: int, block_len: int, **extra) -> list[dict]:
    """One record per origin of a single test cell."""
    records = []
    for i, org in enumerate(origins):
        traj = result.trajectories[i]
        observed_after = cell.capacity[org.t0:]                 # causal series, cycles t0+1..
        pred_rul = result.predicted_rul(i)
        err = pred_rul - org.true_rul
        records.append({
            "model": model, "seed": seed, "dataset": fold.dataset, "cell": cell.name,
            "fold": fold.fold_id, "frac": org.frac, "t0": org.t0,
            "true_eol": org.true_eol, "true_rul": org.true_rul,
            "pred_rul": pred_rul, "pred_eol": org.t0 + pred_rul,
            "censored": int(result.censored(i)),
            "rul_error": err, "abs_rul_error": abs(err),
            "rel_abs_rul_error": abs(err) / max(org.true_rul, 1),
            "alpha_lambda_hit": int(abs(err) <= ALPHA_LAMBDA * org.true_rul),
            "block_rmse": _rmse(traj[:block_len], observed_after[:block_len]),
            "traj_rmse": _rmse(traj[:org.true_rul], observed_after[:org.true_rul]),
            "forecast_min_Ah": float(np.min(traj[:max(org.true_rul, 1)])),
            "observed_min_to_eol_Ah": float(np.min(observed_after[:max(org.true_rul, 1)]))
            if len(observed_after) else float("nan"),
            **extra,
        })
    return records


def seed_average(records: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Average seeds within each (model, unit[, arm]) - one row per unit."""
    group_cols = group_cols or ["model", *UNIT_COLS]
    agg = {m: "mean" for m in METRICS if m in records.columns}
    agg.update({"true_rul": "first", "t0": "first", "seed": "nunique"})
    out = records.groupby(group_cols, as_index=False).agg(agg)
    return out.rename(columns={"seed": "n_seeds"})


def summary_table(units: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Median and mean of each metric across units, per model (and optional groups)."""
    by = by or ["model"]
    g = units.groupby(by)
    table = pd.DataFrame({
        "n_units": g.size(),
        "median_rel_err": g["rel_abs_rul_error"].median(),
        "mean_rel_err": g["rel_abs_rul_error"].mean(),
        "median_abs_err_cycles": g["abs_rul_error"].median(),
        "mean_signed_err_cycles": g["rul_error"].mean(),
        "alpha_lambda_acc": g["alpha_lambda_hit"].mean(),
        "censor_rate": g["censored"].mean(),
        "median_block_rmse_Ah": g["block_rmse"].median(),
        "median_traj_rmse_Ah": g["traj_rmse"].median(),
    }).reset_index()
    return table.sort_values(by[:-1] + ["median_rel_err"] if len(by) > 1 else "median_rel_err")


def wide(units: pd.DataFrame, metric: str, models: list[str], unit_cols: list[str] = UNIT_COLS) -> pd.DataFrame:
    """Units x models matrix, keeping only units every model has (complete blocks)."""
    table = units[units["model"].isin(models)].pivot_table(index=unit_cols, columns="model",
                                                           values=metric, aggfunc="mean")
    return table[[m for m in models if m in table.columns]].dropna()
