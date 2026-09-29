"""Statistical comparisons at the unit of replication.

Protocol (fixed before any result, carried over from the sales-forecasting
project and tightened):

1. Units are (cell, origin) pairs with seeds averaged within each unit.
2. More than two models: Friedman omnibus test on complete blocks.
3. Pairwise: Wilcoxon signed-rank, Holm-corrected within the research
   question's family of comparisons.
4. Effect size: matched-pairs rank-biserial correlation, plus a bootstrap CI
   on the median paired difference.
5. Non-significance means "not detected under this design", never "equivalent".
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .config import ALPHA, N_BOOTSTRAP


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(running, 1.0)
    return out


def rank_biserial(a, b) -> float:
    """Matched-pairs rank-biserial r in [-1, 1]; positive means a > b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    w_pos, w_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def bootstrap_ci(values, statistic=np.median, n_boot: int = N_BOOTSTRAP, level: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.array([statistic(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def wilcoxon(a, b) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank; (nan, 1.0) when every difference is zero."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return float("nan"), 1.0
    res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def friedman(table: pd.DataFrame) -> dict:
    """Friedman test over the columns (models) of a complete units x models table."""
    if table.shape[1] < 3 or table.shape[0] < 2:
        return {"statistic": np.nan, "p_value": np.nan, "n_units": table.shape[0], "k_models": table.shape[1]}
    res = stats.friedmanchisquare(*[table[c].to_numpy() for c in table.columns])
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue),
            "n_units": table.shape[0], "k_models": table.shape[1]}


def pairwise(table: pd.DataFrame, pairs: list[tuple[str, str]] | None = None,
             lower_is_better: bool = True) -> pd.DataFrame:
    """Pairwise Wilcoxon with Holm correction, effect sizes and bootstrap CIs.

    ``table`` is units x models (see :func:`battery_rul.evaluation.wide`). For
    each pair (a, b) the paired difference is ``a - b``.
    """
    pairs = pairs or list(combinations(table.columns, 2))
    rows = []
    for a, b in pairs:
        x, y = table[a].to_numpy(), table[b].to_numpy()
        stat, p = wilcoxon(x, y)
        diff = x - y
        lo, hi = bootstrap_ci(diff, np.median)
        better = (diff < 0) if lower_is_better else (diff > 0)
        rows.append({
            "model_a": a, "model_b": b, "n_units": len(diff),
            "median_a": float(np.median(x)), "median_b": float(np.median(y)),
            "median_diff": float(np.median(diff)), "ci_low": lo, "ci_high": hi,
            "wilcoxon_W": stat, "p_value": p,
            "rank_biserial": rank_biserial(x, y),
            "a_better_in": int(better.sum()), "b_better_in": int(((diff > 0) if lower_is_better else (diff < 0)).sum()),
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"])
    out["significant"] = out["p_holm"] < ALPHA
    return out


def subgroup_difference(diff_a, diff_b) -> dict:
    """Mann-Whitney U on paired differences from two independent subgroups."""
    a, b = np.asarray(diff_a, float), np.asarray(diff_b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"U": np.nan, "p_value": np.nan, "n_a": len(a), "n_b": len(b)}
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"U": float(res.statistic), "p_value": float(res.pvalue), "n_a": len(a), "n_b": len(b),
            "median_a": float(np.median(a)), "median_b": float(np.median(b))}


def spearman(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"rho": np.nan, "p_value": np.nan, "n": int(ok.sum())}
    res = stats.spearmanr(x[ok], y[ok])
    return {"rho": float(res.statistic), "p_value": float(res.pvalue), "n": int(ok.sum())}


def verdict(p_adjusted: float, direction_as_predicted: bool, alpha: float = ALPHA) -> str:
    """Pre-registered verdict wording.

    * supported      - significant after correction, in the predicted direction
    * contradicted   - significant after correction, in the opposite direction
    * not detected   - not significant: insufficient evidence either way
    """
    if not np.isfinite(p_adjusted):
        return "not testable"
    if p_adjusted < alpha:
        return "supported" if direction_as_predicted else "contradicted"
    return "not detected"
