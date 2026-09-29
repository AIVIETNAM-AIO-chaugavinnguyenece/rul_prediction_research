"""Explainability for both families, on the same footing where possible.

Linear models are explained exactly: each trained model *is* a matrix W (see
:func:`battery_rul.models.linear.effective_map`). Summaries here turn W into
numbers a report can quote - how much weight sits on recent cycles, and how the
weight on the last observed value changes with the forecast step.

Tree models are explained with permutation importance (model-agnostic, always
available) and, when the ``shap`` package is installed, TreeSHAP on the first
and last horizon step.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def recency_profile(W: np.ndarray) -> pd.DataFrame:
    """Absolute weight per input position, averaged over forecast steps and
    normalised to sum to 1 (position L-1 = last observed cycle)."""
    a = np.abs(W).mean(axis=0)
    a = a / a.sum() if a.sum() > 0 else a
    L = W.shape[1]
    return pd.DataFrame({"lag": np.arange(L - 1, -1, -1), "position": np.arange(L), "share": a})


def recent_weight_share(W: np.ndarray, frac: float = 0.25) -> float:
    """Share of absolute weight on the most recent ``frac`` of the window."""
    prof = recency_profile(W)
    k = max(1, int(round(frac * W.shape[1])))
    return float(prof["share"].to_numpy()[-k:].sum())


def last_value_weight(W: np.ndarray) -> np.ndarray:
    """Weight on the last observed value for each forecast step."""
    return W[:, -1].copy()


def weight_row_sums(W: np.ndarray) -> np.ndarray:
    """Row sums of W: 1.0 means the step is level-preserving (a shift in the whole
    window shifts the forecast by the same amount); NLinear enforces this."""
    return W.sum(axis=1)


def tree_permutation_importance(forecaster, X_windows: np.ndarray, Y: np.ndarray,
                                n_repeats: int = 5, seed: int = 0, max_rows: int = 2000) -> pd.DataFrame:
    """Permutation importance of the window features, scored by multi-output MSE
    on the targets the model was trained for."""
    from .features import make_targets

    rng = np.random.default_rng(seed)
    if len(X_windows) > max_rows:
        idx = rng.choice(len(X_windows), max_rows, replace=False)
        X_windows, Y = X_windows[idx], Y[idx]
    F = forecaster.features(X_windows)
    T = make_targets(X_windows, Y, forecaster.target_mode)
    res = permutation_importance(forecaster.model, F, T, n_repeats=n_repeats, random_state=seed,
                                 scoring="neg_mean_squared_error")
    out = pd.DataFrame({"feature": F.columns, "importance": res.importances_mean,
                        "std": res.importances_std})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def tree_shap_importance(forecaster, X_windows: np.ndarray, max_rows: int = 500, seed: int = 0
                         ) -> pd.DataFrame | None:
    """Mean |SHAP| per feature for the first and last forecast step.

    Returns None when ``shap`` is not installed or the model type is unsupported.
    """
    try:
        with warnings.catch_warnings():         # shap imports tqdm.auto, which warns without ipywidgets
            warnings.simplefilter("ignore")
            import shap
    except ImportError:
        return None
    rng = np.random.default_rng(seed)
    if len(X_windows) > max_rows:
        X_windows = X_windows[rng.choice(len(X_windows), max_rows, replace=False)]
    F = forecaster.features(X_windows)
    model = forecaster.model
    estimators = getattr(model, "estimators_", None)
    rows = []
    try:
        if type(model).__name__ == "MultiOutputRegressor" and estimators is not None:
            steps = {"h=1": estimators[0], f"h={len(estimators)}": estimators[-1]}
            for step, est in steps.items():
                sv = shap.TreeExplainer(est).shap_values(F)
                rows.append(pd.DataFrame({"feature": F.columns, "step": step,
                                          "mean_abs_shap": np.abs(sv).mean(axis=0)}))
        else:
            sv = shap.TreeExplainer(model).shap_values(F)
            sv = np.asarray(sv)
            if sv.ndim == 3:            # (n, features, outputs) or (outputs, n, features)
                first, last = (sv[..., 0], sv[..., -1]) if sv.shape[-1] == forecaster.horizon else (sv[0], sv[-1])
            else:
                first = last = sv
            for step, s in (("h=1", first), (f"h={forecaster.horizon}", last)):
                rows.append(pd.DataFrame({"feature": F.columns, "step": step,
                                          "mean_abs_shap": np.abs(s).mean(axis=0)}))
    except Exception:  # noqa: BLE001 - SHAP support varies by model and version
        return None
    return pd.concat(rows, ignore_index=True)
