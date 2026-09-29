"""The six tree-based models, wrapped as H-step block forecasters.

Each model predicts all ``H`` future values from one window's features:

* DecisionTree, RandomForest and XGBoost handle multi-output targets natively.
* AdaBoost, GradientBoosting and LightGBM fit one model per horizon step
  (``MultiOutputRegressor``).

Two target modes, declared in :mod:`battery_rul.config`:

* ``absolute`` - features and targets in Ah. Standard practice; a tree cannot
  predict outside the range of targets it saw in training.
* ``delta`` - level-relative features, targets = change from the last value.
  Removes absolute level from the problem, the tree analogue of NLinear.
  Including it separates "model family" from "target normalisation" in RQ2.

XGBoost and LightGBM run single-threaded with fixed seeds so that results are
bit-reproducible (a lesson from the sales-forecasting project, where
multi-threaded histogram construction flipped a significance verdict).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

from ..config import TREE_PARAMS
from ..features import invert_targets, make_targets, window_features


def make_tree_model(name: str, seed: int, params: dict | None = None):
    p = dict(TREE_PARAMS[name] if params is None else params)
    if name == "DecisionTree":
        return DecisionTreeRegressor(random_state=seed, **p)
    if name == "RandomForest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **p)
    if name == "AdaBoost":
        depth = p.pop("base_max_depth", 4)
        return MultiOutputRegressor(AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=depth, random_state=seed),
            random_state=seed, **p), n_jobs=-1)
    if name == "GradientBoosting":
        return MultiOutputRegressor(GradientBoostingRegressor(random_state=seed, **p), n_jobs=-1)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist",
                            multi_strategy="one_output_per_tree", **p)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return MultiOutputRegressor(LGBMRegressor(
            random_state=seed, n_jobs=1, deterministic=True, force_row_wise=True,
            verbose=-1, **p), n_jobs=-1)
    raise ValueError(f"unknown tree model {name!r}")


def _single_threaded_predict(model) -> None:
    """Roll-outs call predict on a handful of windows many times; parallel
    dispatch would cost more than the prediction itself."""
    if isinstance(model, (MultiOutputRegressor, RandomForestRegressor)):
        model.n_jobs = None


def count_tree_nodes(model) -> int:
    """Total decision nodes + leaves across every tree in the fitted model."""
    def _sk_tree(est) -> int:
        return int(est.tree_.node_count)

    if isinstance(model, DecisionTreeRegressor):
        return _sk_tree(model)
    if isinstance(model, RandomForestRegressor):
        return sum(_sk_tree(e) for e in model.estimators_)
    if isinstance(model, MultiOutputRegressor):
        return sum(count_tree_nodes(e) for e in model.estimators_)
    if isinstance(model, AdaBoostRegressor):
        return sum(_sk_tree(e) for e in model.estimators_)
    if isinstance(model, GradientBoostingRegressor):
        return sum(_sk_tree(e) for e in np.ravel(model.estimators_))
    cls = type(model).__name__
    if cls == "XGBRegressor":
        return sum(tree.count("\n") for tree in model.get_booster().get_dump())
    if cls == "LGBMRegressor":
        info = model.booster_.dump_model()["tree_info"]
        return sum(2 * t["num_leaves"] - 1 for t in info)
    return -1


@dataclass
class TreeForecaster:
    """A fitted tree model that forecasts ``H`` cycles from a raw window."""

    name: str
    model: object
    target_mode: str
    input_len: int
    horizon: int
    seed: int
    feature_names: list[str]
    train_seconds: float = 0.0

    @property
    def level_relative(self) -> bool:
        return self.target_mode == "delta"

    @property
    def n_nodes(self) -> int:
        return count_tree_nodes(self.model)

    def features(self, windows: np.ndarray):
        return window_features(np.atleast_2d(windows), level_relative=self.level_relative)

    def predict_block(self, windows: np.ndarray) -> np.ndarray:
        windows = np.atleast_2d(windows)
        T = np.asarray(self.model.predict(self.features(windows)), dtype=np.float64)
        T = T.reshape(len(windows), self.horizon)
        return invert_targets(windows, T, self.target_mode)


def train_tree(name: str, X: np.ndarray, Y: np.ndarray, target_mode: str, seed: int,
               params: dict | None = None) -> TreeForecaster:
    """Fit on all training windows (fit + validation parts; trees do not early-stop)."""
    level_relative = target_mode == "delta"
    F = window_features(X, level_relative=level_relative)
    T = make_targets(X, Y, target_mode)
    model = make_tree_model(name, seed, params)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        # joblib/loky reports idle workers being recycled between fits; harmless
        warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given")
        model.fit(F, T)
    elapsed = time.perf_counter() - t0
    _single_threaded_predict(model)
    return TreeForecaster(name=name, model=model, target_mode=target_mode, input_len=X.shape[1],
                          horizon=Y.shape[1], seed=seed, feature_names=list(F.columns),
                          train_seconds=elapsed)
