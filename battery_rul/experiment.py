"""One experiment runner for every model family.

A *model spec* says how to train a forecaster for a fold; the runner handles
everything else identically for all models - training-set construction,
forecast origins, roll-out, scoring and timing. Notebooks 04-06 differ only in
which specs and which settings they pass in.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from .config import DETERMINISTIC_MODELS, LABELS, LINEAR_TRAIN, WINDOWS, WindowConfig
from .dataset import Cell, Fold, Origin, forecast_origins, rollout_cap
from .evaluation import score_rollout
from .features import TrainingSet, build_training_set, window_at
from .forecast import rollout
from .models.baselines import LinearTrendForecaster, PersistenceForecaster
from .models.trees import train_tree

Logger = Callable[[str], None]


@dataclass(frozen=True)
class ExperimentSettings:
    """Settings shared by every model in one experiment."""

    reference: str = LABELS.primary_reference
    windows: dict = field(default_factory=lambda: dict(WINDOWS))
    arm: str = "full_range"
    range_restrict_soh: float | None = None     # RQ2 extrapolation arm
    max_train_cells: int | None = None          # RQ3 training-size curve
    subset_seed: int = 0

    def with_window(self, dataset: str, input_len: int, horizon: int) -> "ExperimentSettings":
        w = dict(self.windows)
        w[dataset] = WindowConfig(input_len=input_len, horizon=horizon)
        return replace(self, windows=w)


@dataclass
class FoldData:
    fold: Fold
    window: WindowConfig
    train_cells: list[Cell]
    training: TrainingSet
    test_cell: Cell
    origins: list[Origin]
    windows: np.ndarray
    thresholds: np.ndarray
    cap: int


def prepare_fold(cells: dict[str, Cell], fold: Fold, settings: ExperimentSettings) -> FoldData | None:
    """Everything a model needs for one fold. None if the test cell has no valid origin."""
    window = settings.windows[fold.dataset]
    train_names = list(fold.train)
    if settings.max_train_cells is not None and settings.max_train_cells < len(train_names):
        # deterministic subset per (fold, subset seed): same cells for every model
        key = zlib.crc32(f"{fold.fold_id}|{settings.subset_seed}".encode())
        rng = np.random.default_rng(key)
        train_names = sorted(rng.choice(train_names, settings.max_train_cells, replace=False).tolist())
    train_cells = [cells[n] for n in train_names]
    test = cells[fold.test]
    origins = forecast_origins(test, window.input_len, settings.reference)
    if not origins:
        return None
    training = build_training_set(train_cells, window.input_len, window.horizon,
                                  stride=window.stride, reference=settings.reference,
                                  range_restrict_soh=settings.range_restrict_soh)
    windows = np.stack([window_at(test.capacity, o.t0, window.input_len) for o in origins])
    thresholds = np.full(len(origins), test.thresholds[settings.reference])
    return FoldData(fold, window, train_cells, training, test, origins, windows, thresholds,
                    rollout_cap(train_cells, settings.reference))


# --------------------------------------------------------------------------
# Model specs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    label: str                                   # e.g. "NLinear", "LightGBM (delta)"
    family: str                                  # "linear" | "tree" | "baseline"
    base: str                                    # model name without the target mode
    train: Callable[[FoldData, int], object]     # (fold data, seed) -> forecaster
    seeds: tuple[int, ...] | None = None         # None = use the runner's seeds
    target_mode: str = ""


def linear_spec(name: str) -> ModelSpec:
    from .models.linear import train_linear  # imported lazily: torch is only needed here

    def _train(fd: FoldData, seed: int):
        w = fd.window
        return train_linear(name, fd.training, w.input_len, w.horizon, w.ma_kernel, seed, LINEAR_TRAIN)
    return ModelSpec(label=name, family="linear", base=name, train=_train)


def tree_spec(name: str, target_mode: str) -> ModelSpec:
    def _train(fd: FoldData, seed: int):
        return train_tree(name, fd.training.X_all, fd.training.Y_all, target_mode, seed)
    return ModelSpec(label=f"{name} ({target_mode})", family="tree", base=name, train=_train,
                     target_mode=target_mode, seeds=(0,) if name in DETERMINISTIC_MODELS else None)


def baseline_spec(name: str) -> ModelSpec:
    def _train(fd: FoldData, seed: int):
        w = fd.window
        cls = PersistenceForecaster if name == "Persistence" else LinearTrendForecaster
        return cls(w.input_len, w.horizon)
    return ModelSpec(label=name, family="baseline", base=name, train=_train, seeds=(0,))


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def _inference_seconds_per_window(forecaster, fd: FoldData, n: int = 256) -> float:
    X = fd.training.X_all[:n]
    if len(X) == 0:
        return float("nan")
    start = time.perf_counter()
    forecaster.predict_block(X)
    return (time.perf_counter() - start) / len(X)


def _size(forecaster) -> tuple[int, int]:
    """(parameters, tree nodes); -1 where not applicable."""
    params = getattr(forecaster, "n_params", -1)
    nodes = getattr(forecaster, "n_nodes", -1)
    return (params if isinstance(params, int) else -1, nodes if isinstance(nodes, int) else -1)


def run_experiment(cells: dict[str, Cell], folds: list[Fold], specs: list[ModelSpec],
                   seeds: tuple[int, ...], settings: ExperimentSettings = ExperimentSettings(),
                   log: Logger = print, keep_models: bool = False
                   ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train and evaluate every spec on every fold and seed.

    Returns ``(records, fits, models)``:

    * ``records`` - one row per (model, seed, test cell, origin), see evaluation.py
    * ``fits``    - one row per trained model: time, size, epochs, windows used
    * ``models``  - fitted forecasters keyed by (label, fold_id, seed), if requested
    """
    records, fits, kept = [], [], {}
    for fold in folds:
        fd = prepare_fold(cells, fold, settings)
        if fd is None:
            log(f"  {fold.fold_id}: no valid forecast origin for L={settings.windows[fold.dataset].input_len}, skipped")
            continue
        log(f"  {fold.fold_id}: {fd.training.n_windows} training windows from "
            f"{len(fd.train_cells)} cells, {len(fd.origins)} origins, roll-out cap {fd.cap}")
        for spec in specs:
            for seed in (spec.seeds or seeds):
                forecaster = spec.train(fd, seed)
                t_roll = time.perf_counter()
                result = rollout(forecaster, fd.windows, fd.thresholds, fd.cap, LABELS.eol_persist)
                t_roll = time.perf_counter() - t_roll
                extra = {"family": spec.family, "base_model": spec.base, "target_mode": spec.target_mode,
                         "arm": settings.arm, "input_len": fd.window.input_len,
                         "horizon": fd.window.horizon, "n_train_cells": len(fd.train_cells),
                         "reference": settings.reference,
                         "train_target_min_Ah": float(fd.training.Y_all.min()),
                         "train_target_max_Ah": float(fd.training.Y_all.max())}
                records.extend(score_rollout(result, fd.origins, fd.test_cell, fold, spec.label,
                                             seed, fd.window.horizon, **extra))
                params, nodes = _size(forecaster)
                fits.append({
                    "model": spec.label, **extra, "seed": seed, "fold": fold.fold_id,
                    "dataset": fold.dataset, "test_cell": fold.test,
                    "n_train_windows": fd.training.n_windows,
                    "n_removed_by_range": fd.training.n_removed_by_range,
                    "train_seconds": float(getattr(forecaster, "train_seconds", 0.0)),
                    "rollout_seconds": t_roll,
                    "infer_seconds_per_window": _inference_seconds_per_window(forecaster, fd),
                    "n_params": params, "n_tree_nodes": nodes,
                    "best_epoch": getattr(forecaster, "best_epoch", np.nan),
                    "epochs_run": getattr(forecaster, "epochs_run", np.nan),
                })
                if keep_models:
                    kept[(spec.label, fold.fold_id, seed)] = forecaster
    return pd.DataFrame(records), pd.DataFrame(fits), kept


def rollout_examples(models: dict, cells: dict[str, Cell], folds: list[Fold], settings: ExperimentSettings,
                     labels: list[str], seed: int = 0) -> pd.DataFrame:
    """Long table of forecast trajectories for plotting (one seed per model)."""
    rows = []
    for fold in folds:
        fd = prepare_fold(cells, fold, settings)
        if fd is None:
            continue
        for label in labels:
            f = models.get((label, fold.fold_id, seed)) or models.get((label, fold.fold_id, 0))
            if f is None:
                continue
            res = rollout(f, fd.windows, fd.thresholds, fd.cap, LABELS.eol_persist)
            for i, org in enumerate(fd.origins):
                traj = res.trajectories[i]
                rows.append(pd.DataFrame({
                    "model": label, "cell": fold.test, "dataset": fold.dataset, "frac": org.frac,
                    "t0": org.t0, "cycle": org.t0 + np.arange(1, len(traj) + 1), "capacity": traj,
                }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
