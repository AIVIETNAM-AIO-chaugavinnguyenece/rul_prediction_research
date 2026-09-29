"""Single source of truth for every setting the study uses.

Each notebook imports from here instead of re-declaring constants, so the six
notebooks cannot drift apart: one split rule, one metric definition, one set of
seeds, one window geometry per dataset.

Two environment variables change behaviour without editing code:

* ``BATTERY_RUL_DATA``  - folder holding the raw data (default: ``<repo>/data``;
                          relative paths are taken from the repository root)
* ``RUL_QUICK=1``       - smoke-test mode: one seed, small grids, fewer trees.
                          Use it for CI or to check the pipeline end to end;
                          never report numbers produced in quick mode.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(os.environ.get("BATTERY_RUL_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("BATTERY_RUL_DATA", PROJECT_ROOT / "data")).expanduser()
if not DATA_DIR.is_absolute():            # relative paths mean "relative to the repository",
    DATA_DIR = PROJECT_ROOT / DATA_DIR    # whichever folder the notebook kernel starts in
PROCESSED_DIR = PROJECT_ROOT / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

QUICK = os.environ.get("RUL_QUICK", "0") == "1"

# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------

#: Rated capacity (Ah) per cell family; used by the "80% of rated" EOL rule.
RATED_AH = {"NASA": 2.0, "CX2": 1.35, "CS2": 1.1}

#: Cell groups. The study set is ``STUDY_GROUPS``; the rest are loaded and
#: summarised in notebook 01 but kept out of modelling unless opted in.
CELL_GROUPS: dict[str, list[str]] = {
    # NASA PCoE Battery Data Set #5, the standard benchmark cells
    "nasa_core": ["B0005", "B0006", "B0007", "B0018"],
    # CALCE CX2, constant-current 0.5C cycling (CX2_31 is used if present)
    "calce_cc": ["CX2_16", "CX2_31", "CX2_33", "CX2_34", "CX2_35", "CX2_36", "CX2_37", "CX2_38"],
    # CALCE CX2 with non-standard protocols (3C, pulsed, temperature cycling)
    "calce_varied": ["CX2_3", "CX2_4", "CX2_8", "CX2_32"],
}
STUDY_GROUPS: tuple[str, ...] = ("nasa_core", "calce_cc")

#: Which dataset a group belongs to. Folds never mix datasets.
GROUP_DATASET = {"nasa_core": "NASA", "calce_cc": "CALCE", "calce_varied": "CALCE"}

#: CALCE files to ignore: special tests that are not regular cycling.
CALCE_EXCLUDE_PATTERNS = (r"self[\s_-]*discharge",)


def family_of(cell: str) -> str:
    """'CX2_34' -> 'CX2', 'B0005' -> 'NASA'."""
    up = cell.upper()
    if up.startswith("CX2"):
        return "CX2"
    if up.startswith("CS2"):
        return "CS2"
    return "NASA"


def dataset_of(cell: str) -> str:
    """'CX2_34' -> 'CALCE', 'B0005' -> 'NASA'."""
    return "NASA" if family_of(cell) == "NASA" else "CALCE"


def group_of(cell: str) -> str | None:
    for group, cells in CELL_GROUPS.items():
        if cell.upper() in cells:
            return group
    return None


# --------------------------------------------------------------------------
# Cleaning and labels
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CleaningConfig:
    """Outlier handling. Two series are produced from the same raw capacity:

    * ``capacity_input`` - **causal**: every value uses only the current and
      past cycles. This is the only series a model ever sees.
    * ``capacity_label`` - **centred** and smoothed. Used only to decide the
      true end-of-life cycle; never a model input.
    """

    min_capacity_frac: float = 0.05     # drop cycles below 5% of rated (aborted tests)
    max_capacity_frac: float = 1.25     # drop cycles above 125% of rated (glitches)
    hampel_window: int = 9              # cycles in the rolling-median window
    hampel_n_mad: float = 4.0           # flag points this many scaled MADs away
    smooth_window: int = 5              # centred moving average for labels only
    regen_jump_frac: float = 0.01       # rise > 1% of rated capacity = regeneration


@dataclass(frozen=True)
class LabelConfig:
    """End of life = first cycle at which the smoothed capacity stays at or
    below ``eol_frac`` x reference for ``eol_persist`` consecutive cycles."""

    eol_frac: float = 0.80
    eol_persist: int = 5
    initial_cycles: int = 5             # cycles averaged for the 'initial' reference
    primary_reference: str = "rated"    # 'rated' is primary; 'initial' is the sensitivity check


CLEANING = CleaningConfig()
LABELS = LabelConfig()
REFERENCES = ("rated", "initial")

# --------------------------------------------------------------------------
# Windows, horizons and forecast origins
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowConfig:
    """L past cycles in, H future cycles out. Chosen per dataset because NASA
    cells live ~130-170 cycles and CALCE cells over a thousand."""

    input_len: int
    horizon: int
    stride: int = 1

    @property
    def ma_kernel(self) -> int:
        """DLinear moving-average kernel: odd, about a quarter of the window,
        at least 3 (the official 25 assumes L = 336, far longer than ours)."""
        k = max(3, int(round(self.input_len / 4)))
        return k if k % 2 == 1 else k + 1


WINDOWS: dict[str, WindowConfig] = {
    "NASA": WindowConfig(input_len=16, horizon=8),
    "CALCE": WindowConfig(input_len=48, horizon=16),
}

#: Forecast origins as fractions of each cell's true life (EOL cycle).
ORIGIN_FRACS: tuple[float, ...] = (0.25, 0.50, 0.75)

#: Roll-outs stop after this multiple of the longest *training* cell's EOL
#: cycle. A forecast that never crosses the threshold by then is 'censored'.
ROLLOUT_CAP_FACTOR = 1.5

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

LINEAR_MODELS: tuple[str, ...] = ("DLinear", "NLinear", "Linear")
BASELINE_MODELS: tuple[str, ...] = ("Persistence", "LinearTrend")
TREE_MODELS: tuple[str, ...] = (
    "DecisionTree", "RandomForest", "AdaBoost", "GradientBoosting", "XGBoost", "LightGBM",
)
#: Tree targets: 'absolute' predicts future capacity in Ah (standard practice);
#: 'delta' predicts the change from the last observed value (level-free, the
#: tree analogue of NLinear's normalisation).
TREE_TARGET_MODES: tuple[str, ...] = ("absolute", "delta")


@dataclass(frozen=True)
class LinearTrainConfig:
    learning_rate: float = 5e-3         # the official LTSF-Linear rate for small datasets
    batch_size: int = 256
    max_epochs: int = 300 if not QUICK else 30
    patience: int = 30 if not QUICK else 5
    weight_decay: float = 0.0
    val_frac: float = 0.2               # last 20% of each training cell's windows


LINEAR_TRAIN = LinearTrainConfig()


def _n(full: int, quick: int) -> int:
    return quick if QUICK else full


#: Fixed hyper-parameters. This is not a tuning study: every model gets one
#: sensible configuration, declared here and in the report's appendix.
TREE_PARAMS: dict[str, dict] = {
    "DecisionTree": dict(min_samples_leaf=5),
    "RandomForest": dict(n_estimators=_n(300, 30), min_samples_leaf=2, max_features=0.5),
    # AdaBoost's prediction is a weighted median of its trees, so shallow base
    # learners leave only a few distinct output levels and roll-outs stall on a
    # plateau; depth 6 gives it a fair chance to resolve the capacity range.
    "AdaBoost": dict(n_estimators=_n(100, 20), learning_rate=0.05, base_max_depth=6),
    "GradientBoosting": dict(n_estimators=_n(150, 30), learning_rate=0.08, max_depth=3, subsample=0.8),
    "XGBoost": dict(n_estimators=_n(300, 30), learning_rate=0.05, max_depth=4,
                    subsample=0.8, colsample_bytree=0.8),
    "LightGBM": dict(n_estimators=_n(300, 30), learning_rate=0.05, num_leaves=15,
                     min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8),
}

#: Seeds. Linear models train by gradient descent, so seed variance is real
#: and reported. Tree seeds cover bagging/subsampling randomness.
LINEAR_SEEDS: tuple[int, ...] = (0,) if QUICK else (0, 1, 2, 3, 4)
TREE_SEEDS: tuple[int, ...] = (0,) if QUICK else (0, 1, 2)
#: Models whose fit does not depend on the seed run once (seed 0).
DETERMINISTIC_MODELS: tuple[str, ...] = ("DecisionTree", "Persistence", "LinearTrend")
#: The RQ2 range-restricted arm is a stress test; one seed per model keeps it affordable.
RANGE_ARM_SEEDS: tuple[int, ...] = (0,)

# --------------------------------------------------------------------------
# Evaluation and statistics
# --------------------------------------------------------------------------

#: The primary metric, declared once. Relative error lets NASA (RUL ~ tens of
#: cycles) and CALCE (hundreds) be pooled in one paired test.
PRIMARY_METRIC = "rel_abs_rul_error"
ALPHA_LAMBDA = 0.20                     # prediction within +/-20% of true RUL counts as accurate
ALPHA = 0.05                            # significance level, Holm-corrected within each RQ
N_BOOTSTRAP = _n(5000, 200)

#: RQ2 range-restricted arm: training windows whose targets fall below this
#: state of health are removed, so the end-of-life region is unseen in training.
RANGE_RESTRICT_SOH = 0.85

# --------------------------------------------------------------------------
# RQ3 grids
# --------------------------------------------------------------------------

RQ3_INPUT_LENS: dict[str, tuple[int, ...]] = {
    "NASA": (8, 12, 16, 24) if not QUICK else (8, 16),
    "CALCE": (16, 32, 48, 64, 96) if not QUICK else (16, 48),
}
RQ3_HORIZONS: dict[str, tuple[int, ...]] = {
    "NASA": (4, 8, 12) if not QUICK else (8,),
    "CALCE": (8, 16, 32) if not QUICK else (16,),
}
RQ3_SEEDS: tuple[int, ...] = (0,) if QUICK else (0, 1, 2)
#: Training-set sizes: number of training cells used per fold (capped by availability).
RQ3_TRAIN_CELLS: tuple[int, ...] = (1, 2, 3, 5, 7) if not QUICK else (1, 2)
#: Tree models (with target mode) carried into RQ3 next to the linear models.
RQ3_TREE_MODELS: tuple[tuple[str, str], ...] = (("LightGBM", "delta"), ("RandomForest", "absolute"))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

#: Colour follows the model, never its rank. Categorical slots in fixed order
#: (validated for colour-vision deficiency); baselines use neutral ink.
MODEL_COLORS: dict[str, str] = {
    "DLinear": "#2a78d6",
    "NLinear": "#eb6834",
    "DecisionTree": "#1baf7a",
    "RandomForest": "#eda100",
    "AdaBoost": "#e87ba4",
    "GradientBoosting": "#008300",
    "XGBoost": "#4a3aa7",
    "LightGBM": "#e34948",
    "Linear": "#52514e",
    "Persistence": "#898781",
    "LinearTrend": "#898781",
}
MODEL_LINESTYLES: dict[str, str] = {"Linear": "-.", "Persistence": ":", "LinearTrend": "--"}


def snapshot() -> dict:
    """Everything above as a JSON-serialisable dict, saved next to results."""
    return {
        "quick_mode": QUICK,
        "data_dir": str(DATA_DIR),
        "study_groups": list(STUDY_GROUPS),
        "cell_groups": CELL_GROUPS,
        "cleaning": asdict(CLEANING),
        "labels": asdict(LABELS),
        "windows": {k: {**asdict(v), "ma_kernel": v.ma_kernel} for k, v in WINDOWS.items()},
        "origin_fracs": list(ORIGIN_FRACS),
        "rollout_cap_factor": ROLLOUT_CAP_FACTOR,
        "linear_train": asdict(LINEAR_TRAIN),
        "tree_params": TREE_PARAMS,
        "linear_seeds": list(LINEAR_SEEDS),
        "tree_seeds": list(TREE_SEEDS),
        "deterministic_models": list(DETERMINISTIC_MODELS),
        "range_arm_seeds": list(RANGE_ARM_SEEDS),
        "primary_metric": PRIMARY_METRIC,
        "alpha_lambda": ALPHA_LAMBDA,
        "alpha": ALPHA,
        "range_restrict_soh": RANGE_RESTRICT_SOH,
        "rq3_input_lens": RQ3_INPUT_LENS,
        "rq3_horizons": RQ3_HORIZONS,
        "rq3_train_cells": list(RQ3_TRAIN_CELLS),
    }
