"""Processed cells, cross-validation folds and forecast origins.

Everything downstream of notebook 01 reads cells through :func:`load_cells`, so
the modelling notebooks share one view of the data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    GROUP_DATASET, LABELS, ORIGIN_FRACS, PROCESSED_DIR, ROLLOUT_CAP_FACTOR,
    STUDY_GROUPS, dataset_of, group_of,
)


@dataclass
class Cell:
    """One battery cell, as the models see it."""

    name: str
    dataset: str
    group: str | None
    capacity: np.ndarray            # causal input series (Ah) - the only model input
    label: np.ndarray               # centred, smoothed series (Ah) - for true EOL only
    thresholds: dict[str, float]    # EOL threshold per reference
    eol: dict[str, int | None]      # true EOL cycle per reference (1-based)
    references: dict[str, float]    # reference capacity per convention (rated / initial)
    info: dict = field(default_factory=dict)

    @property
    def n_cycles(self) -> int:
        return len(self.capacity)

    def reaches_eol(self, reference: str = LABELS.primary_reference) -> bool:
        return self.eol.get(reference) is not None


def load_cells(processed_dir: Path = PROCESSED_DIR, groups: tuple[str, ...] = STUDY_GROUPS,
               cells: list[str] | None = None) -> dict[str, Cell]:
    """Read the per-cell parquet files written by notebook 01."""
    processed_dir = Path(processed_dir)
    summary = pd.read_csv(processed_dir / "cell_summary.csv")
    if cells is not None:
        summary = summary[summary["cell"].isin([c.upper() for c in cells])]
    else:
        summary = summary[summary["group"].isin(groups)]

    out: dict[str, Cell] = {}
    for _, row in summary.sort_values(["dataset", "cell"]).iterrows():
        path = processed_dir / "cells" / f"{row['cell']}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)

        def _eol(value):
            return None if pd.isna(value) else int(value)

        out[row["cell"]] = Cell(
            name=row["cell"],
            dataset=row["dataset"],
            group=row["group"] if isinstance(row["group"], str) else None,
            capacity=df["capacity_input"].to_numpy(dtype=np.float64),
            label=df["capacity_label"].to_numpy(dtype=np.float64),
            thresholds={"rated": float(row["threshold_rated_Ah"]),
                        "initial": float(row["threshold_initial_Ah"])},
            eol={"rated": _eol(row["eol_rated"]), "initial": _eol(row["eol_initial"])},
            references={"rated": float(row["rated_Ah"]), "initial": float(row["initial_Ah"])},
            info=row.to_dict(),
        )
    return out


def cells_by_dataset(cells: dict[str, Cell]) -> dict[str, list[str]]:
    by: dict[str, list[str]] = {}
    for name, cell in cells.items():
        by.setdefault(cell.dataset, []).append(name)
    return {k: sorted(v) for k, v in sorted(by.items())}


@dataclass(frozen=True)
class Fold:
    dataset: str
    test: str
    train: tuple[str, ...]

    @property
    def fold_id(self) -> str:
        return f"{self.dataset}:{self.test}"


def loco_folds(cells: dict[str, Cell], reference: str = LABELS.primary_reference) -> list[Fold]:
    """Leave-one-cell-out folds within each dataset.

    Every cell is tested once, provided it reaches EOL (otherwise there is no
    true RUL to score). Training uses every other cell of the same dataset,
    including ones that never reach EOL: their trajectories are still informative.
    Folds never mix datasets, and never split a cell by time.
    """
    folds = []
    for dataset, names in cells_by_dataset(cells).items():
        for test in names:
            if not cells[test].reaches_eol(reference):
                continue
            train = tuple(n for n in names if n != test)
            if train:
                folds.append(Fold(dataset, test, train))
    return folds


@dataclass(frozen=True)
class Origin:
    """A forecast made at cycle ``t0``: the model sees cycles 1..t0 only."""

    cell: str
    frac: float
    t0: int                 # last observed cycle (1-based)
    true_eol: int
    true_rul: int           # true_eol - t0


def forecast_origins(cell: Cell, input_len: int, reference: str = LABELS.primary_reference,
                     fracs: tuple[float, ...] = ORIGIN_FRACS) -> list[Origin]:
    """Origins at fixed fractions of the cell's true life.

    An origin is kept only when at least ``input_len`` cycles have been observed
    and end of life has not yet been reached. Skipped origins are reported by
    the notebooks as coverage, not silently dropped.
    """
    eol = cell.eol.get(reference)
    if eol is None:
        return []
    out = []
    for frac in fracs:
        t0 = int(round(frac * eol))
        if t0 >= input_len and t0 < eol and t0 <= cell.n_cycles:
            out.append(Origin(cell.name, frac, t0, eol, eol - t0))
    return out


def rollout_cap(train_cells: list[Cell], reference: str = LABELS.primary_reference) -> int:
    """Maximum roll-out length, from *training* cells only (no test information)."""
    eols = [c.eol[reference] for c in train_cells if c.eol.get(reference) is not None]
    longest = max(eols) if eols else max(c.n_cycles for c in train_cells)
    return int(np.ceil(ROLLOUT_CAP_FACTOR * longest))


def regeneration_regimes(summary: pd.DataFrame) -> pd.DataFrame:
    """Split cells into 'high' / 'low' capacity-regeneration regimes.

    Median split of regeneration events per 100 cycles, *within each dataset*
    (the datasets differ in protocol and length). Defined here, from the data
    summary alone, before any model is run - so the RQ1 subgroup analysis
    cannot be tuned after seeing results.
    """
    out = []
    for dataset, g in summary.groupby("dataset"):
        med = g["regen_per_100_cycles"].median()
        for _, row in g.iterrows():
            out.append({"cell": row["cell"], "dataset": dataset,
                        "regen_per_100_cycles": row["regen_per_100_cycles"],
                        "regime": "high" if row["regen_per_100_cycles"] > med else "low",
                        "dataset_median": med})
    return pd.DataFrame(out)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def dataset_of_group(group: str) -> str:
    return GROUP_DATASET[group]


__all__ = [
    "Cell", "Fold", "Origin", "load_cells", "cells_by_dataset", "loco_folds",
    "forecast_origins", "rollout_cap", "regeneration_regimes", "save_json", "dataset_of", "group_of",
]
