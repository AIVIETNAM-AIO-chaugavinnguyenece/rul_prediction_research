"""Raw files -> processed per-cell tables (the engine behind notebook 01)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from .cleaning import CELL_COLUMNS, clean_capacity, label_cell
from .config import CELL_GROUPS, PROCESSED_DIR, dataset_of, group_of
from .data import discover, load_cell

Logger = Callable[[str], None]


def load_raw_cycles(name: str, kind: str, path: Path, cache_dir: Path, force: bool = False,
                    log: Logger = print) -> pd.DataFrame:
    """Per-cycle table straight from the raw files, cached as parquet.

    Reading ~50 Arbin workbooks per CALCE cell takes minutes; the cache makes
    re-runs take seconds. Delete ``processed/raw_cycles`` or pass ``force=True``
    after changing the loader.
    """
    cache = cache_dir / f"{name}.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    raw = load_cell(kind, path, log)
    cache.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache, index=False)
    return raw


def run_preprocessing(data_dir: Path, processed_dir: Path = PROCESSED_DIR, groups: tuple[str, ...] | None = None,
                      force: bool = False, min_cycles: int = 20, log: Logger = print
                      ) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str]]]:
    """Load, clean and label every known cell found under ``data_dir``.

    Returns ``(summary, drop_report, skipped)``. Cells outside ``CELL_GROUPS``
    (for example NASA's second campaign, B0025 onwards) are listed as skipped
    with the reason, never silently ignored.
    """
    groups = groups or tuple(CELL_GROUPS)
    wanted = {c for g in groups for c in CELL_GROUPS[g]}
    found = discover(data_dir)
    if not found:
        raise FileNotFoundError(f"no .mat files or CX2/CS2 folders under {Path(data_dir).resolve()}")

    infos, drops, skipped = [], [], []
    for name, (kind, path) in sorted(found.items()):
        if name not in wanted:
            skipped.append((name, "not in the configured cell groups"))
            continue
        log(f"{name} ({kind})")
        try:
            raw = load_raw_cycles(name, kind, path, processed_dir / "raw_cycles", force, log)
        except Exception as err:  # noqa: BLE001 - report every unreadable cell
            skipped.append((name, f"read error: {err}"))
            log(f"  ! skipped: {err}")
            continue
        df, report = clean_capacity(raw, name)
        drops.append({"cell": name, **report})
        if len(df) < min_cycles:
            skipped.append((name, f"only {len(df)} usable cycles"))
            log(f"  ! skipped: only {len(df)} usable cycles of {report['n_read']}")
            continue
        df, info = label_cell(df, name)
        info.update({"dataset": dataset_of(name), "group": group_of(name)})
        infos.append(info)
        out = processed_dir / "cells" / f"{name}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df[[c for c in CELL_COLUMNS if c in df.columns]].to_parquet(out, index=False)
        log(f"  {info['n_cycles']} cycles | initial {info['initial_Ah']:.3f} Ah | "
            f"EOL rated {info['eol_rated']} | EOL initial {info['eol_initial']} | "
            f"outliers {info['n_outliers_causal']} | regeneration {info['n_regeneration']}")

    summary = pd.DataFrame(infos)
    if not summary.empty:
        front = ["cell", "dataset", "group", "n_cycles", "rated_Ah", "initial_Ah"]
        summary = summary[front + [c for c in summary.columns if c not in front]]
        summary.to_csv(processed_dir / "cell_summary.csv", index=False)
    drop_report = pd.DataFrame(drops)
    drop_report.to_csv(processed_dir / "drop_report.csv", index=False)
    return summary, drop_report, skipped
