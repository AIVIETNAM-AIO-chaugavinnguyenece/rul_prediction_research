"""Small helpers shared by the notebooks: output paths, tables, environment."""

from __future__ import annotations

import json
import platform
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FIGURES_DIR, PROCESSED_DIR, QUICK, RESULTS_DIR, snapshot


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


@contextmanager
def timer(label: str, log=print):
    start = time.perf_counter()
    yield
    log(f"{label}: {time.perf_counter() - start:.1f}s")


def save_table(df: pd.DataFrame, name: str, directory: Path = RESULTS_DIR, index: bool = False) -> Path:
    """Write a CSV to results/ (or another directory) and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    df.to_csv(path, index=index)
    return path


def load_table(name: str, directory: Path = RESULTS_DIR) -> pd.DataFrame:
    path = directory / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the notebook that produces it first")
    return pd.read_csv(path)


def save_json(obj, name: str, directory: Path = RESULTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def environment() -> dict:
    """Library versions, recorded next to every result set."""
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for dist in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost", "lightgbm", "shap", "matplotlib"):
        try:
            versions[dist] = version(dist)       # reads metadata; does not import the package
        except PackageNotFoundError:
            versions[dist] = "not installed"
    return versions


def record_run(notebook: str) -> dict:
    """Save the environment and configuration used by a notebook run."""
    info = {"notebook": notebook, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": environment(), "config": snapshot()}
    save_json(info, f"run_{notebook}")
    if QUICK:
        print("QUICK MODE (RUL_QUICK=1): one seed, small grids. Do not report these numbers.")
    return info


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """A GitHub-flavoured Markdown table, for pasting into the report."""
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except ImportError:  # tabulate missing
        return df.to_string(index=False)
