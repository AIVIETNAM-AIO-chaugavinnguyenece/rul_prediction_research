"""Readers for the two raw datasets.

Both loaders return one row per discharge cycle with at least::

    cycle        1..N, renumbered consecutively
    capacity_Ah  discharge capacity of that cycle

NASA PCoE (``.mat``)
    One struct per cell with a ``cycle`` array; only ``discharge`` entries carry
    ``Capacity``. Read with ``scipy.io.loadmat(simplify_cells=True)``.

CALCE (folder of Arbin ``.xlsx`` exports per cell)
    ``Discharge_Capacity(Ah)`` **accumulates within a file**, and some cells keep
    ``Cycle_Index`` constant across a file. Grouping by ``Cycle_Index`` therefore
    returns running totals (up to 140 Ah for a 1.35 Ah cell). Capacity is instead
    measured per *discharge segment*: a contiguous run of rows with negative
    current, taking the capacity gained from the reading just before the run to
    its maximum. This is correct whether the column accumulates or resets.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import CALCE_EXCLUDE_PATTERNS

Logger = Callable[[str], None]


def _silent(_msg: str) -> None:
    return None


# --------------------------------------------------------------------------
# NASA PCoE
# --------------------------------------------------------------------------

def load_nasa_mat(path: Path) -> pd.DataFrame:
    """One NASA PCoE ``.mat`` file -> capacity per discharge cycle."""
    from scipy.io import loadmat

    mat = loadmat(str(path), simplify_cells=True)
    keys = [k for k in mat if not k.startswith("__")]
    if not keys:
        raise ValueError(f"{path.name}: no data struct found")
    root = mat[keys[0]]
    cycles = root["cycle"] if isinstance(root, dict) and "cycle" in root else root
    if isinstance(cycles, dict):
        cycles = [cycles]
    try:
        cycles = list(cycles)
    except TypeError as err:
        raise ValueError(f"{path.name}: unexpected structure, no cycle list") from err

    rows = []
    for entry in cycles:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("type", "")).strip().lower() != "discharge":
            continue
        data = entry.get("data", {})
        if not isinstance(data, dict):
            continue
        cap = data.get("Capacity")
        cap = np.ravel(cap) if cap is not None else np.array([])
        if cap.size == 0:
            continue
        temps = np.ravel(data.get("Temperature_measured", []))
        times = np.ravel(data.get("Time", []))
        rows.append({
            "capacity_Ah": float(cap[0]),
            "ambient_C": float(np.ravel(entry.get("ambient_temperature", np.nan))[0]),
            "mean_temp_C": float(np.mean(temps)) if temps.size else np.nan,
            "max_temp_C": float(np.max(temps)) if temps.size else np.nan,
            "discharge_time_s": float(times[-1]) if times.size else np.nan,
        })

    if not rows:
        raise ValueError(f"{path.name}: no discharge cycles with a Capacity field")
    df = pd.DataFrame(rows)
    df.insert(0, "cycle", np.arange(1, len(df) + 1))
    return df


# --------------------------------------------------------------------------
# CALCE
# --------------------------------------------------------------------------

_DCAP_COLS = ["Discharge_Capacity(Ah)", "Discharge_Capacity (Ah)", "Discharge Capacity(Ah)"]
_CYCLE_COLS = ["Cycle_Index", "Cycle Index"]
_TIME_COLS = ["Test_Time(s)", "Test Time(s)", "Test_Time (s)"]
_STEP_COLS = ["Step_Time(s)", "Step Time(s)"]
_CURR_COLS = ["Current(A)", "Current (A)"]
_VOLT_COLS = ["Voltage(V)", "Voltage (V)"]
_DATE_COLS = ["Date_Time", "Date Time", "DateTime"]
_RES_COLS = ["Internal_Resistance(Ohm)", "Internal Resistance(Ohm)"]

CURRENT_TOL = 0.01      # A; below this magnitude the cell is at rest
MIN_SEGMENT_AH = 0.02   # discard micro-segments (noise, aborted steps)


def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lowered = {str(c).lower().replace(" ", "_"): c for c in df.columns}
    for c in candidates:
        hit = lowered.get(c.lower().replace(" ", "_"))
        if hit is not None:
            return hit
    return None


def _read_record_sheet(path: Path, log: Logger) -> pd.DataFrame | None:
    """The per-record sheet of an Arbin workbook (largest sheet with capacity)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sheets = pd.read_excel(path, sheet_name=None)
        except Exception as err:  # noqa: BLE001 - report and skip unreadable files
            log(f"      ! cannot read {path.name}: {err}")
            return None
    best = None
    for df in sheets.values():
        if _pick(df, _DCAP_COLS) is not None and len(df):
            if best is None or len(df) > len(best):
                best = df
    return best


def discharge_segments(df: pd.DataFrame) -> pd.DataFrame:
    """One row per discharge segment of an Arbin record sheet."""
    c_dcap = _pick(df, _DCAP_COLS)
    if c_dcap is None:
        return pd.DataFrame()
    df = df.reset_index(drop=True)
    dcap = pd.to_numeric(df[c_dcap], errors="coerce")
    c_cur, c_vol, c_time = _pick(df, _CURR_COLS), _pick(df, _VOLT_COLS), _pick(df, _TIME_COLS)
    c_step, c_cyc, c_res = _pick(df, _STEP_COLS), _pick(df, _CYCLE_COLS), _pick(df, _RES_COLS)
    cur = pd.to_numeric(df[c_cur], errors="coerce") if c_cur else None

    rising = dcap.diff().fillna(0) > 0
    if cur is not None and (cur < -CURRENT_TOL).any():
        is_dis = cur < -CURRENT_TOL                       # usual sign convention
    elif cur is not None and (cur > CURRENT_TOL).any() and rising.any():
        is_dis = (cur > CURRENT_TOL) & rising             # magnitude-only current column
    else:
        is_dis = rising                                   # capacity column only
    is_dis = is_dis.fillna(False).astype(bool)

    reset = dcap.diff().fillna(0) < -1e-9
    seg_id = ((is_dis != is_dis.shift(fill_value=False)) | reset).cumsum()

    volt = pd.to_numeric(df[c_vol], errors="coerce") if c_vol else None
    tt = pd.to_numeric(df[c_time], errors="coerce") if c_time else None
    st = pd.to_numeric(df[c_step], errors="coerce") if c_step else None
    res = pd.to_numeric(df[c_res], errors="coerce") if c_res else None
    cyc = pd.to_numeric(df[c_cyc], errors="coerce") if c_cyc else None

    rows = []
    for _, idx in df.groupby(seg_id).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2 or not is_dis.iloc[idx].any():
            continue
        seg_cap = dcap.iloc[idx]
        if seg_cap.notna().sum() < 2:
            continue
        # Baseline is the reading just before the segment: some exports log the
        # first in-segment row after the first increment.
        prev = dcap.iloc[idx[0] - 1] if idx[0] > 0 else np.nan
        seg_min, seg_max = float(seg_cap.min()), float(seg_cap.max())
        base = float(prev) if (np.isfinite(prev) and prev < seg_min and prev <= seg_max) else seg_min
        cap = seg_max - base
        if (not np.isfinite(cap) or cap < MIN_SEGMENT_AH) and seg_cap.nunique() == 1:
            cap = seg_max                                 # summary-style export
        if not np.isfinite(cap) or cap < MIN_SEGMENT_AH:
            continue
        row = {"capacity_Ah": cap}
        if tt is not None:
            row["test_time_s"] = float(tt.iloc[idx].max())
        if st is not None:
            row["discharge_time_s"] = float(st.iloc[idx].max())
        if volt is not None:
            row["v_min"] = float(volt.iloc[idx].min())
            row["v_mean"] = float(volt.iloc[idx].mean())
        if cur is not None:
            row["current_mean_A"] = float(cur.iloc[idx].abs().mean())
        if res is not None:
            r = res.iloc[idx]
            r = r[r > 0]
            if len(r):
                row["internal_resistance_ohm"] = float(r.median())
        if cyc is not None:
            row["cycle_index_raw"] = float(cyc.iloc[idx].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _is_excluded(path: Path) -> bool:
    return any(re.search(p, path.name, re.IGNORECASE) for p in CALCE_EXCLUDE_PATTERNS)


def load_calce_cell(cell_dir: Path, log: Logger = _silent) -> pd.DataFrame:
    """Every ``.xlsx`` in one CALCE cell folder, stitched in chronological order."""
    files = sorted(p for p in Path(cell_dir).rglob("*.xls*")
                   if not p.name.startswith(("~$", ".")))
    if not files:
        raise ValueError(f"{cell_dir}: no .xlsx files found")

    parts = []
    for path in files:
        if _is_excluded(path):
            log(f"      {path.name}: excluded (special test, not regular cycling)")
            continue
        sheet = _read_record_sheet(path, log)
        if sheet is None or sheet.empty:
            continue
        cycles = discharge_segments(sheet)
        if cycles.empty:
            log(f"      {path.name}: no discharge segments")
            continue
        cycles["source_file"] = path.name
        c_date = _pick(sheet, _DATE_COLS)
        start = pd.to_datetime(sheet[c_date], errors="coerce").min() if c_date else pd.NaT
        cycles["_start"] = start
        parts.append(cycles)
        log(f"      {path.name}: {len(cycles)} discharge cycles")

    if not parts:
        raise ValueError(f"{cell_dir}: no discharge segments in any file")

    def sort_key(d: pd.DataFrame):
        start = d["_start"].iloc[0]
        if pd.notna(start):
            return (0, pd.Timestamp(start).value, "")
        date = re.findall(r"(\d{1,2})[_-](\d{1,2})[_-](\d{2,4})", d["source_file"].iloc[0])
        if date:
            m, day, y = (int(x) for x in date[-1])
            y = y + 2000 if y < 100 else y
            return (1, y * 10000 + m * 100 + day, "")
        return (2, 0, d["source_file"].iloc[0])

    df = pd.concat(sorted(parts, key=sort_key), ignore_index=True).drop(columns="_start")
    df.insert(0, "cycle", np.arange(1, len(df) + 1))
    return df


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def discover(data_root: Path) -> dict[str, tuple[str, Path]]:
    """Map cell name -> (kind, path) for every ``.mat`` file and CX2/CS2 folder."""
    data_root = Path(data_root)
    found: dict[str, tuple[str, Path]] = {}
    for mat in sorted(data_root.rglob("*.mat")):
        found[mat.stem.upper()] = ("nasa", mat)
    for folder in sorted(p for p in data_root.rglob("*") if p.is_dir()):
        if re.fullmatch(r"(CX2|CS2)[_-]?\d+", folder.name, re.IGNORECASE) and any(folder.rglob("*.xls*")):
            found[folder.name.upper().replace("-", "_")] = ("calce", folder)
    return found


def load_cell(kind: str, path: Path, log: Logger = _silent) -> pd.DataFrame:
    """Dispatch to the right loader."""
    return load_nasa_mat(path) if kind == "nasa" else load_calce_cell(path, log)
