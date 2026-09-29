"""Generate SYNTHETIC battery data in the exact raw formats of NASA PCoE and CALCE.

For testing the pipeline only - never for reported results. The files mimic the
structures the loaders must handle:

* NASA ``.mat``: a struct per cell with a ``cycle`` array of charge / discharge /
  impedance entries; discharge entries carry ``Capacity``.
* CALCE ``.xlsx``: Arbin exports with ``Discharge_Capacity(Ah)`` accumulating
  within each file, ``Cycle_Index`` restarting per file, a decoy "Info" sheet,
  and one self-discharge test file that the loader must exclude.

Usage:
    python scripts/make_synthetic_data.py --out data_synthetic [--calce-cycles 450]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


def fade_curve(n: int, c0: float, end_frac: float, knee: float, rng: np.random.Generator,
               regen_every: int, regen_size: float, noise: float) -> np.ndarray:
    """Capacity fade with a knee, capacity regeneration after rests, noise, outliers."""
    t = np.arange(n) / n
    cap = c0 * (1 - (1 - end_frac) * (0.6 * t + 0.4 * t ** knee))
    regen = np.zeros(n)
    for k in range(regen_every, n, regen_every):
        jump = regen_size * c0 * rng.uniform(0.5, 1.5)
        decay = np.exp(-np.arange(n - k) / rng.uniform(4, 10))
        regen[k:] += jump * decay
    cap = cap + regen + rng.normal(0, noise * c0, n)
    for k in rng.choice(np.arange(10, n), size=max(1, n // 150), replace=False):
        cap[k] *= rng.uniform(0.7, 0.85)       # sensor glitches the cleaning must handle
    return cap


def write_nasa(out: Path, rng: np.random.Generator) -> None:
    folder = out / "NASA_PCoE"
    folder.mkdir(parents=True, exist_ok=True)
    specs = {"B0005": (168, 1.86, 0.68), "B0006": (168, 2.03, 0.58),
             "B0007": (168, 1.89, 0.76), "B0018": (132, 1.86, 0.72)}
    for name, (n, c0, end) in specs.items():
        caps = fade_curve(n, c0, end, knee=2.2, rng=rng, regen_every=int(rng.integers(18, 30)),
                          regen_size=0.025, noise=0.004)
        cycles = []
        for c in caps:
            cycles.append({"type": "charge", "ambient_temperature": 24.0, "time": np.zeros(6),
                           "data": {"Voltage_measured": np.linspace(3.5, 4.2, 20)}})
            cycles.append({"type": "discharge", "ambient_temperature": 24.0, "time": np.zeros(6),
                           "data": {"Capacity": np.array([[c]]),
                                    "Temperature_measured": 24 + rng.normal(8, 1, 40),
                                    "Time": np.linspace(0, 3300 * c / c0, 40)}})
            if rng.random() < 0.1:
                cycles.append({"type": "impedance", "ambient_temperature": 24.0, "time": np.zeros(6),
                               "data": {"Re": 0.05}})
        savemat(folder / f"{name}.mat", {name: {"cycle": np.array(cycles, dtype=object)}})
        print(f"  NASA {name}: {n} discharge cycles")


def _records(caps: np.ndarray, start_cycle: int, stamp0: pd.Timestamp) -> pd.DataFrame:
    rows, cum_d, cum_c = [], 0.0, 0.0
    for local, cap in enumerate(caps, start=1):
        base_t = 3600.0 * (start_cycle + local)
        stamp = stamp0 + pd.Timedelta(hours=local * 3)
        for j in range(3):                                    # CC charge
            cum_c += cap / 3
            rows.append((base_t + 60 * j, stamp, 60 * j, 1, local, 0.675, 3.85 + 0.1 * j, cum_c, cum_d, 0.09))
        for j in range(2):                                    # CV charge
            cum_c += 0.005
            rows.append((base_t + 300 + 60 * j, stamp, 90 * j, 2, local, 0.10, 4.20, cum_c, cum_d, 0.09))
        rows.append((base_t + 500, stamp, 0, 3, local, 0.0, 4.10, cum_c, cum_d, 0.0))   # rest
        for j in range(5):                                    # discharge, capacity accumulates
            cum_d += cap / 5
            rows.append((base_t + 900 + 70 * j, stamp, 70 * j, 4, local, -0.675, 4.0 - 0.25 * j,
                         cum_c, cum_d, 0.09 + 0.0001 * start_cycle))
    cols = ["Test_Time(s)", "Date_Time", "Step_Time(s)", "Step_Index", "Cycle_Index", "Current(A)",
            "Voltage(V)", "Charge_Capacity(Ah)", "Discharge_Capacity(Ah)", "Internal_Resistance(Ohm)"]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "Data_Point", np.arange(1, len(df) + 1))
    return df


def write_calce(out: Path, rng: np.random.Generator, n_cycles: int) -> None:
    folder = out / "CALCE_CX2"
    cells = ["CX2_16", "CX2_33", "CX2_34", "CX2_35", "CX2_36", "CX2_37", "CX2_38"]
    for i, name in enumerate(cells):
        d = folder / name
        d.mkdir(parents=True, exist_ok=True)
        n = int(n_cycles * rng.uniform(0.85, 1.15))
        caps = fade_curve(n, 1.33 - 0.01 * i, rng.uniform(0.45, 0.6), knee=2.8, rng=rng,
                          regen_every=int(rng.integers(40, 70)), regen_size=0.012, noise=0.003)
        start, file_no = 0, 0
        stamp0 = pd.Timestamp("2010-09-07")
        while start < n:
            stop = min(start + 50, n)
            df = _records(caps[start:stop], start, stamp0 + pd.Timedelta(days=7 * file_no))
            day = stamp0 + pd.Timedelta(days=7 * file_no)
            path = d / f"{name}_{day.month}_{day.day}_{str(day.year)[2:]}.xlsx"
            with pd.ExcelWriter(path) as w:
                pd.DataFrame({"Item": ["Cell", "Channel"], "Value": [name, 5]}).to_excel(
                    w, sheet_name="Info", index=False)
                df.to_excel(w, sheet_name=f"Channel_1-00{i % 9}", index=False)
            start, file_no = stop, file_no + 1
        # a special test the loader must skip
        junk = _records(np.full(3, 0.2), 0, stamp0)
        with pd.ExcelWriter(d / f"{name}_2_13_12_self discharge test.xlsx") as w:
            junk.to_excel(w, sheet_name="Channel_1-001", index=False)
        print(f"  CALCE {name}: {n} cycles in {file_no} files (+1 excluded self-discharge file)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_synthetic")
    ap.add_argument("--calce-cycles", type=int, default=450)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    print(f"Writing SYNTHETIC data to {out.resolve()} (for pipeline tests only)")
    write_nasa(out, rng)
    write_calce(out, rng, args.calce_cycles)
    (out / "SYNTHETIC_DATA_NOT_REAL.txt").write_text(
        "Generated by scripts/make_synthetic_data.py for pipeline testing. Not real battery data.\n")


if __name__ == "__main__":
    main()
