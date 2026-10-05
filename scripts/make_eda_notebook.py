"""Generate the EDA notebook (02_EDA.ipynb) programmatically."""
import json
from pathlib import Path

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines()}

def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines()}

cells = []

# ── Title ──
cells.append(md("""# 02 · Exploratory Data Analysis

**Battery RUL prediction — DLinear and NLinear vs tree-based models**

This notebook explores the capacity-fade trajectories of the 11 study cells (4 NASA PCoE, 7 CALCE CX2) to understand the data before any modelling. Each section ends with a design consequence.

| | |
|---|---|
| **Inputs** | `processed/cells/*.parquet`, `processed/cell_summary.csv` (from 01) |
| **Outputs** | `processed/regimes.csv`, `results/eda_*.csv`, `figures/02_*.png` |

Three properties matter most for the research questions:

* **Capacity regeneration** — sudden partial recoveries after rest. NLinear's last-value anchoring and DLinear's trend branch should respond differently (RQ1).
* **Level differences between cells** — cells start at different capacities, so a model trained on some cells must forecast others at levels it has not seen (RQ1 and RQ2).
* **How far the end-of-life region lies from the training data** — the premise of the RQ2 extrapolation test."""))

# ── Setup ──
cells.append(code("""import os, sys
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive; drive.mount("/content/drive")
    os.chdir("/content/battery-rul-linear")
    os.environ.setdefault("BATTERY_RUL_DATA", "/content/drive/MyDrive/battery_rul/data")
ROOT = Path.cwd().resolve()
if not (ROOT / "battery_rul").is_dir(): ROOT = ROOT.parent
os.environ.setdefault("BATTERY_RUL_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT))"""))

cells.append(code("""import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from battery_rul import config
from battery_rul.dataset import cells_by_dataset, forecast_origins, load_cells, loco_folds, regeneration_regimes
from battery_rul.features import build_training_set, make_windows
from battery_rul.plotting import INK, INK_2, MUTED, degradation_curves, savefig, setup_style
from battery_rul.utils import markdown_table, record_run, save_table
pd.set_option("display.width", 160)
setup_style()
record_run("02_EDA")
summary = pd.read_csv(config.PROCESSED_DIR / "cell_summary.csv")
summary = summary[summary["group"].isin(config.STUDY_GROUPS)].reset_index(drop=True)
cells = load_cells()
REF = config.LABELS.primary_reference
print(f"{len(cells)} study cells:", cells_by_dataset(cells))"""))

# ── Section 1: Overview ──
cells.append(md("## 1. Dataset overview"))
cells.append(code("""overview = summary.groupby("dataset").agg(
    cells=("cell", "size"), cycles_min=("n_cycles", "min"), cycles_max=("n_cycles", "max"),
    initial_Ah_min=("initial_Ah", "min"), initial_Ah_max=("initial_Ah", "max"),
    eol_rated_min=("eol_rated", "min"), eol_rated_max=("eol_rated", "max"),
    eol_initial_min=("eol_initial", "min"), eol_initial_max=("eol_initial", "max"),
    reach_eol=("reaches_eol", "sum")).reset_index()
save_table(overview, "eda_overview")
print(markdown_table(overview, floatfmt=".3f"))
overview"""))

# ── Section 2: Capacity fade ──
cells.append(md("""## 2. Capacity fade

Raw capacity per cell (left) and the same trajectories as state of health (right). The spread on the left that collapses on the right is level difference, not a difference in degradation behaviour."""))
cells.append(code("""for ds in cells_by_dataset(cells):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    degradation_curves(cells, ds, REF, ax=axes[0])
    for c in (c for c in cells.values() if c.dataset == ds):
        soh = c.capacity / c.references["initial"]
        axes[1].plot(np.arange(1, c.n_cycles + 1), soh, color=INK_2, lw=1, alpha=0.55)
    axes[1].axhline(config.LABELS.eol_frac, color=INK, lw=1, ls="--")
    axes[1].set(title=f"{ds}: state of health", xlabel="cycle", ylabel="SoH")
    fig.tight_layout(); savefig(fig, f"02_fade_{ds.lower()}"); plt.show()"""))

# ── Section 3: EOL conventions ──
cells.append(md("""## 3. Two end-of-life conventions

How much the choice of reference moves the EOL cycle. Cells whose initial capacity differs most from the rated value move most."""))
cells.append(code("""conv = summary[["cell", "dataset", "initial_over_rated", "eol_rated", "eol_initial"]].copy()
conv["eol_shift_cycles"] = conv["eol_initial"] - conv["eol_rated"]
fig, ax = plt.subplots(figsize=(6.5, 4))
for ds, g in conv.groupby("dataset"):
    ax.scatter(g["initial_over_rated"], g["eol_shift_cycles"], s=40, label=ds,
               color="#2a78d6" if ds == "NASA" else "#eb6834", edgecolor="white")
    for _, r in g.iterrows():
        ax.annotate(r["cell"], (r["initial_over_rated"], r["eol_shift_cycles"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points", color=INK_2)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set(xlabel="initial / rated capacity", ylabel="EOL(initial) - EOL(rated), cycles",
       title="How the reference choice moves end of life"); ax.legend()
savefig(fig, "02_eol_conventions"); plt.show(); conv"""))

# ── Section 4: Regeneration ──
cells.append(md("""## 4. Capacity regeneration and the RQ1 subgroups

A regeneration event is a rise of more than 1% of rated capacity between consecutive cycles. The subgroup split is fixed here, before any model is run: a median split of events per 100 cycles within each dataset."""))
cells.append(code("""regimes = regeneration_regimes(summary)
regimes.to_csv(config.PROCESSED_DIR / "regimes.csv", index=False)
regimes.sort_values(["dataset", "regen_per_100_cycles"])"""))

cells.append(code("""example = max(cells.values(), key=lambda c: c.info["regen_per_100_cycles"])
df = pd.read_parquet(config.PROCESSED_DIR / "cells" / f"{example.name}.parquet")
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(df["cycle"], df["capacity_input"], color=INK_2, lw=1.1, label="capacity (causal input)")
ev = df[df["is_regeneration"]]
ax.scatter(ev["cycle"], ev["capacity_input"], color="#eb6834", s=30, zorder=3,
           label=f"regeneration events ({len(ev)})")
ax.set(title=f"{example.name}: capacity regeneration (highest rate)", xlabel="cycle", ylabel="capacity (Ah)")
ax.legend(); savefig(fig, "02_regeneration_example"); plt.show()"""))

# ── Section 5: Fade rate ──
cells.append(md("""## 5. Degradation rate over life

Fade per cycle, smoothed over 15 cycles and expressed as a share of rated capacity. A rate that accelerates late in life (a knee) is what a straight-line extrapolation will miss."""))
cells.append(code("""fig, axes = plt.subplots(1, len(cells_by_dataset(cells)), figsize=(13, 3.8), squeeze=False)
for ax, ds in zip(axes[0], cells_by_dataset(cells)):
    for c in (c for c in cells.values() if c.dataset == ds):
        rate = -np.gradient(pd.Series(c.label).rolling(15, center=True, min_periods=1).mean())
        life = np.arange(1, c.n_cycles + 1) / (c.eol[REF] or c.n_cycles)
        ax.plot(life, 100 * rate / c.references["rated"], color=INK_2, lw=1, alpha=0.6)
    ax.axvline(1.0, color=INK, lw=0.8, ls="--")
    ax.set(title=f"{ds}: fade rate over life", xlabel="fraction of life (1 = EOL)",
           ylabel="fade, % of rated per cycle", xlim=(0, 1.3))
fig.tight_layout(); savefig(fig, "02_fade_rate"); plt.show()"""))

# ── Section 6: Training range ──
cells.append(md("""## 6. Where the end-of-life region sits relative to the training data

For each leave-one-cell-out fold: the lowest capacity the test cell reaches before its true EOL, against the lowest target value in the training windows. Under full-range training the training data usually extend below the threshold. Under the range-restricted arm, the end-of-life region is outside the training range by construction."""))
cells.append(code("""rows = []
for fold in loco_folds(cells, REF):
    w = config.WINDOWS[fold.dataset]
    test = cells[fold.test]; train = [cells[n] for n in fold.train]
    full = build_training_set(train, w.input_len, w.horizon, reference=REF)
    restricted = build_training_set(train, w.input_len, w.horizon, reference=REF,
                                    range_restrict_soh=config.RANGE_RESTRICT_SOH)
    eol = test.eol[REF]
    rows.append({"dataset": fold.dataset, "test_cell": fold.test,
        "threshold_Ah": test.thresholds[REF],
        "test_min_before_eol_Ah": float(test.capacity[:eol].min()),
        "train_target_min_full_Ah": float(full.Y_all.min()),
        "train_target_min_restricted_Ah": float(restricted.Y_all.min()),
        "windows_full": full.n_windows, "windows_restricted": restricted.n_windows})
coverage = pd.DataFrame(rows)
coverage["below_range_full"] = coverage["test_min_before_eol_Ah"] < coverage["train_target_min_full_Ah"]
coverage["below_range_restricted"] = coverage["test_min_before_eol_Ah"] < coverage["train_target_min_restricted_Ah"]
save_table(coverage, "eda_training_range")
print(f"folds below training range: full {coverage['below_range_full'].sum()}/{len(coverage)}, restricted {coverage['below_range_restricted'].sum()}/{len(coverage)}")
coverage"""))

# ── Section 7: Window budget ──
cells.append(md("""## 7. Window budget

A window needs L + H cycles, and a forecast origin at fraction f of life needs f x EOL >= L observed cycles. Short NASA lives constrain L hard; CALCE allows much longer windows."""))
cells.append(code("""budget_rows = []
for ds, names in cells_by_dataset(cells).items():
    for L in config.RQ3_INPUT_LENS[ds]:
        for H in config.RQ3_HORIZONS[ds]:
            n_windows = [len(make_windows(cells[n].capacity, L, H)[0]) for n in names]
            testable = [n for n in names if cells[n].reaches_eol(REF)]
            n_orig = sum(len(forecast_origins(cells[n], L, REF)) for n in testable)
            budget_rows.append({"dataset": ds, "L": L, "H": H,
                "min_windows_per_cell": min(n_windows), "total_windows": sum(n_windows),
                "origin_coverage": n_orig / max(1, len(testable) * len(config.ORIGIN_FRACS)),
                "is_default": (L, H) == (config.WINDOWS[ds].input_len, config.WINDOWS[ds].horizon)})
budget = pd.DataFrame(budget_rows)
save_table(budget, "eda_window_budget")
budget"""))

# ── Section 8: Design consequences ──
cells.append(md("## 8. Design consequences"))
cells.append(code("""for ds, g in summary.groupby("dataset"):
    spread = g["initial_Ah"].max() - g["initial_Ah"].min()
    print(f"{ds}: initial capacity spread {spread:.3f} Ah ({100*spread/g['rated_Ah'].iloc[0]:.1f}% of rated)")
    print(f"{ds}: regeneration {g['regen_per_100_cycles'].min():.2f}-{g['regen_per_100_cycles'].max():.2f} events/100 cycles")
defaults = budget[budget["is_default"]]
for _, r in defaults.iterrows():
    print(f"{r['dataset']}: default L={r['L']}, H={r['H']} serves {r['origin_coverage']:.0%} of origins")
print(f"range-restricted arm: {coverage['below_range_restricted'].sum()}/{len(coverage)} folds below range")
print("next: 03_feature_engineering.ipynb")"""))

# ── Assemble notebook ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "02_EDA.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Written: {out} ({len(cells)} cells)")
