"""Append a '9. Data caveats and anomalies' section to notebooks/02_EDA.ipynb.

Adds cells that surface three findings not produced by the original EDA
template: the NASA Time-field duplication, the CX2_16 outlier profile, and
the small-sample fragility of the regeneration regime split.
"""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "02_EDA.ipynb"

nb = nbf.read(NB_PATH, as_version=4)


def md(src):
    return nbf.v4.new_markdown_cell(src)


def code(src):
    return nbf.v4.new_code_cell(src)


new_cells = []

new_cells.append(md("""## 9. Data caveats and anomalies

The checks above establish the study design. This section goes one level deeper into the raw files
and flags three properties that the standard overview does not surface, each with a concrete
consequence for modelling or reporting."""))

# --- 9.1 NASA Time duplication -------------------------------------------------
new_cells.append(md("""### 9.1 NASA `Time` field is identical across B0005, B0006 and B0007

`discharge_time_s` (derived from the `.mat` file's `Time` array) is **byte-identical, cycle for
cycle, across B0005, B0006 and B0007** — confirmed directly from the raw `.mat` structs, not a
caching artefact. Voltage and capacity differ correctly between the three cells; only the timing
grid is shared. This points to how this copy of the NASA PCoE set was assembled (a shared sampling
clock recorded once and attached to all three logs), not a corruption in this repository's loader."""))

new_cells.append(code("""import scipy.io as sio

time_cols = {}
for name in ("B0005", "B0006", "B0007", "B0018"):
    df = pd.read_parquet(config.PROCESSED_DIR / "cells" / f"{name}.parquet")
    time_cols[name] = df["discharge_time_s"].to_numpy()

pairs = [("B0005", "B0006"), ("B0005", "B0007"), ("B0006", "B0007"), ("B0005", "B0018")]
rows = [{"pair": f"{a} vs {b}",
         "n_a": len(time_cols[a]), "n_b": len(time_cols[b]),
         "identical_on_overlap": bool(np.array_equal(
             time_cols[a][:min(len(time_cols[a]), len(time_cols[b]))],
             time_cols[b][:min(len(time_cols[a]), len(time_cols[b]))]))}
        for a, b in pairs]
time_dup = pd.DataFrame(rows)
save_table(time_dup, "eda_nasa_time_duplication")
time_dup"""))

new_cells.append(md("""**Consequence:** harmless for the current study. `battery_rul/features.py` builds every model
input from the causal capacity series only (`cap_*`, lags, rolling stats) — `discharge_time_s` is
stored in the processed parquet files but never read as a feature. It only becomes a hazard if a
future extension (RQ3's explainability work, or a teammate's exported-sample pipeline) adds
discharge duration as a predictor without knowing three of the four NASA cells share one timing
array: that feature would then carry zero cell-specific information for 3/4 of the dataset,
silently inflating any model that leans on it as if it were informative."""))

# --- 9.2 CX2_16 outlier profile -------------------------------------------------
new_cells.append(md("""### 9.2 CX2_16 is an outlier within its own CALCE family

CX2_16 differs from its six CALCE siblings on every axis that matters for degradation, not just one:
substantially slower fade, the longest life, the highest regeneration rate, *and* a markedly weaker
capacity/internal-resistance relationship. One anomalous property could be sampling noise; this
combination suggests a different degradation regime, not just a different point on the same curve."""))

new_cells.append(code("""calce_cells = [c for c in cells.values() if c.dataset == "CALCE"]
rows = []
for c in calce_cells:
    df = pd.read_parquet(config.PROCESSED_DIR / "cells" / f"{c.name}.parquet")
    corr_r = df["capacity_input"].corr(df["internal_resistance_ohm"])
    rows.append({
        "cell": c.name,
        "n_cycles": c.n_cycles,
        "eol_rated": c.eol["rated"],
        "fade_rate_mAh_per_cycle": c.info["fade_rate_mAh_per_cycle"],
        "regen_per_100_cycles": c.info["regen_per_100_cycles"],
        "corr_capacity_vs_resistance": round(corr_r, 3),
    })
outlier_profile = pd.DataFrame(rows).sort_values("fade_rate_mAh_per_cycle")
save_table(outlier_profile, "eda_calce_outlier_profile")
outlier_profile"""))

new_cells.append(code("""fig, ax = plt.subplots(figsize=(6.5, 4.2))
for c in calce_cells:
    df = pd.read_parquet(config.PROCESSED_DIR / "cells" / f"{c.name}.parquet")
    color = "#b3261e" if c.name == "CX2_16" else INK_2
    lw = 1.8 if c.name == "CX2_16" else 1.0
    alpha = 1.0 if c.name == "CX2_16" else 0.5
    ax.scatter(df["capacity_input"], df["internal_resistance_ohm"], s=4, color=color, alpha=alpha,
               label=c.name if c.name == "CX2_16" else None)
ax.set(xlabel="capacity (Ah)", ylabel="internal resistance (Ohm)",
       title="CALCE: capacity vs internal resistance (CX2_16 highlighted)")
ax.legend(fontsize=8)
savefig(fig, "02_cx2_16_outlier"); plt.show()"""))

new_cells.append(md("""**Consequence:** under leave-one-cell-out, whichever fold holds out CX2_16 as the *test* cell is
the one most likely to produce the largest RUL error for every model family, linear or tree — the
model trains on six faster-fading, more resistance-coupled cells and is asked to forecast a
slower-fading, decoupled one. Conversely, when CX2_16 is in the *training* set, it pulls the pooled
CALCE training distribution toward "slow fade, high regeneration" in a way the other six do not
support, which can bias every other fold's forecast origin at 75% of life (the region CX2_16's long
tail disproportionately populates). RQ1/RQ2 results should report CX2_16's fold separately from the
other six CALCE folds, or include a with/without-CX2_16 sensitivity row, rather than averaging it
into one CALCE number."""))

# --- 9.3 Sample size and regime-split fragility --------------------------------
new_cells.append(md("""### 9.3 Sample size: regeneration regime split is thin by construction

The whole study runs on 11 cells (33 test units: 11 cells x 3 forecast origins). The regeneration
"high/low" split used for hypothesis H1b is a *median* split within each dataset, which — with only
7 CALCE and 4 NASA cells — cannot produce anything but small, slightly unbalanced groups."""))

new_cells.append(code("""regime_sizes = regimes.groupby(["dataset", "regime"]).size().rename("n_cells").reset_index()
regime_sizes["n_test_units"] = regime_sizes["n_cells"] * len(config.ORIGIN_FRACS)
save_table(regime_sizes, "eda_regime_group_sizes")
regime_sizes"""))

new_cells.append(md("""**Consequence:** the smallest subgroup (NASA, either regime) is 2 cells = 6 test units. Any
H1b verdict ("not detected") on a 2-vs-2 or 3-vs-4 cell split should be read as *underpowered*, not
as evidence of no effect — consistent with the limitation the project README already states in
general terms, but worth stating numerically here, at the point where the split is constructed,
rather than only in the final report's limitations section."""))

new_cells.append(code("""print("Data caveats summary:")
print(f"  - NASA Time duplication: {int(time_dup['identical_on_overlap'].sum())}/{len(time_dup)} "
      f"checked pairs share an identical timing array (not used as a model feature)")
print(f"  - CX2_16 vs CALCE siblings: fade rate "
      f"{outlier_profile.loc[outlier_profile['cell'] == 'CX2_16', 'fade_rate_mAh_per_cycle'].iloc[0]:.3f} "
      f"mAh/cycle vs {outlier_profile.loc[outlier_profile['cell'] != 'CX2_16', 'fade_rate_mAh_per_cycle'].min():.3f}"
      f"-{outlier_profile.loc[outlier_profile['cell'] != 'CX2_16', 'fade_rate_mAh_per_cycle'].max():.3f} for the other six; "
      f"capacity/resistance correlation {outlier_profile.loc[outlier_profile['cell'] == 'CX2_16', 'corr_capacity_vs_resistance'].iloc[0]:.2f} "
      f"vs {outlier_profile.loc[outlier_profile['cell'] != 'CX2_16', 'corr_capacity_vs_resistance'].min():.2f}"
      f"-{outlier_profile.loc[outlier_profile['cell'] != 'CX2_16', 'corr_capacity_vs_resistance'].max():.2f} for the other six "
      f"-> report CX2_16's fold separately in RQ1/RQ2")
print(f"  - smallest regeneration-regime subgroup: {int(regime_sizes['n_cells'].min())} cells "
      f"({int(regime_sizes['n_test_units'].min())} test units) -> treat H1b verdicts as low-powered")"""))

nb.cells.extend(new_cells)
nbf.write(nb, NB_PATH)
print(f"Appended {len(new_cells)} cells. Notebook now has {len(nb.cells)} cells.")
