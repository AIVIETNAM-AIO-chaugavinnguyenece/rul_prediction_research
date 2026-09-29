"""Figures for the notebooks and the report, in one consistent style.

Conventions (validated palette, see config.MODEL_COLORS):

* colour follows the model, never its rank; baselines use neutral ink with a
  distinct line style so identity never rests on colour alone;
* one y-axis per chart; thin marks; hairline grid; legends always present for
  two or more series, and every figure has a results table behind it;
* signed quantities (weights) use a diverging blue-red scale with a grey midpoint.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .config import FIGURES_DIR, MODEL_COLORS, MODEL_LINESTYLES

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#184f95", "#6da7ec", "#f0efec", "#ef8a73", "#b3261e"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 10, "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2.0,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def base_model(label: str) -> str:
    """'LightGBM (delta)' -> 'LightGBM'."""
    return label.split(" (")[0]


def color_of(label: str) -> str:
    return MODEL_COLORS.get(base_model(label), INK_2)


def style_of(label: str) -> dict:
    ls = MODEL_LINESTYLES.get(base_model(label), "-")
    if "(delta)" in label:
        ls = (0, (4, 1.5))            # delta-target trees: dashed variant of their colour
    return {"color": color_of(label), "linestyle": ls}


def savefig(fig, name: str, directory: Path = FIGURES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    return path


# --------------------------------------------------------------------------
# Data figures
# --------------------------------------------------------------------------

def _end_labels(ax, points: list[tuple[float, float, str]], min_gap_frac: float = 0.045) -> None:
    """Direct labels in a right-hand gutter, nudged apart so they never overlap,
    each joined to its line end by a hairline leader."""
    if not points:
        return
    lo, hi = ax.get_ylim()
    gap = min_gap_frac * (hi - lo)
    x_max = max(p[0] for p in points)
    x_lab = x_max * 1.04
    ordered = sorted(points, key=lambda p: p[1])
    placed: list[float] = []
    for _, y, _ in ordered:
        placed.append(y if not placed else max(y, placed[-1] + gap))
    if placed[-1] > hi - gap / 2:            # stack ran off the top: slide it down
        drop = placed[-1] - (hi - gap / 2)
        placed = [y - drop for y in placed]
    for (x, y, text), y_lab in zip(ordered, placed):
        ax.plot([x, x_lab * 0.995], [y, y_lab], color=AXIS, lw=0.5, zorder=1)
        ax.text(x_lab, y_lab, text, fontsize=8, color=INK_2, va="center", ha="left")
    ax.set_xlim(ax.get_xlim()[0], x_max * 1.16)


def degradation_curves(cells: dict, dataset: str, reference: str = "rated", ax=None):
    """Capacity against cycle for every cell of one dataset, with EOL marked."""
    ax = ax or plt.subplots(figsize=(7.5, 4.2))[1]
    members = [c for c in cells.values() if c.dataset == dataset]
    for cell in members:
        cyc = np.arange(1, cell.n_cycles + 1)
        ax.plot(cyc, cell.capacity, color=INK_2, lw=1.0, alpha=0.55)
    _end_labels(ax, [(c.n_cycles, float(c.capacity[-1]), c.name) for c in members])
    for cell in members:
        eol = cell.eol.get(reference)
        if eol:
            ax.plot(eol, cell.label[eol - 1], marker="o", ms=5, color=INK, zorder=3)
    if members:
        thr = np.mean([c.thresholds[reference] for c in members])
        ax.axhline(thr, color=INK, lw=1, ls="--")
        ax.text(ax.get_xlim()[0], thr, f" EOL threshold ({reference}) {thr:.2f} Ah",
                va="bottom", fontsize=8, color=INK)
    ax.set(title=f"{dataset}: capacity fade (dot = true end of life)", xlabel="cycle",
           ylabel="discharge capacity (Ah)")
    return ax


def error_distribution(units: pd.DataFrame, metric: str, models: list[str], ax=None,
                       title: str | None = None, log: bool = False, clip_quantile: float | None = 0.95):
    """Per-unit errors by model: box (median, IQR) plus the individual units.

    Units beyond the ``clip_quantile`` of all values are marked at the axis edge
    with their count rather than stretching the axis (set None to show all).
    """
    ax = ax or plt.subplots(figsize=(7.5, 0.45 * len(models) + 1.5))[1]
    order = (units[units["model"].isin(models)].groupby("model")[metric].median()
             .sort_values(ascending=False).index.tolist())
    data = [units.loc[units["model"] == m, metric].dropna().to_numpy() for m in order]
    bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "lw": 1.5})
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(color_of(m)); patch.set_alpha(0.35); patch.set_edgecolor(color_of(m))
    # Cap the axis at a high percentile so a few extreme units do not flatten the
    # rest; points beyond the cap are drawn at the edge as '>' with a count.
    pooled = np.concatenate([d for d in data if len(d)]) if data else np.array([0.0])
    cap = None if (log or clip_quantile is None) else float(np.nanquantile(pooled, clip_quantile)) * 1.15
    rng = np.random.default_rng(0)
    for i, (vals, m) in enumerate(zip(data, order), start=1):
        inside = vals if cap is None else vals[vals <= cap]
        ax.scatter(inside, i + rng.uniform(-0.15, 0.15, len(inside)), s=10, color=color_of(m),
                   edgecolor=SURFACE, linewidth=0.4, zorder=3)
        if cap is not None and (vals > cap).any():
            n_out = int((vals > cap).sum())
            ax.scatter([cap], [i], marker=">", s=28, color=color_of(m), zorder=4, clip_on=False)
            ax.annotate(f"{n_out}", (cap, i), xytext=(6, 0), textcoords="offset points", fontsize=7,
                        color=INK_2, va="center", annotation_clip=False)
    if cap is not None:
        ax.set_xlim(left=min(0, float(np.nanmin(pooled))), right=cap)
    ax.set_yticks(range(1, len(order) + 1), order)
    if log:
        ax.set_xscale("log")
    xlabel = metric.replace("_", " ")
    if cap is not None:
        xlabel += "  (> = units beyond the axis, with count)"
    ax.set(xlabel=xlabel, title=title or f"{metric.replace('_', ' ')} by model")
    ax.grid(axis="y", visible=False)
    return ax


def rollout_panel(examples: pd.DataFrame, cell, frac: float, reference: str = "rated", ax=None,
                  models: list[str] | None = None, legend_loc: str = "upper right"):
    """Observed capacity, each model's roll-out from one origin, and the threshold."""
    ax = ax or plt.subplots(figsize=(7.5, 4.2))[1]
    sub = examples[(examples["cell"] == cell.name) & (np.isclose(examples["frac"], frac))]
    if models is not None:
        sub = sub[sub["model"].isin(models)]
    cyc = np.arange(1, cell.n_cycles + 1)
    ax.plot(cyc, cell.capacity, color=INK, lw=1.2, label="observed")
    thr = cell.thresholds[reference]
    ax.axhline(thr, color=INK, lw=0.8, ls="--")
    if len(sub):
        t0 = int(sub["t0"].iloc[0])
        ax.axvline(t0, color=MUTED, lw=0.8)
        ax.text(t0, ax.get_ylim()[1], " forecast origin", fontsize=8, color=MUTED, va="top")
        rank = list(MODEL_COLORS)           # legend in the study's fixed model order
        labels = sorted(sub["model"].unique(),
                        key=lambda m: (rank.index(base_model(m)) if base_model(m) in rank else len(rank), m))
        for label in labels:
            g = sub[sub["model"] == label]
            ax.plot(g["cycle"], g["capacity"], lw=1.6, label=label, **style_of(label))
        ax.set_xlim(0, max(cell.n_cycles, int(sub["cycle"].max())) * 1.02)
    eol = cell.eol.get(reference)
    if eol:
        ax.plot(eol, thr, marker="o", ms=6, color=INK, zorder=4)
    lo = min(thr, np.nanmin(cell.capacity)) - 0.05 * cell.references["rated"]
    ax.set_ylim(lo, np.nanmax(cell.capacity) * 1.03)
    ax.text(ax.get_xlim()[1], thr, f"EOL threshold ({reference}) ", ha="right", va="bottom",
            fontsize=8, color=INK)
    ax.set(title=f"{cell.name}: roll-out from {int(frac * 100)}% of life", xlabel="cycle",
           ylabel="capacity (Ah)")
    ax.legend(loc=legend_loc, fontsize=8)
    return ax


def metric_vs_setting(df: pd.DataFrame, x: str, y: str, hue: str = "model", ax=None,
                      title: str = "", band: tuple[str, str] | None = None):
    """Line per model across a setting (window length, training cells...)."""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    for label, g in df.groupby(hue):
        g = g.sort_values(x)
        ax.plot(g[x], g[y], marker="o", ms=5, label=label, **style_of(label))
        if band is not None:
            ax.fill_between(g[x], g[band[0]], g[band[1]], color=color_of(label), alpha=0.12, lw=0)
    ax.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "), title=title)
    ax.legend(fontsize=8)
    return ax


def weight_heatmap(W: np.ndarray, title: str, ax=None, fig=None):
    """Signed weights, horizon step (rows) x input lag (columns), diverging scale."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
    lim = float(np.nanmax(np.abs(W))) or 1.0
    im = ax.imshow(W, aspect="auto", cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim),
                   interpolation="nearest")
    L = W.shape[1]
    ticks = np.linspace(0, L - 1, min(L, 6)).astype(int)
    ax.set_xticks(ticks, [f"t-{L - 1 - t}" if t < L - 1 else "t" for t in ticks])
    ax.set(xlabel="input position (t = last observed cycle)", ylabel="forecast step h", title=title)
    ax.grid(False)
    if fig is not None:
        fig.colorbar(im, ax=ax, shrink=0.85, label="weight")
    return ax


def bar_with_ci(df: pd.DataFrame, value: str, lo: str, hi: str, label_col: str = "model", ax=None,
                title: str = "", xlabel: str = ""):
    ax = ax or plt.subplots(figsize=(7, 0.42 * len(df) + 1.4))[1]
    df = df.sort_values(value, ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    ax.barh(y, df[value], color=[color_of(m) for m in df[label_col]], height=0.6, alpha=0.85)
    ax.errorbar(df[value], y, xerr=[df[value] - df[lo], df[hi] - df[value]], fmt="none",
                ecolor=INK, elinewidth=1, capsize=3)
    ax.set_yticks(y, df[label_col])
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis="y", visible=False)
    return ax
