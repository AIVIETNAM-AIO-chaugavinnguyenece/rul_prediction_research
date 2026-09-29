"""Pre-registered hypotheses and computed verdicts.

Every hypothesis statement lives here, written before any model was run, and
every verdict the notebooks print is computed from a test result - never typed
by hand. That keeps the technical report and the code from disagreeing (the
main failure found when reviewing the sales-forecasting project).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import ALPHA
from .stats import holm, verdict

HYPOTHESES: dict[str, str] = {
    # RQ1 - DLinear vs NLinear
    "H1a": "NLinear achieves lower relative RUL error than DLinear across all units.",
    "H1b": "NLinear's advantage over DLinear is larger in high-regeneration cells than in "
           "low-regeneration cells.",
    # RQ2 - linear vs tree models
    "H2a": "Each linear model (DLinear, NLinear) achieves lower relative RUL error than each "
           "absolute-target tree model under full-range training.",
    "H2b": "When the end-of-life region is removed from training (range-restricted arm), "
           "absolute-target tree models degrade more than NLinear.",
    "H2c": "Under range restriction, delta-target tree models degrade less than the same model "
           "with absolute targets (target normalisation, not model family, drives extrapolation).",
    # RQ3 - practical settings and explainability
    "H3a": "Longer input windows reduce DLinear's error more than NLinear's "
           "(DLinear's trend branch needs context; NLinear is anchored on the last value).",
    "H3b": "Reducing the number of training cells degrades NLinear less than DLinear.",
    "H3c": "DLinear and NLinear use at least 10x fewer parameters than the tree ensembles have "
           "nodes, and train at least 10x faster (descriptive, no significance test).",
}


@dataclass
class HypothesisResult:
    hypothesis: str
    statement: str
    test: str
    n_units: int
    effect: float                  # signed effect in the metric's units (see ``effect_label``)
    effect_label: str
    p_value: float
    direction_as_predicted: bool
    p_holm: float = np.nan
    verdict: str = ""
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def finalize(results: list[HypothesisResult], alpha: float = ALPHA) -> list[HypothesisResult]:
    """Holm-correct within the family and attach verdicts."""
    adjusted = holm([r.p_value for r in results])
    for r, p in zip(results, adjusted):
        r.p_holm = float(p)
        r.verdict = verdict(r.p_holm, r.direction_as_predicted, alpha)
    return results


def describe(results: list[HypothesisResult]) -> str:
    """Plain-text block for the end of a notebook."""
    lines = []
    for r in results:
        lines.append(f"{r.hypothesis}: {r.verdict.upper()}")
        lines.append(f"    {r.statement}")
        lines.append(f"    {r.test}; n = {r.n_units}; {r.effect_label} = {r.effect:+.4f}; "
                     f"p = {r.p_value:.4g}, Holm p = {r.p_holm:.4g}")
        if r.note:
            lines.append(f"    note: {r.note}")
    return "\n".join(lines)
