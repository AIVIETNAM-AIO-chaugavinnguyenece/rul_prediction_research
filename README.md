# Battery RUL by threshold crossing: DLinear and NLinear vs tree-based models

How far can a *linear* forecaster carry a lithium-ion cell's capacity-fade curve
before it crosses the end-of-life threshold, and how does that compare with the
tree-based models that dominate applied RUL work?

This repository contains the complete, reproducible pipeline behind the
technical report: six notebooks, one tested Python package, and pre-registered
hypotheses whose verdicts are computed by code, never typed by hand.

| | |
|---|---|
| **Task** | Forecast discharge capacity cycle by cycle; RUL = first cycle at which the forecast stays below 80% of rated capacity |
| **Data** | NASA PCoE (B0005, B0006, B0007, B0018) and CALCE CX2 constant-current cells (CX2_16, 31, 33–38) |
| **Models** | DLinear, NLinear (Zeng et al., 2023), plain Linear · Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM · Persistence and linear-trend baselines |
| **Evaluation** | Leave-one-cell-out within each dataset, forecasts from 25/50/75% of true life, 5 seeds (linear) / 3 seeds (trees) |
| **Primary metric** | Relative absolute RUL error, \|RUL̂ − RUL\| / RUL |

---

## Research questions

**RQ1 · DLinear vs NLinear.** How do DLinear and NLinear compare in forecasting
capacity degradation and the resulting RUL, and how do degradation patterns
(capacity regeneration, NASA vs CALCE differences) affect their relative performance?

**RQ2 · Linear vs tree models.** How does threshold-crossing RUL accuracy from
DLinear/NLinear compare with the six tree-based models, particularly when capacity
falls below the range seen in training?

**RQ3 · Practical settings and explainability.** How do window length, forecast
horizon and the amount of training data affect DLinear/NLinear, and how efficient
and interpretable are they compared with the tree models?

### Pre-registered hypotheses

Stated in [`battery_rul/report.py`](battery_rul/report.py) before any model was
run. Each is tested on the unit of replication (dataset × test cell × origin),
Holm-corrected within its RQ, and labelled *supported*, *contradicted* or
*not detected* by code.

| ID | Hypothesis | Test |
|---|---|---|
| H1a | NLinear achieves lower relative RUL error than DLinear | Wilcoxon signed-rank |
| H1b | NLinear's advantage is larger in high-regeneration cells | Mann-Whitney on paired differences |
| H2a | Each linear model beats each absolute-target tree model (full-range training) | Wilcoxon, 12 pairs |
| H2b | With the EOL region removed from training, absolute-target trees degrade more than NLinear | Wilcoxon on degradation |
| H2c | Under that restriction, delta-target trees degrade less than the same tree with absolute targets | Wilcoxon on degradation |
| H3a | Longer windows help DLinear more than NLinear | Wilcoxon on per-unit change |
| H3b | Fewer training cells hurt NLinear less than DLinear | Wilcoxon on degradation |
| H3c | Linear models use ≥10× fewer parameters and train ≥10× faster than tree ensembles | Descriptive |

All verdicts are collected in `results/all_hypotheses.csv` by notebook 06.

---

## Method in one picture

```
raw capacity ──► causal Hampel filter ──► capacity_input  (the only thing models see)
             └─► centred Hampel + smoothing ──► capacity_label ──► EOL, RUL (targets only)

window of L past cycles ──► model ──► next H cycles ──► append, repeat ──► first sustained
                                                                          crossing of 80% rated
```

* **One roll-out engine for every model.** Linear models, trees and baselines all
  forecast H cycles at a time from the same L-cycle window and are rolled out by the
  same function ([`forecast.py`](battery_rul/forecast.py)), so horizon, window and
  stopping rule are identical across families.
* **Causal inputs, centred labels.** The input series is filtered with a trailing
  window only; the smoothed, centred series defines EOL and is never used as input.
  EOL is the first of 5 consecutive cycles at or below the threshold, so a single
  noisy dip does not end a cell's life.
* **Leave-one-cell-out.** Each test cell is forecast by models trained on the other
  cells of the same dataset. Validation (early stopping) is the chronological last
  20% of each training cell. Nothing from the test cell touches training or scaling;
  notebook 03 asserts this (`processed/leakage_checks.csv`).
* **Two target modes for trees.** *Absolute* (predict Ah) and *delta* (predict the
  change from the last observed value, from level-relative features). This separates
  "trees vs linear" from "absolute vs normalised target", the real reason trees
  cannot extrapolate below their training range.
* **Range-restricted arm (RQ2).** Training windows whose targets fall below 85% of
  rated capacity are removed, so the EOL region is never seen in training: a direct
  test of extrapolation.
* **Censoring.** A roll-out that has not crossed by 1.5× the longest training cell's
  life is censored and scored at that cap, never dropped.

---

## Repository layout

```
battery-rul-linear/
├── battery_rul/                 the package (all logic lives here, notebooks only orchestrate)
│   ├── config.py                every setting, declared once
│   ├── data.py                  NASA .mat and CALCE .xlsx readers
│   ├── cleaning.py              causal inputs, centred labels, EOL and RUL
│   ├── preprocess.py            raw -> processed/cells/*.parquet + cell_summary.csv
│   ├── dataset.py               cells, leave-one-cell-out folds, forecast origins
│   ├── features.py              windows, tree features, target modes, training sets
│   ├── models/                  linear.py (DLinear, NLinear, Linear), trees.py, baselines.py
│   ├── forecast.py              roll-out and threshold crossing
│   ├── evaluation.py            metrics, seed averaging, unit-level tables
│   ├── stats.py                 Friedman, Wilcoxon + Holm, rank-biserial, bootstrap CIs
│   ├── experiment.py            one runner for every model family and arm
│   ├── explain.py               effective weight maps, recency, permutation importance, SHAP
│   ├── report.py                hypotheses and computed verdicts
│   └── plotting.py              consistent, colour-blind-checked figures
├── notebooks/
│   ├── 01_preprocessing.ipynb   load, clean, label, validate; drop report
│   ├── 02_EDA.ipynb             fade curves, EOL conventions, regeneration regimes, range coverage
│   ├── 03_feature_engineering.ipynb  folds, origins, windows, features, leakage checks
│   ├── 04_modelling_linears.ipynb    RQ1
│   ├── 05_modelling_tree_ml_models.ipynb  RQ2
│   └── 06_window_length_explainable.ipynb RQ3 + all verdicts
├── scripts/make_synthetic_data.py    fake data in the real file formats, for smoke tests
├── .github/workflows/ci.yml   unit tests + quick-mode smoke run of all six notebooks on synthetic data
├── tests/                       pytest suite (36 tests: cleaning, features, models, roll-out, stats, loaders)
├── data/README.md               where to put the raw files
├── requirements.txt
└── pyproject.toml
```

Generated on run (git-ignored by default): `processed/`, `results/` (every table and
JSON behind every number in the report), `figures/` (every figure, 160 dpi PNG).

---

## Reproducing the study

### 1 · Install

```bash
git clone https://github.com/<your-account>/battery-rul-linear.git
cd battery-rul-linear
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                     # 36 passed
```

### 2 · Get the data

Download the NASA PCoE Battery Data Set and the CALCE CX2 cells and arrange them
as described in [`data/README.md`](data/README.md). To keep the data elsewhere:

```bash
export BATTERY_RUL_DATA=/path/to/data
```

### 3 · Run the notebooks in order

```bash
jupyter lab notebooks/
```

Each notebook reads only what earlier notebooks wrote to `processed/` and
`results/`, and records its library versions and full configuration to
`results/run_<notebook>.json`.

Headless alternative:

```bash
for nb in notebooks/0*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb"
done
```

### Google Colab

1. Upload the raw data to Drive as `MyDrive/battery_rul/data/NASA_PCoE/…` and `…/CALCE_CX2/CX2_*/`.
2. In a new Colab session run:
   ```python
   !git clone https://github.com/<your-account>/battery-rul-linear.git /content/battery-rul-linear
   %pip install -q -r /content/battery-rul-linear/requirements.txt
   ```
3. Open the notebooks from `/content/battery-rul-linear/notebooks/`. The first cell
   mounts Drive and sets the data path. Results are written inside the clone, so
   copy `results/` and `figures/` to Drive before the session ends.

### Quick mode (smoke test)

```bash
RUL_QUICK=1 jupyter lab notebooks/
```

One seed, small grids, fewer trees and epochs. It exercises every cell of every
notebook in a few minutes and prints a warning banner. **Never report quick-mode
numbers.** Without the real data, pair it with the synthetic set:

```bash
python scripts/make_synthetic_data.py --out data_synthetic
BATTERY_RUL_DATA=data_synthetic RUL_QUICK=1 jupyter lab notebooks/
```

### Expected runtime

Measured on a 2-core cloud VM (CPU only); full-mode figures are extrapolated from
timed single-fold runs and scale with the length of the CALCE cells. A modern
laptop is typically 2–4× faster. AdaBoost and Gradient Boosting (one model per
forecast step) account for most of notebook 05; the linear models train in about
a second per fold.

| Notebook | Quick mode | Full mode (approx.) |
|---|---|---|
| 01 preprocessing | < 30 s | 1–5 min (first read of the CALCE `.xlsx` files; cached afterwards) |
| 02 EDA / 03 features | < 15 s each | < 1 min each |
| 04 RQ1 (linear models) | ~20 s | 5–10 min |
| 05 RQ2 (trees, two arms) | ~6 min | 1–3 h |
| 06 RQ3 (grid, training size, explainability) | ~2 min | 1.5–3 h |

All models run on CPU and single-threaded where it affects determinism
(XGBoost, LightGBM, PyTorch), so re-running with the same versions reproduces
every forecast and RUL error bit for bit. Only wall-clock timings (H3c) vary between runs.

---

## Design rules

These rules come from reviewing an earlier forecasting project and are enforced
in the code, not just documented:

1. **Horizon parity.** Every model forecasts the same H steps from the same L-cycle window and is rolled out by one shared function.
2. **Honest baselines.** Persistence and least-squares trend extrapolation are always in the table; a model that cannot beat them is reported as such.
3. **Test targets are never altered.** Cleaning of the input is causal; labels are only used for scoring.
4. **One definition of each metric**, in `evaluation.py`, used by every notebook.
5. **Every number in the report comes from a file in `results/`**, written by committed code. Hypothesis verdicts are computed, not typed.
6. **Seeds are averaged within a unit before testing**, so seeds never inflate the sample size.
7. **Paired, same-unit comparisons only**, with Holm correction within each RQ and effect sizes (rank-biserial, bootstrap CIs) next to every p-value.
8. **Configuration lives in one place** (`config.py`) and is snapshotted with every run.

---

## Where to find each result

| Report element | File |
|---|---|
| Cell inventory, EOL, regeneration statistics | `processed/cell_summary.csv`, `processed/drop_report.csv` |
| Regeneration regimes (fixed before modelling) | `processed/regimes.csv` |
| Folds, origins, feature dictionary, leakage checks | `processed/folds.csv`, `origins.csv`, `feature_dictionary.csv`, `training_sets.csv`, `leakage_checks.csv` |
| EDA tables | `results/eda_overview.csv`, `eda_training_range.csv`, `eda_window_budget.csv` |
| RQ1 summary, pairwise tests, hypotheses | `results/rq1_summary*.csv`, `rq1_pairwise.csv`, `rq1_hypotheses.csv` |
| RQ1 sensitivity to the EOL reference | `results/rq1_reference_sensitivity.csv` |
| RQ2 full-range and range-restricted results | `results/rq2_summary_*.csv`, `rq2_degradation.csv`, `rq2_training_floor.csv` |
| RQ3 window/horizon grid, training size, efficiency | `results/rq3_window_*.csv`, `rq3_train_size_*.csv`, `rq3_efficiency.csv` |
| RQ3 explainability | `results/rq3_linear_explain.csv`, `rq3_linear_weights.npz`, `rq3_tree_*.csv` |
| All hypothesis verdicts | `results/all_hypotheses.csv` |

---

## Limitations

* Small number of cells (4 NASA, up to 8 CALCE). Leave-one-cell-out with three
  origins gives 12 and up to 24 units; tests are paired and non-parametric, and
  "not detected" means exactly that, not "no difference".
* Only capacity is used as input. Voltage, temperature and impedance features
  could help every model family and are left for future work.
* One chemistry per dataset and laboratory cycling conditions. Results may not
  transfer to field data with variable load.
* Hyperparameters are fixed sensible defaults for all models (listed in
  `config.py`) rather than tuned per fold, to keep the comparison fair and cheap.
  Notebook 06 reports sensitivity to window length and horizon.

---

## References

* A. Zeng, M. Chen, L. Zhang, Q. Xu. *Are Transformers Effective for Time Series Forecasting?* AAAI 2023. Official code: [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear).
* W. Toner, L. Darlow. *An Analysis of Linear Time Series Forecasting Models.* ICML 2024.
* B. Saha, K. Goebel. *Battery Data Set.* NASA Prognostics Data Repository, NASA Ames Research Center, 2007.
* CALCE Battery Research Group, University of Maryland. *CALCE Battery Data.* https://calce.umd.edu/battery-data
* W. He, N. Williard, M. Osterman, M. Pecht. *Prognostics of lithium-ion batteries based on Dempster–Shafer theory and the Bayesian Monte Carlo method.* Journal of Power Sources 196 (2011). (CX2 data)

## Citation

If you use this code, please cite the technical report (add the BibTeX entry here once published).

## License

Code: add a licence of your choice (MIT is common for research code). The NASA
and CALCE data remain under their providers' terms and are not redistributed here.
