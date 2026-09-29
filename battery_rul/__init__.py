"""battery_rul - threshold-crossing RUL prediction for lithium-ion batteries.

Compares DLinear and NLinear (Zeng et al., 2023) with six tree-based models on
NASA PCoE and CALCE CX2 capacity-fade data. Every model forecasts the capacity
trajectory from the same window, and RUL is the cycle at which the forecast
stays below the end-of-life threshold.

Modules, in pipeline order:

    config       every setting, declared once
    data         raw NASA .mat / CALCE .xlsx readers
    cleaning     causal inputs, centred labels, EOL and RUL
    dataset      processed cells, leave-one-cell-out folds, forecast origins
    features     windows, tree features, training sets
    models       DLinear / NLinear / Linear, six tree models, baselines
    forecast     roll-out and threshold crossing (identical for all models)
    evaluation   metrics and aggregation to the unit of replication
    stats        Friedman, Wilcoxon + Holm, effect sizes, verdicts
    experiment   one runner for every model family
    explain      linear weight maps, permutation importance, SHAP
    plotting     consistent figures
"""

__version__ = "1.0.0"
