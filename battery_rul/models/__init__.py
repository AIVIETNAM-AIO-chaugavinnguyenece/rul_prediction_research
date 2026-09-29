"""Model families: linear (PyTorch), tree-based (scikit-learn / XGBoost / LightGBM)
and naive baselines. All share one interface: ``predict_block(windows) -> (n, H)``
in Ah, so the roll-out and evaluation code never special-cases a model."""

from .baselines import LinearTrendForecaster, PersistenceForecaster
from .trees import TreeForecaster, count_tree_nodes, make_tree_model, train_tree

__all__ = [
    "LinearTrendForecaster", "PersistenceForecaster",
    "TreeForecaster", "count_tree_nodes", "make_tree_model", "train_tree",
]
