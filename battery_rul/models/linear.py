"""DLinear, NLinear and Linear (Zeng et al., AAAI 2023), univariate.

The layers follow the official LTSF-Linear code (cure-lab/LTSF-Linear):

* **DLinear** splits the window into a trend (moving average with replicate
  padding) and a remainder, and maps each with its own linear layer.
* **NLinear** subtracts the last value, applies one linear layer, adds it back.
* **Linear** is a single linear layer - the reference that shows how much of
  any difference comes from parameterisation rather than capacity.

A property worth knowing before reading results (Toner & Darlow, ICML 2024):
all three are *linear maps of the window*, so they share one function class.
With exact least squares they would coincide. They differ in practice because
gradient descent from a default initialisation, early stopping and the
parameterisation give each a different implicit bias - NLinear starts near
"repeat the last value", DLinear near a smoothed level. :func:`effective_map`
recovers the single (W, b) each trained model implements, so the notebooks can
show this directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..config import LINEAR_TRAIN, LinearTrainConfig
from ..features import Standardizer, TrainingSet

torch.set_num_threads(1)  # tiny models: one thread is faster and deterministic


class MovingAvg(nn.Module):
    """Moving average with replicate padding, as in the official DLinear."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (batch, L)
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1].repeat(1, pad)
        back = x[:, -1:].repeat(1, pad)
        padded = torch.cat([front, x, back], dim=1)
        return self.pool(padded.unsqueeze(1)).squeeze(1)


class DLinear(nn.Module):
    def __init__(self, input_len: int, horizon: int, kernel_size: int) -> None:
        super().__init__()
        self.decomp = MovingAvg(kernel_size)
        self.linear_seasonal = nn.Linear(input_len, horizon)
        self.linear_trend = nn.Linear(input_len, horizon)

    def components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.decomp(x)
        return self.linear_seasonal(x - trend), self.linear_trend(trend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.components(x)
        return seasonal + trend


class NLinear(nn.Module):
    def __init__(self, input_len: int, horizon: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_len, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:]
        return self.linear(x - last) + last


class Linear(nn.Module):
    def __init__(self, input_len: int, horizon: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_len, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_linear_model(name: str, input_len: int, horizon: int, kernel_size: int) -> nn.Module:
    if name == "DLinear":
        return DLinear(input_len, horizon, kernel_size)
    if name == "NLinear":
        return NLinear(input_len, horizon)
    if name == "Linear":
        return Linear(input_len, horizon)
    raise ValueError(f"unknown linear model {name!r}")


def n_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


@dataclass
class LinearForecaster:
    """A trained linear model plus its scaler; forecasts in Ah."""

    name: str
    model: nn.Module
    scaler: Standardizer
    input_len: int
    horizon: int
    kernel_size: int
    seed: int
    best_epoch: int = 0
    epochs_run: int = 0
    train_seconds: float = 0.0
    history: dict = field(default_factory=lambda: {"train": [], "val": []})

    @property
    def n_params(self) -> int:
        return n_parameters(self.model)

    def predict_block(self, windows: np.ndarray) -> np.ndarray:
        """(n, L) windows in Ah -> (n, H) forecasts in Ah."""
        windows = np.atleast_2d(windows)
        with torch.no_grad():
            z = torch.as_tensor(self.scaler.transform(windows), dtype=torch.float32)
            out = self.model(z).numpy().astype(np.float64)
        return self.scaler.inverse(out)


def _batches(n: int, batch_size: int, generator: torch.Generator):
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        yield order[start: start + batch_size]


def train_linear(name: str, training: TrainingSet, input_len: int, horizon: int, kernel_size: int,
                 seed: int, cfg: LinearTrainConfig = LINEAR_TRAIN) -> LinearForecaster:
    """Adam + MSE on standardised capacity, early stopping on the validation part.

    The scaler is fitted on the fit windows of the training cells only. The best
    weights (lowest validation loss) are restored at the end.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)

    scaler = Standardizer().fit(training.X_fit)
    Xf = torch.as_tensor(scaler.transform(training.X_fit), dtype=torch.float32)
    Yf = torch.as_tensor(scaler.transform(training.Y_fit), dtype=torch.float32)
    has_val = len(training.X_val) > 0
    if has_val:
        Xv = torch.as_tensor(scaler.transform(training.X_val), dtype=torch.float32)
        Yv = torch.as_tensor(scaler.transform(training.Y_val), dtype=torch.float32)

    model = build_linear_model(name, input_len, horizon, kernel_size)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_state, best_val, best_epoch, bad = None, float("inf"), 0, 0
    history = {"train": [], "val": []}
    t_start = time.perf_counter()
    epoch = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total = 0.0
        for idx in _batches(len(Xf), cfg.batch_size, gen):
            opt.zero_grad()
            loss = loss_fn(model(Xf[idx]), Yf[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history["train"].append(total / len(Xf))

        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(Xv), Yv)) if has_val else history["train"][-1]
        history["val"].append(val)

        if val < best_val - 1e-9:
            best_val, best_epoch, bad = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return LinearForecaster(
        name=name, model=model, scaler=scaler, input_len=input_len, horizon=horizon,
        kernel_size=kernel_size, seed=seed, best_epoch=best_epoch, epochs_run=epoch,
        train_seconds=time.perf_counter() - t_start, history=history,
    )


# --------------------------------------------------------------------------
# Explainability: the single linear map each model implements
# --------------------------------------------------------------------------

def moving_average_matrix(input_len: int, kernel_size: int) -> np.ndarray:
    """The (L, L) matrix A with trend = x @ A.T, built by pushing basis vectors
    through the same MovingAvg layer the model uses."""
    ma = MovingAvg(kernel_size)
    with torch.no_grad():
        eye = torch.eye(input_len)
        return ma(eye).numpy().T.astype(np.float64)   # column j = response to basis j


def effective_map(forecaster: LinearForecaster) -> dict[str, np.ndarray]:
    """(W, b) such that the model computes y = W x + b in standardised space.

    DLinear: W = Ws (I - A) + Wt A,  b = bs + bt
    NLinear: W = Wn + (1 - Wn 1) e_L^T, b = bn   (the last-value term folded in)
    Linear:  W = Wl, b = bl
    For DLinear the branch matrices are returned as well.
    """
    m = forecaster.model
    L = forecaster.input_len
    if isinstance(m, DLinear):
        A = moving_average_matrix(L, forecaster.kernel_size)
        Ws = m.linear_seasonal.weight.detach().numpy().astype(np.float64)
        Wt = m.linear_trend.weight.detach().numpy().astype(np.float64)
        W = Ws @ (np.eye(L) - A) + Wt @ A
        b = (m.linear_seasonal.bias + m.linear_trend.bias).detach().numpy().astype(np.float64)
        return {"W": W, "b": b, "W_seasonal": Ws, "W_trend": Wt, "A": A}
    if isinstance(m, NLinear):
        Wn = m.linear.weight.detach().numpy().astype(np.float64)
        W = Wn.copy()
        W[:, -1] += 1.0 - Wn.sum(axis=1)
        return {"W": W, "b": m.linear.bias.detach().numpy().astype(np.float64), "W_raw": Wn}
    if isinstance(m, Linear):
        return {"W": m.linear.weight.detach().numpy().astype(np.float64),
                "b": m.linear.bias.detach().numpy().astype(np.float64)}
    raise TypeError(type(m))


def dlinear_branch_share(forecaster: LinearForecaster, windows: np.ndarray) -> dict[str, float]:
    """Share of DLinear's output variance carried by the trend vs remainder branch."""
    if not isinstance(forecaster.model, DLinear):
        raise TypeError("branch shares exist for DLinear only")
    with torch.no_grad():
        z = torch.as_tensor(forecaster.scaler.transform(np.atleast_2d(windows)), dtype=torch.float32)
        seasonal, trend = forecaster.model.components(z)
    vs, vt = float(seasonal.var()), float(trend.var())
    total = vs + vt if vs + vt > 0 else 1.0
    return {"trend_share": vt / total, "remainder_share": vs / total}
