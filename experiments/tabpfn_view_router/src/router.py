from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def score_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def uniform_mix(preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.full((preds.shape[0], preds.shape[1]), 1.0 / preds.shape[1], dtype=np.float32)
    return preds.mean(axis=1).astype(np.float32), weights


def sigma2_mix(preds: np.ndarray, sigma2_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv = 1.0 / (np.maximum(sigma2_v, -5.0) + 6.0)
    weights = inv / (inv.sum(axis=1, keepdims=True) + 1e-8)
    pred = (weights * preds).sum(axis=1)
    return pred.astype(np.float32), weights.astype(np.float32)


@dataclass(frozen=True)
class RouterResult:
    pred_val: np.ndarray
    pred_test: np.ndarray
    weights_val: np.ndarray
    weights_test: np.ndarray
    best_epoch: int


class SoftViewRouter(torch.nn.Module):
    def __init__(self, in_dim: int, n_views: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, n_views),
        )
        final = self.net[-1]
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


def fit_soft_router(
    x_val: np.ndarray,
    pred_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    pred_test: np.ndarray,
    seed: int = 42,
    hidden_dim: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 400,
    patience: int = 40,
) -> RouterResult:
    idx = np.arange(len(x_val))
    tr_idx, hold_idx = train_test_split(idx, test_size=0.30, random_state=seed)

    model = SoftViewRouter(in_dim=x_val.shape[1], n_views=pred_val.shape[1], hidden_dim=hidden_dim)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    xtr = torch.from_numpy(x_val[tr_idx]).float()
    ptr = torch.from_numpy(pred_val[tr_idx]).float()
    ytr = torch.from_numpy(y_val[tr_idx]).float()
    xhold = torch.from_numpy(x_val[hold_idx]).float()
    phold = torch.from_numpy(pred_val[hold_idx]).float()
    yhold = torch.from_numpy(y_val[hold_idx]).float()

    best_state = None
    best_rmse = float("inf")
    best_epoch = -1
    wait = patience

    for epoch in range(max_epochs):
        model.train()
        w = model(xtr)
        pred = (w * ptr).sum(dim=1)
        loss = torch.nn.functional.mse_loss(pred, ytr)
        optim.zero_grad()
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            wh = model(xhold)
            ph = (wh * phold).sum(dim=1)
            rmse = float(torch.sqrt(torch.mean((ph - yhold) ** 2)).item())
        if rmse + 1e-8 < best_rmse:
            best_rmse = rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = patience
        else:
            wait -= 1
            if wait <= 0:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        weights_val = model(torch.from_numpy(x_val).float()).cpu().numpy().astype(np.float32)
        weights_test = model(torch.from_numpy(x_test).float()).cpu().numpy().astype(np.float32)
    pred_val_out = (weights_val * pred_val).sum(axis=1).astype(np.float32)
    pred_test_out = (weights_test * pred_test).sum(axis=1).astype(np.float32)
    return RouterResult(
        pred_val=pred_val_out,
        pred_test=pred_test_out,
        weights_val=weights_val,
        weights_test=weights_test,
        best_epoch=best_epoch,
    )
