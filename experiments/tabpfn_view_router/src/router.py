from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


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


def gora_mix(
    preds: np.ndarray,
    sigma2_v: np.ndarray,
    mean_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytical GoRA-style routing — zero free parameters, no val labels used.

    weights = softmax(-sigma2_v / tau)  where tau = 1 / (mean_J + eps)

    High view-agreement (mean_J↑) → smaller tau → sharper softmax → more decisive routing.
    Low sigma2 views (label-predictive neighbours) get higher weight.
    """
    tau = 1.0 / (np.maximum(mean_j, 0.0)[:, None] + 1e-8)   # [N, 1]
    logits = -sigma2_v * tau                                   # [N, V]
    logits -= logits.max(axis=1, keepdims=True)                # numerical stability
    exp_l = np.exp(np.clip(logits, -30.0, 0.0))
    weights = (exp_l / (exp_l.sum(axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    return (weights * preds).sum(axis=1).astype(np.float32), weights


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


def build_trust_gate_features(
    quality_features: np.ndarray,
    pred_full: np.ndarray,
    pred_route: np.ndarray,
    pred_views: np.ndarray,
) -> np.ndarray:
    route_delta = (pred_route - pred_full).astype(np.float32)[:, None]
    abs_route_delta = np.abs(route_delta)
    view_std = pred_views.std(axis=1, keepdims=True).astype(np.float32)
    return np.concatenate(
        [
            quality_features.astype(np.float32),
            route_delta,
            abs_route_delta,
            view_std,
        ],
        axis=1,
    ).astype(np.float32)


def build_foundation_router_features(
    quality_features: np.ndarray,
    pred_views: np.ndarray,
    pred_foundation: np.ndarray,
    *,
    anchor_idx: int = 0,
) -> np.ndarray:
    anchor = pred_views[:, anchor_idx].astype(np.float32)
    pred_foundation = pred_foundation.astype(np.float32)
    mean_internal = pred_views.mean(axis=1).astype(np.float32)
    all_preds = np.concatenate([pred_views.astype(np.float32), pred_foundation[:, None]], axis=1)
    return np.concatenate(
        [
            quality_features.astype(np.float32),
            (pred_foundation - anchor)[:, None],
            np.abs(pred_foundation - anchor)[:, None],
            (pred_foundation - mean_internal)[:, None],
            np.abs(pred_foundation - mean_internal)[:, None],
            pred_views.std(axis=1, keepdims=True).astype(np.float32),
            all_preds.std(axis=1, keepdims=True).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def oracle_full_route_blend(
    y_true: np.ndarray,
    pred_full: np.ndarray,
    pred_route: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    use_route = (np.abs(pred_route - y_true) < np.abs(pred_full - y_true)).astype(np.float32)
    pred = ((1.0 - use_route) * pred_full + use_route * pred_route).astype(np.float32)
    return pred, use_route


@dataclass(frozen=True)
class TrustGateResult:
    pred_val: np.ndarray
    pred_test: np.ndarray
    alpha_val: np.ndarray
    alpha_test: np.ndarray
    best_epoch: int


class SoftTrustGate(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, bias_init: float = -2.0) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        final = self.net[-1]
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.constant_(final.bias, bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def _inner_holdout_size(
    n_rows: int,
    *,
    holdout_fraction: float = 0.30,
    min_holdout: int = 50,
) -> int:
    if n_rows < 2:
        raise ValueError(f"Need at least 2 rows to build an inner holdout, got {n_rows}")
    fractional = int(np.ceil(n_rows * holdout_fraction))
    guarded = max(fractional, min(min_holdout, n_rows // 2))
    return min(max(1, guarded), n_rows - 1)


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
    min_holdout: int = 50,
) -> RouterResult:
    idx = np.arange(len(x_val))
    tr_idx, hold_idx = train_test_split(
        idx,
        test_size=_inner_holdout_size(len(idx), min_holdout=min_holdout),
        random_state=seed,
    )

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


def fit_trust_gate(
    x_val: np.ndarray,
    pred_full_val: np.ndarray,
    pred_route_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    pred_full_test: np.ndarray,
    pred_route_test: np.ndarray,
    seed: int = 42,
    hidden_dim: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 400,
    patience: int = 40,
    bias_init: float = -2.0,
    min_holdout: int = 50,
) -> TrustGateResult:
    idx = np.arange(len(x_val))
    tr_idx, hold_idx = train_test_split(
        idx,
        test_size=_inner_holdout_size(len(idx), min_holdout=min_holdout),
        random_state=seed,
    )

    model = SoftTrustGate(in_dim=x_val.shape[1], hidden_dim=hidden_dim, bias_init=bias_init)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    xtr = torch.from_numpy(x_val[tr_idx]).float()
    full_tr = torch.from_numpy(pred_full_val[tr_idx]).float()
    route_tr = torch.from_numpy(pred_route_val[tr_idx]).float()
    ytr = torch.from_numpy(y_val[tr_idx]).float()
    xhold = torch.from_numpy(x_val[hold_idx]).float()
    full_hold = torch.from_numpy(pred_full_val[hold_idx]).float()
    route_hold = torch.from_numpy(pred_route_val[hold_idx]).float()
    yhold = torch.from_numpy(y_val[hold_idx]).float()

    best_state = None
    best_rmse = float("inf")
    best_epoch = -1
    wait = patience

    for epoch in range(max_epochs):
        model.train()
        alpha = model(xtr).squeeze(1)
        pred = full_tr + alpha * (route_tr - full_tr)
        loss = torch.nn.functional.mse_loss(pred, ytr)
        optim.zero_grad()
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            alpha_hold = model(xhold).squeeze(1)
            pred_hold = full_hold + alpha_hold * (route_hold - full_hold)
            rmse = float(torch.sqrt(torch.mean((pred_hold - yhold) ** 2)).item())
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
        alpha_val = model(torch.from_numpy(x_val).float()).cpu().numpy().astype(np.float32).reshape(-1)
        alpha_test = model(torch.from_numpy(x_test).float()).cpu().numpy().astype(np.float32).reshape(-1)
    pred_val = (pred_full_val + alpha_val * (pred_route_val - pred_full_val)).astype(np.float32)
    pred_test = (pred_full_test + alpha_test * (pred_route_test - pred_full_test)).astype(np.float32)
    return TrustGateResult(
        pred_val=pred_val,
        pred_test=pred_test,
        alpha_val=alpha_val,
        alpha_test=alpha_test,
        best_epoch=best_epoch,
    )


@dataclass(frozen=True)
class CrossfitRouterResult:
    pred_val_oof: np.ndarray   # clean OOF val predictions — unbiased val RMSE
    weights_val_oof: np.ndarray
    pred_test: np.ndarray      # final model predictions on test (router trained on all val)
    weights_test: np.ndarray   # final model weights on test
    n_splits: int


def fit_crossfit_router(
    x_val: np.ndarray,
    pred_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    pred_test: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    hidden_dim: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 400,
    patience: int = 40,
) -> CrossfitRouterResult:
    """
    5-fold cross-fit soft router.

    OOF val predictions: each val row predicted by a router never trained on it →
    val RMSE is a clean, unbiased estimate (no val-label leakage into the reported metric).

    Final test predictions: router trained on all val rows via fit_soft_router (70/30 split
    for early stopping) — identical protocol to P0_router, so test RMSEs are comparable.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y_val), dtype=np.float32)
    oof_weights = np.zeros_like(pred_val, dtype=np.float32)

    for fold, (fold_tr_idx, fold_oof_idx) in enumerate(kf.split(x_val)):
        # fit_soft_router internally splits fold_tr 70/30 for training / early-stopping
        fold_result = fit_soft_router(
            x_val=x_val[fold_tr_idx],
            pred_val=pred_val[fold_tr_idx],
            y_val=y_val[fold_tr_idx],
            x_test=x_val[fold_oof_idx],
            pred_test=pred_val[fold_oof_idx],
            seed=seed + fold,          # different seed per fold avoids systematic bias
            hidden_dim=hidden_dim,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
        )
        oof_preds[fold_oof_idx] = fold_result.pred_test   # OOF slot used as "test" here
        oof_weights[fold_oof_idx] = fold_result.weights_test

    # Final router on all val → actual test predictions
    final = fit_soft_router(
        x_val=x_val,
        pred_val=pred_val,
        y_val=y_val,
        x_test=x_test,
        pred_test=pred_test,
        seed=seed,
        hidden_dim=hidden_dim,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
    )

    return CrossfitRouterResult(
        pred_val_oof=oof_preds,
        weights_val_oof=oof_weights,
        pred_test=final.pred_test,
        weights_test=final.weights_test,
        n_splits=n_splits,
    )
