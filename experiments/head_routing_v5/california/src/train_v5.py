"""
train_v5.py — Training and inference loop for v5 CA experiments.

Supports both FlatRegressionHead (A0) and HeadGatedRegressor (A1+).
V5 batch includes optional quality_score, J_flat, mean_J, sigma2_v arrays.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from experiments.head_routing_v5.california.src.data_v5 import denormalise_target, normalise_target
from experiments.head_routing_v5.shared.src.losses_v5 import composite_loss


@dataclass(frozen=True)
class TrainConfigV5:
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 150
    patience: int = 30
    loss_type: str = "huber"
    # Loss weights
    lambda_orth: float = 0.0
    lambda_pdiv: float = 0.0
    lambda_lb: float = 0.0
    lambda_cons: float = 0.0
    lambda_nll: float = 0.0


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def standardise_observers(g: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, dict]:
    mean = g[train_idx].mean(axis=0)
    std = g[train_idx].std(axis=0) + 1e-8
    normed = ((g - mean) / std).astype(np.float32)
    return normed, {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def maybe_truncate_splits(
    train_idx, val_idx, test_idx, smoke, smoke_train=500, smoke_val=200, smoke_test=200
):
    if not smoke:
        return train_idx, val_idx, test_idx
    return train_idx[:smoke_train], val_idx[:smoke_val], test_idx[:smoke_test]


# ---------------------------------------------------------------------------
# Batch fetching
# ---------------------------------------------------------------------------

def _to_tensor(arr, device, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype, device=device)


def fetch_batch(
    batch_idx: np.ndarray,
    view_feats: Dict[str, np.ndarray],
    per_view_knn: Dict[str, Tuple[np.ndarray, np.ndarray]],
    g_scaled: np.ndarray,
    y_norm: np.ndarray,
    device: torch.device,
    # V5 extras (all optional)
    quality_score: Optional[np.ndarray] = None,   # [N, V]
    quality_norm_arr: Optional[np.ndarray] = None, # [N, V, 3]
    J_flat: Optional[np.ndarray] = None,           # [N, n_pairs]
    mean_J: Optional[np.ndarray] = None,           # [N]
    sigma2_v: Optional[np.ndarray] = None,         # [N, V]
) -> dict:
    x_anchor_by_view = {}
    x_nei_by_view = {}
    ew_by_view = {}
    for view_name, X_view in view_feats.items():
        neigh_idx, edge_wts = per_view_knn[view_name]
        x_anchor_by_view[view_name] = _to_tensor(X_view[batch_idx], device)
        x_nei_by_view[view_name] = _to_tensor(X_view[neigh_idx[batch_idx]], device)
        ew_by_view[view_name] = _to_tensor(edge_wts[batch_idx], device)

    batch = {
        "x_anchor_by_view": x_anchor_by_view,
        "x_nei_by_view": x_nei_by_view,
        "ew_by_view": ew_by_view,
        "g": _to_tensor(g_scaled[batch_idx], device),
        "y": _to_tensor(y_norm[batch_idx], device),
        "quality_score": _to_tensor(quality_score[batch_idx], device) if quality_score is not None else None,
        "quality_norm": _to_tensor(quality_norm_arr[batch_idx], device) if quality_norm_arr is not None else None,
        "J_flat": _to_tensor(J_flat[batch_idx], device) if J_flat is not None else None,
        "mean_J": _to_tensor(mean_J[batch_idx], device) if mean_J is not None else None,
        "sigma2_v": _to_tensor(sigma2_v[batch_idx], device) if sigma2_v is not None else None,
    }
    return batch


# ---------------------------------------------------------------------------
# Model wrapper: backbone + head, with unified forward interface
# ---------------------------------------------------------------------------

class V5Model(nn.Module):
    """
    Wraps HeadRoutingBackboneV5 + a task head (flat or head-gated).
    Provides a single forward() for both training and inference.
    """

    def __init__(self, backbone, task_head):
        super().__init__()
        self.backbone = backbone
        self.task_head = task_head

    def forward(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        head_repr, aux = self.backbone(
            x_anchor_by_view=batch["x_anchor_by_view"],
            x_nei_by_view=batch["x_nei_by_view"],
            ew_by_view=batch["ew_by_view"],
            g=batch["g"],
            quality_score=batch.get("quality_score"),
            quality_norm=batch.get("quality_norm"),
            J_flat=batch.get("J_flat"),
            mean_J=batch.get("mean_J"),
            sigma2_v=batch.get("sigma2_v"),
        )

        # Determine which task head type we have
        from experiments.head_routing_v5.shared.src.task_heads_v5 import (
            HeadGatedRegressor, FlatRegressionHead
        )
        if isinstance(self.task_head, HeadGatedRegressor):
            pred, head_out = self.task_head(
                head_repr, aux,
                g=batch["g"],
                quality_score=batch.get("quality_score"),
            )
        else:
            pred, head_out = self.task_head(head_repr, aux)

        aux.update(head_out)
        return pred, aux


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model: V5Model,
    indices: np.ndarray,
    view_feats: Dict[str, np.ndarray],
    per_view_knn: Dict[str, Tuple[np.ndarray, np.ndarray]],
    g_scaled: np.ndarray,
    target_stats: dict,
    batch_size: int = 512,
    quality_score: Optional[np.ndarray] = None,
    quality_norm_arr: Optional[np.ndarray] = None,
    J_flat: Optional[np.ndarray] = None,
    mean_J: Optional[np.ndarray] = None,
    sigma2_v: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    device = get_device()
    model = model.to(device).eval()

    dummy_y = np.zeros(next(iter(view_feats.values())).shape[0], dtype=np.float32)
    preds_norm, pi_all, beta_all, gate_h_all = [], [], [], []

    for start in range(0, len(indices), batch_size):
        bi = indices[start: start + batch_size]
        batch = fetch_batch(
            bi, view_feats, per_view_knn, g_scaled, dummy_y, device,
            quality_score=quality_score,
            quality_norm_arr=quality_norm_arr,
            J_flat=J_flat,
            mean_J=mean_J,
            sigma2_v=sigma2_v,
        )
        pred_norm, aux = model(batch)
        preds_norm.append(pred_norm.cpu().numpy())
        pi_all.append(aux["pi"].detach().cpu().numpy())
        beta_all.append(aux["beta"].detach().cpu().numpy())
        gh = aux.get("gate_h")
        if gh is not None:
            gate_h_all.append(gh.detach().cpu().numpy())

    preds_norm_np = np.concatenate(preds_norm, axis=0).astype(np.float32)
    preds = denormalise_target(preds_norm_np, target_stats)

    routing = {
        "pi": np.concatenate(pi_all, axis=0),
        "beta": np.concatenate(beta_all, axis=0),
    }
    if gate_h_all:
        routing["gate_h"] = np.concatenate(gate_h_all, axis=0)

    return preds, routing


def train(
    model: V5Model,
    view_feats: Dict[str, np.ndarray],
    per_view_knn: Dict[str, Tuple[np.ndarray, np.ndarray]],
    g_scaled: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    target_stats: dict,
    config: TrainConfigV5,
    quality_score: Optional[np.ndarray] = None,
    quality_norm_arr: Optional[np.ndarray] = None,
    J_flat: Optional[np.ndarray] = None,
    mean_J: Optional[np.ndarray] = None,
    sigma2_v: Optional[np.ndarray] = None,
) -> V5Model:
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    y_norm = normalise_target(y, target_stats)
    best_state = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    best_epoch = -1
    wait = 0
    stop_epoch = -1

    for epoch in range(config.max_epochs):
        model.train()
        perm = np.random.permutation(train_idx)
        for start in range(0, len(perm), config.batch_size):
            bi = perm[start: start + config.batch_size]
            batch = fetch_batch(
                bi, view_feats, per_view_knn, g_scaled, y_norm, device,
                quality_score=quality_score,
                quality_norm_arr=quality_norm_arr,
                J_flat=J_flat,
                mean_J=mean_J,
                sigma2_v=sigma2_v,
            )
            optimizer.zero_grad()
            pred_norm, aux = model(batch)

            loss, _ = composite_loss(
                pred=pred_norm,
                target=batch["y"],
                task="regression",
                loss_type=config.loss_type,
                pred_h=aux.get("pred_h"),
                logvar_h=aux.get("logvar_h"),
                gate_h=aux.get("gate_h"),
                pi=aux.get("pi"),
                neighbor_attn=aux.get("neighbor_attn"),
                cross_view_attn=aux.get("cross_view_attn"),
                lambda_orth=config.lambda_orth,
                lambda_pdiv=config.lambda_pdiv,
                lambda_lb=config.lambda_lb,
                lambda_cons=config.lambda_cons,
                lambda_nll=config.lambda_nll,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_preds, _ = predict(
            model, val_idx, view_feats, per_view_knn, g_scaled, target_stats,
            batch_size=config.batch_size,
            quality_score=quality_score,
            quality_norm_arr=quality_norm_arr,
            J_flat=J_flat,
            mean_J=mean_J,
            sigma2_v=sigma2_v,
        )
        val_rmse = float(np.sqrt(np.mean((val_preds - y[val_idx]) ** 2)))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        stop_epoch = epoch
        if wait >= config.patience:
            break

    model.load_state_dict(best_state)
    model.training_metadata = {
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "best_val_rmse": float(best_val_rmse),
        "collapsed": bool(stop_epoch <= 5),
    }
    return model
