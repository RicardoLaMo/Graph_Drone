"""
train.py — Training and inference for MV-TabR-GoRA.

Batch structure
---------------
Each training step samples B anchors and fetches their per-view neighbours.
We do NOT precompute z_nei because per-view encoders change during training;
instead we fetch raw view features for both anchors and neighbours, then
encode everything in the forward pass (standard retrieval-model protocol).

Batch dict keys (all float32 Tensors on device):
    x_anc             [B, F]           anchor raw features
    x_anc_v           {name: [B, d_v]} anchor raw features per view
    x_nei             {name: [B, K, d_v]} neighbour raw features per view
    y_nei             {name: [B, K]}   normalised neighbour labels
    ew                {name: [B, K]}   neighbour edge weights
    sigma2_v_anc      [B, V]           anchor sigma2_v (quality priors)
    sigma2_v_nei      {name: [B, K]}   neighbour sigma2_v in each view
    mean_J            [B]              mean Jaccard (view agreement)
    y_true            [B]              anchor normalised labels (for loss)
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import MVDataBundle
from .model import MVTabrGoraModel


EPS = 1e-8


@dataclass
class TrainConfig:
    batch_size: int = 512
    max_epochs: int = 150
    patience: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    loss: str = "huber"          # "mse" | "huber" | "mae"
    huber_delta: float = 1.0
    grad_clip: float = 1.0
    seed: int = 0


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Batch fetching
# ---------------------------------------------------------------------------

def _t(arr: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(arr, dtype=dtype, device=device)


def fetch_batch(
    anchor_idx: np.ndarray,             # [B]  global indices of anchors
    bundle: MVDataBundle,
    device: torch.device,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Fetch a complete batch dict from global anchor indices.
    All neighbour indices in per_view_knn are GLOBAL (train-referenced).
    """
    B = len(anchor_idx)
    V = len(bundle.view_names)
    K = bundle.K

    x_anc = _t(bundle.X[anchor_idx], device)                  # [B, F]
    y_true = _t(bundle.y_norm[anchor_idx], device)             # [B]

    # Per-view anchor features
    x_anc_v = {
        name: _t(bundle.view_feats[name][anchor_idx], device)  # [B, d_v]
        for name in bundle.view_names
    }

    # Per-view neighbours
    x_nei: Dict[str, torch.Tensor] = {}
    y_nei: Dict[str, torch.Tensor] = {}
    ew: Dict[str, torch.Tensor] = {}
    sigma2_v_nei: Dict[str, torch.Tensor] = {}

    for v_idx, name in enumerate(bundle.view_names):
        nei_idx, nei_wt = bundle.per_view_knn[name]   # [N, K] each
        an = nei_idx[anchor_idx]                       # [B, K]
        aw = nei_wt[anchor_idx]                        # [B, K]

        # Raw view features of neighbours
        x_nei[name] = _t(bundle.view_feats[name][an], device)   # [B, K, d_v]

        # Normalised labels of neighbours (training labels → no leakage since
        # kNN is train-only referenced; at eval time x_anc is val/test, but
        # the NEIGHBOURS are always from training set)
        y_nei[name] = _t(bundle.y_norm[an], device)             # [B, K]

        ew[name] = _t(aw, device)                               # [B, K]

        # Per-view label variance of each NEIGHBOUR in this view
        sigma2_v_nei[name] = _t(bundle.sigma2_v[an, v_idx], device)  # [B, K]

    # Anchor sigma2_v (quality priors across all views)
    sigma2_v_anc = _t(bundle.sigma2_v[anchor_idx], device)     # [B, V]
    mean_J = _t(bundle.mean_J[anchor_idx], device)              # [B]

    return {
        "x_anc": x_anc,
        "x_anc_v": x_anc_v,
        "x_nei": x_nei,
        "y_nei": y_nei,
        "ew": ew,
        "sigma2_v_anc": sigma2_v_anc,
        "sigma2_v_nei": sigma2_v_nei,
        "mean_J": mean_J,
        "y_true": y_true,
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(pred: torch.Tensor, y_true: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    if cfg.loss == "huber":
        return F.huber_loss(pred, y_true, delta=cfg.huber_delta)
    elif cfg.loss == "mae":
        return F.l1_loss(pred, y_true)
    else:
        return F.mse_loss(pred, y_true)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, EPS)
    return {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2": float(r2),
    }


# ---------------------------------------------------------------------------
# Inference (full split)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_split(
    model: MVTabrGoraModel,
    idx: np.ndarray,
    bundle: MVDataBundle,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict raw-scale targets for all points in idx.
    Returns (y_true_raw [n], y_pred_raw [n]).
    """
    model.eval()
    mean_t = float(bundle.target_stats["mean"])
    std_t = float(bundle.target_stats["std"])

    preds_norm: List[np.ndarray] = []
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start : start + batch_size]
        batch = fetch_batch(batch_idx, bundle, device)
        pred_norm, _ = model(batch)
        preds_norm.append(pred_norm.detach().cpu().numpy())

    y_pred_norm = np.concatenate(preds_norm)
    y_pred_raw = y_pred_norm * std_t + mean_t
    y_true_raw = bundle.y[idx]
    return y_true_raw.astype(np.float32), y_pred_raw.astype(np.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_mv_tabr_gora(
    model: MVTabrGoraModel,
    bundle: MVDataBundle,
    cfg: TrainConfig,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """
    Train model with early stopping on val RMSE.
    Returns result dict with train/val/test metrics and timing.
    """
    model.to(device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs, eta_min=cfg.lr * 0.05
    )

    train_idx = bundle.train_idx
    val_idx = bundle.val_idx

    best_val_rmse = float("inf")
    best_state: Optional[dict] = None
    patience_left = cfg.patience
    best_epoch = -1
    epoch_results: List[dict] = []
    t_start = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        perm = rng.permutation(len(train_idx))
        epoch_loss = 0.0
        n_steps = 0

        for start in range(0, len(train_idx), cfg.batch_size):
            batch_local = perm[start : start + cfg.batch_size]
            batch_idx = train_idx[batch_local]
            batch = fetch_batch(batch_idx, bundle, device)

            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(batch)
            loss = compute_loss(pred, batch["y_true"], cfg)
            loss.backward()

            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()
            epoch_loss += float(loss.item())
            n_steps += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_steps, 1)

        # Validation
        y_val_true, y_val_pred = predict_split(model, val_idx, bundle, device, cfg.batch_size)
        val_rmse = rmse(y_val_true, y_val_pred)

        if verbose and (epoch % 10 == 0 or epoch < 5):
            print(f"  epoch {epoch:4d}  loss={avg_loss:.4f}  val_rmse={val_rmse:.4f}  "
                  f"patience={patience_left}")

        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"  Early stop at epoch {epoch} (best={best_epoch})")
                break

        epoch_results.append({"epoch": epoch, "loss": avg_loss, "val_rmse": val_rmse})

    # Restore best checkpoint
    assert best_state is not None
    model.load_state_dict(best_state)

    duration = time.time() - t_start
    if verbose:
        print(f"  Training done in {duration:.1f}s  best_epoch={best_epoch}  "
              f"best_val_rmse={best_val_rmse:.4f}")

    # Final metrics on all splits
    y_tr_true, y_tr_pred = predict_split(model, train_idx, bundle, device, cfg.batch_size)
    y_va_true, y_va_pred = predict_split(model, val_idx, bundle, device, cfg.batch_size)
    y_te_true, y_te_pred = predict_split(model, bundle.test_idx, bundle, device, cfg.batch_size)

    return {
        "best_epoch": best_epoch,
        "duration_seconds": round(duration, 2),
        "train": regression_metrics(y_tr_true, y_tr_pred),
        "val": regression_metrics(y_va_true, y_va_pred),
        "test": regression_metrics(y_te_true, y_te_pred),
        "epoch_history": epoch_results,
    }
