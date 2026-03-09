"""
manifold_teacher_v4.py — v4 teacher training.

Key differences from v3:
  - optional `skip_centroid_loss` for regression-safe teacher-lite
  - longer default epoch budget for regression
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.gora_tabular.src.manifold_teacher import (
    ManifoldTeacher,
    _precompute_label_centroids,
    _precompute_neighbour_centroid,
    get_device,
)

__all__ = ["ManifoldTeacher", "train_teacher_v4"]


def train_teacher_v4(
    teacher: ManifoldTeacher,
    X: np.ndarray,
    y: np.ndarray,
    neigh_idx: np.ndarray,
    edge_wts: np.ndarray,
    view_mask: np.ndarray,
    agree_score: np.ndarray,
    tr_i: np.ndarray,
    task: str = "regression",
    n_classes: int = 1,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 512,
    lam_agree: float = 1.0,
    lam_label: float = 0.5,
    lam_centroid: float = 0.1,
    skip_centroid_loss: bool = False,
) -> np.ndarray:
    t0 = time.time()
    dev = get_device()
    n_rows, d_x = X.shape
    n_views = edge_wts.shape[2]
    d_z = teacher.d_z
    regression = task == "regression"

    print("  [teacher-v4] Precomputing label centroids...")
    y_bar = _precompute_label_centroids(y.astype(np.float32), neigh_idx, edge_wts, view_mask, n_classes)

    if not skip_centroid_loss:
        print("  [teacher-v4] Precomputing neighbour centroids...")
        x_bar = _precompute_neighbour_centroid(X, neigh_idx, edge_wts, primary_view=0)
    else:
        x_bar = None

    ag_mean = agree_score[tr_i].mean()
    ag_std = agree_score[tr_i].std() + 1e-8
    ag_norm = ((agree_score - ag_mean) / ag_std).astype(np.float32)

    teacher = teacher.to(dev)
    agree_head = nn.Linear(d_z, 1).to(dev)
    if regression:
        label_heads = nn.ModuleList([nn.Linear(d_z, 1) for _ in range(n_views)]).to(dev)
    else:
        label_heads = nn.ModuleList([nn.Linear(d_z, n_classes) for _ in range(n_views)]).to(dev)
    centroid_head = nn.Linear(d_z, d_x).to(dev)

    params = list(teacher.parameters()) + list(agree_head.parameters()) + list(label_heads.parameters())
    if not skip_centroid_loss:
        params += list(centroid_head.parameters())

    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    print(
        f"  [teacher-v4] Training for {epochs} epochs on {len(tr_i)} rows "
        f"({'lite: L_agree+L_label' if skip_centroid_loss else 'full: L_agree+L_label+L_centroid'})..."
    )

    train_indices = tr_i.copy()
    for ep in range(epochs):
        np.random.shuffle(train_indices)
        teacher.train()
        agree_head.train()
        label_heads.train()
        centroid_head.train()

        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_indices), batch_size):
            b_idx = train_indices[start : start + batch_size]
            x_b = torch.from_numpy(X[b_idx]).float().to(dev)
            ag_b = torch.from_numpy(ag_norm[b_idx]).float().to(dev)
            z_b = teacher(x_b)

            l_agree = F.mse_loss(agree_head(z_b).squeeze(-1), ag_b)

            if regression:
                target = torch.from_numpy(y_bar[b_idx]).float().to(dev)
                l_label = torch.tensor(0.0, device=dev)
                for view_idx, head in enumerate(label_heads):
                    l_label = l_label + F.mse_loss(head(z_b).squeeze(-1), target[:, view_idx])
                l_label = l_label / n_views
            else:
                target = torch.from_numpy(y_bar[b_idx]).float().to(dev)
                l_label = torch.tensor(0.0, device=dev)
                for view_idx, head in enumerate(label_heads):
                    pred_log = F.log_softmax(head(z_b), dim=-1)
                    tgt = target[:, view_idx, :].clamp(min=1e-9)
                    tgt = tgt / tgt.sum(dim=-1, keepdim=True)
                    l_label = l_label + F.kl_div(pred_log, tgt, reduction="batchmean")
                l_label = l_label / n_views

            loss = lam_agree * l_agree + lam_label * l_label

            if not skip_centroid_loss and x_bar is not None:
                xb_bar = torch.from_numpy(x_bar[b_idx]).float().to(dev)
                l_centroid = F.mse_loss(centroid_head(z_b), xb_bar)
                loss = loss + lam_centroid * l_centroid

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1

        sched.step()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  [teacher-v4] ep={ep:4d} loss={total_loss / max(n_batches, 1):.4f}")

    elapsed = time.time() - t0
    print(f"  [teacher-v4] Done in {elapsed:.1f}s")

    teacher.eval()
    all_z = []
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            x_b = torch.from_numpy(X[start : start + batch_size]).float().to(dev)
            all_z.append(teacher(x_b).cpu().numpy())
    z_arr = np.concatenate(all_z, axis=0)
    print(f"  [teacher-v4] z_arr: {z_arr.shape}, mean_norm={np.linalg.norm(z_arr, axis=1).mean():.3f}")
    return z_arr
