"""
manifold_teacher_v4.py — V4 teacher training.

Key changes vs v3:
  train_teacher_v4():
    - skip_centroid_loss (CA_FIX_3): drops L_centroid for regression stability
    - epochs parameter (CA_FIX_4): default 200 for regression, 100 for classification

The ManifoldTeacher architecture itself is unchanged; only training is modified.
"""
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3_SRC = os.path.normpath(os.path.join(_HERE, '..', '..', '..', 'gora_tabular', 'src'))
if _V3_SRC not in sys.path:
    sys.path.insert(0, _V3_SRC)

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from manifold_teacher import (
    ManifoldTeacher,
    _precompute_label_centroids,
    _precompute_neighbour_centroid,
    get_device,
)

__all__ = ['ManifoldTeacher', 'train_teacher_v4']


def train_teacher_v4(
    teacher: ManifoldTeacher,
    X: np.ndarray,               # [N, d_x]
    y: np.ndarray,               # [N]  float (reg) or int (clf)
    neigh_idx: np.ndarray,       # [N, P]
    edge_wts: np.ndarray,        # [N, P, M]
    view_mask: np.ndarray,       # [N, P, M]
    agree_score: np.ndarray,     # [N]
    tr_i: np.ndarray,
    task: str = "regression",
    n_classes: int = 1,
    epochs: int = 200,           # CA_FIX_4: 200 epochs default for regression
    lr: float = 1e-3,
    batch_size: int = 512,
    lam_agree: float = 1.0,
    lam_label: float = 0.5,
    lam_centroid: float = 0.1,
    skip_centroid_loss: bool = False,   # CA_FIX_3: disable L_centroid
) -> np.ndarray:
    """
    Train ManifoldTeacher and return z_arr [N, d_z] for all rows.

    V4 changes vs v3:
      - skip_centroid_loss=True removes L_centroid (neighbour feature centroid).
        Recommended for regression: centroid loss on raw feature space adds
        noise and the loss doesn't converge well on low-dim tabular data.
      - epochs defaults to 200 to allow the regression teacher more room.

    Loss schedule (v4 lite mode, skip_centroid_loss=True):
      L = lam_agree * L_agree + lam_label * L_label_m

    Loss schedule (full mode):
      L = lam_agree * L_agree + lam_label * L_label_m + lam_centroid * L_centroid
    """
    t0 = time.time()
    dev = get_device()
    N, d_x = X.shape
    M = edge_wts.shape[2]

    print(f"  [teacher-v4] Precomputing label centroids...")
    y_bar = _precompute_label_centroids(y, neigh_idx, edge_wts, view_mask, n_classes)

    if not skip_centroid_loss:
        print(f"  [teacher-v4] Precomputing neighbour centroids...")
        x_bar = _precompute_neighbour_centroid(X, neigh_idx, edge_wts, primary_view=0)
    else:
        x_bar = None

    teacher = teacher.to(dev)
    tr_idx_set = set(tr_i.tolist())
    tr_mask = np.array([i in tr_idx_set for i in range(N)], dtype=bool)

    opt = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"  [teacher-v4] Training for {epochs} epochs on {len(tr_i)} rows "
          f"({'lite: L_agree+L_label' if skip_centroid_loss else 'full: L_agree+L_label+L_centroid'})...")

    tr_i_arr = tr_i.copy()
    for ep in range(epochs):
        teacher.train()
        np.random.shuffle(tr_i_arr)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(tr_i_arr), batch_size):
            b_idx = tr_i_arr[start:start + batch_size]
            x_b   = torch.from_numpy(X[b_idx]).float().to(dev)
            ag_b  = torch.from_numpy(agree_score[b_idx]).float().to(dev)

            z = teacher(x_b)                                 # [B, d_z]

            # L_agree: predict per-row agree_score
            l_agree = F.mse_loss(teacher.head_agree(z).squeeze(-1), ag_b)

            # L_label: predict per-view label centroid
            if n_classes == 1:
                # Regression: y_bar [N, M] → predict each view centroid
                yb_b = torch.from_numpy(y_bar[b_idx]).float().to(dev)   # [B, M]
                pred_label = teacher.head_label(z)                        # [B, M]
                l_label = F.mse_loss(pred_label, yb_b)
            else:
                # Classification: y_bar [N, M, C] → cross-entropy per view
                yb_b = torch.from_numpy(y_bar[b_idx]).float().to(dev)   # [B, M, C]
                pred_label = teacher.head_label(z).view(len(b_idx), M, n_classes)
                l_label = F.cross_entropy(
                    pred_label.reshape(-1, n_classes),
                    yb_b.reshape(-1, n_classes).argmax(-1),
                )

            loss = lam_agree * l_agree + lam_label * l_label

            if not skip_centroid_loss and x_bar is not None:
                xb_b = torch.from_numpy(x_bar[b_idx]).float().to(dev)   # [B, d_x]
                pred_centroid = teacher.head_centroid(z)                  # [B, d_x]
                l_centroid = F.mse_loss(pred_centroid, xb_b)
                loss = loss + lam_centroid * l_centroid

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if ep % 20 == 0:
            avg = total_loss / max(n_batches, 1)
            print(f"  [teacher-v4] ep={ep:4d} loss={avg:.4f}")

    elapsed = time.time() - t0
    print(f"  [teacher-v4] Done in {elapsed:.1f}s")

    # Precompute z for all rows
    teacher.eval()
    all_z = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            x_b = torch.from_numpy(X[start:start + batch_size]).float().to(dev)
            all_z.append(teacher(x_b).cpu().numpy())
    z_arr = np.concatenate(all_z, axis=0)
    print(f"  [teacher-v4] z_arr: {z_arr.shape}, mean_norm={np.linalg.norm(z_arr, axis=1).mean():.3f}")
    return z_arr
