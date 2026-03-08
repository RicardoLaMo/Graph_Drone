"""
manifold_teacher.py — ManifoldTeacher: self-supervised manifold encoder for GoRA v3.

The teacher f_T: x_i → z_i ∈ R^{d_z} is trained with three simultaneous tasks
using the precomputed joint-kNN neighbourhood:

  L_agree    = MSE( f_T(x_i), agree_score_i )
               Teaches z_i to encode local view-agreement geometry.
               agree_score_i ∈ [0,1]: fraction of union pool shared by ≥2 views.

  L_label_m  = (1/M) Σ_m MSE( g_m(z_i), y̅^(m)_i )
               g_m: per-view linear head R^{d_z} → R^1 (regression) or R^C (clf).
               y̅^(m)_i = Σ_{j∈kNN_m(i)} w^(m)_ij · y_j  — per-view label centroid.
               Teaches z_i to anticipate the label density each view sees.

  L_centroid = MSE( h_T(z_i), x̄^(0)_i )
               h_T: Linear R^{d_z} → R^{d_x}.
               x̄^(0)_i = Σ_{j∈kNN_0(i)} w^(0)_ij · x_j  — primary-view centroid.
               Teaches z_i to project onto the local manifold centroid.

After training:
  - f_T weights frozen
  - Projection heads g_m, h_T discarded
  - z_arr = f_T(X_all) precomputed and returned as [N, d_z]

Usage:
  teacher = ManifoldTeacher(d_x, d_z=64)
  z_arr = train_teacher(teacher, X, y, neigh_idx, edge_wts, view_mask, agree_score,
                        task="regression", n_views=M)
  # z_arr shape [N, d_z], used downstream as router context and cross-attn query
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class ManifoldTeacher(nn.Module):
    """
    f_T: x_i → z_i ∈ R^{d_z}

    Three-layer MLP with LayerNorm output, trained jointly on:
      1. agree_score regression
      2. per-view label centroid regression
      3. primary-view neighbourhood centroid regression
    """

    def __init__(self, d_x: int, d_z: int = 64, hidden: int = 128):
        super().__init__()
        self.d_z = d_z
        self.encoder = nn.Sequential(
            nn.Linear(d_x, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_z),
            nn.LayerNorm(d_z),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d_x] → z: [B, d_z]"""
        return self.encoder(x)


def _precompute_label_centroids(
    y: np.ndarray,
    neigh_idx: np.ndarray,    # [N, P]  union pool, -1=padding
    edge_wts: np.ndarray,     # [N, P, M]  per-view Gaussian weights
    view_mask: np.ndarray,    # [N, P, M]  binary: j ∈ kNN_m(i)
    n_classes: int = 1,
) -> np.ndarray:
    """
    Precompute per-view label centroid for each anchor.
    Returns y_bar [N, M] (regression) or [N, M, C] (classification).
    """
    N, P, M = edge_wts.shape
    regression = (n_classes == 1)

    if regression:
        y_bar = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            for vi in range(M):
                wts = edge_wts[i, :, vi] * view_mask[i, :, vi]   # [P]
                w_sum = wts.sum() + 1e-8
                valid = neigh_idx[i] >= 0
                y_vals = np.where(valid, y[np.clip(neigh_idx[i], 0, None)], 0.0)
                y_bar[i, vi] = (wts * y_vals).sum() / w_sum
    else:
        y_bar = np.zeros((N, M, n_classes), dtype=np.float32)
        y_onehot = np.eye(n_classes, dtype=np.float32)[y]   # [N, C]
        for i in range(N):
            for vi in range(M):
                wts = edge_wts[i, :, vi] * view_mask[i, :, vi]   # [P]
                w_sum = wts.sum() + 1e-8
                valid = neigh_idx[i] >= 0
                valid_idx = np.where(valid, neigh_idx[i], 0)
                y_vals = y_onehot[valid_idx]   # [P, C]
                y_bar[i, vi] = (wts[:, None] * y_vals).sum(0) / w_sum

    return y_bar


def _precompute_neighbour_centroid(
    X: np.ndarray,
    neigh_idx: np.ndarray,    # [N, P]
    edge_wts: np.ndarray,     # [N, P, M]
    primary_view: int = 0,
) -> np.ndarray:
    """
    x̄^(0)_i = Σ_{j∈pool} w^(0)_ij · x_j  — primary-view weighted centroid.
    Returns [N, d_x].
    """
    N, P, M = edge_wts.shape
    centroid = np.zeros((N, X.shape[1]), dtype=np.float32)
    wts_pv = edge_wts[:, :, primary_view]   # [N, P]
    for i in range(N):
        total = wts_pv[i].sum() + 1e-8
        for pi_idx in range(P):
            j = neigh_idx[i, pi_idx]
            if j >= 0:
                centroid[i] += wts_pv[i, pi_idx] * X[j]
        centroid[i] /= total
    return centroid


def train_teacher(
    teacher: ManifoldTeacher,
    X: np.ndarray,               # [N, d_x]
    y: np.ndarray,               # [N] int (clf) or float (reg)
    neigh_idx: np.ndarray,       # [N, P]
    edge_wts: np.ndarray,        # [N, P, M]
    view_mask: np.ndarray,       # [N, P, M]
    agree_score: np.ndarray,     # [N]
    tr_i: np.ndarray,            # training indices
    task: str = "regression",
    n_classes: int = 1,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 512,
    lam_agree: float = 1.0,
    lam_label: float = 0.5,
    lam_centroid: float = 0.1,
) -> np.ndarray:
    """
    Train ManifoldTeacher and return z_arr [N, d_z] (all rows).

    The teacher is frozen after this call — caller should not pass it to
    the student optimizer.
    """
    t0 = time.time()
    dev = get_device()
    N, d_x = X.shape
    M = edge_wts.shape[2]
    d_z = teacher.d_z
    regression = (task == "regression")

    print(f"  [teacher] Precomputing label centroids...")
    y_float = y.astype(np.float32)
    y_bar = _precompute_label_centroids(y_float, neigh_idx, edge_wts, view_mask, n_classes)
    # y_bar: [N, M] regression | [N, M, C] classification

    print(f"  [teacher] Precomputing neighbour centroids...")
    x_bar = _precompute_neighbour_centroid(X, neigh_idx, edge_wts, primary_view=0)
    # x_bar: [N, d_x]

    # Normalise agree_score to stabilise loss scale
    ag_mean = agree_score[tr_i].mean()
    ag_std = agree_score[tr_i].std() + 1e-8
    ag_norm = ((agree_score - ag_mean) / ag_std).astype(np.float32)

    teacher = teacher.to(dev)

    # Per-view label heads + centroid head (trained jointly, discarded after)
    if regression:
        label_heads = nn.ModuleList([nn.Linear(d_z, 1) for _ in range(M)])
    else:
        label_heads = nn.ModuleList([nn.Linear(d_z, n_classes) for _ in range(M)])
    centroid_head = nn.Linear(d_z, d_x)

    label_heads = label_heads.to(dev)
    centroid_head = centroid_head.to(dev)

    all_params = list(teacher.parameters()) + list(label_heads.parameters()) + list(centroid_head.parameters())
    opt = torch.optim.Adam(all_params, lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    print(f"  [teacher] Training for {epochs} epochs on {len(tr_i)} rows...")
    for ep in range(epochs):
        teacher.train(); label_heads.train(); centroid_head.train()
        perm = np.random.permutation(tr_i)
        ep_loss = 0.0; n_batches = 0

        for start in range(0, len(perm), batch_size):
            b_idx = perm[start:start + batch_size]
            x_b = torch.tensor(X[b_idx], dtype=torch.float32).to(dev)
            ag_b = torch.tensor(ag_norm[b_idx], dtype=torch.float32).to(dev)

            z_b = teacher(x_b)   # [B, d_z]

            # L_agree
            ag_pred = z_b.mean(-1)   # use first scalar of z as agree predictor
            # Better: a small linear head dedicated to agree
            l_agree = F.mse_loss(z_b[:, 0], ag_b)

            # L_label
            l_label = torch.tensor(0.0, device=dev)
            if regression:
                yb_bar = torch.tensor(y_bar[b_idx], dtype=torch.float32).to(dev)   # [B, M]
                for vi, lh in enumerate(label_heads):
                    l_label = l_label + F.mse_loss(lh(z_b).squeeze(-1), yb_bar[:, vi])
            else:
                yb_bar = torch.tensor(y_bar[b_idx], dtype=torch.float32).to(dev)   # [B, M, C]
                for vi, lh in enumerate(label_heads):
                    # KL-div: predicted log-softmax vs target soft distribution
                    pred_log = F.log_softmax(lh(z_b), dim=-1)   # [B, C]
                    tgt = yb_bar[:, vi, :].clamp(min=1e-9)
                    tgt = tgt / tgt.sum(-1, keepdim=True)
                    l_label = l_label + F.kl_div(pred_log, tgt, reduction="batchmean")
            l_label = l_label / M

            # L_centroid
            xb_bar = torch.tensor(x_bar[b_idx], dtype=torch.float32).to(dev)   # [B, d_x]
            l_centroid = F.mse_loss(centroid_head(z_b), xb_bar)

            loss = lam_agree * l_agree + lam_label * l_label + lam_centroid * l_centroid
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_batches += 1

        sched.step()
        if ep % 20 == 0:
            print(f"  [teacher] ep={ep:4d} loss={ep_loss/n_batches:.4f}")

    print(f"  [teacher] Done in {time.time()-t0:.1f}s")

    # Freeze teacher, discard heads
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # Precompute z_arr for all N rows
    print(f"  [teacher] Precomputing z_arr for all {N} rows...")
    z_arr = np.zeros((N, d_z), dtype=np.float32)
    teacher_cpu = teacher.cpu()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            b_idx = np.arange(start, min(start + batch_size, N))
            x_b = torch.tensor(X[b_idx], dtype=torch.float32)
            z_arr[b_idx] = teacher_cpu(x_b).numpy()
    teacher.to(dev)

    return z_arr
