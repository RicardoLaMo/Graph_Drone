"""
train.py — GoRA-Tabular v2 training with joint-view kNN neighbourhood.

KEY UPGRADE from v1: build_joint_neighbourhood()
  Instead of one primary view determining neighbour indices,
  each view contributes k_per_view neighbours independently.
  Union pool has ≤ M × k_per_view candidates per anchor row.

  This makes routing semantics correct:
    - Isolation (peaked π at view m): Ã_{ij} ≈ w^(m)_{ij} which is 0
      for j ∉ kNN_m(i). Head only attends to rows genuinely close in view m.
    - Interaction (flat π): Ã averages all views. Rows close in any view
      receive some attention weight.

  view_mask [N, P, M]: binary indicator j ∈ kNN_m(i), used to compute
  per-row view-agreement score (disagreement → routing should specialise).

Decisions:
  K_each = 5 per view (pool ≤ 20 for CA M=4, ≤ 15 for MNIST M=3)
  TabPFN: California subsampled to 8k train; MNIST full 7k (PCA-200 features)
"""
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Tuple


SEED = 42; torch.manual_seed(SEED)


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# ─── Joint-view kNN neighbourhood ──────────────────────────────────────────────

def build_joint_neighbourhood(
    X_views: Dict[str, np.ndarray],
    k_per_view: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a joint-view neighbourhood pool.

    For each view m, find k_per_view nearest neighbours independently.
    Union the pools across views, deduplicate, pad to fixed size P.

    Returns:
      neigh_idx:  [N, P]    — padded neighbour pool indices (self=-1 mask)
      edge_wts:   [N, P, M] — Gaussian edge weights (0 if j ∉ kNN_m(i))
      view_mask:  [N, P, M] — binary: j ∈ kNN_m(i)
      agree_score:[N]       — fraction of pool shared by ≥2 views per anchor
    """
    view_names = list(X_views.keys())
    M = len(view_names)
    N = list(X_views.values())[0].shape[0]
    max_pool = k_per_view * M   # generous upper bound

    print(f"  [joint-kNN] Views={view_names} k_per_view={k_per_view} max_pool={max_pool}")

    # Per-view kNN
    per_view_knn = {}     # {vname: [N, k_per_view]}
    per_view_sigma = {}   # for Gaussian bandwidth
    for vname, Xv in X_views.items():
        nb = NearestNeighbors(n_neighbors=k_per_view + 1, n_jobs=-1).fit(Xv)
        dists, idxs = nb.kneighbors(Xv)
        per_view_knn[vname] = idxs[:, 1:].astype(np.int64)   # drop self  [N, k]
        per_view_sigma[vname] = float(np.median(dists[:, 1:]) + 1e-8)
        print(f"  [joint-kNN] View '{vname}' sigma={per_view_sigma[vname]:.4f}")

    # Build union pool per anchor
    neigh_idx = np.full((N, max_pool), -1, dtype=np.int64)   # -1 = padding
    edge_wts = np.zeros((N, max_pool, M), dtype=np.float32)
    view_mask = np.zeros((N, max_pool, M), dtype=np.float32)

    for i in range(N):
        # Collect union set across views
        pool_set = set()
        for vname in view_names:
            pool_set.update(per_view_knn[vname][i].tolist())
        pool = sorted(pool_set)[:max_pool]   # deterministic ordering
        P_i = len(pool)
        neigh_idx[i, :P_i] = pool

        for vi, vname in enumerate(view_names):
            Xv = X_views[vname]
            knn_set_i = set(per_view_knn[vname][i].tolist())
            sigma_v = per_view_sigma[vname]
            for pi_idx, j in enumerate(pool):
                diff_sq = float(np.sum((Xv[i] - Xv[j]) ** 2))
                w = float(np.exp(-diff_sq / sigma_v ** 2))
                edge_wts[i, pi_idx, vi] = w if j in knn_set_i else 0.0
                view_mask[i, pi_idx, vi] = 1.0 if j in knn_set_i else 0.0

        # Row-normalise per view (only positions j ∈ kNN_m(i))
        for vi in range(M):
            row_sum = edge_wts[i, :P_i, vi].sum() + 1e-8
            edge_wts[i, :P_i, vi] /= row_sum

    # Agreement score: fraction of pool positions covered by ≥ 2 views per anchor
    view_coverage = view_mask.sum(-1)   # [N, P] — how many views claim j
    in_pool = (neigh_idx >= 0).astype(np.float32)   # [N, P]
    shared = ((view_coverage >= 2) * in_pool).sum(-1)   # [N]
    total = in_pool.sum(-1).clip(min=1)               # [N]
    agree_score = (shared / total).astype(np.float32)   # [N]
    print(f"  [joint-kNN] mean agree_score={agree_score.mean():.3f} "
          f"(0=all different, 1=all views agree on the same neighbours)")

    return neigh_idx, edge_wts, view_mask, agree_score


def build_neighbourhood(X_views, k=15, primary_key=None):
    """V1 fallback: single primary view — kept for G2 backward compat."""
    view_names = list(X_views.keys())
    M = len(view_names)
    primary = primary_key or view_names[0]
    X_prim = X_views[primary]
    N = X_prim.shape[0]
    nb = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X_prim)
    _, neigh_idx_raw = nb.kneighbors(X_prim)
    neigh_idx = neigh_idx_raw[:, 1:].astype(np.int64)
    edge_wts = np.zeros((N, k, M), dtype=np.float32)
    for mi, vname in enumerate(view_names):
        Xv = X_views[vname]
        diff_sq = ((Xv[np.arange(N)[:, None], :] - Xv[neigh_idx]) ** 2).sum(-1)
        sigma = np.median(np.sqrt(diff_sq)) + 1e-8
        edge_wts[:, :, mi] = np.exp(-diff_sq / sigma ** 2)
        row_sum = edge_wts[:, :, mi].sum(-1, keepdims=True) + 1e-8
        edge_wts[:, :, mi] /= row_sum
    return neigh_idx, edge_wts


# ─── Mini-batch fetch (handles -1 padding) ────────────────────────────────────

def fetch_batch(batch_idx, X_all, g_all, y_all, neigh_idx, edge_wts, device,
                view_mask=None, agree_score=None):
    b_ni = neigh_idx[batch_idx]        # [B, P]  may contain -1
    # Replace -1 (padding) with 0 for gather, then zero the embeddings after
    pad_mask = (b_ni == -1)
    b_ni_safe = np.where(pad_mask, 0, b_ni)

    x_anc = torch.tensor(X_all[batch_idx], dtype=torch.float32).to(device)
    g_anc = torch.tensor(g_all[batch_idx], dtype=torch.float32).to(device)
    x_nei = torch.tensor(X_all[b_ni_safe], dtype=torch.float32).to(device)  # [B, P, d]
    # Zero out padded positions
    pad_t = torch.tensor(pad_mask, dtype=torch.float32).to(device).unsqueeze(-1)
    x_nei = x_nei * (1 - pad_t)

    ew = torch.tensor(edge_wts[batch_idx], dtype=torch.float32).to(device)  # [B, P, M]
    y_b = torch.tensor(y_all[batch_idx]).to(device)

    vm = torch.tensor(view_mask[batch_idx],
                      dtype=torch.float32).to(device) if view_mask is not None else None
    ag = torch.tensor(agree_score[batch_idx],
                      dtype=torch.float32).to(device) if agree_score is not None else None
    return x_anc, g_anc, x_nei, ew, y_b, vm, ag


# ─── Disagreement-aligned routing loss ────────────────────────────────────────

def routing_disagreement_loss(pi: torch.Tensor, agree: torch.Tensor, lam: float = 0.01):
    """
    Geometry-grounded routing regularisation.

    pi:    [B, H, M]  routing weights
    agree: [B]         per-row fraction of pool shared by ≥2 views

    L_route = lam * mean(
        agree_i * H(pi_i)                        ← high agreement → encourage diffuse
       + (1 - agree_i) * (log(M) - H(pi_i))      ← low agreement  → encourage peaked
    )

    This aligns routing entropy with the geometry: the router is told to
    specialise exactly when the views disagree about who is close.
    """
    M = pi.shape[-1]
    H = -(pi * (pi + 1e-9).log()).sum(-1).mean(-1)   # [B]  avg over heads
    log_M = float(torch.tensor(M).log())
    loss = (agree * H + (1 - agree) * (log_M - H)).mean()
    return lam * loss


# ─── Training ─────────────────────────────────────────────────────────────────

def train_gora(
    model: nn.Module,
    X_all: np.ndarray,
    g_all: np.ndarray,
    y_all: np.ndarray,
    neigh_idx: np.ndarray,
    edge_wts: np.ndarray,
    tr_i: np.ndarray,
    va_i: np.ndarray,
    task: str,
    n_classes: int = 1,
    epochs: int = 150,
    patience: int = 20,
    lr: float = 3e-4,
    batch_size: int = 512,
    name: str = "model",
    view_mask: np.ndarray = None,
    agree_score: np.ndarray = None,
    routing_lam: float = 0.0,   # 0 = disabled (G5); >0 = G6
) -> nn.Module:
    dev = get_device()
    model = model.to(dev)
    crit = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    best, bst, wait = 1e9, None, 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr_i)
        ep_loss = 0.0; n_batches = 0
        for start in range(0, len(perm), batch_size):
            b_idx = perm[start:start + batch_size]
            x_a, g_a, x_n, ew_a, y_b, vm, ag = fetch_batch(
                b_idx, X_all, g_all, y_all, neigh_idx, edge_wts, dev, view_mask, agree_score)
            opt.zero_grad()
            out, pi, tau = model(x_a, g_a, x_n, ew_a)
            pred_loss = (crit(out.squeeze(-1), y_b.float()) if task == "regression"
                         else crit(out, y_b.long()))
            loss = pred_loss
            if routing_lam > 0 and pi is not None and ag is not None:
                loss = loss + routing_disagreement_loss(pi, ag, routing_lam)
            loss.backward(); opt.step()
            ep_loss += pred_loss.item(); n_batches += 1
        sched.step()

        model.eval()
        val_loss = 0.0; val_batches = 0
        with torch.no_grad():
            for start in range(0, len(va_i), batch_size):
                b_idx = va_i[start:start + batch_size]
                x_a, g_a, x_n, ew_a, y_b, _, _ = fetch_batch(
                    b_idx, X_all, g_all, y_all, neigh_idx, edge_wts, dev)
                out, *_ = model(x_a, g_a, x_n, ew_a)
                vl = (crit(out.squeeze(-1), y_b.float()) if task == "regression"
                      else crit(out, y_b.long())).item()
                val_loss += vl; val_batches += 1
        val_loss /= max(val_batches, 1)

        if val_loss < best:
            best = val_loss; wait = 0
            bst = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= patience: break
        if ep % 10 == 0:
            print(f"  [{name}] ep={ep:4d} tr={ep_loss/n_batches:.4f} val={val_loss:.4f}")

    model.load_state_dict(bst)
    print(f"  [{name}] Done in {time.time()-t0:.1f}s | best_val={best:.4f}")
    return model


@torch.no_grad()
def predict_gora(model, X_all, g_all, y_all, neigh_idx, edge_wts, te_i, task, batch_size=512):
    dev = get_device()
    model = model.to(dev).eval()
    all_out, all_pi = [], []
    for start in range(0, len(te_i), batch_size):
        b_idx = te_i[start:start + batch_size]
        x_a, g_a, x_n, ew_a, _, _, _ = fetch_batch(
            b_idx, X_all, g_all, y_all, neigh_idx, edge_wts, dev)
        out, pi, tau = model(x_a, g_a, x_n, ew_a)
        all_out.append(out.cpu())
        all_pi.append(pi.cpu() if pi is not None else None)

    out_all = torch.cat(all_out, dim=0)
    pi_all = (torch.cat([p for p in all_pi if p is not None], dim=0).numpy()
              if all_pi and all_pi[0] is not None else None)
    if task == "classification":
        proba = torch.softmax(out_all, -1).numpy(); preds = proba.argmax(-1)
    else:
        preds = out_all.squeeze(-1).numpy(); proba = None
    tau_np = tau.cpu().numpy() if tau is not None else None
    return preds, proba, pi_all, tau_np


# ─── v3: Label context precomputation ─────────────────────────────────────────

def compute_label_ctx_per_view(
    y_all: np.ndarray,
    neigh_idx: np.ndarray,    # [N, P]
    edge_wts: np.ndarray,     # [N, P, M]
    view_mask: np.ndarray,    # [N, P, M]
) -> np.ndarray:
    """
    Precompute lbl_nei [N, P, M]: per-slot weighted neighbour label
    for each view m.  Used in fetch_batch_v3 for LabelContextEncoder.

    lbl_nei[i, p, m] = y_all[neigh_idx[i,p]] * view_mask[i,p,m]
                       (0 if slot is padded or not in view m's kNN)

    Regression: y_all float, n_classes=1
    Classification: caller should pass one-hot or integer y (LabelContextEncoder
                    handles normalisation internally).
    """
    N, P, M = edge_wts.shape
    lbl_nei = np.zeros((N, P, M), dtype=np.float32)
    for i in range(N):
        for pi_idx in range(P):
            j = neigh_idx[i, pi_idx]
            if j >= 0:
                for vi in range(M):
                    lbl_nei[i, pi_idx, vi] = float(y_all[j]) * view_mask[i, pi_idx, vi]
    return lbl_nei


# ─── v3: Extended fetch_batch ──────────────────────────────────────────────────

def fetch_batch_v3(
    batch_idx: np.ndarray,
    X_all: np.ndarray,
    g_all: np.ndarray,
    y_all: np.ndarray,
    neigh_idx: np.ndarray,
    edge_wts: np.ndarray,
    device: torch.device,
    view_mask: np.ndarray = None,
    agree_score: np.ndarray = None,
    z_arr: np.ndarray = None,       # [N, d_z]  teacher embeddings
    lbl_nei: np.ndarray = None,     # [N, P, M] label context
):
    """
    Extended fetch_batch for v3: adds z_anc and lbl_nei to the batch.
    All new fields are optional — falls back to fetch_batch behaviour when None.
    """
    x_anc, g_anc, x_nei, ew, y_b, vm, ag = fetch_batch(
        batch_idx, X_all, g_all, y_all, neigh_idx, edge_wts, device,
        view_mask=view_mask, agree_score=agree_score,
    )
    z_b = (torch.tensor(z_arr[batch_idx], dtype=torch.float32).to(device)
           if z_arr is not None else None)
    lb = (torch.tensor(lbl_nei[batch_idx], dtype=torch.float32).to(device)
          if lbl_nei is not None else None)
    return x_anc, g_anc, x_nei, ew, y_b, vm, ag, z_b, lb


# ─── v3: MQGoraTransformer training ───────────────────────────────────────────

def train_gora_v3(
    model: nn.Module,
    X_all: np.ndarray,
    g_all: np.ndarray,
    y_all: np.ndarray,
    neigh_idx: np.ndarray,
    edge_wts: np.ndarray,
    tr_i: np.ndarray,
    va_i: np.ndarray,
    task: str,
    n_classes: int = 1,
    epochs: int = 150,
    patience: int = 20,
    lr: float = 3e-4,
    batch_size: int = 512,
    name: str = "model",
    view_mask: np.ndarray = None,
    agree_score: np.ndarray = None,
    routing_lam: float = 0.0,
    z_arr: np.ndarray = None,       # [N, d_z]  teacher embeddings (frozen)
    lbl_nei: np.ndarray = None,     # [N, P, M] label context per view
) -> nn.Module:
    """
    Training loop for MQGoraTransformer (G7-G10).

    Compatible with standard GoraTransformer too (z_arr=None, lbl_nei=None
    → falls back to fetch_batch semantics, aux_losses={}).

    Returns the best-val-loss model.
    """
    dev = get_device()
    model = model.to(dev)
    crit = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    # Exclude frozen teacher parameters from student optimizer
    student_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(student_params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    best, bst, wait = 1e9, None, 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr_i)
        ep_loss = 0.0; n_batches = 0

        for start in range(0, len(perm), batch_size):
            b_idx = perm[start:start + batch_size]
            x_a, g_a, x_n, ew_a, y_b, vm, ag, z_b, lb = fetch_batch_v3(
                b_idx, X_all, g_all, y_all, neigh_idx, edge_wts, dev,
                view_mask=view_mask, agree_score=agree_score,
                z_arr=z_arr, lbl_nei=lbl_nei,
            )
            opt.zero_grad()

            # MQGoraTransformer returns (pred, pi, tau, aux_losses)
            # Standard GoraTransformer returns (pred, pi, tau) — handle both
            result = model(x_a, g_a, x_n, ew_a,
                           view_mask=vm, z_anc=z_b, lbl_nei=lb, agree_score=ag)
            if len(result) == 4:
                out, pi, tau, aux_losses = result
            else:
                out, pi, tau = result; aux_losses = {}

            pred_loss = (crit(out.squeeze(-1), y_b.float()) if task == "regression"
                         else crit(out, y_b.long()))
            loss = pred_loss
            if routing_lam > 0 and pi is not None and ag is not None:
                loss = loss + routing_disagreement_loss(pi, ag, routing_lam)
            for aux_val in aux_losses.values():
                loss = loss + aux_val

            loss.backward(); opt.step()
            ep_loss += pred_loss.item(); n_batches += 1
        sched.step()

        # Validation (no label context at val time — mirrors inference)
        model.eval()
        val_loss = 0.0; val_batches = 0
        with torch.no_grad():
            for start in range(0, len(va_i), batch_size):
                b_idx = va_i[start:start + batch_size]
                x_a, g_a, x_n, ew_a, y_b, vm, ag, z_b, _ = fetch_batch_v3(
                    b_idx, X_all, g_all, y_all, neigh_idx, edge_wts, dev,
                    view_mask=view_mask, agree_score=agree_score, z_arr=z_arr,
                    lbl_nei=None,   # ← inference mode: no labels
                )
                result = model(x_a, g_a, x_n, ew_a, view_mask=vm, z_anc=z_b)
                out = result[0]
                vl = (crit(out.squeeze(-1), y_b.float()) if task == "regression"
                      else crit(out, y_b.long())).item()
                val_loss += vl; val_batches += 1
        val_loss /= max(val_batches, 1)

        if val_loss < best:
            best = val_loss; wait = 0
            bst = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= patience: break
        if ep % 10 == 0:
            print(f"  [{name}] ep={ep:4d} tr={ep_loss/n_batches:.4f} val={val_loss:.4f}")

    model.load_state_dict(bst)
    print(f"  [{name}] Done in {time.time()-t0:.1f}s | best_val={best:.4f}")
    return model


@torch.no_grad()
def predict_gora_v3(
    model, X_all, g_all, y_all, neigh_idx, edge_wts, te_i, task,
    view_mask=None, z_arr=None, batch_size=512,
):
    """
    Prediction for MQGoraTransformer.  lbl_nei is always None at inference.
    """
    dev = get_device()
    model = model.to(dev).eval()
    all_out, all_pi = [], []

    for start in range(0, len(te_i), batch_size):
        b_idx = te_i[start:start + batch_size]
        x_a, g_a, x_n, ew_a, _, vm, _, z_b, _ = fetch_batch_v3(
            b_idx, X_all, g_all, y_all, neigh_idx, edge_wts, dev,
            view_mask=view_mask, z_arr=z_arr, lbl_nei=None,
        )
        result = model(x_a, g_a, x_n, ew_a, view_mask=vm, z_anc=z_b)
        out, pi = result[0], result[1]
        all_out.append(out.cpu())
        all_pi.append(pi.cpu() if pi is not None else None)

    out_all = torch.cat(all_out, dim=0)
    pi_all = (torch.cat([p for p in all_pi if p is not None], dim=0).numpy()
              if all_pi and all_pi[0] is not None else None)
    if task == "classification":
        proba = torch.softmax(out_all, -1).numpy(); preds = proba.argmax(-1)
    else:
        preds = out_all.squeeze(-1).numpy(); proba = None
    return preds, proba, pi_all
