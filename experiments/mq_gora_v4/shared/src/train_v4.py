"""
train_v4.py — Split-track training functions for MQ-GoRA v4.

CA track (train_gora_v4_ca):
  - increased patience (default 40 vs v3's 20)           CA_FIX_5
  - optional cosine annealing scheduler                   CA_FIX_5

MNIST track (train_gora_v4_mn):
  - same as v3 train_gora_v3 (classification-safe)
  - wrapped here for a clean import path

Shared helpers:
  compute_y_norm_stats — compute mu/std from training y for CA label normalisation
  predict_gora_v4      — unified predict (wraps v3 predict_gora_v3)
"""
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3_SRC = os.path.normpath(os.path.join(_HERE, '..', '..', '..', 'gora_tabular', 'src'))
if _V3_SRC not in sys.path:
    sys.path.insert(0, _V3_SRC)

import time
import numpy as np
import torch
import torch.nn.functional as F

from train import (
    get_device, fetch_batch_v3,
    build_joint_neighbourhood, compute_label_ctx_per_view,
)

__all__ = [
    'compute_y_norm_stats',
    'train_gora_v4_ca', 'train_gora_v4_mn',
    'predict_gora_v4',
]


# ─── Y-normalisation helpers ──────────────────────────────────────────────────

def compute_y_norm_stats(y_all: np.ndarray, tr_i: np.ndarray):
    """Return (mu, std) computed from training indices."""
    mu  = float(y_all[tr_i].mean())
    std = float(y_all[tr_i].std()) + 1e-8
    return mu, std


def normalise_lbl_nei(lbl_nei: np.ndarray, y_mu: float, y_std: float) -> np.ndarray:
    """
    Scale lbl_nei values (raw regression labels masked per view) to zero-mean,
    unit-variance.  Zero-slots (padding / view-mask=0) stay zero because
    0 - mu normalised is -mu/std which is nonzero; instead we only scale the
    magnitude without shifting (divide by std), assuming the
    compute_label_ctx_per_view already zeroed invalid slots.

    Rationale: lbl_nei[i,p,m] = y[j] * view_mask — it can be 0 for masked
    positions which represent "no label" not "label=0". A mean-subtraction
    would corrupt those zeros.  Std-only scaling preserves the zero-sentinel.
    """
    return (lbl_nei / y_std).astype(np.float32)


# ─── California track ─────────────────────────────────────────────────────────

def train_gora_v4_ca(
    name: str,
    model,
    X_all:      np.ndarray,
    g_all:      np.ndarray,
    y_all:      np.ndarray,
    neigh_idx:  np.ndarray,
    edge_wts:   np.ndarray,
    tr_i:       np.ndarray,
    va_i:       np.ndarray,
    task:       str   = "regression",
    view_mask:  np.ndarray = None,
    agree_score: np.ndarray = None,
    z_arr:      np.ndarray = None,
    lbl_nei:    np.ndarray = None,    # already y-normalised by caller if regression
    epochs:     int   = 150,
    lr:         float = 1e-3,
    batch_size: int   = 512,
    patience:   int   = 40,           # CA_FIX_5: increased from 20
    use_cosine: bool  = False,        # CA_FIX_5: cosine annealing
    T_max:      int   = 100,
    weight_decay: float = 1e-4,
) -> object:
    """
    Training loop for California v4 variants.

    Key differences vs v3 train_gora_v3:
      - patience=40 (was 20) to avoid pathological early stopping
      - optional CosineAnnealingLR for healthier LR schedule
      - lbl_nei is passed in pre-normalised (y / y_std) by the runner
      - all other semantics identical
    """
    dev = get_device()
    model = model.to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T_max, eta_min=lr * 0.01)
    else:
        scheduler = None

    best_val, best_ep, no_improve, t0 = float('inf'), 0, 0, time.time()

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr_i)
        for start in range(0, len(perm), batch_size):
            b_idx = perm[start:start + batch_size]
            x_a, g_a, x_n, ew_a, y_b, vm_b, ag_b = fetch_batch_v3(
                b_idx, X_all, g_all, y_all, neigh_idx, edge_wts, dev,
                view_mask=view_mask, agree_score=agree_score,
                z_arr=None, lbl_nei=None,
            )
            z_b  = torch.from_numpy(z_arr[b_idx]).float().to(dev)   if z_arr   is not None else None
            lb_b = torch.from_numpy(lbl_nei[b_idx]).float().to(dev) if lbl_nei is not None else None

            result = model(x_a, g_a, x_n, ew_a,
                           view_mask=vm_b, z_anc=z_b,
                           lbl_nei=lb_b, agree_score=ag_b)
            pred, pi, tau, aux = result

            loss = F.mse_loss(pred.squeeze(-1), y_b)
            for v in aux.values():
                loss = loss + v

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if scheduler is not None:
            scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(va_i), batch_size):
                b_idx = va_i[start:start + batch_size]
                x_a, g_a, x_n, ew_a, y_b, vm_b, ag_b = fetch_batch_v3(
                    b_idx, X_all, g_all, y_all, neigh_idx, edge_wts, dev,
                    view_mask=view_mask, agree_score=agree_score,
                )
                pred, _, _, _ = model(x_a, g_a, x_n, ew_a, view_mask=vm_b)
                val_losses.append(F.mse_loss(pred.squeeze(-1), y_b).item())

        val_loss = float(np.mean(val_losses))
        if ep % 10 == 0:
            tr_l = loss.item()
            print(f"  [{name}] ep={ep:4d} tr={tr_l:.4f} val={val_loss:.4f}")

        if val_loss < best_val - 1e-5:
            best_val, best_ep, no_improve = val_loss, ep, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    elapsed = time.time() - t0
    print(f"  [{name}] Done in {elapsed:.1f}s | best_val={best_val:.4f} @ ep={best_ep}")
    return model


# ─── MNIST track ─────────────────────────────────────────────────────────────

def train_gora_v4_mn(
    name: str,
    model,
    X_all:      np.ndarray,
    g_all:      np.ndarray,
    y_all:      np.ndarray,
    neigh_idx:  np.ndarray,
    edge_wts:   np.ndarray,
    tr_i:       np.ndarray,
    va_i:       np.ndarray,
    task:       str   = "classification",
    view_mask:  np.ndarray = None,
    agree_score: np.ndarray = None,
    z_arr:      np.ndarray = None,
    lbl_nei:    np.ndarray = None,
    n_classes:  int   = 10,
    epochs:     int   = 100,
    lr:         float = 1e-3,
    batch_size: int   = 512,
    patience:   int   = 20,
    weight_decay: float = 1e-4,
    lam_diversity: float = 0.0,   # MN_v4c/d: inter-head diversity regulariser
) -> object:
    """
    Training loop for MNIST v4 variants.

    Identical to v3 train_gora_v3 except:
      - optional lam_diversity: adds −lam * mean KL(head_pi || mean_pi)
        to encourage head specialisation without hurting accuracy.
    """
    dev = get_device()
    model = model.to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    best_val, best_ep, no_improve, t0 = float('inf'), 0, 0, time.time()

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr_i)
        for start in range(0, len(perm), batch_size):
            b_idx = perm[start:start + batch_size]
            x_a, g_a, x_n, ew_a, y_b, vm_b, ag_b = fetch_batch_v3(
                b_idx, X_all, g_all, y_all, neigh_idx, edge_wts, dev,
                view_mask=view_mask, agree_score=agree_score,
            )
            z_b  = torch.from_numpy(z_arr[b_idx]).float().to(dev)   if z_arr   is not None else None
            lb_b = torch.from_numpy(lbl_nei[b_idx]).float().to(dev) if lbl_nei is not None else None

            result = model(x_a, g_a, x_n, ew_a,
                           view_mask=vm_b, z_anc=z_b,
                           lbl_nei=lb_b, agree_score=ag_b)
            pred, pi, tau, aux = result

            y_long = y_b.long()
            loss = F.cross_entropy(pred, y_long)

            # Diversity regulariser: −KL(pi_h || pi_mean) averaged over heads
            if lam_diversity > 0 and pi is not None:
                pi_mean = pi.mean(dim=0, keepdim=True)          # [1, H, M]
                kl = (pi * (torch.log(pi + 1e-8) - torch.log(pi_mean + 1e-8))).sum(-1)
                loss = loss - lam_diversity * kl.mean()

            for v in aux.values():
                loss = loss + v

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(va_i), batch_size):
                b_idx = va_i[start:start + batch_size]
                x_a, g_a, x_n, ew_a, y_b, vm_b, _ = fetch_batch_v3(
                    b_idx, X_all, g_all, y_all, neigh_idx, edge_wts, dev,
                    view_mask=view_mask,
                )
                pred, _, _, _ = model(x_a, g_a, x_n, ew_a, view_mask=vm_b)
                val_losses.append(F.cross_entropy(pred, y_b.long()).item())

        val_loss = float(np.mean(val_losses))
        if ep % 10 == 0:
            print(f"  [{name}] ep={ep:4d} tr={loss.item():.4f} val={val_loss:.4f}")

        if val_loss < best_val - 1e-5:
            best_val, best_ep, no_improve = val_loss, ep, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    elapsed = time.time() - t0
    print(f"  [{name}] Done in {elapsed:.1f}s | best_val={best_val:.4f} @ ep={best_ep}")
    return model, best_ep


# ─── Unified predict ─────────────────────────────────────────────────────────

@torch.no_grad()
def predict_gora_v4(
    model,
    X_all:      np.ndarray,
    g_all:      np.ndarray,
    y_all:      np.ndarray,
    neigh_idx:  np.ndarray,
    edge_wts:   np.ndarray,
    te_i:       np.ndarray,
    task:       str,
    view_mask:  np.ndarray = None,
    z_arr:      np.ndarray = None,
    batch_size: int = 512,
):
    """
    Returns (preds, proba, pi_all, tau_np).
    lbl_nei is always None at inference (safe: W_lbl_v zero-init → lbl_delta=0).
    """
    dev = get_device()
    model = model.to(dev).eval()
    all_out, all_pi, last_tau = [], [], None

    for start in range(0, len(te_i), batch_size):
        b_idx = te_i[start:start + batch_size]
        x_a, g_a, x_n, ew_a, _, vm_b, _ = fetch_batch_v3(
            b_idx, X_all, g_all, y_all, neigh_idx, edge_wts, dev,
            view_mask=view_mask,
        )
        z_b = torch.from_numpy(z_arr[b_idx]).float().to(dev) if z_arr is not None else None
        out, pi, tau, _ = model(x_a, g_a, x_n, ew_a, view_mask=vm_b, z_anc=z_b)
        all_out.append(out.cpu())
        all_pi.append(pi.cpu() if pi is not None else None)
        last_tau = tau

    out_all = torch.cat(all_out, 0)
    pi_all = (torch.cat([p for p in all_pi if p is not None], 0).numpy()
              if all_pi and all_pi[0] is not None else None)
    tau_np = last_tau.cpu().numpy() if last_tau is not None else None

    if task == "classification":
        proba = torch.softmax(out_all, -1).numpy()
        preds = proba.argmax(-1)
    else:
        preds = out_all.squeeze(-1).numpy()
        proba = None

    return preds, proba, pi_all, tau_np
