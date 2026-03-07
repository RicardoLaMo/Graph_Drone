"""
train.py — GoRA-Tabular sparse mini-batch training loop.

KEY DESIGN: Mini-batch subgraph sampling.
  1. Precompute for all N rows: neigh_idx [N, K], edge_wts [N, K, M]
     This is done once on CPU and is O(N*K) — very cheap.
  2. At each training step, sample B anchor indices.
  3. Fetch x_nei = X[neigh_idx[batch]]  → [B, K, d]
  4. Fetch ew  = edge_wts[batch]         → [B, K, M]
  5. Forward: anchor attends to K neighbours with geometry-shaped logits.

Memory: O(B × K × d_model) per step — tractable at any N.
Routing structure: still causal (anchor geometry → which view's edge weights it trusts)
                  across the SAME neighbourhood structure.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple


SEED = 42; torch.manual_seed(SEED)


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# ─── Precompute neighbourhood + per-view edge weights ─────────────────────────

def build_neighbourhood(
    X_views: Dict[str, np.ndarray],   # {view_name: X_topo [N, d_topo]}
    k: int = 15,
    primary_key: str = None,          # view used to anchor the neighbourhood indices
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      neigh_idx: [N, K]   — universal neighbour indices (from primary view or union)
      edge_wts:  [N, K, M]— Gaussian edge weights per view for those K neighbours
    """
    view_names = list(X_views.keys())
    M = len(view_names)
    primary = primary_key or view_names[0]
    X_prim = X_views[primary]
    N = X_prim.shape[0]

    print(f"  [neighbourhood] Fitting kNN (k={k}) on primary view '{primary}'...")
    nb = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X_prim)
    _, neigh_idx_raw = nb.kneighbors(X_prim)
    neigh_idx = neigh_idx_raw[:, 1:].astype(np.int64)   # [N, K] drop self

    edge_wts = np.zeros((N, k, M), dtype=np.float32)
    for mi, vname in enumerate(view_names):
        Xv = X_views[vname]
        # Gaussian weights for each (i, neigh_idx[i,j]) pair
        diff_sq = ((Xv[np.arange(N)[:, None], :] -
                    Xv[neigh_idx]) ** 2).sum(-1)   # [N, K]
        sigma = np.median(np.sqrt(diff_sq)) + 1e-8
        edge_wts[:, :, mi] = np.exp(-diff_sq / sigma ** 2)
        # row-normalise per view
        row_sum = edge_wts[:, :, mi].sum(-1, keepdims=True) + 1e-8
        edge_wts[:, :, mi] /= row_sum
        print(f"  [neighbourhood] View '{vname}': mean_ew={edge_wts[:,:,mi].mean():.4f}")

    return neigh_idx, edge_wts


# ─── Mini-batch fetch ──────────────────────────────────────────────────────────

def fetch_batch(
    batch_idx: np.ndarray,
    X_all: np.ndarray,
    g_all: np.ndarray,
    y_all: np.ndarray,
    neigh_idx: np.ndarray,
    edge_wts: np.ndarray,
    device,
):
    """Fetch anchor + neighbour tensors for a mini-batch."""
    b_ni = neigh_idx[batch_idx]            # [B, K]
    x_anc = torch.tensor(X_all[batch_idx], dtype=torch.float32).to(device)
    g_anc = torch.tensor(g_all[batch_idx], dtype=torch.float32).to(device)
    x_nei = torch.tensor(X_all[b_ni], dtype=torch.float32).to(device)   # [B, K, d]
    ew    = torch.tensor(edge_wts[batch_idx], dtype=torch.float32).to(device)  # [B, K, M]
    y_b   = torch.tensor(y_all[batch_idx]).to(device)
    return x_anc, g_anc, x_nei, ew, y_b


# ─── Training ─────────────────────────────────────────────────────────────────

def train_gora(
    model: nn.Module,
    X_all: np.ndarray,
    g_all: np.ndarray,
    y_all: np.ndarray,
    neigh_idx: np.ndarray,   # [N, K]
    edge_wts: np.ndarray,    # [N, K, M]
    tr_i: np.ndarray,
    va_i: np.ndarray,
    task: str,
    n_classes: int = 1,
    epochs: int = 100,
    patience: int = 15,
    lr: float = 3e-4,
    batch_size: int = 512,
    name: str = "model",
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
            x_a, g_a, x_n, ew_a, y_b = fetch_batch(b_idx, X_all, g_all, y_all,
                                                      neigh_idx, edge_wts, dev)
            opt.zero_grad()
            out, pi, tau = model(x_a, g_a, x_n, ew_a)
            if task == "regression":
                loss = crit(out.squeeze(-1), y_b.float())
            else:
                loss = crit(out, y_b.long())
            loss.backward(); opt.step()
            ep_loss += loss.item(); n_batches += 1
        sched.step()

        # Validation (full pass in batches)
        model.eval()
        val_loss = 0.0; val_batches = 0
        with torch.no_grad():
            for start in range(0, len(va_i), batch_size):
                b_idx = va_i[start:start + batch_size]
                x_a, g_a, x_n, ew_a, y_b = fetch_batch(b_idx, X_all, g_all, y_all,
                                                          neigh_idx, edge_wts, dev)
                out, *_ = model(x_a, g_a, x_n, ew_a)
                if task == "regression":
                    vl = crit(out.squeeze(-1), y_b.float()).item()
                else:
                    vl = crit(out, y_b.long()).item()
                val_loss += vl; val_batches += 1
        val_loss /= max(val_batches, 1)

        if val_loss < best:
            best = val_loss
            bst = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience: break
        if ep % 10 == 0:
            print(f"  [{name}] ep={ep:4d} tr={ep_loss/n_batches:.4f} val={val_loss:.4f}")

    model.load_state_dict(bst)
    print(f"  [{name}] Done in {time.time()-t0:.1f}s | best_val={best:.4f}")
    return model


@torch.no_grad()
def predict_gora(model, X_all, g_all, y_all, neigh_idx, edge_wts, te_i, task,
                 batch_size=512):
    dev = get_device()
    model = model.to(dev).eval()
    all_out, all_pi = [], []
    for start in range(0, len(te_i), batch_size):
        b_idx = te_i[start:start + batch_size]
        x_a, g_a, x_n, ew_a, _ = fetch_batch(b_idx, X_all, g_all, y_all,
                                               neigh_idx, edge_wts, dev)
        out, pi, tau = model(x_a, g_a, x_n, ew_a)
        all_out.append(out.cpu()); all_pi.append(pi.cpu() if pi is not None else None)

    out_all = torch.cat(all_out, dim=0)
    pi_all = torch.cat([p for p in all_pi if p is not None], dim=0).numpy() \
             if all_pi[0] is not None else None

    if task == "classification":
        proba = torch.softmax(out_all, -1).numpy(); preds = proba.argmax(-1)
    else:
        preds = out_all.squeeze(-1).numpy(); proba = None

    tau_np = tau.cpu().numpy() if tau is not None else None
    return preds, proba, pi_all, tau_np
