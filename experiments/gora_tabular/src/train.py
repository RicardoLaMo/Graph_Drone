"""
train.py — GoRA-Tabular training loop.

Key design:
  - Full-batch forward pass over ALL N rows (for full graph context)
  - Loss computed only on tr_i indices
  - adj matrices [N, N] are precomputed and passed each forward
  - For large N, adjacency [N,N] may be memory-intensive; use float16 or
    edge-index variant for scale-up (left as future work)
"""
import time
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict


SEED = 42; torch.manual_seed(SEED)


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def precompute_adj_dense(views: Dict, N: int, device) -> List[torch.Tensor]:
    """
    Build dense [N, N] adjacency per view from (edge_index, edge_weight) tuples.
    Row-normalised; each entry A[i,j] = Gaussian kNN weight (or 0).
    """
    adjs = []
    for name, (ei, ew) in views.items():
        A = torch.zeros(N, N, dtype=torch.float32)
        A[ei[0], ei[1]] = ew
        adjs.append(A.to(device))
        print(f"  [adj {name}] sparsity={(A == 0).float().mean():.4f}")
    return adjs


def train_gora(
    model: nn.Module,
    X_all: np.ndarray,        # [N, d]
    g_all: np.ndarray,        # [N, obs_dim]
    y_all: np.ndarray,        # [N]
    adjs_cpu: List[torch.Tensor],
    tr_i: np.ndarray,
    va_i: np.ndarray,
    task: str,
    n_classes: int = 1,
    epochs: int = 500,
    patience: int = 40,
    lr: float = 3e-4,
    name: str = "model",
) -> nn.Module:
    dev = get_device()
    model = model.to(dev)
    adjs = [A.to(dev) for A in adjs_cpu]

    X_t = torch.tensor(X_all, dtype=torch.float32).to(dev)
    g_t = torch.tensor(g_all, dtype=torch.float32).to(dev)
    y_t = torch.tensor(y_all,
                        dtype=torch.long if task == "classification" else torch.float32).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    crit = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    tr_t = torch.tensor(tr_i, dtype=torch.long)
    va_t = torch.tensor(va_i, dtype=torch.long)
    best, bst, wait = 1e9, None, 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        # Full-batch forward (all N rows so attention has full graph context)
        out, pi, tau, _ = model(X_t, g_t, adjs)     # [N, out_dim]
        # Loss on train indices only
        pred_tr = out[tr_t]
        yt_tr = y_t[tr_t]
        if task == "regression":
            loss = crit(pred_tr.squeeze(-1), yt_tr)
        else:
            loss = crit(pred_tr, yt_tr)
        loss.backward(); opt.step(); sched.step()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                vout, *_ = model(X_t, g_t, adjs)
                pred_va = vout[va_t]
                yv = y_t[va_t]
                if task == "regression":
                    vl = crit(pred_va.squeeze(-1), yv).item()
                else:
                    vl = crit(pred_va, yv).item()
            if vl < best:
                best = vl
                bst = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 10
            if wait >= patience: break
            if ep % 100 == 0:
                print(f"  [{name}] ep={ep:4d} train_loss={loss.item():.4f} val_loss={vl:.4f}")

    model.load_state_dict(bst)
    print(f"  [{name}] Done in {time.time()-t0:.1f}s")
    return model


@torch.no_grad()
def predict_gora(model, X_all, g_all, adjs_cpu, te_i, task, device):
    dev = device
    model = model.to(dev).eval()
    X_t = torch.tensor(X_all, dtype=torch.float32).to(dev)
    g_t = torch.tensor(g_all, dtype=torch.float32).to(dev)
    adjs = [A.to(dev) for A in adjs_cpu]
    out, pi, tau, _ = model(X_t, g_t, adjs, return_attn=False)
    out = out.cpu(); pi_np = pi.cpu().numpy() if pi is not None else None
    tau_np = tau.cpu().numpy() if tau is not None else None
    te_out = out[te_i]
    if task == "classification":
        proba = torch.softmax(te_out, -1).numpy(); preds = proba.argmax(-1)
    else:
        preds = te_out.squeeze(-1).numpy(); proba = None
    return preds, proba, pi_np, tau_np


@torch.no_grad()
def get_head_attn_maps(model, X_all, g_all, adjs_cpu, device):
    """Extract attention maps per head from first layer (for head specialisation analysis)."""
    dev = device
    model = model.to(dev).eval()
    X_t = torch.tensor(X_all, dtype=torch.float32).to(dev)
    g_t = torch.tensor(g_all, dtype=torch.float32).to(dev)
    adjs = [A.to(dev) for A in adjs_cpu]
    _, _, _, attn_maps = model(X_t, g_t, adjs, return_attn=True)
    return {h: A.cpu().numpy() for h, A in attn_maps.items()}
