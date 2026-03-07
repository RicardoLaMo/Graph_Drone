"""
train.py — Shared training loop for Model A and Model B.

Key invariant (contract §Fairness constraints):
  - Both A and B use the SAME per-view encoders (frozen representations after encoding)
  - Only the combiner/router differs
  - Same splits, same views, same observer vector
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from .models import ViewEncoder, A_posthoc_combiner, B_intended_router

SEED = 42; torch.manual_seed(SEED)


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# ─── Step 1: Train shared per-view encoders ───────────────────────────────────

def train_view_encoders(
    graphs: Dict,          # {tag: PyG Data (full N nodes)}
    tr_i: np.ndarray,
    va_i: np.ndarray,
    rep_dim: int,
    task: str,             # "regression" | "classification"
    n_classes: int = 1,
    epochs: int = 150,
    patience: int = 20,
    lr: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """
    Trains one ViewEncoder per view, returns frozen reps [N, rep_dim] dict.
    Encoders are trained independently inside this function, then detached.
    """
    dev = get_device()
    out_dim = n_classes if task == "classification" else 1
    all_reps = {}

    for tag, data in graphs.items():
        data = data.to(dev)
        in_dim = data.x.shape[1]
        enc = ViewEncoder(in_dim, rep_dim).to(dev)
        head = nn.Linear(rep_dim, out_dim).to(dev)
        params = list(enc.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        tr_t, va_t = torch.tensor(tr_i), torch.tensor(va_i)
        best, bst, wait = 1e9, None, 0
        for ep in range(epochs):
            enc.train(); head.train(); opt.zero_grad()
            reps = enc(data.x, data.edge_index)
            out = head(reps)[tr_t]
            yt = data.y[tr_t].long() if task == "classification" else data.y[tr_t].float()
            crit(out, yt).backward(); opt.step()
            enc.eval(); head.eval()
            with torch.no_grad():
                vr = enc(data.x, data.edge_index)
                vout = head(vr)[va_t]
                yo = data.y[va_t].long() if task == "classification" else data.y[va_t].float()
                vl = crit(vout, yo).item()
            if vl < best: best=vl; bst={"enc":enc.state_dict(),"head":head.state_dict()}; wait=0
            else: wait+=1
            if wait >= patience: break
        enc.load_state_dict(bst["enc"]); enc.eval()
        with torch.no_grad():
            reps_all = enc(data.x, data.edge_index).cpu()  # [N, rep_dim] frozen
        all_reps[tag] = reps_all
        print(f"  [ViewEncoder {tag}] Done. rep shape={reps_all.shape}")

    return all_reps  # {tag: Tensor[N, rep_dim]}


# ─── Step 2: Stack reps into [N, V, D] ───────────────────────────────────────

def stack_reps(all_reps: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(all_reps.values()), dim=1)  # [N, V, D]


# ─── Step 3: Train combiner/router ────────────────────────────────────────────

def _train_combiner(
    name: str,
    model: nn.Module,
    forward_fn,       # (model, idx) -> prediction
    y_all: torch.Tensor,
    tr_i, va_i,
    task: str,
    epochs: int = 600,
    patience: int = 40,
    lr: float = 5e-3,
) -> Dict:
    dev = get_device()
    model = model.to(dev)
    y_all = y_all.to(dev)
    crit = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_t, va_t = torch.tensor(tr_i), torch.tensor(va_i)
    best, bst, wait = 1e9, None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = forward_fn(model, tr_t)
        yt = y_all[tr_t].long() if task == "classification" else y_all[tr_t].float()
        crit(out if out.dim()==1 or out.shape[-1]>1 else out.squeeze(-1), yt).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vout = forward_fn(model, va_t)
            yv = y_all[va_t].long() if task == "classification" else y_all[va_t].float()
            vout2 = vout if vout.dim()==1 or vout.shape[-1]>1 else vout.squeeze(-1)
            vl = crit(vout2, yv).item()
        if vl < best: best=vl; bst={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else: wait+=1
        if wait >= patience: break
    model.load_state_dict(bst)
    print(f"  [{name}] Trained {ep+1} epochs in {time.time()-t0:.1f}s")
    return model


def train_model_a(
    reps_t: torch.Tensor, obs_t: torch.Tensor, y_all: np.ndarray,
    tr_i, va_i, n_views, rep_dim, obs_dim, out_dim, task
):
    dev = get_device()
    y_t = torch.tensor(y_all, dtype=torch.long if task=="classification" else torch.float32)
    r, o = reps_t.to(dev), obs_t.to(dev)
    m = A_posthoc_combiner(n_views, rep_dim, obs_dim, out_dim)
    def fwd(model, idx):
        out = model(r[idx], o[idx])
        return out.squeeze(-1) if task=="regression" else out
    m = _train_combiner("A_posthoc", m, fwd, y_t, tr_i, va_i, task)
    return m


def train_model_b(
    reps_t: torch.Tensor, obs_t: torch.Tensor, y_all: np.ndarray,
    tr_i, va_i, n_views, rep_dim, obs_dim, out_dim, task
):
    dev = get_device()
    y_t = torch.tensor(y_all, dtype=torch.long if task=="classification" else torch.float32)
    r, o = reps_t.to(dev), obs_t.to(dev)
    m = B_intended_router(obs_dim, n_views, rep_dim, out_dim)
    def fwd(model, idx):
        out, pi, beta, iso_rep, inter_rep, final_rep = model(r[idx], o[idx])
        return out.squeeze(-1) if task=="regression" else out
    m = _train_combiner("B_router", m, fwd, y_t, tr_i, va_i, task)
    return m
