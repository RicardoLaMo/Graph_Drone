"""
train_v4.py — Split-track training functions for MQ-GoRA v4.

Shared fixes relative to the initial scaffold:
  - robust package imports
  - best-checkpoint restore (matching v3 semantics)
  - validation passes `z_anc` when teacher-query models need it
  - prediction returns explicit beta plus debug tensors for diagnostics
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.gora_tabular.src.train import (
    build_joint_neighbourhood,
    compute_label_ctx_per_view,
    fetch_batch_v3,
    get_device,
)

__all__ = [
    "compute_y_norm_stats",
    "normalise_lbl_nei",
    "train_gora_v4_ca",
    "train_gora_v4_mn",
    "predict_gora_v4",
]


def compute_y_norm_stats(y_all: np.ndarray, tr_i: np.ndarray):
    mu = float(y_all[tr_i].mean())
    std = float(y_all[tr_i].std()) + 1e-8
    return mu, std


def normalise_lbl_nei(lbl_nei: np.ndarray, y_mu: float, y_std: float) -> np.ndarray:
    _ = y_mu
    return (lbl_nei / y_std).astype(np.float32)


def _extract_v4_result(result):
    if len(result) == 6:
        pred, pi, beta, tau, aux_losses, debug = result
    elif len(result) == 5:
        pred, pi, beta, tau, aux_losses = result
        debug = {}
    elif len(result) == 4:
        pred, pi, tau, aux_losses = result
        beta, debug = None, {}
    else:
        pred, pi, tau = result
        beta, aux_losses, debug = None, {}, {}
    return pred, pi, beta, tau, aux_losses, debug


def _attach_training_metadata(model, best_ep: int, stop_ep: int, epochs: int):
    collapsed = stop_ep <= max(10, int(0.35 * epochs))
    model.training_metadata = {
        "best_ep": int(best_ep),
        "stop_ep": int(stop_ep),
        "collapsed": bool(collapsed),
    }


def train_gora_v4_ca(
    name: str,
    model,
    X_all: np.ndarray,
    g_all: np.ndarray,
    y_all: np.ndarray,
    neigh_idx: np.ndarray,
    edge_wts: np.ndarray,
    tr_i: np.ndarray,
    va_i: np.ndarray,
    task: str = "regression",
    view_mask: np.ndarray = None,
    agree_score: np.ndarray = None,
    z_arr: np.ndarray = None,
    lbl_nei: np.ndarray = None,
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 512,
    patience: int = 40,
    use_cosine: bool = False,
    T_max: int = 100,
    weight_decay: float = 1e-4,
):
    dev = get_device()
    model = model.to(dev)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=lr * 0.01)
        if use_cosine
        else None
    )

    best_val = float("inf")
    best_ep = 0
    stop_ep = 0
    no_improve = 0
    best_state = None
    t0 = time.time()

    for ep in range(epochs):
        stop_ep = ep
        model.train()
        perm = np.random.permutation(tr_i)
        train_loss = 0.0
        n_batches = 0

        for start in range(0, len(perm), batch_size):
            b_idx = perm[start : start + batch_size]
            x_a, g_a, x_n, ew_a, y_b, vm_b, ag_b, _, _ = fetch_batch_v3(
                b_idx,
                X_all,
                g_all,
                y_all,
                neigh_idx,
                edge_wts,
                dev,
                view_mask=view_mask,
                agree_score=agree_score,
            )
            z_b = torch.from_numpy(z_arr[b_idx]).float().to(dev) if z_arr is not None else None
            lb_b = torch.from_numpy(lbl_nei[b_idx]).float().to(dev) if lbl_nei is not None else None
            pred, _, _, _, aux_losses, _ = _extract_v4_result(
                model(
                    x_a,
                    g_a,
                    x_n,
                    ew_a,
                    view_mask=vm_b,
                    z_anc=z_b,
                    lbl_nei=lb_b,
                    agree_score=ag_b,
                )
            )
            loss = F.mse_loss(pred.squeeze(-1), y_b)
            for aux_val in aux_losses.values():
                loss = loss + aux_val
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.item())
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(va_i), batch_size):
                b_idx = va_i[start : start + batch_size]
                x_a, g_a, x_n, ew_a, y_b, vm_b, _, _, _ = fetch_batch_v3(
                    b_idx,
                    X_all,
                    g_all,
                    y_all,
                    neigh_idx,
                    edge_wts,
                    dev,
                    view_mask=view_mask,
                    agree_score=agree_score,
                )
                z_b = torch.from_numpy(z_arr[b_idx]).float().to(dev) if z_arr is not None else None
                pred, _, _, _, _, _ = _extract_v4_result(
                    model(x_a, g_a, x_n, ew_a, view_mask=vm_b, z_anc=z_b)
                )
                val_losses.append(F.mse_loss(pred.squeeze(-1), y_b).item())

        val_loss = float(np.mean(val_losses))
        if ep % 10 == 0:
            print(
                f"  [{name}] ep={ep:4d} tr={train_loss / max(n_batches, 1):.4f} "
                f"val={val_loss:.4f}"
            )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_ep = ep
            no_improve = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _attach_training_metadata(model, best_ep=best_ep, stop_ep=stop_ep, epochs=epochs)
    print(f"  [{name}] Done in {time.time() - t0:.1f}s | best_val={best_val:.4f} @ ep={best_ep}")
    return model


def train_gora_v4_mn(
    name: str,
    model,
    X_all: np.ndarray,
    g_all: np.ndarray,
    y_all: np.ndarray,
    neigh_idx: np.ndarray,
    edge_wts: np.ndarray,
    tr_i: np.ndarray,
    va_i: np.ndarray,
    task: str = "classification",
    view_mask: np.ndarray = None,
    agree_score: np.ndarray = None,
    z_arr: np.ndarray = None,
    lbl_nei: np.ndarray = None,
    n_classes: int = 10,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 512,
    patience: int = 20,
    weight_decay: float = 1e-4,
    lam_diversity: float = 0.0,
):
    _ = n_classes
    dev = get_device()
    model = model.to(dev)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val = float("inf")
    best_ep = 0
    stop_ep = 0
    no_improve = 0
    best_state = None
    t0 = time.time()

    for ep in range(epochs):
        stop_ep = ep
        model.train()
        perm = np.random.permutation(tr_i)
        train_loss = 0.0
        n_batches = 0

        for start in range(0, len(perm), batch_size):
            b_idx = perm[start : start + batch_size]
            x_a, g_a, x_n, ew_a, y_b, vm_b, ag_b, _, _ = fetch_batch_v3(
                b_idx,
                X_all,
                g_all,
                y_all,
                neigh_idx,
                edge_wts,
                dev,
                view_mask=view_mask,
                agree_score=agree_score,
            )
            z_b = torch.from_numpy(z_arr[b_idx]).float().to(dev) if z_arr is not None else None
            lb_b = torch.from_numpy(lbl_nei[b_idx]).float().to(dev) if lbl_nei is not None else None
            pred, pi, _, _, aux_losses, _ = _extract_v4_result(
                model(
                    x_a,
                    g_a,
                    x_n,
                    ew_a,
                    view_mask=vm_b,
                    z_anc=z_b,
                    lbl_nei=lb_b,
                    agree_score=ag_b,
                )
            )
            loss = F.cross_entropy(pred, y_b.long())
            if lam_diversity > 0 and pi is not None:
                pi_mean = pi.mean(dim=0, keepdim=True)
                kl = (pi * (torch.log(pi + 1e-8) - torch.log(pi_mean + 1e-8))).sum(-1)
                loss = loss - lam_diversity * kl.mean()
            for aux_val in aux_losses.values():
                loss = loss + aux_val
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.item())
            n_batches += 1

        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(va_i), batch_size):
                b_idx = va_i[start : start + batch_size]
                x_a, g_a, x_n, ew_a, y_b, vm_b, _, _, _ = fetch_batch_v3(
                    b_idx,
                    X_all,
                    g_all,
                    y_all,
                    neigh_idx,
                    edge_wts,
                    dev,
                    view_mask=view_mask,
                )
                z_b = torch.from_numpy(z_arr[b_idx]).float().to(dev) if z_arr is not None else None
                pred, _, _, _, _, _ = _extract_v4_result(
                    model(x_a, g_a, x_n, ew_a, view_mask=vm_b, z_anc=z_b)
                )
                val_losses.append(F.cross_entropy(pred, y_b.long()).item())

        val_loss = float(np.mean(val_losses))
        if ep % 10 == 0:
            print(
                f"  [{name}] ep={ep:4d} tr={train_loss / max(n_batches, 1):.4f} "
                f"val={val_loss:.4f}"
            )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_ep = ep
            no_improve = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _attach_training_metadata(model, best_ep=best_ep, stop_ep=stop_ep, epochs=epochs)
    print(f"  [{name}] Done in {time.time() - t0:.1f}s | best_val={best_val:.4f} @ ep={best_ep}")
    return model


@torch.no_grad()
def predict_gora_v4(
    model,
    X_all: np.ndarray,
    g_all: np.ndarray,
    y_all: np.ndarray,
    neigh_idx: np.ndarray,
    edge_wts: np.ndarray,
    te_i: np.ndarray,
    task: str,
    view_mask: np.ndarray = None,
    z_arr: np.ndarray = None,
    batch_size: int = 512,
):
    dev = get_device()
    model = model.to(dev).eval()
    all_out = []
    all_pi = []
    all_beta = []
    all_view_ctxs = []
    all_alpha = []
    last_tau = None

    for start in range(0, len(te_i), batch_size):
        b_idx = te_i[start : start + batch_size]
        x_a, g_a, x_n, ew_a, _, vm_b, _, _, _ = fetch_batch_v3(
            b_idx,
            X_all,
            g_all,
            y_all,
            neigh_idx,
            edge_wts,
            dev,
            view_mask=view_mask,
        )
        z_b = torch.from_numpy(z_arr[b_idx]).float().to(dev) if z_arr is not None else None
        pred, pi, beta, tau, _, debug = _extract_v4_result(
            model(x_a, g_a, x_n, ew_a, view_mask=vm_b, z_anc=z_b)
        )
        all_out.append(pred.cpu())
        if pi is not None:
            all_pi.append(pi.cpu())
        if beta is not None:
            all_beta.append(beta.cpu())
        if "view_ctxs" in debug:
            all_view_ctxs.append(debug["view_ctxs"].cpu())
        if "alpha" in debug:
            all_alpha.append(debug["alpha"].cpu())
        last_tau = tau

    out_all = torch.cat(all_out, dim=0)
    routing = {
        "pi_all": torch.cat(all_pi, dim=0).numpy() if all_pi else None,
        "beta_all": torch.cat(all_beta, dim=0).numpy() if all_beta else None,
        "tau_np": last_tau.cpu().numpy() if last_tau is not None else None,
        "view_ctxs": torch.cat(all_view_ctxs, dim=0).numpy() if all_view_ctxs else None,
        "alpha": torch.cat(all_alpha, dim=0).numpy() if all_alpha else None,
    }

    if task == "classification":
        proba = torch.softmax(out_all, dim=-1).numpy()
        preds = proba.argmax(-1)
    else:
        preds = out_all.squeeze(-1).numpy()
        proba = None
    return preds, proba, routing
