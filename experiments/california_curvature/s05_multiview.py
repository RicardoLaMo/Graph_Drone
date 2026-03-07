"""
05_multiview.py — M6: Uniform ensemble, M7: Learned combiner (no curvature).
Requires predictions from 04_graphsage.py to be saved first.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

ARTIFACTS = Path("artifacts")
SEED = 42
torch.manual_seed(SEED)


def evaluate(name, y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  [{name}]  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def load_view_preds(split="all"):
    """Load all-row predictions from M3, M4, M5."""
    p_full = np.load(ARTIFACTS / f"preds_M3_FULL_{split}.npy")
    p_geo = np.load(ARTIFACTS / f"preds_M4_GEO_{split}.npy")
    p_socio = np.load(ARTIFACTS / f"preds_M5_SOCIO_{split}.npy")
    return p_full, p_geo, p_socio


# ─── M6: Uniform ensemble ─────────────────────────────────────────────────────

def run_m6(idx_te, y_all):
    p_full, p_geo, p_socio = load_view_preds("all")
    preds_all = (p_full + p_geo + p_socio) / 3.0
    preds_te = preds_all[idx_te]
    np.save(ARTIFACTS / "preds_M6_Uniform_test.npy", preds_te)
    np.save(ARTIFACTS / "preds_M6_Uniform_all.npy", preds_all)
    m = evaluate("M6_Uniform", y_all[idx_te], preds_te)
    m.update({"train_time_s": 0.0, "inference_time_s": 0.0})
    return m, preds_all


# ─── M7: Learned combiner (no curvature) ─────────────────────────────────────

class ViewCombiner(nn.Module):
    """
    Takes [pred_full, pred_geo, pred_socio] and outputs per-row weighted sum.
    Learns global softmax weights (no row-level context yet — that's M9).
    """
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, preds_stack):
        w = torch.softmax(self.weights, dim=0)
        return (preds_stack * w).sum(dim=-1)


def run_m7(idx_tr, idx_val, idx_te, y_all):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    p_full, p_geo, p_socio = load_view_preds("all")
    preds_stack = np.stack([p_full, p_geo, p_socio], axis=-1)  # (N, 3)

    preds_t = torch.tensor(preds_stack, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_all, dtype=torch.float32).to(device)

    model = ViewCombiner().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()

    idx_tr_t = torch.tensor(idx_tr, dtype=torch.long)
    idx_val_t = torch.tensor(idx_val, dtype=torch.long)

    best_val = float("inf")
    best_state = None
    wait = 0
    start = time.time()

    for epoch in range(500):
        model.train()
        optim.zero_grad()
        out = model(preds_t[idx_tr_t])
        loss = loss_fn(out, y_t[idx_tr_t])
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(preds_t[idx_val_t])
            val_loss = loss_fn(val_pred, y_t[idx_val_t]).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 30:
                break

    train_time = time.time() - start
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        preds_all = model(preds_t).cpu().numpy()

    preds_te = preds_all[idx_te]
    np.save(ARTIFACTS / "preds_M7_LearnedCombiner_test.npy", preds_te)
    np.save(ARTIFACTS / "preds_M7_LearnedCombiner_all.npy", preds_all)

    learned_weights = torch.softmax(model.weights, dim=0).detach().cpu().numpy()
    print(f"  [M7] Learned view weights: FULL={learned_weights[0]:.3f}  GEO={learned_weights[1]:.3f}  SOCIO={learned_weights[2]:.3f}")

    metrics = evaluate("M7_LearnedCombiner", y_all[idx_te], preds_te)
    metrics["train_time_s"] = train_time
    metrics["inference_time_s"] = 0.0
    torch.save(model.state_dict(), ARTIFACTS / "model_M7_LearnedCombiner.pt")
    return metrics, preds_all


def run_multiview():
    print("[05_multiview] Loading indices and labels...")
    idx_tr = np.load(ARTIFACTS / "idx_train.npy")
    idx_val = np.load(ARTIFACTS / "idx_val.npy")
    idx_te = np.load(ARTIFACTS / "idx_test.npy")
    y_all = np.load(ARTIFACTS / "y_all.npy")

    results = []

    print("[05_multiview] M6: Uniform ensemble...")
    m6, _ = run_m6(idx_te, y_all)
    results.append(m6)

    print("[05_multiview] M7: Learned combiner (no curvature)...")
    m7, _ = run_m7(idx_tr, idx_val, idx_te, y_all)
    results.append(m7)

    print("[05_multiview] Done.")
    return results


if __name__ == "__main__":
    run_multiview()
