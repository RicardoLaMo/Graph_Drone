"""
s05_multiview.py — M8: Uniform ensemble, M9: Learned combiner (no curvature).
"""

import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, log_loss

EXP_ROOT = Path(__file__).parent.parent
ARTIFACTS = EXP_ROOT / "artifacts"
N_CLASSES = 10
SEED = 42
torch.manual_seed(SEED)


def evaluate_clf(name, y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ll = float(log_loss(y_true, y_proba)) if y_proba is not None else float("nan")
    print(f"  [{name}]  Acc={acc:.4f}  F1={f1:.4f}  LogLoss={ll:.4f}")
    return {"model": name, "accuracy": acc, "macro_f1": f1, "log_loss": ll}


def load_view_proba():
    p5 = np.load(ARTIFACTS / "proba_M5_FULL_all.npy")
    p6 = np.load(ARTIFACTS / "proba_M6_BLOCK_all.npy")
    p7 = np.load(ARTIFACTS / "proba_M7_PCA_all.npy")
    return p5, p6, p7   # each (N, 10)


# ── M8: Uniform ensemble ──────────────────────────────────────────────────────

def run_m8(idx_te, y_all):
    p5, p6, p7 = load_view_proba()
    proba_all = (p5 + p6 + p7) / 3.0
    preds_all = proba_all.argmax(axis=-1)
    preds_te = preds_all[idx_te]
    proba_te = proba_all[idx_te]
    np.save(ARTIFACTS / "preds_M8_Uniform_test.npy", preds_te)
    np.save(ARTIFACTS / "proba_M8_Uniform_all.npy", proba_all)
    m = evaluate_clf("M8_Uniform", y_all[idx_te], preds_te, proba_te)
    m.update({"train_time_s": 0.0, "inference_time_s": 0.0})
    return m, proba_all


# ── M9: Learned combiner (no curvature) ──────────────────────────────────────

class SoftViewCombiner(nn.Module):
    """Global learned softmax weights over 3 views' log-proba."""
    def __init__(self):
        super().__init__()
        self.logw = nn.Parameter(torch.zeros(3))

    def forward(self, proba_stack):
        # proba_stack: (N, 3, C)
        w = torch.softmax(self.logw, dim=0)  # (3,)
        return (proba_stack * w.view(1, 3, 1)).sum(dim=1)  # (N, C)


def run_m9(idx_tr, idx_val, idx_te, y_all):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    p5, p6, p7 = load_view_proba()
    # Stack: (N, 3, C)
    proba_stack = np.stack([p5, p6, p7], axis=1)
    proba_t = torch.tensor(proba_stack, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_all, dtype=torch.long).to(device)
    idx_tr_t = torch.tensor(idx_tr, dtype=torch.long)
    idx_val_t = torch.tensor(idx_val, dtype=torch.long)

    model = SoftViewCombiner().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state, wait = float("inf"), None, 0
    start = time.time()
    for epoch in range(500):
        model.train()
        optim.zero_grad()
        out = model(proba_t[idx_tr_t])
        criterion(out, y_t[idx_tr_t]).backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(proba_t[idx_val_t]), y_t[idx_val_t]).item()
        if vl < best_val:
            best_val = vl
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
        proba_all = torch.softmax(model(proba_t), dim=-1).cpu().numpy()
    preds_all = proba_all.argmax(-1)
    preds_te = preds_all[idx_te]
    proba_te = proba_all[idx_te]

    w = torch.softmax(model.logw, dim=0).detach().cpu().numpy()
    print(f"  [M9] View weights: FULL={w[0]:.3f} BLOCK={w[1]:.3f} PCA={w[2]:.3f}")

    np.save(ARTIFACTS / "preds_M9_Combiner_test.npy", preds_te)
    np.save(ARTIFACTS / "proba_M9_Combiner_all.npy", proba_all)
    torch.save(model.state_dict(), ARTIFACTS / "model_M9_Combiner.pt")
    m = evaluate_clf("M9_Combiner", y_all[idx_te], preds_te, proba_te)
    m.update({"train_time_s": train_time, "inference_time_s": 0.0})
    return m, proba_all


def run_multiview():
    idx_tr = np.load(ARTIFACTS / "idx_train.npy")
    idx_val = np.load(ARTIFACTS / "idx_val.npy")
    idx_te = np.load(ARTIFACTS / "idx_test.npy")
    y_all = np.load(ARTIFACTS / "y_all.npy")
    results = []

    print("[s05] M8: Uniform ensemble...")
    m8, _ = run_m8(idx_te, y_all)
    results.append(m8)

    print("[s05] M9: Learned combiner (no curvature)...")
    m9, _ = run_m9(idx_tr, idx_val, idx_te, y_all)
    results.append(m9)

    print("[s05] Multi-view done.")
    return results


if __name__ == "__main__":
    run_multiview()
