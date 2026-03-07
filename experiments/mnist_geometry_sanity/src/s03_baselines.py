"""
s03_baselines.py — M0: Majority, M1: MLP, M2: HGBR, M3: XGBoost, M4: TabPFN
"""

import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

EXP_ROOT = Path(__file__).parent.parent
ARTIFACTS = EXP_ROOT / "artifacts"
SEED = 42
N_CLASSES = 10
torch.manual_seed(SEED)


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_splits():
    return (
        np.load(ARTIFACTS / "X_train.npy"), np.load(ARTIFACTS / "X_val.npy"), np.load(ARTIFACTS / "X_test.npy"),
        np.load(ARTIFACTS / "y_train.npy"), np.load(ARTIFACTS / "y_val.npy"), np.load(ARTIFACTS / "y_test.npy"),
    )


def evaluate_clf(name, y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ll = float(log_loss(y_true, y_proba)) if y_proba is not None else float("nan")
    print(f"  [{name}]  Acc={acc:.4f}  F1={f1:.4f}  LogLoss={ll:.4f}")
    return {"model": name, "accuracy": acc, "macro_f1": f1, "log_loss": ll}


# ── M0: Majority class ────────────────────────────────────────────────────────

def train_m0(y_tr, y_te):
    majority = np.bincount(y_tr).argmax()
    preds = np.full(len(y_te), majority)
    proba = np.zeros((len(y_te), N_CLASSES))
    proba[:, majority] = 1.0
    m = evaluate_clf("M0_Majority", y_te, preds, proba)
    m.update({"train_time_s": 0.0, "inference_time_s": 0.0})
    np.save(ARTIFACTS / "preds_M0_Majority_test.npy", preds)
    return m


# ── M1: MLP ──────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=256, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr, X_val, X_te, y_tr, y_val, y_te):
    device = get_device()
    print(f"  [M1_MLP] Device={device} hidden=256")
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)

    model = MLP(X_tr.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state, wait = float("inf"), None, 0
    start = time.time()

    from torch.utils.data import TensorDataset, DataLoader
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True)

    for epoch in range(100):
        model.train()
        for xb, yb in loader:
            optim.zero_grad()
            criterion(model(xb), yb).backward()
            optim.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_val_t), y_val_t).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 15:
                print(f"  [M1_MLP] Early stop epoch {epoch}")
                break

    train_time = time.time() - start
    model.load_state_dict(best_state)
    model.eval()
    start_inf = time.time()
    with torch.no_grad():
        logits = model(X_te_t)
        proba = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
    inf_time = time.time() - start_inf

    np.save(ARTIFACTS / "preds_M1_MLP_test.npy", preds)
    np.save(ARTIFACTS / "proba_M1_MLP_test.npy", proba)
    torch.save(model.state_dict(), ARTIFACTS / "model_M1_MLP.pt")
    m = evaluate_clf("M1_MLP", y_te, preds, proba)
    m.update({"train_time_s": train_time, "inference_time_s": inf_time})
    return m


# ── M2: HGBR ─────────────────────────────────────────────────────────────────

def train_hgb(X_tr, X_val, X_te, y_tr, y_val, y_te):
    start = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.05, random_state=SEED,
        validation_fraction=0.1, n_iter_no_change=15, verbose=0,
    )
    clf.fit(np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val]))
    train_time = time.time() - start

    start_inf = time.time()
    preds = clf.predict(X_te)
    proba = clf.predict_proba(X_te)
    inf_time = time.time() - start_inf

    np.save(ARTIFACTS / "preds_M2_HGBR_test.npy", preds)
    np.save(ARTIFACTS / "proba_M2_HGBR_test.npy", proba)
    import pickle
    with open(ARTIFACTS / "model_M2_HGBR.pkl", "wb") as f:
        pickle.dump(clf, f)
    m = evaluate_clf("M2_HGBR", y_te, preds, proba)
    m.update({"train_time_s": train_time, "inference_time_s": inf_time})
    return m


# ── M3: XGBoost ──────────────────────────────────────────────────────────────

def train_xgboost(X_tr, X_val, X_te, y_tr, y_val, y_te):
    import xgboost as xgb
    start = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        tree_method="hist", objective="multi:softmax", num_class=N_CLASSES,
        eval_metric="mlogloss", random_state=SEED, verbosity=0,
        early_stopping_rounds=15,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.time() - start

    start_inf = time.time()
    preds = clf.predict(X_te)
    proba = clf.predict_proba(X_te)
    inf_time = time.time() - start_inf

    np.save(ARTIFACTS / "preds_M3_XGB_test.npy", preds)
    np.save(ARTIFACTS / "proba_M3_XGB_test.npy", proba)
    clf.save_model(str(ARTIFACTS / "model_M3_XGB.json"))
    m = evaluate_clf("M3_XGBoost", y_te, preds, proba)
    m.update({"train_time_s": train_time, "inference_time_s": inf_time})
    return m


# ── M4: TabPFN ───────────────────────────────────────────────────────────────

def train_tabpfn(X_tr, X_te, y_tr, y_te):
    """
    TabPFN v2.5 has a hard practical limit at ~1024 training samples for public access.
    We document this limitation clearly and run on a fixed 1024-sample subset.
    """
    MAX_TRAIN = 1024
    try:
        from tabpfn import TabPFNClassifier
        print(f"  [M4_TabPFN] Using first {MAX_TRAIN} train samples (model limit). DOCUMENTED LIMITATION.")
        X_sub = X_tr[:MAX_TRAIN]
        y_sub = y_tr[:MAX_TRAIN]
        # Force CPU for TabPFN — MPS OOM with 784-d × 512 test samples (8 GiB limit).
        # Documented: MPS allocated 2.06 GiB + 6.19 GiB other, attempted 165 MiB extra → OOM.
        MAX_TEST = 128
        X_te_sub = X_te[:MAX_TEST]
        y_te_sub = y_te[:MAX_TEST]

        start = time.time()
        clf = TabPFNClassifier(device="cpu")
        clf.fit(X_sub, y_sub)
        train_time = time.time() - start

        start_inf = time.time()
        preds = clf.predict(X_te_sub)
        proba = clf.predict_proba(X_te_sub)
        inf_time = time.time() - start_inf

        np.save(ARTIFACTS / "preds_M4_TabPFN_test_subset.npy", preds)
        np.save(ARTIFACTS / "proba_M4_TabPFN_test_subset.npy", proba)
        m = evaluate_clf("M4_TabPFN_subset", y_te_sub, preds, proba)
        m.update({
            "train_time_s": train_time, "inference_time_s": inf_time,
            "note": f"Limited to {MAX_TRAIN} train samples (TabPFN v2.5 public limit). (n_test_eval={len(y_te_sub)})",
        })
        return m
    except Exception as e:
        print(f"  [M4_TabPFN] FAILED: {e}")
        return {
            "model": "M4_TabPFN", "accuracy": float("nan"), "macro_f1": float("nan"),
            "log_loss": float("nan"), "train_time_s": 0.0, "inference_time_s": 0.0,
            "note": f"Failed: {str(e)[:200]}",
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_baselines():
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_splits()
    results = []

    print("[s03] M0: Majority class")
    results.append(train_m0(y_tr, y_te))

    print("[s03] M1: MLP")
    results.append(train_mlp(X_tr, X_val, X_te, y_tr, y_val, y_te))

    print("[s03] M2: HistGradientBoosting")
    results.append(train_hgb(X_tr, X_val, X_te, y_tr, y_val, y_te))

    print("[s03] M3: XGBoost")
    results.append(train_xgboost(X_tr, X_val, X_te, y_tr, y_val, y_te))

    print("[s03] M4: TabPFN")
    results.append(train_tabpfn(X_tr, X_te, y_tr, y_te))

    print("[s03] Baselines done.")
    return results


if __name__ == "__main__":
    run_baselines()
