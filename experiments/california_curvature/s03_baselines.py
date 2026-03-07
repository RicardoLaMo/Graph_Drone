"""
03_baselines.py — B0: Mean predictor, B1: MLP, B2: HistGradientBoostingRegressor
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ARTIFACTS = Path("artifacts")
SEED = 42
torch.manual_seed(SEED)


def load_splits():
    X_tr = np.load(ARTIFACTS / "X_train.npy")
    X_val = np.load(ARTIFACTS / "X_val.npy")
    X_te = np.load(ARTIFACTS / "X_test.npy")
    y_tr = np.load(ARTIFACTS / "y_train.npy")
    y_val = np.load(ARTIFACTS / "y_val.npy")
    y_te = np.load(ARTIFACTS / "y_test.npy")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def evaluate(name, y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  [{name}]  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


# ─── B0: Mean predictor ───────────────────────────────────────────────────────

def train_mean(X_tr, X_te, y_tr, y_te):
    mean_val = y_tr.mean()
    preds = np.full(len(y_te), mean_val)
    return evaluate("B0_Mean", y_te, preds), preds


# ─── B1: MLP ─────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr, X_val, X_te, y_tr, y_val, y_te):
    import time
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"  [B1_MLP] Device: {device}")

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)

    model = MLP(X_tr.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = 15
    wait = 0
    start = time.time()

    for epoch in range(200):
        model.train()
        optim.zero_grad()
        out = model(X_tr_t)
        loss = loss_fn(out, y_tr_t)
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [B1_MLP] Early stop at epoch {epoch}")
                break

    train_time = time.time() - start
    model.load_state_dict(best_state)
    model.eval()

    start_inf = time.time()
    with torch.no_grad():
        preds = model(X_te_t).cpu().numpy()
    inf_time = time.time() - start_inf

    metrics = evaluate("B1_MLP", y_te, preds)
    metrics["train_time_s"] = train_time
    metrics["inference_time_s"] = inf_time
    torch.save(model.state_dict(), ARTIFACTS / "model_B1_MLP.pt")
    return metrics, preds


# ─── B2: HistGradientBoostingRegressor ───────────────────────────────────────

def train_hgbr(X_tr, X_val, X_te, y_tr, y_val, y_te):
    import time
    start = time.time()
    hgbr = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=SEED,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=0,
    )
    hgbr.fit(np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val]))
    train_time = time.time() - start

    start_inf = time.time()
    preds = hgbr.predict(X_te).astype(np.float32)
    inf_time = time.time() - start_inf

    metrics = evaluate("B2_HGBR", y_te, preds)
    metrics["train_time_s"] = train_time
    metrics["inference_time_s"] = inf_time

    import pickle
    with open(ARTIFACTS / "model_B2_HGBR.pkl", "wb") as f:
        pickle.dump(hgbr, f)
    return metrics, preds


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_baselines():
    print("[03_baselines] Loading splits...")
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_splits()
    results = []

    print("[03_baselines] B0: Mean predictor")
    m, preds_b0 = train_mean(X_tr, X_te, y_tr, y_te)
    m.update({"train_time_s": 0.0, "inference_time_s": 0.0})
    results.append(m)
    np.save(ARTIFACTS / "preds_B0_Mean_test.npy", preds_b0)

    print("[03_baselines] B1: MLP")
    m, preds_b1 = train_mlp(X_tr, X_val, X_te, y_tr, y_val, y_te)
    results.append(m)
    np.save(ARTIFACTS / "preds_B1_MLP_test.npy", preds_b1)

    print("[03_baselines] B2: HistGradientBoostingRegressor")
    m, preds_b2 = train_hgbr(X_tr, X_val, X_te, y_tr, y_val, y_te)
    results.append(m)
    np.save(ARTIFACTS / "preds_B2_HGBR_test.npy", preds_b2)

    print("[03_baselines] Done.")
    return results


if __name__ == "__main__":
    run_baselines()
