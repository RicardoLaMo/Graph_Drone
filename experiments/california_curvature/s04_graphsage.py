"""
04_graphsage.py — GraphSAGE on individual graph views.
Node features = full 8-d scaled features (topology changes, not features).
Models: M3=FULL, M4=GEO, M5=SOCIO
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.nn import SAGEConv

ARTIFACTS = Path("artifacts")
SEED = 42
torch.manual_seed(SEED)
EPOCHS = 200
PATIENCE = 20
LR = 1e-3
HIDDEN = 64


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def evaluate(name, y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  [{name}]  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


class GraphSAGERegressor(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden)
        self.sage2 = SAGEConv(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.sage1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.sage2(x, edge_index))
        return self.head(x).squeeze(-1)


def train_graphsage(model_name, graph_path, idx_tr, idx_val, idx_te, y_all):
    device = get_device()
    print(f"  [{model_name}] Device: {device}")

    data = torch.load(graph_path, weights_only=False)
    data = data.to(device)

    y_tr = torch.tensor(y_all[idx_tr], dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_all[idx_val], dtype=torch.float32).to(device)

    in_dim = data.x.shape[1]
    model = GraphSAGERegressor(in_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    idx_tr_t = torch.tensor(idx_tr, dtype=torch.long)
    idx_val_t = torch.tensor(idx_val, dtype=torch.long)

    best_val = float("inf")
    best_state = None
    wait = 0
    start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        optim.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[idx_tr_t], y_tr)
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(data.x, data.edge_index)[idx_val_t]
            val_loss = loss_fn(val_pred, y_val_t).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  [{model_name}] Early stop at epoch {epoch}")
                break

    train_time = time.time() - start
    model.load_state_dict(best_state)
    model.eval()

    start_inf = time.time()
    with torch.no_grad():
        all_preds = model(data.x, data.edge_index).cpu().numpy()
    inf_time = time.time() - start_inf

    preds_te = all_preds[idx_te]
    y_te = y_all[idx_te]

    metrics = evaluate(model_name, y_te, preds_te)
    metrics["train_time_s"] = train_time
    metrics["inference_time_s"] = inf_time
    torch.save(model.state_dict(), ARTIFACTS / f"model_{model_name}.pt")
    np.save(ARTIFACTS / f"preds_{model_name}_all.npy", all_preds)
    np.save(ARTIFACTS / f"preds_{model_name}_test.npy", preds_te)
    return metrics, all_preds


def run_graphsage():
    print("[04_graphsage] Loading indices and labels...")
    idx_tr = np.load(ARTIFACTS / "idx_train.npy")
    idx_val = np.load(ARTIFACTS / "idx_val.npy")
    idx_te = np.load(ARTIFACTS / "idx_test.npy")
    y_all = np.load(ARTIFACTS / "y_all.npy")

    results = []

    for tag, gname in [("M3_FULL", "FULL"), ("M4_GEO", "GEO"), ("M5_SOCIO", "SOCIO")]:
        gpath = ARTIFACTS / f"graph_{gname}.pt"
        print(f"[04_graphsage] Training {tag}...")
        m, _ = train_graphsage(tag, gpath, idx_tr, idx_val, idx_te, y_all)
        results.append(m)

    print("[04_graphsage] Done.")
    return results


if __name__ == "__main__":
    run_graphsage()
