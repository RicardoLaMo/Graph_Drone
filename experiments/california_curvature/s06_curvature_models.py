"""
06_curvature_models.py — Curvature-augmented models.

M8A: kappa_i appended to node features → GraphSAGE on FULL graph
M8B: kappa_i as 4th explicit view in learned multi-view combiner
M9:  Observer-driven per-row combiner: [kappa, LID, LOF, density] → row-level
     weights over M3/M4/M5 predictions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

from torch_geometric.nn import SAGEConv

ARTIFACTS = Path("artifacts")
SEED = 42
torch.manual_seed(SEED)
LR = 1e-3
HIDDEN = 64
EPOCHS = 200
PATIENCE = 20


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def evaluate(name, y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  [{name}]  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


# ─── Shared GraphSAGE architecture ───────────────────────────────────────────

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


def train_sage(model_name, data, idx_tr, idx_val, idx_te, y_all, device):
    data = data.to(device)
    y_tr_t = torch.tensor(y_all[idx_tr], dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_all[idx_val], dtype=torch.float32).to(device)
    idx_tr_t = torch.tensor(idx_tr, dtype=torch.long)
    idx_val_t = torch.tensor(idx_val, dtype=torch.long)

    model = GraphSAGERegressor(data.x.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0
    start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        optim.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[idx_tr_t], y_tr_t)
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(data.x, data.edge_index)[idx_val_t], y_val_t).item()

        if vl < best_val:
            best_val = vl
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
    with torch.no_grad():
        all_preds = model(data.x, data.edge_index).cpu().numpy()

    preds_te = all_preds[idx_te]
    metrics = evaluate(model_name, y_all[idx_te], preds_te)
    metrics["train_time_s"] = train_time
    metrics["inference_time_s"] = 0.0
    torch.save(model.state_dict(), ARTIFACTS / f"model_{model_name}.pt")
    np.save(ARTIFACTS / f"preds_{model_name}_all.npy", all_preds)
    np.save(ARTIFACTS / f"preds_{model_name}_test.npy", preds_te)
    return metrics, all_preds


# ─── M8A: kappa as appended node feature ─────────────────────────────────────

def run_m8a(idx_tr, idx_val, idx_te, y_all):
    device = get_device()
    print(f"  [M8A] Device: {device}")

    data = torch.load(ARTIFACTS / "graph_FULL.pt", weights_only=False)
    kappa = np.load(ARTIFACTS / "kappa.npy")[:, None].astype(np.float32)  # (N, 1)

    X_all = np.load(ARTIFACTS / "X_all.npy")
    X_augmented = np.concatenate([X_all, kappa], axis=1)  # (N, 9)

    from torch_geometric.data import Data
    data_aug = Data(
        x=torch.tensor(X_augmented, dtype=torch.float32),
        edge_index=data.edge_index.clone(),
        y=data.y.clone(),
    )

    return train_sage("M8A_KappaFeature", data_aug, idx_tr, idx_val, idx_te, y_all, device)


# ─── M8B: kappa as 4th view in learned combiner ──────────────────────────────

class FourViewCombiner(nn.Module):
    """Learns per-row softmax weights over 4 views using their predictions."""
    def __init__(self):
        super().__init__()
        # small MLP that maps 4 view predictions → 4 softmax weights
        self.w_net = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, preds_stack):
        # preds_stack: (N, 4)
        weights = torch.softmax(self.w_net(preds_stack), dim=-1)
        return (preds_stack * weights).sum(dim=-1)


def run_m8b(idx_tr, idx_val, idx_te, y_all):
    device = get_device()
    print(f"  [M8B] Device: {device}")

    p_full = np.load(ARTIFACTS / "preds_M3_FULL_all.npy")
    p_geo = np.load(ARTIFACTS / "preds_M4_GEO_all.npy")
    p_socio = np.load(ARTIFACTS / "preds_M5_SOCIO_all.npy")

    # CURVATURE view: predictions from GraphSAGE on curvature-weighted graph
    data_curv = torch.load(ARTIFACTS / "graph_CURVATURE.pt", weights_only=False)
    temp_data = data_curv
    curv_model_path = ARTIFACTS / "model_M8B_curvature_view_temp.pt"

    # Train a GraphSAGE on the curvature graph for its predictions
    temp_m, p_curv_all = train_sage(
        "M8B_CurvView", temp_data, idx_tr, idx_val, idx_te, y_all, device
    )

    # Now combine all 4 views
    preds_stack = np.stack([p_full, p_geo, p_socio, p_curv_all], axis=-1)  # (N, 4)

    preds_t = torch.tensor(preds_stack, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_all, dtype=torch.float32).to(device)
    idx_tr_t = torch.tensor(idx_tr, dtype=torch.long)
    idx_val_t = torch.tensor(idx_val, dtype=torch.long)

    model = FourViewCombiner().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()

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
            vl = loss_fn(model(preds_t[idx_val_t]), y_t[idx_val_t]).item()

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
        preds_all = model(preds_t).cpu().numpy()

    preds_te = preds_all[idx_te]
    np.save(ARTIFACTS / "preds_M8B_FourViewCombiner_test.npy", preds_te)
    np.save(ARTIFACTS / "preds_M8B_FourViewCombiner_all.npy", preds_all)

    metrics = evaluate("M8B_FourViewCombiner", y_all[idx_te], preds_te)
    metrics["train_time_s"] = train_time
    metrics["inference_time_s"] = 0.0
    torch.save(model.state_dict(), ARTIFACTS / "model_M8B_FourViewCombiner.pt")
    return metrics, preds_all


# ─── M9: Observer-driven per-row combiner ────────────────────────────────────

class ObserverCombiner(nn.Module):
    """
    Maps per-row observer features [kappa, LID, LOF, density] to softmax
    weights over 3 views. This allows the model to specialise view selection
    based on local structure.
    """
    def __init__(self, n_observers=4, n_views=3):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(n_observers, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, n_views),
        )

    def forward(self, observer_feat, preds_stack):
        # observer_feat: (N, n_observers), preds_stack: (N, n_views)
        row_weights = torch.softmax(self.weight_net(observer_feat), dim=-1)
        return (preds_stack * row_weights).sum(dim=-1)


def run_m9(idx_tr, idx_val, idx_te, y_all):
    device = get_device()
    print(f"  [M9] Device: {device}")

    observer = np.load(ARTIFACTS / "observer_features.npy")  # (N, 5): kappa, lid, lof, density, forman
    obs_4 = observer[:, :4]  # use kappa, LID, LOF, density

    p_full = np.load(ARTIFACTS / "preds_M3_FULL_all.npy")
    p_geo = np.load(ARTIFACTS / "preds_M4_GEO_all.npy")
    p_socio = np.load(ARTIFACTS / "preds_M5_SOCIO_all.npy")
    preds_stack = np.stack([p_full, p_geo, p_socio], axis=-1)  # (N, 3)

    # Normalise observer features
    obs_mean = obs_4[idx_tr].mean(axis=0)
    obs_std = obs_4[idx_tr].std(axis=0) + 1e-8
    obs_4_sc = ((obs_4 - obs_mean) / obs_std).astype(np.float32)

    obs_t = torch.tensor(obs_4_sc, dtype=torch.float32).to(device)
    preds_t = torch.tensor(preds_stack, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_all, dtype=torch.float32).to(device)
    idx_tr_t = torch.tensor(idx_tr, dtype=torch.long)
    idx_val_t = torch.tensor(idx_val, dtype=torch.long)

    model = ObserverCombiner(n_observers=4, n_views=3).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0
    start = time.time()

    for epoch in range(500):
        model.train()
        optim.zero_grad()
        out = model(obs_t[idx_tr_t], preds_t[idx_tr_t])
        loss = loss_fn(out, y_t[idx_tr_t])
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(obs_t[idx_val_t], preds_t[idx_val_t]), y_t[idx_val_t]).item()

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
        preds_all = model(obs_t, preds_t).cpu().numpy()

    preds_te = preds_all[idx_te]
    np.save(ARTIFACTS / "preds_M9_ObserverCombiner_test.npy", preds_te)
    np.save(ARTIFACTS / "preds_M9_ObserverCombiner_all.npy", preds_all)

    metrics = evaluate("M9_ObserverCombiner", y_all[idx_te], preds_te)
    metrics["train_time_s"] = train_time
    metrics["inference_time_s"] = 0.0
    torch.save(model.state_dict(), ARTIFACTS / "model_M9_ObserverCombiner.pt")
    return metrics, preds_all


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_curvature_models():
    print("[06_curvature_models] Loading indices and labels...")
    idx_tr = np.load(ARTIFACTS / "idx_train.npy")
    idx_val = np.load(ARTIFACTS / "idx_val.npy")
    idx_te = np.load(ARTIFACTS / "idx_test.npy")
    y_all = np.load(ARTIFACTS / "y_all.npy")

    results = []

    print("[06_curvature_models] M8A: kappa as node feature...")
    m8a, _ = run_m8a(idx_tr, idx_val, idx_te, y_all)
    results.append(m8a)

    print("[06_curvature_models] M8B: 4-view learned combiner with kappa view...")
    m8b, _ = run_m8b(idx_tr, idx_val, idx_te, y_all)
    results.append(m8b)

    print("[06_curvature_models] M9: Observer-driven per-row combiner...")
    m9, _ = run_m9(idx_tr, idx_val, idx_te, y_all)
    results.append(m9)

    print("[06_curvature_models] Done.")
    return results


if __name__ == "__main__":
    run_curvature_models()
