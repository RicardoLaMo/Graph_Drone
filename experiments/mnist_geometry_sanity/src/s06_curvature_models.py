"""
s06_curvature_models.py — M10: kappa as node feature, M11: observer-driven combiner.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

EXP_ROOT = Path(__file__).parent.parent
ARTIFACTS = EXP_ROOT / "artifacts"
N_CLASSES = 10
HIDDEN = 128
EPOCHS = 150
PATIENCE = 20
torch.manual_seed(42)


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def evaluate_clf(name, y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ll = float(log_loss(y_true, y_proba)) if y_proba is not None else float("nan")
    print(f"  [{name}]  Acc={acc:.4f}  F1={f1:.4f}  LogLoss={ll:.4f}")
    return {"model": name, "accuracy": acc, "macro_f1": f1, "log_loss": ll}


class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, n_classes=N_CLASSES):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden)
        self.sage2 = SAGEConv(hidden, hidden // 2)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden // 2)
        self.head = nn.Linear(hidden // 2, n_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.sage1(x, edge_index)))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.bn2(self.sage2(x, edge_index)))
        return self.head(x)


def _train_sage(name, data, idx_tr, idx_val, idx_te, y_all, device):
    data = data.to(device)
    y_t = torch.tensor(y_all, dtype=torch.long).to(device)
    tr_t = torch.tensor(idx_tr, dtype=torch.long)
    val_t = torch.tensor(idx_val, dtype=torch.long)
    model = GraphSAGEClassifier(data.x.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best, best_st, wait = float("inf"), None, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); optim.zero_grad()
        crit(model(data.x, data.edge_index)[tr_t], y_t[tr_t]).backward(); optim.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(data.x, data.edge_index)[val_t], y_t[val_t]).item()
        if vl < best:
            best = vl; best_st = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= PATIENCE: print(f"  [{name}] Early stop ep {ep}"); break
    model.load_state_dict(best_st); model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        proba_all = torch.softmax(logits, -1).cpu().numpy()
        preds_all = logits.argmax(-1).cpu().numpy()
    p_te, pr_te = preds_all[idx_te], proba_all[idx_te]
    np.save(ARTIFACTS / f"preds_{name}_test.npy", p_te)
    np.save(ARTIFACTS / f"proba_{name}_all.npy", proba_all)
    torch.save(model.state_dict(), ARTIFACTS / f"model_{name}.pt")
    m = evaluate_clf(name, y_all[idx_te], p_te, pr_te)
    m["train_time_s"] = time.time() - t0; m["inference_time_s"] = 0.0
    return m, proba_all


def run_m10(idx_tr, idx_val, idx_te, y_all):
    device = get_device()
    base = torch.load(ARTIFACTS / "graph_FULL.pt", weights_only=False)
    kappa = np.load(ARTIFACTS / "kappa.npy")[:, None].astype(np.float32)
    X_aug = np.concatenate([np.load(ARTIFACTS / "X_all.npy"), kappa], axis=1)
    data = Data(x=torch.tensor(X_aug, dtype=torch.float32),
                edge_index=base.edge_index.clone(), y=base.y.clone())
    return _train_sage("M10_KappaFeature", data, idx_tr, idx_val, idx_te, y_all, device)


class ObserverCombiner(nn.Module):
    def __init__(self, n_obs=4, n_views=3):
        super().__init__()
        self.wnet = nn.Sequential(nn.Linear(n_obs, 32), nn.ReLU(), nn.Linear(32, n_views))

    def forward(self, obs, ps):
        w = torch.softmax(self.wnet(obs), -1)
        return (ps * w.unsqueeze(-1)).sum(1)


def run_m11(idx_tr, idx_val, idx_te, y_all):
    device = get_device()
    obs = np.load(ARTIFACTS / "observer_features.npy")
    obs_sc = ((obs - obs[idx_tr].mean(0)) / (obs[idx_tr].std(0) + 1e-8)).astype(np.float32)
    p5 = np.load(ARTIFACTS / "proba_M5_FULL_all.npy")
    p6 = np.load(ARTIFACTS / "proba_M6_BLOCK_all.npy")
    p7 = np.load(ARTIFACTS / "proba_M7_PCA_all.npy")
    ps = np.stack([p5, p6, p7], 1)
    obs_t = torch.tensor(obs_sc).to(device)
    ps_t = torch.tensor(ps, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_all, dtype=torch.long).to(device)
    tr_t = torch.tensor(idx_tr, dtype=torch.long)
    val_t = torch.tensor(idx_val, dtype=torch.long)
    model = ObserverCombiner().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-3)
    crit = nn.CrossEntropyLoss()
    best, best_st, wait = float("inf"), None, 0
    t0 = time.time()
    for ep in range(500):
        model.train(); optim.zero_grad()
        crit(model(obs_t[tr_t], ps_t[tr_t]), y_t[tr_t]).backward(); optim.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(obs_t[val_t], ps_t[val_t]), y_t[val_t]).item()
        if vl < best:
            best = vl; best_st = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= 30: break
    model.load_state_dict(best_st); model.eval()
    with torch.no_grad():
        proba_all = torch.softmax(model(obs_t, ps_t), -1).cpu().numpy()
    preds_te = proba_all.argmax(-1)[idx_te]
    proba_te = proba_all[idx_te]
    np.save(ARTIFACTS / "preds_M11_ObsCombiner_test.npy", preds_te)
    np.save(ARTIFACTS / "proba_M11_ObsCombiner_all.npy", proba_all)
    torch.save(model.state_dict(), ARTIFACTS / "model_M11_ObsCombiner.pt")
    m = evaluate_clf("M11_ObsCombiner", y_all[idx_te], preds_te, proba_te)
    m["train_time_s"] = time.time() - t0; m["inference_time_s"] = 0.0
    return m, proba_all


def run_curvature_models():
    idx_tr = np.load(ARTIFACTS / "idx_train.npy")
    idx_val = np.load(ARTIFACTS / "idx_val.npy")
    idx_te = np.load(ARTIFACTS / "idx_test.npy")
    y_all = np.load(ARTIFACTS / "y_all.npy")
    results = []
    print("[s06] M10: kappa as node feature...")
    m10, _ = run_m10(idx_tr, idx_val, idx_te, y_all)
    results.append(m10)
    print("[s06] M11: Observer combiner...")
    m11, _ = run_m11(idx_tr, idx_val, idx_te, y_all)
    results.append(m11)
    print("[s06] Done.")
    return results


if __name__ == "__main__":
    run_curvature_models()
