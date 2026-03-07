"""
s04_graphsage.py — GraphSAGE classifiers on FULL, BLOCK, PCA graph views.
All use 784-d node features; only topology changes per view.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch_geometric.nn import SAGEConv

EXP_ROOT = Path(__file__).parent.parent
ARTIFACTS = EXP_ROOT / "artifacts"
SEED = 42
N_CLASSES = 10
HIDDEN = 128
EPOCHS = 150
PATIENCE = 20
torch.manual_seed(SEED)


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


def train_sage_view(model_name, graph_path, idx_tr, idx_val, idx_te, y_all):
    device = get_device()
    print(f"  [{model_name}] Device={device}")

    data = torch.load(graph_path, weights_only=False).to(device)
    y_all_t = torch.tensor(y_all, dtype=torch.long).to(device)
    idx_tr_t = torch.tensor(idx_tr, dtype=torch.long)
    idx_val_t = torch.tensor(idx_val, dtype=torch.long)

    model = GraphSAGEClassifier(data.x.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state, wait = float("inf"), None, 0
    start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        optim.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[idx_tr_t], y_all_t[idx_tr_t])
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(out[idx_val_t].detach(), y_all_t[idx_val_t]).item()
            # Re-run forward for eval to be accurate
            vl = criterion(model(data.x, data.edge_index)[idx_val_t], y_all_t[idx_val_t]).item()

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  [{model_name}] Early stop epoch {epoch}")
                break

    train_time = time.time() - start
    model.load_state_dict(best_state)
    model.eval()

    start_inf = time.time()
    with torch.no_grad():
        all_logits = model(data.x, data.edge_index)
        all_proba = torch.softmax(all_logits, dim=-1).cpu().numpy()
        all_preds = all_logits.argmax(dim=-1).cpu().numpy()
    inf_time = time.time() - start_inf

    preds_te = all_preds[idx_te]
    proba_te = all_proba[idx_te]

    np.save(ARTIFACTS / f"preds_{model_name}_all.npy", all_preds)
    np.save(ARTIFACTS / f"proba_{model_name}_all.npy", all_proba)
    np.save(ARTIFACTS / f"preds_{model_name}_test.npy", preds_te)
    np.save(ARTIFACTS / f"proba_{model_name}_test.npy", proba_te)
    torch.save(model.state_dict(), ARTIFACTS / f"model_{model_name}.pt")

    m = evaluate_clf(model_name, y_all[idx_te], preds_te, proba_te)
    m.update({"train_time_s": train_time, "inference_time_s": inf_time})
    return m, all_proba


def run_graphsage():
    idx_tr = np.load(ARTIFACTS / "idx_train.npy")
    idx_val = np.load(ARTIFACTS / "idx_val.npy")
    idx_te = np.load(ARTIFACTS / "idx_test.npy")
    y_all = np.load(ARTIFACTS / "y_all.npy")
    results = []

    for tag, gname in [("M5_FULL", "FULL"), ("M6_BLOCK", "BLOCK"), ("M7_PCA", "PCA")]:
        print(f"[s04] Training {tag}...")
        m, _ = train_sage_view(tag, ARTIFACTS / f"graph_{gname}.pt", idx_tr, idx_val, idx_te, y_all)
        results.append(m)

    print("[s04] GraphSAGE done.")
    return results


if __name__ == "__main__":
    run_graphsage()
