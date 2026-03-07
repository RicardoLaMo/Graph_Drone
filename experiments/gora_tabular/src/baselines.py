"""
baselines.py — B0 (MLP), B1 (HGBR), B2 (XGBoost) tabular baselines for GoRA-Tabular audit.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score


def train_mlp(X_tr, y_tr, X_va, y_va, out_dim, task, epochs=200, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    d = X_tr.shape[1]
    model = nn.Sequential(
        nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, out_dim)
    )
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    from torch.utils.data import TensorDataset, DataLoader
    Xt = torch.tensor(X_tr, dtype=torch.float32).to(dev)
    yt_t = torch.tensor(y_tr, dtype=torch.long if task == "classification" else torch.float32).to(dev)
    Xv = torch.tensor(X_va, dtype=torch.float32).to(dev)
    yv_t = torch.tensor(y_va, dtype=torch.long if task == "classification" else torch.float32).to(dev)
    loader = DataLoader(TensorDataset(Xt, yt_t), batch_size=512, shuffle=True)
    best, bst, wait = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = crit(out.squeeze(-1) if task == "regression" else out, yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vout = model(Xv)
            vl = crit(vout.squeeze(-1) if task == "regression" else vout, yv_t).item()
        if vl < best: best = vl; bst = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
        else: wait += 1
        if wait >= 20: break
    model.load_state_dict(bst); model.eval()
    with torch.no_grad():
        Xte = torch.tensor(X_va, dtype=torch.float32).to(dev)
        preds = model(Xte)
    if task == "classification":
        proba = torch.softmax(preds, -1).cpu().numpy()
        return model, proba.argmax(-1), proba
    else:
        return model, preds.squeeze(-1).cpu().numpy(), None


def train_hgbr(X_tr, y_tr, X_va, y_va, task):
    X_fit = np.vstack([X_tr, X_va]); y_fit = np.concatenate([y_tr, y_va])
    if task == "classification":
        m = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.05,
                                            random_state=42, n_iter_no_change=15)
    else:
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05,
                                           random_state=42, n_iter_no_change=15)
    m.fit(X_fit, y_fit)
    return m
