"""
baselines.py — B0 (MLP), B1 (HGBR), B2 (TabPFN) tabular baselines.

TabPFN constraints:
  - Max 10k training samples → California subsampled to 8k (documented)
  - Max 500 features → MNIST (784d) reduced via PCA-200 first
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.decomposition import PCA
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


def train_tabpfn(X_tr, y_tr, X_te, y_te, task, seed=42,
                 max_train_samples=8000, max_features=200, pca_features=True):
    """
    TabPFN v2 (PriorLabs) baseline.

    Handling constraints:
      - If len(X_tr) > max_train_samples: subsample (documented, seeds fixed)
      - If X_tr.shape[1] > max_features: PCA reduction first

    Returns: preds, proba (None for regression)
    """
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    # Feature reduction if needed
    X_tr_use, X_te_use = X_tr, X_te
    pca_note = ""
    if X_tr.shape[1] > max_features and pca_features:
        pca = PCA(n_components=max_features, random_state=seed)
        X_tr_use = pca.fit_transform(X_tr).astype(np.float32)
        X_te_use = pca.transform(X_te).astype(np.float32)
        pca_note = f" (PCA {X_tr.shape[1]}→{max_features})"

    # Training subsample if needed
    subsample_note = ""
    if len(X_tr_use) > max_train_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_tr_use), max_train_samples, replace=False)
        X_tr_use = X_tr_use[idx]; y_tr_sub = y_tr[idx]
        subsample_note = f" (subsampled {len(X_tr)} → {max_train_samples})"
    else:
        y_tr_sub = y_tr

    print(f"  [TabPFN] N_train={len(X_tr_use)} d={X_tr_use.shape[1]}{pca_note}{subsample_note}")

    # Always run TabPFN on CPU — its in-context computation allocates
    # O(N_train × N_test × n_classes) tensors that overflow 8GB MPS on M-series.
    import os; os.environ.pop("TABPFN_DEVICE", None)
    tabpfn_device = "cpu"

    if task == "classification":
        clf = TabPFNClassifier(n_estimators=8, random_state=seed, device=tabpfn_device)
        clf.fit(X_tr_use, y_tr_sub)
        # Batch predict_proba to avoid residual OOM even on CPU for very large test sets
        chunk = 500
        proba_chunks = []
        for start in range(0, len(X_te_use), chunk):
            proba_chunks.append(clf.predict_proba(X_te_use[start:start + chunk]))
        proba = np.vstack(proba_chunks)
        preds = proba.argmax(-1)
        # TabPFN returns classes in sorted order — remap to original labels
        classes = np.sort(np.unique(y_tr_sub))
        label_map = {i: c for i, c in enumerate(classes)}
        preds = np.array([label_map[p] for p in preds])
        return preds, proba
    else:
        reg = TabPFNRegressor(n_estimators=8, random_state=seed, device=tabpfn_device)
        reg.fit(X_tr_use, y_tr_sub)
        preds = reg.predict(X_te_use).astype(np.float32)
        return preds, None
