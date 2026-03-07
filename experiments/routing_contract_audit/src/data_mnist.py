"""
data_mnist.py — MNIST-784 loader for routing audit.
"""
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

SEED = 42


def load_mnist(artifact_dir: Path, n_subset=10000):
    print(f"[MN] Loading MNIST (n_subset={n_subset})...")
    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = data.data.astype(np.float32), data.target.astype(np.int64)
    if n_subset and n_subset < len(X):
        sss = StratifiedShuffleSplit(1, train_size=n_subset, random_state=SEED)
        idx, _ = next(sss.split(X, y)); X, y = X[idx], y[idx]
    sc = StandardScaler(); X = sc.fit_transform(X).astype(np.float32)
    all_i = np.arange(len(X))
    tr_i, tmp_i = train_test_split(all_i, test_size=0.30, random_state=SEED, stratify=y)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, random_state=SEED, stratify=y[tmp_i])
    for nm, arr in [("X_all",X),("y_all",y),("idx_train",tr_i),("idx_val",va_i),("idx_test",te_i)]:
        np.save(artifact_dir/f"mn_{nm}.npy", arr)
    print(f"[MN] n={len(X)} train={len(tr_i)} val={len(va_i)} test={len(te_i)}")
    return X, y, tr_i, va_i, te_i
