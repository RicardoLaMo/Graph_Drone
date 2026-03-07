"""
s00_data.py — Load MNIST_784, preprocess, split, save to artifacts/.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EXP_ROOT = Path(__file__).parent.parent
ARTIFACTS = EXP_ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CLASSES = 10


def load_and_preprocess(n_subset=10000):
    print(f"[s00_data] Loading MNIST_784 from OpenML (n_subset={n_subset})...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data.astype(np.float32), mnist.target.astype(np.int64)

    if n_subset and n_subset < len(X):
        # Stratified subsample for class balance
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=None, train_size=n_subset, random_state=SEED)
        idx, _ = next(sss.split(X, y))
        X, y = X[idx], y[idx]
        print(f"[s00_data] Subset: {X.shape[0]} rows, class distribution: {np.bincount(y)}")

    print(f"[s00_data] Features: {X.shape[1]}, Classes: {len(np.unique(y))}")

    # ── Splits (70/15/15) ──────────────────────────────────────────────
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

    # ── Scale: fit on train only ──────────────────────────────────────
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_sc = scaler.transform(X_val).astype(np.float32)
    X_te_sc = scaler.transform(X_te).astype(np.float32)

    # ── Full-dataset scaled (needed for graphs over all nodes) ────────
    X_all_sc = scaler.transform(X).astype(np.float32)

    # ── Save ──────────────────────────────────────────────────────────
    np.save(ARTIFACTS / "X_all.npy", X_all_sc)
    np.save(ARTIFACTS / "y_all.npy", y)
    np.save(ARTIFACTS / "X_train.npy", X_tr_sc)
    np.save(ARTIFACTS / "X_val.npy", X_val_sc)
    np.save(ARTIFACTS / "X_test.npy", X_te_sc)
    np.save(ARTIFACTS / "y_train.npy", y_tr)
    np.save(ARTIFACTS / "y_val.npy", y_val)
    np.save(ARTIFACTS / "y_test.npy", y_te)

    # Save indices into the (possibly subset) X_all for traceability
    all_idx = np.arange(len(X))
    tr_idx, tmp_idx = train_test_split(all_idx, test_size=0.30, random_state=SEED, stratify=y)
    val_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, random_state=SEED, stratify=y[tmp_idx])
    np.save(ARTIFACTS / "idx_train.npy", tr_idx)
    np.save(ARTIFACTS / "idx_val.npy", val_idx)
    np.save(ARTIFACTS / "idx_test.npy", te_idx)

    meta = {
        "n_total": int(len(X)),
        "n_train": int(len(X_tr_sc)),
        "n_val": int(len(X_val_sc)),
        "n_test": int(len(X_te_sc)),
        "n_features": int(X.shape[1]),
        "n_classes": N_CLASSES,
        "seed": SEED,
    }
    with open(ARTIFACTS / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[s00_data] Train:{X_tr_sc.shape} Val:{X_val_sc.shape} Test:{X_te_sc.shape}")
    print("[s00_data] Done.")
    return meta


if __name__ == "__main__":
    load_and_preprocess()
