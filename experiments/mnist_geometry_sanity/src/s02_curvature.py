"""
s02_curvature.py — Curvature proxy + observer features for MNIST rows.
Same methodology as California experiment; adapted for 784-d inputs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
import torch
from torch_geometric.data import Data

EXP_ROOT = Path(__file__).parent.parent
ARTIFACTS = EXP_ROOT / "artifacts"
K_BASE = 15
K_SCALES = [10, 20, 30]
R_DIMS = 2


def knn_dists_indices(X, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
    d, idx = nbrs.kneighbors(X)
    return d[:, 1:], idx[:, 1:]


def local_pca_residual(X, indices, r=R_DIMS):
    n = X.shape[0]
    kappa = np.zeros(n, dtype=np.float32)
    for i in range(n):
        nbr = X[indices[i]]
        centered = nbr - nbr.mean(0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        tv = (s**2).sum()
        kappa[i] = 0.0 if tv < 1e-12 else float(1.0 - (s[:r]**2).sum() / tv)
    return kappa


def lid_proxy(dists):
    k = dists.shape[1]
    ks = np.arange(1, k + 1, dtype=np.float64)
    lid = np.zeros(dists.shape[0], dtype=np.float32)
    for i in range(dists.shape[0]):
        d = np.maximum(dists[i], 1e-12)
        lid[i] = float(max(np.polyfit(np.log(ks), np.log(d), 1)[0], 0.0))
    return lid


def local_density(dists):
    return (1.0 / np.maximum(dists.mean(1), 1e-12)).astype(np.float32)


def multiscale_stability(X):
    kappas = {}
    for k in K_SCALES:
        _, idx = knn_dists_indices(X, k)
        kappas[k] = local_pca_residual(X, idx, R_DIMS)

    results = {}
    n = X.shape[0]
    top_n = max(1, n // 5)
    top_sets = {k: set(np.argsort(kappas[k])[-top_n:]) for k in K_SCALES}
    for i, k1 in enumerate(K_SCALES):
        for k2 in K_SCALES[i+1:]:
            rho, pval = spearmanr(kappas[k1], kappas[k2])
            results[f"spearman_k{k1}_k{k2}"] = float(rho)
            results[f"pval_k{k1}_k{k2}"] = float(pval)
            overlap = len(top_sets[k1] & top_sets[k2]) / top_n
            results[f"top20pct_overlap_k{k1}_k{k2}"] = float(overlap)
    return results, kappas


def build_curvature_graph(X, kappa, y_all):
    dists, indices = knn_dists_indices(X, K_BASE)
    src = np.repeat(np.arange(X.shape[0]), K_BASE)
    dst = indices.flatten()
    ew = 1.0 / (1.0 + np.abs(kappa[src] - kappa[dst]))
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_attr=torch.tensor(ew, dtype=torch.float32).unsqueeze(-1),
        y=torch.tensor(y_all, dtype=torch.long),
    )
    torch.save(data, ARTIFACTS / "graph_CURVATURE.pt")
    print(f"[s02_curvature] CURVATURE graph: {data.edge_index.shape[1]} edges")


def compute_all():
    print("[s02_curvature] Loading data...")
    X_all = np.load(ARTIFACTS / "X_all.npy")
    y_all = np.load(ARTIFACTS / "y_all.npy")
    n = X_all.shape[0]

    # Use PCA features for curvature in high-d to avoid noise dominance
    X_curv = np.load(ARTIFACTS / "X_pca.npy")
    print(f"[s02_curvature] Using PCA-50 features for curvature (avoids curse-of-dim in kNN). n={n}")

    print("[s02_curvature] kNN distances...")
    dists, indices = knn_dists_indices(X_curv, K_BASE)

    print("[s02_curvature] local PCA curvature (kappa)...")
    kappa = local_pca_residual(X_curv, indices, R_DIMS)
    assert kappa.std() > 0, "Curvature proxy zero variance!"
    print(f"[s02_curvature] kappa: mean={kappa.mean():.4f} std={kappa.std():.4f} range=[{kappa.min():.4f},{kappa.max():.4f}]")

    print("[s02_curvature] LID...")
    lid = lid_proxy(dists)

    print("[s02_curvature] LOF (may take ~30s)...")
    lof_m = LocalOutlierFactor(n_neighbors=K_BASE, novelty=False, n_jobs=-1)
    lof_m.fit(X_curv)
    lof = (-lof_m.negative_outlier_factor_).astype(np.float32)

    print("[s02_curvature] local density...")
    density = local_density(dists)

    print("[s02_curvature] multi-scale stability...")
    stab, kappas = multiscale_stability(X_curv)
    print("[s02_curvature] Stability:", stab)

    # Observer features array: kappa, LID, LOF, density
    observer = np.stack([kappa, lid, lof, density], axis=1).astype(np.float32)
    np.save(ARTIFACTS / "observer_features.npy", observer)
    np.save(ARTIFACTS / "kappa.npy", kappa)

    low_t = np.percentile(kappa, 33)
    high_t = np.percentile(kappa, 67)
    bins = np.where(kappa <= low_t, "low", np.where(kappa <= high_t, "medium", "high"))

    df = pd.DataFrame({
        "row_idx": np.arange(n), "kappa": kappa, "lid": lid,
        "lof": lof, "density": density, "curvature_bin": bins, "label": y_all,
    })
    df.to_csv(ARTIFACTS / "curvature_bins.csv", index=False)
    pd.DataFrame([stab]).to_csv(ARTIFACTS / "curvature_stability.csv", index=False)

    build_curvature_graph(X_curv, kappa, y_all)
    print("[s02_curvature] Done.")
    return observer, df


if __name__ == "__main__":
    compute_all()
