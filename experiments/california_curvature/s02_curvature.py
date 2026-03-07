"""
02_curvature.py — Compute curvature proxies and observer features.

Curvature proxy: local PCA residual ratio.
  kappa_i = fraction of variance in row i's kNN neighborhood
             NOT explained by the top-r principal directions.
  This is a practical non-flatness proxy — not exact Riemannian curvature.

Observer features computed for every row:
  - kappa (local PCA residual, required)
  - LID (Local Intrinsic Dimension via log-slope of kNN distances)
  - LOF (via sklearn)
  - local_density (inverse mean neighbor distance)
  - degree (= k, constant here but extensible)
  - kappa_forman (optional Forman-style proxy)
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor

ARTIFACTS = Path("artifacts")
K_BASE = 15
K_SCALES = [10, 20, 30]
R_DIMS = 2   # top-r PCs to keep for PCA residual
SEED = 42


# ─── Core kNN utility ────────────────────────────────────────────────────────

def knn_distances_indices(X, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1).fit(X)
    dists, indices = nbrs.kneighbors(X)
    return dists[:, 1:], indices[:, 1:]   # exclude self


# ─── Curvature proxy: local PCA residual ratio ───────────────────────────────

def local_pca_residual(X: np.ndarray, indices: np.ndarray, r: int = R_DIMS) -> np.ndarray:
    """
    For each row i, compute the residual variance ratio in its kNN neighborhood
    after projecting out the top-r principal directions.

    kappa_i ∈ [0, 1].  High value = neighborhood is not well-described by r directions = more curved.
    """
    n = X.shape[0]
    kappa = np.zeros(n, dtype=np.float32)

    for i in range(n):
        nbr_idx = indices[i]
        neighborhood = X[nbr_idx]  # shape: (k, d)
        centered = neighborhood - neighborhood.mean(axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        total_var = (s ** 2).sum()
        if total_var < 1e-12:
            kappa[i] = 0.0
            continue
        top_var = (s[:r] ** 2).sum()
        kappa[i] = float(1.0 - top_var / total_var)

    return kappa


# ─── LID proxy (via log-log slope of kNN distances) ──────────────────────────

def local_intrinsic_dim(dists: np.ndarray, k_half: int = None) -> np.ndarray:
    """
    Estimate LID using the log-log slope of kNN distances.
    Uses the half-distance regression approach.
    """
    k = dists.shape[1]
    if k_half is None:
        k_half = max(1, k // 2)

    lid = np.zeros(dists.shape[0], dtype=np.float32)
    ks = np.arange(1, k + 1, dtype=np.float64)

    for i in range(dists.shape[0]):
        d = dists[i].astype(np.float64)
        d = np.maximum(d, 1e-12)
        # log-log regression
        log_k = np.log(ks)
        log_d = np.log(d)
        # simple least-squares slope
        slope = np.polyfit(log_k, log_d, 1)[0]
        lid[i] = float(max(slope, 0.0))

    return lid


# ─── Local density: inverse mean neighbor distance ───────────────────────────

def local_density(dists: np.ndarray) -> np.ndarray:
    mean_dist = dists.mean(axis=1)
    mean_dist = np.maximum(mean_dist, 1e-12)
    return (1.0 / mean_dist).astype(np.float32)


# ─── Forman-style proxy ───────────────────────────────────────────────────────

def forman_proxy(indices: np.ndarray) -> np.ndarray:
    """
    Forman–Ricci curvature proxy (simplified graph version):
    F_ij ≈ 4 - deg(i) - deg(j) + |triangle_support|
    We use the node-level version: 4 - deg(i) - avg_deg(neighbors).
    Does NOT require computing triangles explicitly.
    Normalised to [0, 1] via sigmoid.
    """
    n = indices.shape[0]
    degree = np.full(n, indices.shape[1], dtype=np.float32)  # uniform k

    forman = np.zeros(n, dtype=np.float32)
    for i in range(n):
        nbr_degrees = degree[indices[i]]
        forman[i] = 4.0 - degree[i] - nbr_degrees.mean()

    # soft-normalise with sigmoid so range is consistent
    forman_norm = 1.0 / (1.0 + np.exp(-forman / 4.0))
    return forman_norm.astype(np.float32)


# ─── Multi-scale stability ────────────────────────────────────────────────────

def multiscale_stability(X: np.ndarray, r: int = R_DIMS) -> dict:
    """
    Compute kappa for k ∈ K_SCALES, then assess Spearman rank correlation
    of the top-20% high-curvature row identities across scales.
    """
    kappas = {}
    for k in K_SCALES:
        _, idx = knn_distances_indices(X, k)
        kappas[k] = local_pca_residual(X, idx, r)

    results = {}
    for i, k1 in enumerate(K_SCALES):
        for k2 in K_SCALES[i + 1 :]:
            rho, pval = spearmanr(kappas[k1], kappas[k2])
            results[f"spearman_k{k1}_k{k2}"] = float(rho)
            results[f"pval_k{k1}_k{k2}"] = float(pval)

    # Top-20% stability: fraction of top-20% nodes shared across scales
    n = X.shape[0]
    top_n = max(1, n // 5)
    top_sets = {k: set(np.argsort(kappas[k])[-top_n:]) for k in K_SCALES}
    for i, k1 in enumerate(K_SCALES):
        for k2 in K_SCALES[i + 1:]:
            overlap = len(top_sets[k1] & top_sets[k2]) / top_n
            results[f"top20pct_overlap_k{k1}_k{k2}"] = float(overlap)

    return results, kappas


# ─── Build curvature graph view ───────────────────────────────────────────────

def build_curvature_graph(X: np.ndarray, kappa: np.ndarray):
    """
    Build a curvature-similarity graph:
    kNN in feature space (FULL), edge-weighted by curvature similarity
    (edges between nodes with similar kappa get higher weight).
    This intentionally emphasises structurally similar rows.
    """
    from torch_geometric.data import Data

    nbrs = NearestNeighbors(n_neighbors=K_BASE + 1, algorithm="auto", n_jobs=-1).fit(X)
    dists, indices = nbrs.kneighbors(X)
    dists = dists[:, 1:]
    indices = indices[:, 1:]

    src = np.repeat(np.arange(X.shape[0]), K_BASE)
    dst = indices.flatten()

    # Edge weight: curvature similarity = 1 / (1 + |kappa_i - kappa_j|)
    kappa_diff = np.abs(kappa[src] - kappa[dst])
    edge_weight = 1.0 / (1.0 + kappa_diff)

    y_all = np.load(ARTIFACTS / "y_all.npy")
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_attr=torch.tensor(edge_weight, dtype=torch.float32).unsqueeze(-1),
        y=torch.tensor(y_all, dtype=torch.float32),
    )
    torch.save(data, ARTIFACTS / "graph_CURVATURE.pt")
    print(f"[02_curvature] CURVATURE graph: {data.edge_index.shape[1]} edges")
    return data


# ─── Main ─────────────────────────────────────────────────────────────────────

def compute_all():
    print("[02_curvature] Loading data...")
    X_all = np.load(ARTIFACTS / "X_all.npy")
    y_all = np.load(ARTIFACTS / "y_all.npy")
    n = X_all.shape[0]

    print(f"[02_curvature] n={n}. Computing kNN distances (k={K_BASE})...")
    dists, indices = knn_distances_indices(X_all, K_BASE)

    # --- curvature proxy ---
    print("[02_curvature] Computing local PCA curvature proxy (kappa)...")
    kappa = local_pca_residual(X_all, indices, r=R_DIMS)
    assert kappa.std() > 0, "Curvature proxy has zero variance — check inputs!"
    print(f"[02_curvature] kappa: mean={kappa.mean():.4f} std={kappa.std():.4f} range=[{kappa.min():.4f},{kappa.max():.4f}]")

    # --- LID ---
    print("[02_curvature] Computing LID...")
    lid = local_intrinsic_dim(dists)

    # --- LOF ---
    print("[02_curvature] Computing LOF (may take ~30s)...")
    lof_model = LocalOutlierFactor(n_neighbors=K_BASE, novelty=False, n_jobs=-1)
    lof_model.fit(X_all)
    lof_scores = -lof_model.negative_outlier_factor_  # positive: higher = more outlier-like
    lof_scores = lof_scores.astype(np.float32)

    # --- local density ---
    print("[02_curvature] Computing local density...")
    density = local_density(dists)

    # --- Forman proxy ---
    print("[02_curvature] Computing Forman-style proxy...")
    forman = forman_proxy(indices)

    # --- multi-scale stability ---
    print("[02_curvature] Computing multi-scale stability (k=10,20,30)...")
    stability_results, kappas = multiscale_stability(X_all, r=R_DIMS)
    print("[02_curvature] Stability:", stability_results)

    # --- Save observer features array ---
    observer = np.stack([kappa, lid, lof_scores, density, forman], axis=1).astype(np.float32)
    np.save(ARTIFACTS / "observer_features.npy", observer)
    np.save(ARTIFACTS / "kappa.npy", kappa)

    # --- Curvature bins (on full dataset) ---
    low_thresh = np.percentile(kappa, 33)
    high_thresh = np.percentile(kappa, 67)
    bins = np.where(kappa <= low_thresh, "low",
           np.where(kappa <= high_thresh, "medium", "high"))

    df_bins = pd.DataFrame({
        "row_idx": np.arange(n),
        "kappa": kappa,
        "lid": lid,
        "lof": lof_scores,
        "density": density,
        "forman": forman,
        "curvature_bin": bins,
        "y": y_all,
    })
    df_bins.to_csv(ARTIFACTS / "curvature_bins.csv", index=False)
    print("[02_curvature] curvature_bins.csv saved.")

    # --- Save stability results ---
    pd.DataFrame([stability_results]).to_csv(ARTIFACTS / "curvature_stability.csv", index=False)

    # --- Build curvature-weighted graph ---
    build_curvature_graph(X_all, kappa)

    print("[02_curvature] Done. Observer features saved.")
    return observer, df_bins


if __name__ == "__main__":
    compute_all()
