"""
mnist_pipeline.py — Full MNIST-784 routing curvature pipeline.
"""

import sys, os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from torch_geometric.nn import SAGEConv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from _shared_routing_curvature.src.routing_models import ObserverRouter, ViewCombiner, NoRouterCombiner
from _shared_routing_curvature.src.observer import compute_observers, multiscale_stability, bin_curvature
from _shared_routing_curvature.src.graph_builder import build_knn_graph
from _shared_routing_curvature.src.eval_utils import eval_classification

EXP = Path(__file__).parent.parent
ART = EXP / "artifacts"; ART.mkdir(exist_ok=True)
FIG = EXP / "figures"; FIG.mkdir(exist_ok=True)
REP = EXP / "reports"; REP.mkdir(exist_ok=True)
SEED = 42; K = 15; NC = 10; torch.manual_seed(SEED)

def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# ═══════════════════════════ DATA ════════════════════════════════════════════

def load_data(n_subset=10000):
    print(f"[MN] Loading MNIST (n_subset={n_subset})...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data.astype(np.float32), mnist.target.astype(np.int64)
    if n_subset and n_subset < len(X):
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_subset, random_state=SEED)
        idx, _ = next(sss.split(X, y)); X, y = X[idx], y[idx]
    sc = StandardScaler(); X = sc.fit_transform(X).astype(np.float32)
    all_i = np.arange(len(X))
    tr_i, tmp_i = train_test_split(all_i, test_size=0.30, random_state=SEED, stratify=y)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, random_state=SEED, stratify=y[tmp_i])
    for nm, arr in [("X_all",X),("y_all",y),("idx_train",tr_i),("idx_val",va_i),("idx_test",te_i)]:
        np.save(ART/f"{nm}.npy", arr)
    print(f"[MN] Train:{len(tr_i)} Val:{len(va_i)} Test:{len(te_i)}")
    return X, y, tr_i, va_i, te_i


# ═══════════════════════════ VIEWS ═══════════════════════════════════════════

def spatial_block_features(X, n_blocks=16):
    n = X.shape[0]; imgs = X.reshape(n, 28, 28); s = 7
    blocks = [imgs[:, r*s:(r+1)*s, c*s:(c+1)*s].mean(axis=(1,2)) for r in range(4) for c in range(4)]
    return np.stack(blocks, axis=1).astype(np.float32)


def build_views(X, y):
    print("[MN] Building views...")
    g = build_knn_graph(X, X, y, K); torch.save(g, ART/"graph_FULL.pt")
    print(f"  FULL: {g.edge_index.shape[1]} edges")
    Xb = spatial_block_features(X)
    g = build_knn_graph(Xb, X, y, K); torch.save(g, ART/"graph_BLOCK.pt")
    print(f"  BLOCK: {g.edge_index.shape[1]} edges")
    pca = PCA(n_components=50, random_state=SEED)
    Xp = pca.fit_transform(X).astype(np.float32)
    np.save(ART/"X_pca.npy", Xp)
    g = build_knn_graph(Xp, X, y, K); torch.save(g, ART/"graph_PCA.pt")
    print(f"  PCA: {g.edge_index.shape[1]} edges  var={pca.explained_variance_ratio_.sum():.3f}")


# ═══════════════════════════ OBSERVERS ═══════════════════════════════════════

def compute_obs(X, y):
    Xp = np.load(ART/"X_pca.npy")
    print("[MN] Computing observers on PCA-50...")
    obs, kappa = compute_observers(Xp, k=K)
    np.save(ART/"observer_features.npy", obs); np.save(ART/"kappa.npy", kappa)
    stab = multiscale_stability(Xp); pd.DataFrame([stab]).to_csv(ART/"curvature_stability.csv", index=False)
    df = bin_curvature(kappa, y, {"lid":obs[:,1],"lof":obs[:,2],"density":obs[:,3],"label":y})
    df.to_csv(ART/"curvature_bins.csv", index=False)
    print(f"[MN] kappa std={kappa.std():.4f}")


# ═══════════════════════════ BASELINES ═══════════════════════════════════════

class MLP_Clf(nn.Module):
    def __init__(self, d, nc=NC):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(256,128), nn.ReLU(), nn.Linear(128, nc))
    def forward(self, x): return self.net(x)


def run_baselines(X, y, tr_i, va_i, te_i):
    results = []; dev = device()
    Xtr, Xv, Xte = X[tr_i], X[va_i], X[te_i]
    ytr, yv, yte = y[tr_i], y[va_i], y[te_i]
    # M0
    maj = np.bincount(ytr).argmax(); p0 = np.full(len(yte), maj)
    results.append(eval_classification("M0_Majority", yte, p0)); np.save(ART/"preds_M0_test.npy", p0)
    # M1 MLP
    model = MLP_Clf(X.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    from torch.utils.data import TensorDataset, DataLoader
    Xt = torch.tensor(Xtr).to(dev); yt_t = torch.tensor(ytr).to(dev)
    Xvt = torch.tensor(Xv).to(dev); yvt = torch.tensor(yv).to(dev)
    loader = DataLoader(TensorDataset(Xt, yt_t), batch_size=256, shuffle=True)
    best, bst, wait = 1e9, None, 0
    for ep in range(100):
        model.train()
        for xb, yb in loader: opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(Xvt), yvt).item()
        if vl < best: best=vl; bst={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=15: break
    model.load_state_dict(bst); model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte).to(dev)); p1 = logits.argmax(-1).cpu().numpy()
        pr1 = torch.softmax(logits,-1).cpu().numpy()
    results.append(eval_classification("M1_MLP", yte, p1, pr1)); np.save(ART/"preds_M1_test.npy", p1)
    # M2 HGBR
    clf = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.05, random_state=SEED, n_iter_no_change=15)
    clf.fit(np.vstack([Xtr,Xv]), np.concatenate([ytr,yv]))
    p2 = clf.predict(Xte); pr2 = clf.predict_proba(Xte)
    results.append(eval_classification("M2_HGBR", yte, p2, pr2)); np.save(ART/"preds_M2_test.npy", p2)
    # M3 XGBoost
    try:
        import xgboost as xgb
        xm = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
                                objective="multi:softmax", num_class=NC, random_state=SEED, verbosity=0, early_stopping_rounds=15)
        xm.fit(Xtr, ytr, eval_set=[(Xv,yv)], verbose=False)
        p3 = xm.predict(Xte); pr3 = xm.predict_proba(Xte)
        results.append(eval_classification("M3_XGBoost", yte, p3, pr3)); np.save(ART/"preds_M3_test.npy", p3)
    except Exception as e:
        print(f"  [M3] FAILED: {e}")
        results.append({"model":"M3_XGBoost","accuracy":float("nan"),"macro_f1":float("nan"),"log_loss":float("nan")})
    # M4 TabPFN
    try:
        os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET']='1'
        from tabpfn import TabPFNClassifier
        clf4 = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
        clf4.fit(Xtr[:1024], ytr[:1024]); p4 = clf4.predict(Xte[:128])
        results.append(eval_classification("M4_TabPFN_sub", yte[:128], p4))
        np.save(ART/"preds_M4_test_sub.npy", p4)
    except Exception as e:
        print(f"  [M4] FAILED: {e}")
        results.append({"model":"M4_TabPFN","accuracy":float("nan"),"macro_f1":float("nan"),"log_loss":float("nan"),"note":str(e)[:200]})
    return results


# ═══════════════════════════ GRAPHSAGE ═══════════════════════════════════════

class SAGEClf(nn.Module):
    def __init__(self, d, h=128, nc=NC):
        super().__init__()
        self.s1 = SAGEConv(d,h); self.bn1 = nn.BatchNorm1d(h)
        self.s2 = SAGEConv(h,h//2); self.bn2 = nn.BatchNorm1d(h//2)
        self.head = nn.Linear(h//2, nc)
    def forward(self, x, ei):
        x = F.relu(self.bn1(self.s1(x,ei))); x = F.dropout(x,0.1,self.training)
        x = F.relu(self.bn2(self.s2(x,ei))); return self.head(x)
    def encode(self, x, ei):
        x = F.relu(self.bn1(self.s1(x,ei))); x = F.dropout(x,0.1,self.training)
        return F.relu(self.bn2(self.s2(x,ei)))


def train_sage_clf(name, gpath, tr_i, va_i, te_i, y_all):
    dev = device(); data = torch.load(gpath, weights_only=False).to(dev)
    y_t = torch.tensor(y_all, dtype=torch.long).to(dev)
    model = SAGEClf(data.x.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(); best, bst, wait = 1e9, None, 0
    tr_t, va_t = torch.tensor(tr_i), torch.tensor(va_i)
    for ep in range(150):
        model.train(); opt.zero_grad()
        crit(model(data.x, data.edge_index)[tr_t], y_t[tr_t]).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(data.x, data.edge_index)[va_t], y_t[va_t]).item()
        if vl < best: best=vl; bst={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=20: break
    model.load_state_dict(bst); model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        proba = torch.softmax(logits,-1).cpu().numpy()
        preds = logits.argmax(-1).cpu().numpy()
        reps = model.encode(data.x, data.edge_index).cpu().numpy()
    np.save(ART/f"proba_{name}_all.npy", proba)
    np.save(ART/f"preds_{name}_test.npy", preds[te_i])
    np.save(ART/f"reps_{name}_all.npy", reps)
    torch.save(model.state_dict(), ART/f"model_{name}.pt")
    return eval_classification(name, y_all[te_i], preds[te_i], proba[te_i])


def run_graph_views(tr_i, va_i, te_i, y_all):
    results = []
    for tag, gn in [("M5_FULL","FULL"),("M5_BLOCK","BLOCK"),("M5_PCA","PCA")]:
        print(f"[MN] {tag}..."); results.append(train_sage_clf(tag, ART/f"graph_{gn}.pt", tr_i, va_i, te_i, y_all))
    return results


# ═══════════════════════════ ROUTING ═════════════════════════════════════════

VIEW_TAGS = ["M5_FULL", "M5_BLOCK", "M5_PCA"]
N_VIEWS = len(VIEW_TAGS); REP_DIM = 64  # h//2


def run_m6_uniform(te_i, y_all):
    proba = np.stack([np.load(ART/f"proba_{t}_all.npy") for t in VIEW_TAGS], axis=0).mean(0)
    preds = proba.argmax(-1)
    np.save(ART/"preds_M6_test.npy", preds[te_i])
    return eval_classification("M6_Uniform", y_all[te_i], preds[te_i], proba[te_i])


def run_m7_learned(tr_i, va_i, te_i, y_all):
    reps = np.stack([np.load(ART/f"reps_{t}_all.npy") for t in VIEW_TAGS], axis=1)  # [N,V,D]
    reps_t = torch.tensor(reps, dtype=torch.float32)
    y_t = torch.tensor(y_all, dtype=torch.long)
    dev = device()
    m = NoRouterCombiner(N_VIEWS, REP_DIM, NC).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3); crit = nn.CrossEntropyLoss()
    r = reps_t.to(dev); yd = y_t.to(dev)
    tr_t, va_t = torch.tensor(tr_i), torch.tensor(va_i)
    best, bst, wait = 1e9, None, 0
    for ep in range(500):
        m.train(); opt.zero_grad(); crit(m(r[tr_t]), yd[tr_t]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = crit(m(r[va_t]), yd[va_t]).item()
        if vl < best: best=vl; bst={k:v.clone() for k,v in m.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=30: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad(): proba = torch.softmax(m(r), -1).cpu().numpy()
    preds = proba.argmax(-1)
    np.save(ART/"preds_M7_test.npy", preds[te_i])
    torch.save(m.state_dict(), ART/"model_M7.pt")
    w = torch.softmax(m.logw, 0).detach().cpu().numpy()
    print(f"  [M7] Weights: {dict(zip(VIEW_TAGS, [f'{v:.3f}' for v in w]))}")
    return eval_classification("M7_Learned", y_all[te_i], preds[te_i], proba[te_i])


def run_m8_routed(tr_i, va_i, te_i, y_all):
    reps = np.stack([np.load(ART/f"reps_{t}_all.npy") for t in VIEW_TAGS], axis=1)
    obs = np.load(ART/"observer_features.npy")
    obs_sc = ((obs - obs[tr_i].mean(0)) / (obs[tr_i].std(0)+1e-8)).astype(np.float32)
    dev = device()

    class M8(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = ObserverRouter(4, N_VIEWS, 32)
            self.head = nn.Linear(REP_DIM, NC)
        def forward(self, reps, obs):
            pi, _ = self.router(obs)
            weighted = (reps * pi.unsqueeze(-1)).sum(1)
            return self.head(weighted)

    m = M8().to(dev)
    r_t = torch.tensor(reps, dtype=torch.float32).to(dev)
    o_t = torch.tensor(obs_sc).to(dev)
    y_t = torch.tensor(y_all, dtype=torch.long).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3); crit = nn.CrossEntropyLoss()
    tr_t, va_t = torch.tensor(tr_i), torch.tensor(va_i)
    best, bst, wait = 1e9, None, 0
    for ep in range(500):
        m.train(); opt.zero_grad(); crit(m(r_t[tr_t], o_t[tr_t]), y_t[tr_t]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = crit(m(r_t[va_t], o_t[va_t]), y_t[va_t]).item()
        if vl < best: best=vl; bst={k:v.clone() for k,v in m.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=30: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad(): proba = torch.softmax(m(r_t, o_t), -1).cpu().numpy()
    preds = proba.argmax(-1)
    np.save(ART/"preds_M8_test.npy", preds[te_i])
    return eval_classification("M8_Routed", y_all[te_i], preds[te_i], proba[te_i])


def run_m9_routed_gate(tr_i, va_i, te_i, y_all):
    reps = np.stack([np.load(ART/f"reps_{t}_all.npy") for t in VIEW_TAGS], axis=1)
    obs = np.load(ART/"observer_features.npy")
    obs_sc = ((obs - obs[tr_i].mean(0)) / (obs[tr_i].std(0)+1e-8)).astype(np.float32)
    dev = device()

    class M9(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = ObserverRouter(4, N_VIEWS, 32)
            self.combiner = ViewCombiner(REP_DIM, N_VIEWS, NC)
        def forward(self, reps, obs):
            pi, beta = self.router(obs)
            return self.combiner(reps, pi, beta)

    m = M9().to(dev)
    r_t = torch.tensor(reps, dtype=torch.float32).to(dev)
    o_t = torch.tensor(obs_sc).to(dev)
    y_t = torch.tensor(y_all, dtype=torch.long).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3); crit = nn.CrossEntropyLoss()
    tr_t, va_t = torch.tensor(tr_i), torch.tensor(va_i)
    best, bst, wait = 1e9, None, 0
    for ep in range(500):
        m.train(); opt.zero_grad(); crit(m(r_t[tr_t], o_t[tr_t]), y_t[tr_t]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = crit(m(r_t[va_t], o_t[va_t]), y_t[va_t]).item()
        if vl < best: best=vl; bst={k:v.clone() for k,v in m.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=30: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad(): proba = torch.softmax(m(r_t, o_t), -1).cpu().numpy()
    preds = proba.argmax(-1)
    np.save(ART/"preds_M9_test.npy", preds[te_i])
    return eval_classification("M9_RoutedGate", y_all[te_i], preds[te_i], proba[te_i])


# ═══════════════════════════ ANALYSIS ════════════════════════════════════════

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ALL_MODELS = ["M0_Majority","M1_MLP","M2_HGBR","M3_XGBoost","M4_TabPFN_sub",
              "M5_FULL","M5_BLOCK","M5_PCA","M6_Uniform","M7_Learned","M8_Routed","M9_RoutedGate"]


def run_analysis(y_all, te_i):
    y_te = y_all[te_i]; kappa = np.load(ART/"kappa.npy")
    df_bins = pd.read_csv(ART/"curvature_bins.csv")
    stab_df = pd.read_csv(ART/"curvature_stability.csv")
    preds = {}
    for nm in ALL_MODELS:
        for suf in ["test","test_sub"]:
            p = ART/f"preds_{nm}_{suf}.npy" if suf=="test_sub" else ART/f"preds_{nm}_test.npy"
            if p.exists(): preds[nm] = np.load(p); break
    rows = []
    for nm, p in preds.items():
        yt = y_te[:len(p)]
        rows.append(eval_classification(nm, yt, p))
    mdf = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    mdf.to_csv(ART/"metrics.csv", index=False)
    print("[MN] Metrics:\n", mdf[["model","accuracy","macro_f1"]].to_string(index=False))

    # bin analysis
    df_te = df_bins.iloc[te_i].copy().reset_index(drop=True)
    bin_rows = []
    for nm in preds:
        if "sub" in nm: continue
        p = preds[nm]
        if len(p) != len(y_te): continue
        for b in ["low","medium","high"]:
            mask = df_te.curvature_bin.values == b
            if mask.sum()==0: continue
            r = eval_classification(f"{nm}_{b}", y_te[mask], p[mask])
            r.update({"model":nm,"bin":b,"n_rows":int(mask.sum())}); bin_rows.append(r)
    bdf = pd.DataFrame(bin_rows); bdf.to_csv(ART/"metrics_by_bin.csv", index=False)

    # verdict
    def get_acc(m): r=mdf[mdf.model==m]; return r.accuracy.values[0] if len(r) else None
    mlp = get_acc("M1_MLP"); m7 = get_acc("M7_Learned")
    m8 = get_acc("M8_Routed"); m9 = get_acc("M9_RoutedGate")
    best_sage = max([get_acc(m) for m in ["M5_FULL","M5_BLOCK","M5_PCA"] if get_acc(m)], default=None)
    routing_helps = m8 is not None and m7 is not None and m8 > m7
    gate_helps = m9 is not None and m8 is not None and m9 > m8
    graph_beats_mlp = best_sage is not None and mlp is not None and best_sage > mlp

    if routing_helps and graph_beats_mlp:
        verdict = "Hidden geometry exploited through routing"
    elif graph_beats_mlp and not routing_helps:
        verdict = "View complementarity present, curvature routing weak"
    elif not graph_beats_mlp and not routing_helps:
        verdict = "Warning sign: routing failed to exploit hidden geometry meaningfully"
    else:
        verdict = "Hidden geometry only partially exploited"

    # figures
    fig, ax = plt.subplots(figsize=(10,5))
    sub = mdf[~mdf.model.str.contains("TabPFN")]
    ax.barh(sub.model, sub.accuracy, color=["#DD8452" if "Rout" in m else "#4C72B0" for m in sub.model])
    ax.set_xlabel("Accuracy"); ax.set_title("MNIST Routing Curvature"); plt.tight_layout()
    plt.savefig(FIG/"model_comparison.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(kappa, bins=60, color="#4C72B0", alpha=0.8); ax.set_xlabel("kappa"); ax.set_title("Curvature Dist")
    plt.tight_layout(); plt.savefig(FIG/"curvature_hist.png", dpi=150); plt.close()

    key = [m for m in ["M1_MLP","M5_FULL","M7_Learned","M8_Routed","M9_RoutedGate"] if m in preds and len(preds[m])==len(y_te)]
    if key and not bdf.empty:
        fig, ax = plt.subplots(figsize=(12,5))
        pal = sns.color_palette("muted", len(key)); x = np.arange(3); w = 0.8/len(key)
        for i, nm in enumerate(key):
            sub2 = bdf[bdf.model==nm]
            vals = [sub2[sub2.bin==b].accuracy.values[0] if (sub2.bin==b).any() else 0 for b in ["low","medium","high"]]
            ax.bar(x+i*w-0.4+w/2, vals, w, label=nm, color=pal[i])
        ax.set_xticks(x); ax.set_xticklabels(["Low κ","Medium κ","High κ"])
        ax.set_ylabel("Accuracy"); ax.legend(fontsize=7); plt.tight_layout()
        plt.savefig(FIG/"error_by_bin.png", dpi=150); plt.close()

    if not stab_df.empty:
        fig, ax = plt.subplots(figsize=(8,3))
        keys = [c for c in stab_df.columns if "spearman" in c or "overlap" in c]
        vals = [stab_df[k].values[0] for k in keys]
        ax.barh(keys, vals); ax.set_xlim(0,1.1); ax.set_title("Curvature Stability")
        plt.tight_layout(); plt.savefig(FIG/"stability.png", dpi=150); plt.close()

    import datetime
    lines = [
        f"# MNIST-784: Routing Curvature Report",
        f"*{datetime.datetime.now().strftime('%Y-%m-%d')}* | Branch: `feature/routing-curvature-dual-datasets`\n",
        "> Curvature used as ROUTING PRIOR only.\n",
        "## Results\n", mdf[["model","accuracy","macro_f1"]].to_markdown(index=False),
        f"\n## Verdict: **{verdict.upper()}**\n",
        f"routing_helps={routing_helps}, gate_helps={gate_helps}, graph>mlp={graph_beats_mlp}\n",
        "## Curvature", f"kappa mean={kappa.mean():.4f} std={kappa.std():.4f}\n",
        "## Stability\n", stab_df.to_markdown(index=False),
        "\n## Per-Bin Accuracy\n", bdf[["model","bin","accuracy","n_rows"]].to_markdown(index=False) if not bdf.empty else "N/A",
    ]
    (REP/"report.md").write_text("\n".join(lines))
    print(f"\n[MN] VERDICT: {verdict.upper()}")


# ═══════════════════════════ MAIN ════════════════════════════════════════════

def run_mnist(n_subset=10000):
    t0 = time.time()
    X, y, tr_i, va_i, te_i = load_data(n_subset)
    build_views(X, y)
    compute_obs(X, y)
    run_baselines(X, y, tr_i, va_i, te_i)
    run_graph_views(tr_i, va_i, te_i, y)
    print("[MN] M6 Uniform..."); run_m6_uniform(te_i, y)
    print("[MN] M7 Learned..."); run_m7_learned(tr_i, va_i, te_i, y)
    print("[MN] M8 Routed..."); run_m8_routed(tr_i, va_i, te_i, y)
    print("[MN] M9 RoutedGate..."); run_m9_routed_gate(tr_i, va_i, te_i, y)
    run_analysis(y, te_i)
    print(f"[MN] Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    p.add_argument("--n", type=int, default=10000)
    a = p.parse_args()
    run_mnist(None if a.full else a.n)
