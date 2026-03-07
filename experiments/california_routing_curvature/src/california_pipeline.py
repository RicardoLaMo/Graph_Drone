"""
california_pipeline.py — Full California Housing routing curvature pipeline.

Stages: data → views → observers → baselines → graph models → routing → analysis
"""

import sys, time, json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from torch_geometric.nn import SAGEConv

# Add shared code
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from _shared_routing_curvature.src.routing_models import ObserverRouter, ViewCombiner, NoRouterCombiner
from _shared_routing_curvature.src.observer import compute_observers, multiscale_stability, bin_curvature
from _shared_routing_curvature.src.graph_builder import build_knn_graph
from _shared_routing_curvature.src.eval_utils import eval_regression

EXP = Path(__file__).parent.parent
ART = EXP / "artifacts"; ART.mkdir(exist_ok=True)
FIG = EXP / "figures"; FIG.mkdir(exist_ok=True)
REP = EXP / "reports"; REP.mkdir(exist_ok=True)
SEED = 42; K = 15; torch.manual_seed(SEED)

def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# ═══════════════════════════ STAGE 0: DATA ═══════════════════════════════════

def load_data():
    print("[CA] Loading California Housing...")
    cal = fetch_california_housing()
    X, y = cal.data.astype(np.float32), cal.target.astype(np.float32)
    # log1p skewed features
    for j, name in enumerate(cal.feature_names):
        if name in ("Population", "AveOccup", "AveRooms"):
            X[:, j] = np.log1p(X[:, j])
    sc = RobustScaler(); X = sc.fit_transform(X).astype(np.float32)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
    Xv, Xte, yv, yte = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=SEED)
    all_idx = np.arange(len(X))
    tr_i, tmp_i = train_test_split(all_idx, test_size=0.30, random_state=SEED)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, random_state=SEED)
    for nm, arr in [("X_all",X),("y_all",y),("X_train",Xtr),("X_val",Xv),("X_test",Xte),
                    ("y_train",ytr),("y_val",yv),("y_test",yte),
                    ("idx_train",tr_i),("idx_val",va_i),("idx_test",te_i)]:
        np.save(ART/f"{nm}.npy", arr)
    fnames = list(cal.feature_names)
    json.dump(fnames, open(ART/"feature_names.json","w"))
    print(f"[CA] Train:{Xtr.shape} Val:{Xv.shape} Test:{Xte.shape}")
    return X, y, Xtr, Xv, Xte, ytr, yv, yte, tr_i, va_i, te_i, fnames


# ═══════════════════════════ STAGE 1: VIEWS ══════════════════════════════════

GEO_COLS = [6, 7]  # Latitude, Longitude
SOCIO_COLS = [0, 1, 2, 3, 4]  # MedInc, HouseAge, AveRooms, AveBedrms, AveOccup

def build_views(X, y):
    print("[CA] Building views...")
    graphs = {}
    for tag, cols in [("FULL", None), ("GEO", GEO_COLS), ("SOCIO", SOCIO_COLS)]:
        Xt = X if cols is None else X[:, cols]
        g = build_knn_graph(Xt, X, y, k=K)
        torch.save(g, ART/f"graph_{tag}.pt")
        graphs[tag] = g
        print(f"  {tag}: {g.edge_index.shape[1]} edges")
    # LOWRANK
    pca = PCA(n_components=4, random_state=SEED)
    Xp = pca.fit_transform(X).astype(np.float32)
    g = build_knn_graph(Xp, X, y, k=K)
    torch.save(g, ART/"graph_LOWRANK.pt"); graphs["LOWRANK"] = g
    np.save(ART/"X_pca.npy", Xp)
    print(f"  LOWRANK: {g.edge_index.shape[1]} edges  (var={pca.explained_variance_ratio_.sum():.3f})")
    return graphs


# ═══════════════════════════ STAGE 2: OBSERVERS ══════════════════════════════

def compute_obs(X, y):
    print("[CA] Computing observers on PCA space for stability...")
    Xp = np.load(ART/"X_pca.npy") if (ART/"X_pca.npy").exists() else X
    obs, kappa = compute_observers(Xp, k=K)
    np.save(ART/"observer_features.npy", obs)
    np.save(ART/"kappa.npy", kappa)
    stab = multiscale_stability(Xp)
    pd.DataFrame([stab]).to_csv(ART/"curvature_stability.csv", index=False)
    df = bin_curvature(kappa, y)
    obs_cols = {"lid": obs[:,1], "lof": obs[:,2], "density": obs[:,3]}
    for k2, v in obs_cols.items(): df[k2] = v
    df.to_csv(ART/"curvature_bins.csv", index=False)
    print(f"[CA] Observers done. kappa std={kappa.std():.4f}")
    return obs, kappa


# ═══════════════════════════ STAGE 3: BASELINES ══════════════════════════════

class MLP_Reg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,128), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x): return self.net(x).squeeze(-1)


def run_baselines(Xtr, Xv, Xte, ytr, yv, yte):
    results = []
    # C0
    pred0 = np.full(len(yte), ytr.mean())
    results.append(eval_regression("C0_Mean", yte, pred0))
    np.save(ART/"preds_C0_test.npy", pred0)
    # C1 MLP
    dev = device()
    model = MLP_Reg(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt, yt = torch.tensor(Xtr).to(dev), torch.tensor(ytr).to(dev)
    Xvt, yvt = torch.tensor(Xv).to(dev), torch.tensor(yv).to(dev)
    best, bst, wait = 1e9, None, 0
    for ep in range(100):
        model.train(); opt.zero_grad(); nn.MSELoss()(model(Xt), yt).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = nn.MSELoss()(model(Xvt), yvt).item()
        if vl < best: best=vl; bst={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=15: break
    model.load_state_dict(bst); model.eval()
    with torch.no_grad(): p1 = model(torch.tensor(Xte).to(dev)).cpu().numpy()
    results.append(eval_regression("C1_MLP", yte, p1)); np.save(ART/"preds_C1_test.npy", p1)
    # C2 HGBR
    clf = HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05,
                                        random_state=SEED, n_iter_no_change=15)
    clf.fit(np.vstack([Xtr,Xv]), np.concatenate([ytr,yv]))
    p2 = clf.predict(Xte)
    results.append(eval_regression("C2_HGBR", yte, p2)); np.save(ART/"preds_C2_test.npy", p2)
    # C3 XGBoost
    try:
        import xgboost as xgb
        xm = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                               tree_method="hist", random_state=SEED, verbosity=0, early_stopping_rounds=15)
        xm.fit(Xtr, ytr, eval_set=[(Xv,yv)], verbose=False)
        p3 = xm.predict(Xte)
        results.append(eval_regression("C3_XGBoost", yte, p3)); np.save(ART/"preds_C3_test.npy", p3)
    except Exception as e:
        print(f"  [C3_XGBoost] FAILED: {e}")
        results.append({"model":"C3_XGBoost","rmse":float("nan"),"mae":float("nan"),"r2":float("nan")})
    # C4 TabPFN
    try:
        import os; os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET']='1'
        from tabpfn import TabPFNRegressor
        MAX_TR = 1024
        clf4 = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
        clf4.fit(Xtr[:MAX_TR], ytr[:MAX_TR])
        p4 = clf4.predict(Xte[:256])
        results.append(eval_regression("C4_TabPFN_sub", yte[:256], p4))
        np.save(ART/"preds_C4_test_sub.npy", p4)
    except Exception as e:
        print(f"  [C4_TabPFN] FAILED: {e}")
        results.append({"model":"C4_TabPFN","rmse":float("nan"),"mae":float("nan"),"r2":float("nan"),
                         "note":str(e)[:200]})
    print("[CA] Baselines done.")
    return results


# ═══════════════════════════ STAGE 4: GRAPHSAGE ══════════════════════════════

class SAGEReg(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.s1 = SAGEConv(d, h); self.s2 = SAGEConv(h, h//2); self.head = nn.Linear(h//2, 1)
    def forward(self, x, ei):
        x = F.relu(self.s1(x, ei)); x = F.dropout(x, 0.1, self.training)
        x = F.relu(self.s2(x, ei)); return self.head(x).squeeze(-1)
    def encode(self, x, ei):
        x = F.relu(self.s1(x, ei)); x = F.dropout(x, 0.1, self.training)
        return F.relu(self.s2(x, ei))  # [N, h//2] representations


def train_sage_reg(name, graph_path, tr_i, va_i, te_i, y_all):
    dev = device()
    data = torch.load(graph_path, weights_only=False).to(dev)
    y_t = torch.tensor(y_all, dtype=torch.float32).to(dev)
    model = SAGEReg(data.x.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(); best, bst, wait = 1e9, None, 0
    tr_t, va_t = torch.tensor(tr_i), torch.tensor(va_i)
    for ep in range(200):
        model.train(); opt.zero_grad()
        loss_fn(model(data.x, data.edge_index)[tr_t], y_t[tr_t]).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = loss_fn(model(data.x, data.edge_index)[va_t], y_t[va_t]).item()
        if vl < best: best=vl; bst={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=20: break
    model.load_state_dict(bst); model.eval()
    with torch.no_grad():
        preds_all = model(data.x, data.edge_index).cpu().numpy()
        reps_all = model.encode(data.x, data.edge_index).cpu().numpy()
    np.save(ART/f"preds_{name}_all.npy", preds_all)
    np.save(ART/f"preds_{name}_test.npy", preds_all[te_i])
    np.save(ART/f"reps_{name}_all.npy", reps_all)
    torch.save(model.state_dict(), ART/f"model_{name}.pt")
    return eval_regression(name, y_all[te_i], preds_all[te_i]), reps_all


def run_graph_views(tr_i, va_i, te_i, y_all):
    results = []
    for tag, gname in [("C5_FULL","FULL"),("C5_GEO","GEO"),("C5_SOCIO","SOCIO"),("C5_LOWRANK","LOWRANK")]:
        print(f"[CA] {tag}...")
        m, _ = train_sage_reg(tag, ART/f"graph_{gname}.pt", tr_i, va_i, te_i, y_all)
        results.append(m)
    return results


# ═══════════════════════════ STAGE 5: ROUTING ════════════════════════════════

VIEW_TAGS = ["C5_FULL", "C5_GEO", "C5_SOCIO", "C5_LOWRANK"]
N_VIEWS = len(VIEW_TAGS)
REP_DIM = 32  # h//2 from SAGEReg


def _load_reps_preds(y_all):
    reps = np.stack([np.load(ART/f"reps_{t}_all.npy") for t in VIEW_TAGS], axis=1)  # [N, V, D]
    preds = np.stack([np.load(ART/f"preds_{t}_all.npy") for t in VIEW_TAGS], axis=1)  # [N, V]
    return reps, preds


def run_c6_uniform(te_i, y_all):
    preds = np.stack([np.load(ART/f"preds_{t}_all.npy") for t in VIEW_TAGS], axis=1)
    p_uni = preds.mean(1)
    np.save(ART/"preds_C6_test.npy", p_uni[te_i])
    return eval_regression("C6_Uniform", y_all[te_i], p_uni[te_i])


def _train_combiner(name, model, inputs_fn, tr_i, va_i, te_i, y_all, lr=5e-3, epochs=500, pat=30):
    dev = device()
    model = model.to(dev)
    y_t = torch.tensor(y_all, dtype=torch.float32).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(); best, bst, wait = 1e9, None, 0
    all_inputs = inputs_fn(dev)
    tr_t, va_t = torch.tensor(tr_i), torch.tensor(va_i)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = all_inputs["forward_train"](model, tr_t)
        loss_fn(out, y_t[tr_t]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(all_inputs["forward_train"](model, va_t), y_t[va_t]).item()
        if vl < best: best=vl; bst={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=pat: break
    model.load_state_dict(bst); model.eval()
    with torch.no_grad():
        preds_all = all_inputs["forward_all"](model)
    p = preds_all.cpu().numpy()
    np.save(ART/f"preds_{name}_all.npy", p)
    np.save(ART/f"preds_{name}_test.npy", p[te_i])
    torch.save(model.state_dict(), ART/f"model_{name}.pt")
    return eval_regression(name, y_all[te_i], p[te_i])


def run_c7_learned(tr_i, va_i, te_i, y_all):
    reps, _ = _load_reps_preds(y_all)
    reps_t = torch.tensor(reps, dtype=torch.float32)
    m = NoRouterCombiner(N_VIEWS, REP_DIM, 1)
    def inputs_fn(dev):
        r = reps_t.to(dev)
        return {
            "forward_train": lambda model, idx: model(r[idx]).squeeze(-1),
            "forward_all": lambda model: model(r).squeeze(-1),
        }
    return _train_combiner("C7_Learned", m, inputs_fn, tr_i, va_i, te_i, y_all)


def run_c8_routed(tr_i, va_i, te_i, y_all):
    """Observer-routed combiner: router outputs pi, uses fixed beta=0 (isolation only)."""
    reps, _ = _load_reps_preds(y_all)
    obs = np.load(ART/"observer_features.npy")
    obs_sc = ((obs - obs[tr_i].mean(0)) / (obs[tr_i].std(0) + 1e-8)).astype(np.float32)
    reps_t = torch.tensor(reps, dtype=torch.float32)
    obs_t = torch.tensor(obs_sc, dtype=torch.float32)

    class C8Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = ObserverRouter(4, N_VIEWS, 32)
            self.head = nn.Linear(REP_DIM, 1)
        def forward(self, reps, obs):
            pi, _ = self.router(obs)  # ignore beta → pure isolation routing
            weighted = (reps * pi.unsqueeze(-1)).sum(1)
            return self.head(weighted).squeeze(-1)

    m = C8Model()
    def inputs_fn(dev):
        r, o = reps_t.to(dev), obs_t.to(dev)
        return {
            "forward_train": lambda model, idx: model(r[idx], o[idx]),
            "forward_all": lambda model: model(r, o),
        }
    return _train_combiner("C8_Routed", m, inputs_fn, tr_i, va_i, te_i, y_all)


def run_c9_routed_gate(tr_i, va_i, te_i, y_all):
    """Full routing: observer → pi + beta (isolation/interaction gate)."""
    reps, _ = _load_reps_preds(y_all)
    obs = np.load(ART/"observer_features.npy")
    obs_sc = ((obs - obs[tr_i].mean(0)) / (obs[tr_i].std(0) + 1e-8)).astype(np.float32)
    reps_t = torch.tensor(reps, dtype=torch.float32)
    obs_t = torch.tensor(obs_sc, dtype=torch.float32)

    class C9Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = ObserverRouter(4, N_VIEWS, 32)
            self.combiner = ViewCombiner(REP_DIM, N_VIEWS, 1)
        def forward(self, reps, obs):
            pi, beta = self.router(obs)
            return self.combiner(reps, pi, beta).squeeze(-1)

    m = C9Model()
    def inputs_fn(dev):
        r, o = reps_t.to(dev), obs_t.to(dev)
        return {
            "forward_train": lambda model, idx: model(r[idx], o[idx]),
            "forward_all": lambda model: model(r, o),
        }
    return _train_combiner("C9_RoutedGate", m, inputs_fn, tr_i, va_i, te_i, y_all)


# ═══════════════════════════ STAGE 6: ANALYSIS ═══════════════════════════════

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ALL_MODELS = ["C0_Mean","C1_MLP","C2_HGBR","C3_XGBoost","C4_TabPFN_sub",
              "C5_FULL","C5_GEO","C5_SOCIO","C5_LOWRANK",
              "C6_Uniform","C7_Learned","C8_Routed","C9_RoutedGate"]


def run_analysis(y_all, te_i):
    from sklearn.metrics import mean_squared_error
    y_te = y_all[te_i]
    df_bins = pd.read_csv(ART/"curvature_bins.csv")
    stab_df = pd.read_csv(ART/"curvature_stability.csv")
    kappa = np.load(ART/"kappa.npy")

    preds = {}
    for nm in ALL_MODELS:
        for suffix in ["test","test_sub"]:
            p = ART/f"preds_{nm}_{suffix}.npy" if suffix == "test_sub" else ART/f"preds_{nm}_test.npy"
            if p.exists(): preds[nm] = np.load(p); break

    rows = []
    for nm, p in preds.items():
        yt = y_te[:len(p)]
        rows.append(eval_regression(nm, yt, p))
    metrics_df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    metrics_df.to_csv(ART/"metrics.csv", index=False)
    print("[CA] Metrics:\n", metrics_df[["model","rmse","mae","r2"]].to_string(index=False))

    # bin analysis
    df_test = df_bins.iloc[te_i].copy().reset_index(drop=True)
    bin_rows = []
    for nm in preds:
        if nm.endswith("_sub"): continue
        if nm not in preds: continue
        p = preds[nm]
        if len(p) != len(y_te): continue
        for b in ["low","medium","high"]:
            mask = df_test["curvature_bin"].values == b
            if mask.sum() == 0: continue
            r =  eval_regression(f"{nm}_{b}", y_te[mask], p[mask])
            r.update({"model":nm,"bin":b,"n_rows":int(mask.sum())})
            bin_rows.append(r)
    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(ART/"metrics_by_bin.csv", index=False)

    # Verdict
    def get_rmse(m): r=metrics_df[metrics_df.model==m]; return r.rmse.values[0] if len(r) else None
    c2 = get_rmse("C2_HGBR"); c7 = get_rmse("C7_Learned")
    c8 = get_rmse("C8_Routed"); c9 = get_rmse("C9_RoutedGate")
    routing_helps = c8 is not None and c7 is not None and c8 < c7
    gate_helps = c9 is not None and c8 is not None and c9 < c8
    graph_beats_boost = min([get_rmse(m) for m in ALL_MODELS if get_rmse(m) and m.startswith("C5")], default=1e9) < (c2 or 1e9)

    if routing_helps and (c8 or 1e9) < (c2 or 1e9):
        verdict = "Local non-flatness helps routing on California"
    elif routing_helps:
        verdict = "Routing signal present, but graph stack still below strong tabular baseline"
    elif not routing_helps and not graph_beats_boost:
        verdict = "No justification for curvature routing on California"
    else:
        verdict = "Curvature-as-routing weak on California under current setup"

    reasons = [f"C2_HGBR={c2:.4f}" if c2 else "C2: N/A",
               f"C7_Learned={c7:.4f}" if c7 else "C7: N/A",
               f"C8_Routed={c8:.4f}" if c8 else "C8: N/A",
               f"C9_RoutedGate={c9:.4f}" if c9 else "C9: N/A",
               f"routing_helps={routing_helps}",
               f"gate_helps={gate_helps}",
               f"graph_beats_boost={graph_beats_boost}"]

    # Figures
    fig, ax = plt.subplots(figsize=(10,5))
    sub = metrics_df[~metrics_df.model.str.contains("TabPFN")]
    ax.barh(sub.model, sub.rmse, color=["#DD8452" if "Rout" in m else "#4C72B0" for m in sub.model])
    ax.set_xlabel("RMSE"); ax.set_title("California Routing Curvature"); plt.tight_layout()
    plt.savefig(FIG/"model_comparison.png", dpi=150); plt.close()

    # curvature hist
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(kappa, bins=60, color="#4C72B0", alpha=0.8); ax.set_xlabel("kappa"); ax.set_title("Curvature Distribution")
    plt.tight_layout(); plt.savefig(FIG/"curvature_hist.png", dpi=150); plt.close()

    # error by bin
    key = [m for m in ["C2_HGBR","C5_FULL","C7_Learned","C8_Routed","C9_RoutedGate"] if m in preds and len(preds[m])==len(y_te)]
    if key and not bin_df.empty:
        fig, ax = plt.subplots(figsize=(12,5))
        pal = sns.color_palette("muted", len(key)); x = np.arange(3); w = 0.8/len(key)
        for i, nm in enumerate(key):
            sub = bin_df[bin_df.model==nm]
            vals = [sub[sub.bin==b].rmse.values[0] if (sub.bin==b).any() else 0 for b in ["low","medium","high"]]
            ax.bar(x+i*w-0.4+w/2, vals, w, label=nm, color=pal[i])
        ax.set_xticks(x); ax.set_xticklabels(["Low κ","Medium κ","High κ"])
        ax.set_ylabel("RMSE"); ax.legend(fontsize=7); plt.tight_layout()
        plt.savefig(FIG/"error_by_bin.png", dpi=150); plt.close()

    # stability
    if not stab_df.empty:
        fig, ax = plt.subplots(figsize=(8,3))
        keys = [c for c in stab_df.columns if "spearman" in c or "overlap" in c]
        vals = [stab_df[k].values[0] for k in keys]
        ax.barh(keys, vals); ax.set_xlim(0,1.1); ax.set_title("Curvature Stability")
        plt.tight_layout(); plt.savefig(FIG/"stability.png", dpi=150); plt.close()

    # Write report
    import datetime
    lines = [
        f"# California Housing: Routing Curvature Report",
        f"*{datetime.datetime.now().strftime('%Y-%m-%d')}* | Branch: `feature/routing-curvature-dual-datasets`\n",
        "> Curvature used as ROUTING PRIOR only, not as direct predictor.\n",
        "## Results\n", metrics_df[["model","rmse","mae","r2"]].to_markdown(index=False),
        f"\n## Verdict: **{verdict.upper()}**\n",
        "```\n" + "\n".join(reasons) + "\n```\n",
        "## Curvature Stats",
        f"kappa mean={kappa.mean():.4f} std={kappa.std():.4f}\n",
        "## Stability\n", stab_df.to_markdown(index=False),
        "\n## Per-Bin RMSE\n", bin_df[["model","bin","rmse","n_rows"]].to_markdown(index=False) if not bin_df.empty else "N/A",
    ]
    (REP/"report.md").write_text("\n".join(lines))
    print(f"\n[CA] VERDICT: {verdict.upper()}")
    print(f"[CA] Report: {REP/'report.md'}")


# ═══════════════════════════ MAIN ════════════════════════════════════════════

def run_california():
    t0 = time.time()
    X, y, Xtr, Xv, Xte, ytr, yv, yte, tr_i, va_i, te_i, fnames = load_data()
    build_views(X, y)
    compute_obs(X, y)
    baselines = run_baselines(Xtr, Xv, Xte, ytr, yv, yte)
    graph_results = run_graph_views(tr_i, va_i, te_i, y)
    print("[CA] C6: Uniform...")
    run_c6_uniform(te_i, y)
    print("[CA] C7: Learned combiner (no observer)...")
    run_c7_learned(tr_i, va_i, te_i, y)
    print("[CA] C8: Observer-routed combiner...")
    run_c8_routed(tr_i, va_i, te_i, y)
    print("[CA] C9: Observer-routed + isolation/interaction gate...")
    run_c9_routed_gate(tr_i, va_i, te_i, y)
    run_analysis(y, te_i)
    print(f"[CA] Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run_california()
