# ============================================================================
# CELL CB-1 -- ACCOUNT-CLUSTER bootstrap for the burden ratio R_{S,L} (HI-Small)
# Addresses the dependence flaw: the delta-method / txn bootstrap treat innocent
# transactions as independent, but transactions cluster within accounts. This
# cell resamples ACCOUNTS within each legal-form stratum (carrying all their
# transactions), recomputes R end-to-end on each resample with the deployed
# threshold HELD FIXED, and compares against the independent-binomial delta
# method on the SAME cells. Reuses EB1's exact loader so the point estimates
# reproduce the locked anchors (GATE).
# Output: per detector x budget -> R, naive 95% CI + p, cluster 95% CI + p,
# design effect, BH(0.05) significance naive-vs-cluster. Saves a CSV.
# CPU fine; ~2-5 min at B=2000. Paste the full printed report.
# ============================================================================
import os, glob, numpy as np, pandas as pd
from math import erfc, sqrt, log, exp
from google.colab import drive
if not os.path.ismount("/content/drive"): drive.mount("/content/drive")

GD    = "/content/drive/MyDrive/Kaggle/gnn_outputs"
CACHE = "/content/drive/MyDrive/Kaggle/aml_cache"
HI_VAL  = (3_248_921, 4_214_444); HI_TEST = (4_214_445, 5_078_344)
ANCH = {"thin": 1.125297147359424, "rich": 2.9984894229358963, "gnn": 2.483790728335002}
# locked R grid (seed-1 gnn) for soft cross-checks at 1/5/20%:
GRID = {"thin":[1.04,1.04,1.13,1.27], "rich":[1.22,2.69,3.00,2.07], "gnn":[0.8044,2.3355,2.4838,1.6897]}
BUDGETS = [0.01, 0.05, 0.10, 0.20]
B = 2000
SEED = 42
rng = np.random.default_rng(SEED)

# ---- data (identical to EB1) ----
fmt = pd.read_csv(f"{CACHE}/Small_HI_formatted_transactions.csv",
                  usecols=["from_id", "Is Laundering"])
assert len(fmt) == 5_078_345, f"formatted rows {len(fmt):,}"
from_id_all = fmt["from_id"].values
y_hi = fmt["Is Laundering"].values.astype("int8")
g_hi = np.load(f"{CACHE}/g_sorted_hismall.npy")
assert len(g_hi) == len(fmt)
print(f"[cache] formatted {len(fmt):,} rows | groups loaded")

def find_one(pat):
    c = sorted(glob.glob(pat), key=lambda s: (len(os.path.basename(s)), s))
    assert c, f"no file matching {pat}"
    return c[0]

def load_scores(path, lo, hi):
    d = pd.read_csv(path)
    ic = next((c for c in ("gid","EdgeID","te_global_idx","va_global_idx",
                           "global_idx","idx") if c in d.columns), None)
    assert ic is not None, f"{path}: no index col in {list(d.columns)}"
    v = d[ic].values.astype(np.int64)
    gid = v if (v.min() >= lo and v.max() <= hi) else v + lo
    pc = next(c for c in ("prob","score","p","pred") if c in d.columns)
    yc = next((c for c in ("y","label","Is Laundering") if c in d.columns), None)
    y = d[yc].values.astype("int8") if yc else y_hi[gid]
    out = (pd.DataFrame({"gid": gid, "prob": d[pc].values, "y": y})
           .sort_values("gid").reset_index(drop=True))
    assert out.gid.min() >= lo and out.gid.max() <= hi and out.gid.is_unique, path
    assert (out.y.values == y_hi[out.gid.values]).mean() > 0.999, f"{path}: y mismatch"
    return out

DETS = {
 "thin": (find_one(f"{GD}/thin*val*score*.csv"), find_one(f"{GD}/thin*test*score*.csv")),
 "rich": (find_one(f"{GD}/rich*val*score*.csv"), find_one(f"{GD}/rich*test*score*.csv")),
 "gnn":  (find_one(f"{GD}/gnn_val_scores*.csv"), find_one(f"{GD}/gnn_test_scores*.csv")),
}

def norm_sf(z):                       # two-sided normal tail, no scipy dependency
    return erfc(abs(z)/sqrt(2.0))

def bh(pvals, alpha=0.05):            # Benjamini-Hochberg; returns boolean reject vector
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    thr = alpha*(np.arange(1, m+1))/m
    passed = p[order] <= thr
    k = np.where(passed)[0]
    rej = np.zeros(m, bool)
    if len(k):
        cut = order[:k.max()+1]; rej[cut] = True
    return rej

rows = []
boot = {}    # (det,b) -> array of log R*
for det, (vp, tp) in DETS.items():
    va = load_scores(vp, *HI_VAL); te = load_scores(tp, *HI_TEST)
    p, y, gid = te.prob.values, te.y.values, te.gid.values
    g = g_hi[gid]; fid = from_id_all[gid]
    taus = {b: np.quantile(va.prob.values, 1-b) for b in BUDGETS}

    # per-account sufficient stats per stratum (S=0, L=2)
    stat = {}
    for code in (0, 2):
        m = (g == code)
        sub = pd.DataFrame({"fid": fid[m], "y": y[m], "p": p[m]})
        gb = sub.groupby("fid")
        acc = pd.DataFrame({"n": gb.size(),
                            "pos": gb["y"].sum(),
                            "n0": gb.apply(lambda d: int((d.y == 0).sum()))})
        fб = {}
        for b in BUDGETS:
            al = (sub.y.values == 0) & (sub.p.values >= taus[b])
            fб[b] = (pd.Series(al.astype(np.int64), index=sub.fid.values)
                       .groupby(level=0).sum().reindex(acc.index).fillna(0).values)
        stat[code] = dict(n=acc["n"].values.astype(np.float64),
                          pos=acc["pos"].values.astype(np.float64),
                          n0=acc["n0"].values.astype(np.float64),
                          f={b: fб[b].astype(np.float64) for b in BUDGETS},
                          k=len(acc))
    S, L = stat[0], stat[2]
    print(f"\n[{det}] S accounts {S['k']:,} | L accounts {L['k']:,} | "
          f"S innocent-txn {int(S['n0'].sum()):,} | L innocent-txn {int(L['n0'].sum()):,}")

    # point R per budget + gate
    point = {}
    for j, b in enumerate(BUDGETS):
        FPR_S = S["f"][b].sum()/S["n0"].sum(); FPR_L = L["f"][b].sum()/L["n0"].sum()
        piS = S["pos"].sum()/S["n"].sum();     piL = L["pos"].sum()/L["n"].sum()
        R = (FPR_S/FPR_L)/(piS/piL)
        point[b] = dict(R=R, FPR_S=FPR_S, FPR_L=FPR_L, piS=piS, piL=piL)
        tag = "" if b != 0.10 else (" [GATE OK]" if abs(R-ANCH[det])/ANCH[det] < 1e-3 else " [GATE FAIL]")
        print(f"   R@{int(b*100):>2}% = {R:.4f}   (locked grid ~{GRID[det][j]}){tag}")
    assert abs(point[0.10]["R"]-ANCH[det])/ANCH[det] < 1e-3, f"{det} gate fail"

    # cluster bootstrap (common random accounts reused across budgets)
    logs = {b: np.empty(B) for b in BUDGETS}
    nS, nL = S["k"], L["k"]
    for it in range(B):
        iS = rng.integers(0, nS, nS); iL = rng.integers(0, nL, nL)
        dS_pos = S["pos"][iS].sum(); dS_n = S["n"][iS].sum(); dS_n0 = S["n0"][iS].sum()
        dL_pos = L["pos"][iL].sum(); dL_n = L["n"][iL].sum(); dL_n0 = L["n0"][iL].sum()
        piS = dS_pos/dS_n; piL = dL_pos/dL_n
        for b in BUDGETS:
            fs = S["f"][b][iS].sum(); fl = L["f"][b][iL].sum()
            if fs == 0 or fl == 0 or piS == 0 or piL == 0:
                logs[b][it] = np.nan; continue
            logs[b][it] = log(((fs/dS_n0)/(fl/dL_n0))/(piS/piL))
    boot[det] = logs

    for b in BUDGETS:
        pt = point[b]; R = pt["R"]
        # naive delta-method log-R variance (independent binomial)
        var_naive = ((1-pt["FPR_S"])/(S["n0"].sum()*pt["FPR_S"])
                     + (1-pt["FPR_L"])/(L["n0"].sum()*pt["FPR_L"])
                     + (1-pt["piS"])/(S["n"].sum()*pt["piS"])
                     + (1-pt["piL"])/(L["n"].sum()*pt["piL"]))
        se_naive = sqrt(var_naive)
        ci_naive = (exp(log(R)-1.96*se_naive), exp(log(R)+1.96*se_naive))
        z_naive  = log(R)/se_naive; p_naive = norm_sf(z_naive)
        # cluster bootstrap
        lr = logs[b][~np.isnan(logs[b])]
        se_clu = lr.std(ddof=1)
        ci_clu = (exp(np.percentile(lr, 2.5)), exp(np.percentile(lr, 97.5)))
        z_clu  = log(R)/se_clu; p_clu = norm_sf(z_clu)
        deff   = (se_clu/se_naive)**2
        rows.append(dict(detector=det, budget=b, R=R,
                         naive_lo=ci_naive[0], naive_hi=ci_naive[1], p_naive=p_naive,
                         clu_lo=ci_clu[0], clu_hi=ci_clu[1], p_cluster=p_clu,
                         se_naive=se_naive, se_cluster=se_clu, design_effect=deff,
                         n_boot=int(len(lr))))

res = pd.DataFrame(rows)
res["bh_naive"]   = bh(res["p_naive"].values, 0.05)
res["bh_cluster"] = bh(res["p_cluster"].values, 0.05)
res.to_csv(f"{GD}/cluster_bootstrap_hismall.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 30)
print("\n================ R_{S,L}: naive (independent) vs account-cluster ================")
for _, r in res.iterrows():
    print(f"{r.detector:>4} @{int(r.budget*100):>2}%  R={r.R:5.2f} | "
          f"naive [{r.naive_lo:4.2f},{r.naive_hi:5.2f}] p={r.p_naive:7.1e} "
          f"{'*' if r.bh_naive else ' '} | "
          f"cluster [{r.clu_lo:4.2f},{r.clu_hi:5.2f}] p={r.p_cluster:7.1e} "
          f"{'*' if r.bh_cluster else ' '} | deff={r.design_effect:5.1f}")

nN, nC = int(res.bh_naive.sum()), int(res.bh_cluster.sum())
flips = res[res.bh_naive & ~res.bh_cluster]
print(f"\nBH(0.05) over {len(res)} HI-Small S/L cells: naive {nN} sig -> cluster {nC} sig")
if len(flips):
    print("Cells that LOSE significance under clustering:")
    for _, r in flips.iterrows():
        print(f"   {r.detector} @{int(r.budget*100)}%  (R={r.R:.2f}, deff={r.design_effect:.1f})")
else:
    print("No cell loses significance under clustering.")
print(f"median design effect = {res.design_effect.median():.1f} "
      f"(>1 means independence understated variance)")
print("\nNOTE: family here = HI-Small S-vs-L, 3 detectors x 4 budgets = 12 cells. "
      "The full 24-cell BH (currency contexts / SAML-D) should be recomputed with "
      "these clustered p-values once those cells' per-account stats are appended.")
print("[saved] cluster_bootstrap_hismall.csv -> gnn_outputs")
print("[DONE CB-1]")
