# ============================================================================
# CELL CB-1c -- cluster bootstrap, BOTH inferential frames side by side.
# Decides how severe the replan must be after CB-1 (pi-random) collapsed.
#
#   FRAME A (pi random): resample accounts; recompute FPR_S,FPR_L,pi_S,pi_L on
#     each resample. = CB-1 v2 behavior (conservative; base rates estimated).
#   FRAME B (pi fixed): resample accounts for the FPR ratio only; hold the
#     base-rate ratio pi_S/pi_L at its full-sample value (base rates treated as
#     a KNOWN population normalization constant). Narrower; isolates the FPR
#     ratio the detector actually controls.
#
# Reporting both is the rigorous move; B is arguably the primary audit frame
# (prevalence is a population property, not the audit's sampling target), A is
# the conservative sensitivity bound. Reuses EB1/CB1 loader -> gates reproduce
# the locked anchors. CPU; ~3-5 min at B=2000. Paste the full report.
# ============================================================================
import os, glob, numpy as np, pandas as pd
from math import erfc, sqrt, log, exp
from google.colab import drive
if not os.path.ismount("/content/drive"): drive.mount("/content/drive")

GD    = "/content/drive/MyDrive/Kaggle/gnn_outputs"
CACHE = "/content/drive/MyDrive/Kaggle/aml_cache"
HI_VAL  = (3_248_921, 4_214_444); HI_TEST = (4_214_445, 5_078_344)
ANCH = {"thin": 1.125297147359424, "rich": 2.9984894229358963, "gnn": 2.483790728335002}
BUDGETS = [0.01, 0.05, 0.10, 0.20]
B = 2000
rng = np.random.default_rng(42)

fmt = pd.read_csv(f"{CACHE}/Small_HI_formatted_transactions.csv",
                  usecols=["from_id", "Is Laundering"])
assert len(fmt) == 5_078_345
from_id_all = fmt["from_id"].values
y_hi = fmt["Is Laundering"].values.astype("int8")
g_hi = np.load(f"{CACHE}/g_sorted_hismall.npy")
print(f"[cache] formatted {len(fmt):,} rows | groups loaded")

def find_one(pat):
    c = sorted(glob.glob(pat), key=lambda s: (len(os.path.basename(s)), s))
    assert c, f"no file matching {pat}"
    return c[0]

def load_scores(path, lo, hi):
    d = pd.read_csv(path)
    ic = next((c for c in ("gid","EdgeID","te_global_idx","va_global_idx","global_idx","idx") if c in d.columns), None)
    assert ic is not None, f"{path}: no index col"
    v = d[ic].values.astype(np.int64)
    gid = v if (v.min() >= lo and v.max() <= hi) else v + lo
    pc = next(c for c in ("prob","score","p","pred") if c in d.columns)
    yc = next((c for c in ("y","label","Is Laundering") if c in d.columns), None)
    y = d[yc].values.astype("int8") if yc else y_hi[gid]
    out = (pd.DataFrame({"gid": gid, "prob": d[pc].values, "y": y})
           .sort_values("gid").reset_index(drop=True))
    assert out.gid.min() >= lo and out.gid.max() <= hi and out.gid.is_unique, path
    return out

DETS = {
 "thin": (find_one(f"{GD}/thin*val*score*.csv"), find_one(f"{GD}/thin*test*score*.csv")),
 "rich": (find_one(f"{GD}/rich*val*score*.csv"), find_one(f"{GD}/rich*test*score*.csv")),
 "gnn":  (find_one(f"{GD}/gnn_val_scores*.csv"), find_one(f"{GD}/gnn_test_scores*.csv")),
}

def norm_sf(z): return erfc(abs(z)/sqrt(2.0))
def bh(p, alpha=0.05):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    passed = p[o] <= alpha*(np.arange(1, m+1))/m
    k = np.where(passed)[0]; rej = np.zeros(m, bool)
    if len(k): rej[o[:k.max()+1]] = True
    return rej

rows = []
for det, (vp, tp) in DETS.items():
    va = load_scores(vp, *HI_VAL); te = load_scores(tp, *HI_TEST)
    p, y, gid = te.prob.values, te.y.values, te.gid.values
    g = g_hi[gid]; fid = from_id_all[gid]
    taus = {b: np.quantile(va.prob.values, 1-b) for b in BUDGETS}
    stat = {}
    for code in (0, 2):
        m = (g == code)
        sub = pd.DataFrame({"fid": fid[m], "y": y[m].astype(np.int64), "p": p[m]})
        sub["inn"] = (sub["y"].values == 0).astype(np.int64)
        acc = sub.groupby("fid").agg(n=("y","size"), pos=("y","sum"), n0=("inn","sum"))
        fb = {}
        for b in BUDGETS:
            al = ((sub["y"].values == 0) & (sub["p"].values >= taus[b])).astype(np.int64)
            fb[b] = (pd.Series(al, index=sub["fid"].values).groupby(level=0).sum()
                       .reindex(acc.index).fillna(0).values.astype(np.float64))
        stat[code] = dict(n=acc["n"].values.astype(np.float64),
                          pos=acc["pos"].values.astype(np.float64),
                          n0=acc["n0"].values.astype(np.float64), f=fb, k=len(acc))
    S, L = stat[0], stat[2]
    piS_pt = S["pos"].sum()/S["n"].sum(); piL_pt = L["pos"].sum()/L["n"].sum()
    pir_pt = piS_pt/piL_pt
    print(f"\n[{det}] S acc {S['k']:,} | L acc {L['k']:,} | base-rate ratio piS/piL = {pir_pt:.4f}")

    point = {}
    for b in BUDGETS:
        FPR_S = S["f"][b].sum()/S["n0"].sum(); FPR_L = L["f"][b].sum()/L["n0"].sum()
        R = (FPR_S/FPR_L)/pir_pt
        point[b] = dict(R=R, FPR_S=FPR_S, FPR_L=FPR_L)
        tag = " [GATE OK]" if (b==0.10 and abs(R-ANCH[det])/ANCH[det] < 1e-3) else ""
        if b == 0.10: print(f"   R@10% {R:.4f} vs locked {ANCH[det]:.4f}{tag}")
    assert abs(point[0.10]["R"]-ANCH[det])/ANCH[det] < 1e-3

    logsA = {b: np.empty(B) for b in BUDGETS}   # pi random
    logsB = {b: np.empty(B) for b in BUDGETS}   # pi fixed
    nS, nL = S["k"], L["k"]
    for it in range(B):
        iS = rng.integers(0, nS, nS); iL = rng.integers(0, nL, nL)
        dS_pos = S["pos"][iS].sum(); dS_n = S["n"][iS].sum(); dS_n0 = S["n0"][iS].sum()
        dL_pos = L["pos"][iL].sum(); dL_n = L["n"][iL].sum(); dL_n0 = L["n0"][iL].sum()
        piS_b = dS_pos/dS_n; piL_b = dL_pos/dL_n
        for b in BUDGETS:
            fs = S["f"][b][iS].sum(); fl = L["f"][b][iL].sum()
            if fs == 0 or fl == 0:
                logsA[b][it] = np.nan; logsB[b][it] = np.nan; continue
            fpr_ratio = (fs/dS_n0)/(fl/dL_n0)
            logsB[b][it] = log(fpr_ratio/pir_pt)                       # pi fixed
            logsA[b][it] = (log(fpr_ratio/(piS_b/piL_b))
                            if (piS_b>0 and piL_b>0) else np.nan)      # pi random

    for b in BUDGETS:
        R = point[b]["R"]; lR = log(R)
        rec = dict(detector=det, budget=b, R=R)
        for frame, logs in (("rand", logsA), ("fix", logsB)):
            lr = logs[b][~np.isnan(logs[b])]
            se = lr.std(ddof=1)
            lo, hi = exp(np.percentile(lr, 2.5)), exp(np.percentile(lr, 97.5))
            rec[f"{frame}_lo"] = lo; rec[f"{frame}_hi"] = hi
            rec[f"{frame}_p"] = norm_sf(lR/se); rec[f"{frame}_se"] = se
        rows.append(rec)

res = pd.DataFrame(rows)
res["bh_rand"] = bh(res["rand_p"].values)
res["bh_fix"]  = bh(res["fix_p"].values)
res.to_csv(f"{GD}/cluster_bootstrap_both_frames_hismall.csv", index=False)

print("\n===== R_{S,L}: FRAME A (pi random) vs FRAME B (pi fixed = base rates known) =====")
for _, r in res.iterrows():
    print(f"{r.detector:>4} @{int(r.budget*100):>2}%  R={r.R:5.2f} | "
          f"A [{r.rand_lo:4.2f},{r.rand_hi:5.2f}] p={r.rand_p:7.1e}{'*' if r.bh_rand else ' '} | "
          f"B [{r.fix_lo:4.2f},{r.fix_hi:5.2f}] p={r.fix_p:7.1e}{'*' if r.bh_fix else ' '}")
print(f"\nBH(0.05), 12 HI-Small S/L cells:  FRAME A (pi random) {int(res.bh_rand.sum())} sig"
      f"  |  FRAME B (pi fixed) {int(res.bh_fix.sum())} sig")
surv = res[res.bh_fix]
if len(surv):
    print("Cells significant under FRAME B (base rates known):")
    for _, r in surv.iterrows():
        print(f"   {r.detector} @{int(r.budget*100)}%  R={r.R:.2f}  CI[{r.fix_lo:.2f},{r.fix_hi:.2f}]")
else:
    print("No cell significant even under FRAME B -> major replan confirmed.")
print("[saved] cluster_bootstrap_both_frames_hismall.csv -> gnn_outputs")
print("[DONE CB-1c]")
