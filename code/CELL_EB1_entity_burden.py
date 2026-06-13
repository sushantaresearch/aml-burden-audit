# CELL EB-1 -- entity-level burden, HI-Small (CPU or T4; no staging, Drive only).
# Question answered: at each review budget, what share of INNOCENT accounts of
# each legal form is flagged at least once? Primary print: 10% budget, S vs L.
# Gates: transaction-level R@10% must reproduce the locked anchors per detector.

import os, glob, numpy as np, pandas as pd
from google.colab import drive
if not os.path.ismount("/content/drive"): drive.mount("/content/drive")
GD = "/content/drive/MyDrive/Kaggle/gnn_outputs"
CACHE = "/content/drive/MyDrive/Kaggle/aml_cache"
HI_VAL = (3_248_921, 4_214_444); HI_TEST = (4_214_445, 5_078_344)
ANCH = {"thin": 1.125297147359424, "rich": 2.9984894229358963, "gnn": 2.483790728335002}

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
    ic = next((c for c in ("gid", "EdgeID", "te_global_idx", "va_global_idx",
                           "global_idx", "idx") if c in d.columns), None)
    assert ic is not None, f"{path}: no index col in {list(d.columns)}"
    v = d[ic].values.astype(np.int64)
    gid = v if (v.min() >= lo and v.max() <= hi) else v + lo
    pc = next(c for c in ("prob", "score", "p", "pred") if c in d.columns)
    yc = next((c for c in ("y", "label", "Is Laundering") if c in d.columns), None)
    y = d[yc].values.astype("int8") if yc else y_hi[gid]
    out = (pd.DataFrame({"gid": gid, "prob": d[pc].values, "y": y})
           .sort_values("gid").reset_index(drop=True))
    assert out.gid.min() >= lo and out.gid.max() <= hi and out.gid.is_unique, path
    assert (out.y.values == y_hi[out.gid.values]).mean() > 0.999, f"{path}: y mismatch"
    return out

def Rpoint(al, y, g):
    inn = y == 0
    return ((al[inn & (g == 0)].mean() / al[inn & (g == 2)].mean())
            / (y[g == 0].mean() / y[g == 2].mean()))

DETS = {
    "thin": (find_one(f"{GD}/thin*val*score*.csv"), find_one(f"{GD}/thin*test*score*.csv")),
    "rich": (find_one(f"{GD}/rich*val*score*.csv"), find_one(f"{GD}/rich*test*score*.csv")),
    "gnn":  (find_one(f"{GD}/gnn_val_scores*.csv"), find_one(f"{GD}/gnn_test_scores*.csv")),
}
for k, (vp, tp) in DETS.items():
    print(f"[files] {k}: {os.path.basename(vp)} | {os.path.basename(tp)}")

rows_out = []
for det, (vp, tp) in DETS.items():
    va = load_scores(vp, *HI_VAL); te = load_scores(tp, *HI_TEST)
    p, y, gid = te.prob.values, te.y.values, te.gid.values
    g = g_hi[gid]; fid = from_id_all[gid]
    R10 = Rpoint(p >= np.quantile(va.prob.values, 0.90), y, g)
    ok = abs(R10 - ANCH[det]) / ANCH[det] < 1e-3
    print(f"\n[{det}] txn-level R@10% {R10:.6f} vs locked {ANCH[det]:.6f} "
          f"{'[GATE OK]' if ok else '[GATE FAIL]'}")
    assert ok
    df = pd.DataFrame({"fid": fid, "g": g, "y": y, "p": p})
    base = df.groupby("fid").agg(g=("g", "first"), g_n=("g", "nunique"),
                                 n=("y", "size"), any_y=("y", "max"))
    incon = int((base.g_n > 1).sum())
    if det == "gnn":
        for grp, code in (("S", 0), ("M", 1), ("L", 2)):
            sub = base[base.g == code]
            print(f"[context] {grp}: {len(sub):,} accounts in test band, "
                  f"{int((sub.any_y == 0).sum()):,} innocent | "
                  f"group-inconsistent accounts overall: {incon}")
    for b in (0.01, 0.05, 0.10, 0.20):
        tau = np.quantile(va.prob.values, 1 - b)
        anyal = (df.assign(al=df.p >= tau).groupby("fid")["al"].max())
        agg = base.join(anyal.rename("any_al"))
        inn = agg[agg.any_y == 0]
        for grp, code in (("S", 0), ("M", 1), ("L", 2)):
            sub = inn[inn.g == code]
            rows_out.append(dict(detector=det, budget=b, group=grp,
                                 n_innocent=int(len(sub)),
                                 n_flagged=int(sub.any_al.sum()),
                                 share_flagged=float(sub.any_al.mean()),
                                 mean_test_tx=float(sub.n.mean())))
        if b == 0.10:
            s = inn[inn.g == 0]; l = inn[inn.g == 2]
            rs, rl = s.any_al.mean(), l.any_al.mean()
            print(f"[{det} @10%] innocent flagged>=1x: "
                  f"S {100*rs:.2f}% ({int(s.any_al.sum()):,}/{len(s):,})  "
                  f"L {100*rl:.2f}% ({int(l.any_al.sum()):,}/{len(l):,})  "
                  f"entity ratio {rs/rl:.3f}")
            print(f"[{det} @10%] mean test txns per innocent account: "
                  f"S {s.n.mean():.1f}  L {l.n.mean():.1f}")

out = pd.DataFrame(rows_out)
out.to_csv(f"{GD}/entity_burden_hismall.csv", index=False)
print(f"\n[saved] entity_burden_hismall.csv ({len(out)} rows) -> gnn_outputs")
gn = out[(out.detector == "gnn") & (out.budget == 0.10)].set_index("group")
print("\nDraft sentence (fill into Section V/VII):")
print(f"At the 10% review budget, {100*gn.loc['S','share_flagged']:.1f}% of innocent "
      f"sole-proprietorship accounts are flagged at least once by the GNN detector, "
      f"versus {100*gn.loc['L','share_flagged']:.1f}% of corporate accounts "
      f"(entity-level ratio "
      f"{gn.loc['S','share_flagged']/gn.loc['L','share_flagged']:.2f}), despite "
      f"comparable per-account transaction counts "
      f"({gn.loc['S','mean_test_tx']:.1f} vs {gn.loc['L','mean_test_tx']:.1f}).")
