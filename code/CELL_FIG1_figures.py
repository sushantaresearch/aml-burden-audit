# CELL FIG-1 -- two publication figures (CPU; Drive only).
# FIG A: 24-cell BH forest plot from bh_family_results.csv (canonical CI source).
# FIG B: reweighting dose-response, R@10% + AUPRC retention vs w (seed-2 series;
#        regenerate as v2 with error bars once 4C-3 seed replicates land).
# Outputs: fig_bh_forest.pdf/.png, fig_dose_response.pdf/.png -> gnn_outputs.

import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from google.colab import drive
if not os.path.ismount("/content/drive"): drive.mount("/content/drive")
GD = "/content/drive/MyDrive/Kaggle/gnn_outputs"

# ---------- FIG A: forest plot ----------
bh = pd.read_csv(f"{GD}/bh_family_results.csv")
print(f"[bh] {len(bh)} rows | columns: {list(bh.columns)}")
assert len(bh) == 24, "expected the 24-cell family"

def pick(df, *names):
    c = next((c for c in names if c in df.columns), None)
    assert c is not None, f"none of {names} in {list(df.columns)}"
    return c

cB = pick(bh, "benchmark", "bench", "dataset", "data")
cD = pick(bh, "detector", "det", "model")
cb = pick(bh, "budget", "b", "alpha")
cR = pick(bh, "R", "R_point", "r", "R_hat")
cL = pick(bh, "lo", "ci_lo", "R_lo", "lower", "ci_lower")
cH = pick(bh, "hi", "ci_hi", "R_hi", "upper", "ci_upper")
try:
    cS = pick(bh, "bh_significant", "significant", "bh_reject", "reject", "sig")
    sig = bh[cS].astype(bool).values
except AssertionError:
    cp = pick(bh, "p", "pval", "p_value", "p_boot")
    sig = (bh[cp].values <= 0.0066)
    print("[bh] no significance column; derived from p <= 0.0066")

bench_raw = bh[cB].astype(str).str.lower()
bh["_bench"] = np.where(bench_raw.str.contains("hi"), "HI-Small", "SAML-D")
bh["_det"] = bh[cD].astype(str).str.lower().str.replace("xgb", "", regex=False).str.strip("_- ")
bud = bh[cb].astype(float)
bh["_bud"] = (bud * 100).round().astype(int) if bud.max() <= 0.5 else bud.round().astype(int)
bh["_R"], bh["_lo"], bh["_hi"], bh["_sig"] = bh[cR], bh[cL], bh[cH], sig

order = [(B, D, b) for B in ("HI-Small", "SAML-D") for D in ("thin", "rich", "gnn")
         for b in (1, 5, 10, 20)]
key = {t: i for i, t in enumerate(order)}
bh["_k"] = [key[(r._bench, r._det, r._bud)] for r in bh.itertuples()]
assert bh["_k"].nunique() == 24, "cell labeling mismatch; check _bench/_det/_bud above"
bh = bh.sort_values("_k").reset_index(drop=True)

DLAB = {"thin": "Thin XGB", "rich": "Rich XGB", "gnn": "Multi-GNN"}
fig, ax = plt.subplots(figsize=(7.0, 9.0))
ypos = np.arange(len(bh))[::-1]
for i, r in bh.iterrows():
    yp = ypos[i]
    color = "#1a4f8b" if r._bench == "HI-Small" else "#8b3a1a"
    ax.plot([r._lo, r._hi], [yp, yp], color=color, lw=1.4, zorder=2)
    ax.plot(r._R, yp, "o", ms=5.5, zorder=3, color=color,
            markerfacecolor=(color if r._sig else "white"), markeredgewidth=1.2)
ax.axvline(1.0, color="0.35", lw=1.0, ls="--", zorder=1)
ax.set_yticks(ypos)
ax.set_yticklabels([f"{r._bench} · {DLAB[r._det]} · {r._bud}%" for _, r in bh.iterrows()],
                   fontsize=8)
for cut in (12,):
    ax.axhline(ypos[cut] + 0.5, color="0.85", lw=0.8)
for j in (4, 8, 16, 20):
    ax.axhline(ypos[j] + 0.5, color="0.93", lw=0.6)
ax.set_xscale("log")
ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
ax.set_xticklabels(["0.5", "0.75", "1", "1.5", "2", "3"], fontsize=9)
ax.set_xlabel("Prevalence-normalized FP-rate ratio  $R$  (log scale)", fontsize=10)
ax.set_title("Burden ratios with 95% bootstrap CIs; filled = BH-significant (q = 0.05)",
             fontsize=9.5)
ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{GD}/fig_bh_forest.pdf"); fig.savefig(f"{GD}/fig_bh_forest.png", dpi=300)
plt.close(fig)
print("[fig A] fig_bh_forest.pdf/.png saved")

# ---------- FIG B: dose-response ----------
W_GRID = [1.0, 1.5, 2.5, 4.0]
R10_LOCKED = {1.0: 2.4471, 1.5: 2.5284, 2.5: 2.7402, 4.0: 3.0703}
RET_LOCKED = {1.0: 100.0, 1.5: 97.7, 2.5: 93.3, 4.0: 100.0}
BASE_ENV = (2.4104, 2.5289)  # baseline multi-seed envelope, seeds 1-4
R10, RET, src = dict(R10_LOCKED), dict(RET_LOCKED), "locked constants"
try:
    for w in (1.5, 2.5, 4.0):
        lad = pd.read_csv(f"{GD}/amlworld_gnn_4c_w{w}_ladder.csv")
        cb2 = pick(lad, "budget", "b"); cR2 = pick(lad, "R", "R_point", "r")
        row = lad[np.isclose(lad[cb2].astype(float), 0.10)]
        if len(row) == 0:
            row = lad[lad[cb2].astype(float).round().astype(int) == 10]
        R10[w] = float(row.iloc[0][cR2])
        ac = next((c for c in ("auprc_4c", "auprc", "auprc_w") if c in lad.columns), None)
        ab = next((c for c in ("auprc_seed2", "auprc_base", "auprc_baseline")
                   if c in lad.columns), None)
        if ac and ab:
            RET[w] = 100.0 * float(lad.iloc[0][ac]) / float(lad.iloc[0][ab])
    src = "ladder CSVs"
except Exception as e:
    print(f"[fig B] CSV read fallback ({type(e).__name__}: {e}); using locked constants")
print(f"[fig B] source: {src} | R10 {dict((k, round(v,4)) for k,v in R10.items())}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
xs = W_GRID
a1.axhspan(*BASE_ENV, color="0.88", zorder=1,
           label="baseline seed envelope (w = 1)")
a1.plot(xs, [R10[w] for w in xs], "o-", color="#1a4f8b", lw=1.6, ms=6, zorder=3)
for w in xs:
    a1.annotate(f"{R10[w]:.2f}", (w, R10[w]), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=8)
a1.set_xlabel("group reweighting strength  $w$", fontsize=10)
a1.set_ylabel("$R_{SL}$ at 10% budget", fontsize=10)
a1.set_title("(a) Burden ratio vs reweighting dose", fontsize=10)
a1.legend(fontsize=8, frameon=False, loc="upper left")
a2.plot(xs, [RET[w] for w in xs], "s-", color="#8b3a1a", lw=1.6, ms=6, zorder=3)
a2.axhline(90.0, color="0.4", lw=1.0, ls="--")
a2.text(3.95, 90.4, "90% retention floor", fontsize=8, ha="right", color="0.3")
for w in xs:
    a2.annotate(f"{RET[w]:.1f}", (w, RET[w]), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=8)
a2.set_xlabel("group reweighting strength  $w$", fontsize=10)
a2.set_ylabel("AUPRC retention vs baseline (%)", fontsize=10)
a2.set_ylim(min(88, min(RET.values()) - 2), 102.5)
a2.set_title("(b) Detection retention vs dose", fontsize=10)
for ax in (a1, a2):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xticks(xs)
fig.tight_layout()
fig.savefig(f"{GD}/fig_dose_response.pdf"); fig.savefig(f"{GD}/fig_dose_response.png", dpi=300)
plt.close(fig)
print("[fig B] fig_dose_response.pdf/.png saved")
print("\n[done] open both PNGs from Drive and confirm they render before LaTeX inclusion")
