# ============================================================
# CELL FIG-1 v2 -- FINAL MANUSCRIPT FIGURES (replaces CELL_FIG1)
# FIG A: 24-cell BH/FDR forest plot (reads bh_family_results.csv)
# FIG B v2: mitigation dose-response with seed replicates.
#   Constants are adjudicated values from gate-validated run logs;
#   provenance: TCSS_4C_multiseed_lock_v1.txt.
# CPU-only, ~1-2 min. Outputs PDF+PNG to gnn_outputs on Drive.
# Paste the final printed block back.
# ============================================================
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
import os, glob
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})

# ---------- locate gnn_outputs ----------
hits = sorted(glob.glob('/content/drive/MyDrive/**/gnn_outputs', recursive=True))
assert hits, '[FAIL] gnn_outputs folder not found'
OUT = hits[0]
print('[out]', OUT)

# ---------- FIG A: BH forest ----------
bhp = sorted(glob.glob('/content/drive/MyDrive/**/bh_family_results.csv', recursive=True))
assert bhp, '[FAIL] bh_family_results.csv not found'
bh = pd.read_csv(bhp[0])
print('[bh]', bhp[0], '| rows', len(bh), '| cols', list(bh.columns))

def pick(df, *cands):
    low = {c.lower(): c for c in df.columns}
    for n in cands:
        if n in low: return low[n]
    for n in cands:
        for c in df.columns:
            if n in c.lower(): return c
    return None

cR  = pick(bh, 'r_point','r_hat','ratio','estimate','point','r')
cLo = pick(bh, 'ci_lo','ci_lower','lower','l95','lo')
cHi = pick(bh, 'ci_hi','ci_upper','upper','u95','hi')
cSig= pick(bh, 'significant','reject','sig')
cQ  = pick(bh, 'q_value','qval','q')
cDs = pick(bh, 'dataset','bench','data')
cDt = pick(bh, 'detector','model','det')
cBu = pick(bh, 'budget','pct','alpha','review')
assert cR is not None, '[FAIL] no R column; columns: %s' % list(bh.columns)

if cSig is not None:
    sig = bh[cSig].astype(str).str.lower().isin(['true','1','yes','reject'])
elif cQ is not None:
    sig = bh[cQ] <= 0.05
else:
    sig = pd.Series([True]*len(bh))

def lab(r):
    parts = [str(r[c]) for c in (cDs, cDt, cBu) if c is not None]
    return ' | '.join(parts) if parts else str(r.name)
labels = bh.apply(lab, axis=1)
order  = np.argsort(labels.values)
bhs, labs, sigs = bh.iloc[order].reset_index(drop=True), labels.iloc[order].reset_index(drop=True), sig.iloc[order].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(6.6, 0.28*len(bhs)+1.2))
y = np.arange(len(bhs))[::-1]
for i in range(len(bhs)):
    r  = bhs.loc[i, cR]
    lo = bhs.loc[i, cLo] if cLo else r
    hi = bhs.loc[i, cHi] if cHi else r
    ax.plot([lo, hi], [y[i], y[i]], lw=1.1, color='0.25')
    ax.plot(r, y[i], marker='o', ms=4.5, mfc=('0.1' if sigs[i] else 'white'), mec='0.1')
ax.axvline(1.0, color='0.5', lw=0.9, ls='--')
ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7)
ax.set_xscale('log')
ax.set_xlabel('Prevalence-normalized FP ratio R (log scale)')
ax.set_title('Burden ratios with bootstrap CIs; filled = BH-significant (q<=0.05)', fontsize=8)
fig.tight_layout()
for ext in ('pdf','png'):
    fig.savefig(os.path.join(OUT, 'fig_bh_forest.'+ext), dpi=300, bbox_inches='tight')
plt.show()

# ---------- FIG B v2: dose-response + seed replicates ----------
W        = [1.0, 1.5, 2.5, 4.0]            # seed-2 dose series; w=1.0 = baseline
R10      = [2.447, 2.528, 2.740, 3.070]
AUPRCRT  = [100.0, 97.7, 93.3, 100.0]
BASE_BAND = (2.4104, 2.5289)               # baseline R@10% range, seeds 1-4
REPL = {'seed 3': (2.5, 2.5679, '^'), 'seed 4': (2.5, 2.3524, 'v')}

fig, ax1 = plt.subplots(figsize=(4.6, 3.2))
ax1.axhspan(BASE_BAND[0], BASE_BAND[1], color='0.85', zorder=0,
            label='baseline R@10% range (seeds 1-4)')
ax1.plot(W, R10, marker='o', color='0.1', lw=1.4,
         label='dose series, seed 2')
for name,(x,v,m) in REPL.items():
    ax1.plot(x, v, marker=m, ms=7, ls='none', mfc='white', mec='0.1',
             label='replicate at w=2.5, '+name)
ax1.axhline(1.0, color='0.6', lw=0.9, ls=':')
ax1.set_xlabel('Group reweighting dose w')
ax1.set_ylabel('R_SL(10%)')
ax2 = ax1.twinx()
ax2.plot(W, AUPRCRT, marker='s', ms=4, color='0.45', lw=1.0, ls='--',
         label='AUPRC retention (%, seed-2 series)')
ax2.set_ylabel('AUPRC retention (%)'); ax2.set_ylim(85, 112)
h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, fontsize=6, loc='upper left', frameon=False)
fig.tight_layout()
for ext in ('pdf','png'):
    fig.savefig(os.path.join(OUT, 'fig_dose_response_v2.'+ext), dpi=300, bbox_inches='tight')
plt.show()

print('\n[OK] saved:')
for f in ('fig_bh_forest.pdf','fig_bh_forest.png',
          'fig_dose_response_v2.pdf','fig_dose_response_v2.png'):
    p = os.path.join(OUT, f)
    print('  ', p, '| exists:', os.path.exists(p), '| bytes:', os.path.getsize(p) if os.path.exists(p) else 0)
print('[CONSTANTS] W', W, '| R10', R10, '| AUPRC%', AUPRCRT, '| band', BASE_BAND)
print('[DONE FIG-1 v2]')
