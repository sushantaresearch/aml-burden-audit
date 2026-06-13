# =====================================================================
# CELL_CX4_ieee_cis_deff.py
# THIRD (REAL-DATA) dependence point for the cross-domain inference claim.
# Measures the design effect of the false-positive-rate estimate under
# CARD-level clustering on IEEE-CIS (real e-commerce transaction fraud),
# to sit beside customs (deff~1.0) and synthetic AML (deff~60).
#
# PURPOSE: defuse the "both datasets synthetic -> deff contrast is a
# cross-simulator artifact" critique. This is the single highest-value
# change for acceptance. No protected-group / fairness claim is made here;
# the cell isolates the DEPENDENCE phenomenon (do FPs cluster within the
# natural account unit?) on REAL transaction data.
#
# HONESTY: report deff as-is. If IEEE-CIS shows HIGH deff -> reinforces
# "transaction data clusters, customs declarations do not." If LOW -> report
# honestly (would mean the AML high-deff is partly simulator-specific). Either
# outcome is a legitimate finding for the paper.
#
# DATA: Kaggle competition 'ieee-fraud-detection' (one-time rule acceptance
# on kaggle.com required). Provide kaggle.json when prompted.
# =====================================================================
import os, subprocess, numpy as np, pandas as pd

# ---- 0. deps ----
try:
    import lightgbm as lgb
except Exception:
    subprocess.run(["pip","-q","install","lightgbm"],check=True); import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

# ---- 1. get IEEE-CIS (Kaggle API) ----
# Upload kaggle.json first:  from google.colab import files; files.upload()
if not os.path.exists("train_transaction.csv"):
    try:
        subprocess.run(["pip","-q","install","kaggle"],check=True)
        os.makedirs(os.path.expanduser("~/.kaggle"),exist_ok=True)
        if os.path.exists("kaggle.json"):
            subprocess.run(["cp","kaggle.json",os.path.expanduser("~/.kaggle/kaggle.json")],check=True)
            os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"),0o600)
        subprocess.run(["kaggle","competitions","download","-c","ieee-fraud-detection",
                        "-f","train_transaction.csv"],check=True)
        if os.path.exists("train_transaction.csv.zip"):
            subprocess.run(["unzip","-o","train_transaction.csv.zip"],check=True)
    except Exception as e:
        raise SystemExit("Download IEEE-CIS manually (accept competition rules on kaggle.com, "
                         "then upload train_transaction.csv to the Colab session). Error: %r"%e)

df=pd.read_csv("train_transaction.csv")
print("IEEE-CIS train_transaction:",df.shape,"| isFraud rate=%.4f"%df["isFraud"].mean())

# ---- 2. clustering unit + features ----
CLUSTER="card1"                      # card identifier ~ account/customer cluster
df[CLUSTER]=df[CLUSTER].fillna(-1).astype(str)
y=df["isFraud"].astype(int).values
drop={"isFraud","TransactionID","TransactionDT"}
num=[c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
X=df[num].fillna(-999).astype(float)
print("using %d numeric features; clustering unit=%s"%(len(num),CLUSTER))

# ---- 3. time-ordered split 60/20/20 (no leakage) ----
order=np.argsort(df["TransactionDT"].values); n=len(df)
i1,i2=int(0.6*n),int(0.8*n)
tr,va,te=order[:i1],order[i1:i2],order[i2:]
LGB=dict(objective="binary",n_estimators=300,learning_rate=0.05,num_leaves=63,
         min_child_samples=50,subsample=0.8,colsample_bytree=0.8,random_state=0,n_jobs=-1,verbose=-1)
clf=lgb.LGBMClassifier(**LGB); clf.fit(X.iloc[tr],y[tr])
s_va=clf.predict_proba(X.iloc[va])[:,1]; s_te=clf.predict_proba(X.iloc[te])[:,1]
print("detector TEST AUPRC=%.3f AUROC=%.3f (prevalence %.4f)"%(
    average_precision_score(y[te],s_te),roc_auc_score(y[te],s_te),y[te].mean()))

# ---- 4. card clustering structure on TEST (compare to customs 1.52 / 72% singletons) ----
ct=df.iloc[te]; g=ct.groupby(CLUSTER).size()
print("\n[TEST] cards=%d rows=%d | txns/card mean=%.2f median=%.0f p95=%.0f max=%d singletons=%.1f%%"%(
    len(g),len(ct),g.mean(),g.median(),g.quantile(.95),g.max(),100*(g==1).mean()))
legit=ct[y[te]==0]; gl=legit.groupby(CLUSTER).size().sort_values(ascending=False)
print("[non-fraud TEST] top-1%% cards hold %.1f%% of legitimate txns"%(
    100*gl.head(max(1,int(0.01*len(gl)))).sum()/gl.sum()))

# ---- 5. deff of the FPR estimate: clustered (by card) vs naive (by row) ----
te_card=ct[CLUSTER].values; te_y=y[te]
B=1000; RNG=np.random.default_rng(0)
rows_by_card={k:np.where(te_card==k)[0] for k in np.unique(te_card)}
card_keys=np.array(list(rows_by_card.keys()))
def fpr_at(thr, idx):
    yy=te_y[idx]; ss=s_te[idx]; leg=yy==0
    return (ss[leg]>=thr).mean() if leg.sum()>0 else np.nan
def ci_width(arr):
    a=arr[~np.isnan(arr)]; return (np.percentile(a,97.5)-np.percentile(a,2.5)) if len(a)>50 else np.nan
print("\n===== IEEE-CIS design effect (FPR estimate, card-clustered vs naive) =====")
out=[]
for b in [0.01,0.05,0.10,0.20]:
    thr=np.quantile(s_va,1-b)
    clus=[]; naive=[]
    for _ in range(B):
        pick=RNG.choice(card_keys,size=len(card_keys),replace=True)
        ridx=np.concatenate([rows_by_card[k] for k in pick]); clus.append(fpr_at(thr,ridx))
        naive.append(fpr_at(thr,RNG.integers(0,len(te_y),len(te_y))))
    wc=ci_width(np.array(clus)); wn=ci_width(np.array(naive))
    deff=(wc/wn)**2 if wn and wc==wc else np.nan
    base=fpr_at(thr,np.arange(len(te_y)))
    print(" budget %4.0f%%  FPR=%.4f | clustered CI width=%.4f | naive CI width=%.4f | deff=%.2f"%(
        100*b,base,wc,wn,deff))
    out.append(dict(budget=b,fpr=base,clus_w=wc,naive_w=wn,deff=deff))
res=pd.DataFrame(out)
OUT="/content/drive/MyDrive/Kaggle/gnn_outputs/"
try: os.makedirs(OUT,exist_ok=True)
except Exception: OUT="./"
res.to_csv(OUT+"ieee_cis_deff.csv",index=False)
print("\n[interpretation] customs deff~1.0, synthetic AML deff~60. IEEE-CIS lands at the")
print("values above -> the THIRD, REAL-data point for the dependence-unit argument.")
print("[saved] ieee_cis_deff.csv ->",OUT,"| paste the printed block back.")
