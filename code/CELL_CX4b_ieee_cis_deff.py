# =====================================================================
# CELL_CX4b_ieee_cis_deff.py  (robust Kaggle auth + clean failure)
# Third REAL-data dependence point: design effect of the FPR estimate under
# CARD-level clustering on IEEE-CIS (real e-commerce fraud), beside customs
# (deff~1.0) and synthetic AML (deff~60). No fairness/group claim; isolates
# the DEPENDENCE phenomenon on real data to defuse the cross-simulator critique.
#
# PREREQS (one-time, on kaggle.com):
#   1) Accept rules: https://www.kaggle.com/competitions/ieee-fraud-detection/rules
#   2) kaggle.json token (Kaggle > Settings > Create New Token).
# The cell will prompt to upload kaggle.json if not found, and will HALT CLEANLY
# (no traceback) with exact remediation if auth/rules are missing.
# =====================================================================
import os, json, glob, subprocess, numpy as np, pandas as pd
try:
    import lightgbm as lgb
except Exception:
    subprocess.run(["pip","-q","install","lightgbm"],check=True); import lightgbm as lgb
subprocess.run(["pip","-q","install","kaggle"],check=False)
from sklearn.metrics import average_precision_score, roc_auc_score

# ---- robust token loader: sets env vars (most reliable) AND places the file ----
def ensure_kaggle_token():
    cands=[os.path.expanduser("~/.kaggle/kaggle.json"),"/content/kaggle.json","kaggle.json"]
    path=next((p for p in cands if os.path.exists(p)),None)
    if path is None:
        try:
            from google.colab import files
            print("Upload kaggle.json (Kaggle > Settings > Create New Token):")
            up=files.upload()
            path="kaggle.json" if os.path.exists("kaggle.json") else (list(up.keys())[0] if up else None)
        except Exception as e:
            print("No kaggle.json found and not in Colab:",repr(e)); return False
    if not path or not os.path.exists(path):
        print("kaggle.json not found."); return False
    try:
        with open(path) as f: cfg=json.load(f)
        os.environ["KAGGLE_USERNAME"]=cfg["username"]; os.environ["KAGGLE_KEY"]=cfg["key"]
        os.makedirs(os.path.expanduser("~/.kaggle"),exist_ok=True)
        dst=os.path.expanduser("~/.kaggle/kaggle.json")
        with open(dst,"w") as f: json.dump(cfg,f)
        os.chmod(dst,0o600)
        print("kaggle token loaded for user:",cfg["username"]); return True
    except Exception as e:
        print("Could not parse kaggle.json:",repr(e)); return False

# ---- download train_transaction.csv, halting cleanly on failure ----
DATA_OK=os.path.exists("train_transaction.csv")
if not DATA_OK and ensure_kaggle_token():
    try:
        import kaggle  # env vars are set before import -> clean auth
        kaggle.api.competition_download_file("ieee-fraud-detection","train_transaction.csv",path=".")
        for z in glob.glob("*.zip"): subprocess.run(["unzip","-o",z],check=False)
        DATA_OK=os.path.exists("train_transaction.csv")
    except Exception as e:
        m=str(e).lower()
        print("DOWNLOAD FAILED:",repr(e))
        if "403" in m or "forbidden" in m or "not accept" in m:
            print(">>> ACCEPT THE RULES first: https://www.kaggle.com/competitions/ieee-fraud-detection/rules")
            print(">>> Click 'I Understand and Accept', then rerun this cell.")
        else:
            print(">>> Check kaggle.json validity and that rules are accepted, then rerun.")

if not DATA_OK:
    print("\n[CX4 halted cleanly: IEEE-CIS not available yet. Fix the above and rerun this cell.]")
else:
    # ================= analysis (unchanged logic) =================
    df=pd.read_csv("train_transaction.csv")
    print("IEEE-CIS train_transaction:",df.shape,"| isFraud rate=%.4f"%df["isFraud"].mean())
    CLUSTER="card1"; df[CLUSTER]=df[CLUSTER].fillna(-1).astype(str)
    y=df["isFraud"].astype(int).values
    drop={"isFraud","TransactionID","TransactionDT"}
    num=[c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    X=df[num].fillna(-999).astype(float)
    print("using %d numeric features; clustering unit=%s"%(len(num),CLUSTER))

    order=np.argsort(df["TransactionDT"].values); n=len(df); i1,i2=int(0.6*n),int(0.8*n)
    tr,va,te=order[:i1],order[i1:i2],order[i2:]
    LGB=dict(objective="binary",n_estimators=300,learning_rate=0.05,num_leaves=63,
             min_child_samples=50,subsample=0.8,colsample_bytree=0.8,random_state=0,n_jobs=-1,verbose=-1)
    clf=lgb.LGBMClassifier(**LGB); clf.fit(X.iloc[tr],y[tr])
    s_va=clf.predict_proba(X.iloc[va])[:,1]; s_te=clf.predict_proba(X.iloc[te])[:,1]
    print("detector TEST AUPRC=%.3f AUROC=%.3f (prevalence %.4f)"%(
        average_precision_score(y[te],s_te),roc_auc_score(y[te],s_te),y[te].mean()))

    ct=df.iloc[te]; g=ct.groupby(CLUSTER).size()
    print("\n[TEST] cards=%d rows=%d | txns/card mean=%.2f median=%.0f p95=%.0f max=%d singletons=%.1f%%"%(
        len(g),len(ct),g.mean(),g.median(),g.quantile(.95),g.max(),100*(g==1).mean()))
    gl=ct[y[te]==0].groupby(CLUSTER).size().sort_values(ascending=False)
    print("[non-fraud TEST] top-1%% cards hold %.1f%% of legitimate txns"%(
        100*gl.head(max(1,int(0.01*len(gl)))).sum()/gl.sum()))

    te_card=ct[CLUSTER].values; te_y=y[te]; B=1000; RNG=np.random.default_rng(0)
    rows_by_card={k:np.where(te_card==k)[0] for k in np.unique(te_card)}
    card_keys=np.array(list(rows_by_card.keys()))
    def fpr_at(thr,idx):
        yy=te_y[idx]; ss=s_te[idx]; leg=yy==0
        return (ss[leg]>=thr).mean() if leg.sum()>0 else np.nan
    def ci_w(a):
        a=a[~np.isnan(a)]; return (np.percentile(a,97.5)-np.percentile(a,2.5)) if len(a)>50 else np.nan
    print("\n===== IEEE-CIS design effect (FPR estimate, card-clustered vs naive) =====")
    out=[]
    for b in [0.01,0.05,0.10,0.20]:
        thr=np.quantile(s_va,1-b); clus=[]; naive=[]
        for _ in range(B):
            pick=RNG.choice(card_keys,size=len(card_keys),replace=True)
            ridx=np.concatenate([rows_by_card[k] for k in pick]); clus.append(fpr_at(thr,ridx))
            naive.append(fpr_at(thr,RNG.integers(0,len(te_y),len(te_y))))
        wc=ci_w(np.array(clus)); wn=ci_w(np.array(naive))
        deff=(wc/wn)**2 if wn and wc==wc else np.nan
        print(" budget %4.0f%%  FPR=%.4f | clustered CI width=%.4f | naive CI width=%.4f | deff=%.2f"%(
            100*b,fpr_at(thr,np.arange(len(te_y))),wc,wn,deff))
        out.append(dict(budget=b,clus_w=wc,naive_w=wn,deff=deff))
    OUT="/content/drive/MyDrive/Kaggle/gnn_outputs/"
    try: os.makedirs(OUT,exist_ok=True)
    except Exception: OUT="./"
    pd.DataFrame(out).to_csv(OUT+"ieee_cis_deff.csv",index=False)
    print("\n[interpretation] customs deff~1.0, synthetic AML deff~60; IEEE-CIS lands above.")
    print("[saved] ieee_cis_deff.csv ->",OUT,"| paste the printed block back.")
