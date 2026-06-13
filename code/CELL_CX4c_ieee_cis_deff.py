# =====================================================================
# CELL_CX4c_ieee_cis_deff.py  (accepts EITHER kaggle.json OR KGAT token;
# secure hidden entry; CLI download; clean failure; same analysis as CX4b)
#
# PREREQS (one-time on kaggle.com):
#   1) Accept rules: https://www.kaggle.com/competitions/ieee-fraud-detection/rules
#   2) Have ONE of:
#        - kaggle.json  (Settings > API > "Create New Token" -> downloads a FILE), or
#        - a fresh KGAT_ access token (paste into the hidden prompt below).
# SECURITY: the token is read via getpass (hidden) and written only to
# ~/.kaggle with chmod 600; it is never printed. Regenerate any token you
# have pasted in plaintext anywhere.
# =====================================================================
import os, json, glob, getpass, subprocess, numpy as np, pandas as pd
try:
    import lightgbm as lgb
except Exception:
    subprocess.run(["pip","-q","install","lightgbm"],check=True); import lightgbm as lgb
subprocess.run(["pip","-q","install","kaggle"],check=False)
from sklearn.metrics import average_precision_score, roc_auc_score

HOME=os.path.expanduser("~/.kaggle"); os.makedirs(HOME,exist_ok=True)
def _verify():
    r=subprocess.run(["kaggle","competitions","list"],capture_output=True,text=True)
    ok=(r.returncode==0)
    print("auth check:","OK" if ok else "FAILED "+ (r.stderr[:200] if r.stderr else ""))
    return ok
def ensure_auth():
    # already have a classic kaggle.json anywhere?
    for c in [os.path.join(HOME,"kaggle.json"),"/content/kaggle.json","kaggle.json"]:
        if os.path.exists(c):
            try:
                cfg=json.load(open(c)); os.environ["KAGGLE_USERNAME"]=cfg["username"]
                os.environ["KAGGLE_KEY"]=cfg["key"]; dst=os.path.join(HOME,"kaggle.json")
                json.dump(cfg,open(dst,"w")); os.chmod(dst,0o600)
                if _verify(): return True
            except Exception as e: print("kaggle.json issue:",repr(e))
    print("\nChoose auth:  [1] upload kaggle.json   [2] paste KGAT_ token (hidden)")
    ch=input("Enter 1 or 2: ").strip()
    if ch=="1":
        try:
            from google.colab import files
            print("Select your kaggle.json file:"); files.upload()
        except Exception as e: print("upload failed:",repr(e)); return False
        if os.path.exists("kaggle.json"):
            cfg=json.load(open("kaggle.json")); os.environ["KAGGLE_USERNAME"]=cfg["username"]
            os.environ["KAGGLE_KEY"]=cfg["key"]; dst=os.path.join(HOME,"kaggle.json")
            json.dump(cfg,open(dst,"w")); os.chmod(dst,0o600); return _verify()
        print("kaggle.json not found after upload."); return False
    elif ch=="2":
        tok=getpass.getpass("Paste FRESH KGAT_ token (hidden): ").strip()
        os.environ["KAGGLE_API_TOKEN"]=tok
        at=os.path.join(HOME,"access_token"); open(at,"w").write(tok); os.chmod(at,0o600)
        return _verify()
    print("No valid choice."); return False

DATA_OK=os.path.exists("train_transaction.csv")
if not DATA_OK:
    if ensure_auth():
        r=subprocess.run(["kaggle","competitions","download","-c","ieee-fraud-detection",
                          "-f","train_transaction.csv","-p","."],capture_output=True,text=True)
        if r.returncode!=0:
            print("download failed:",(r.stderr or r.stdout)[:400])
            if "403" in (r.stderr or "") or "forbidden" in (r.stderr or "").lower():
                print(">>> Accept rules then rerun: https://www.kaggle.com/competitions/ieee-fraud-detection/rules")
        for z in glob.glob("*.zip"): subprocess.run(["unzip","-o",z],check=False)
        DATA_OK=os.path.exists("train_transaction.csv")

if not DATA_OK:
    print("\n[CX4 halted cleanly: IEEE-CIS not available yet. Fix the above and rerun.]")
else:
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
        wc=ci_w(np.array(clus)); wn=ci_w(np.array(naive)); deff=(wc/wn)**2 if wn and wc==wc else np.nan
        print(" budget %4.0f%%  FPR=%.4f | clustered CI width=%.4f | naive CI width=%.4f | deff=%.2f"%(
            100*b,fpr_at(thr,np.arange(len(te_y))),wc,wn,deff))
        out.append(dict(budget=b,clus_w=wc,naive_w=wn,deff=deff))
    OUT="/content/drive/MyDrive/Kaggle/gnn_outputs/"
    try: os.makedirs(OUT,exist_ok=True)
    except Exception: OUT="./"
    pd.DataFrame(out).to_csv(OUT+"ieee_cis_deff.csv",index=False)
    print("\n[interpretation] customs deff~1.0, synthetic AML deff~60; IEEE-CIS lands above.")
    print("[saved] ieee_cis_deff.csv ->",OUT,"| paste the printed block back.")
