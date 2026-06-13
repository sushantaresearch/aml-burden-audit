# =====================================================================
# CELL_CX1_customs_load_detector_diagnostics.py
# Thesis C, step 1: load Korea/Jeong customs data, freeze the LightGBM
# detector, score valid+test, and DUMP the diagnostics needed to build
# the clustered burden-ratio audit (CX2): country-of-origin coding,
# importer clustering / design-effect preview, detector sanity.
# Self-contained. Run top to bottom in Colab. No synthcity needed.
# =====================================================================
import os, subprocess, numpy as np, pandas as pd

# ---- 0. deps (lightgbm only) ----
try:
    import lightgbm as lgb
except Exception:
    subprocess.run(["pip","-q","install","lightgbm"], check=True)
    import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

# ---- 1. data ----
if not os.path.isdir("Customs-Declaration-Datasets"):
    subprocess.run(["git","clone","-b","en",
        "https://github.com/Seondong/Customs-Declaration-Datasets.git"], check=True)
DATA="Customs-Declaration-Datasets/data/"
train=pd.read_csv(DATA+"df_syn_train_eng.csv")
valid=pd.read_csv(DATA+"df_syn_valid_eng.csv")
test =pd.read_csv(DATA+"df_syn_test_eng.csv")
print("shapes:",train.shape,valid.shape,test.shape,
      "| EXPECT (37385,22)(8134,22)(8481,22)")

# ---- GATE: known base rates ----
fr=train["Fraud"].mean(); c2=(train["Critical Fraud"]==2).mean()
print("Fraud rate train: %.4f (EXPECT ~0.2170)  [%s]"%(fr,"OK" if abs(fr-0.217)<0.01 else "CHECK"))
print("Critical2 rate train: %.4f (EXPECT ~0.0105) [%s]"%(c2,"OK" if abs(c2-0.0105)<0.004 else "CHECK"))

# ---- 2. frozen detector (exact notebook config) ----
ID_COLS=["Declarant ID","Importer ID","Seller ID","Courier ID"]
CAT_COLS=["Office ID","Process Type","Import Type","Import Use","Payment Type",
          "Mode of Transport","HS6 Code","Country of Departure","Country of Origin",
          "Tax Type","Country of Origin Indicator"]
DET_NUM=["Tax Rate","Net Mass","log_item_price","duty_proxy"]; DET_CAT=CAT_COLS
LGB=dict(objective="binary",n_estimators=300,learning_rate=0.05,num_leaves=63,
         min_child_samples=50,subsample=0.8,colsample_bytree=0.8,
         random_state=0,n_jobs=-1,verbose=-1)

def add_feats(df):
    d=df.copy()
    ip=pd.to_numeric(d["Item Price"],errors="coerce").fillna(0).clip(lower=0)
    tr=pd.to_numeric(d["Tax Rate"],errors="coerce").fillna(0).clip(lower=0)
    d["Item Price"]=ip; d["Tax Rate"]=tr
    d["log_item_price"]=np.log1p(ip); d["duty_proxy"]=ip*tr/100.0
    return d
def fit_enc(df):
    return {c:{v:i for i,v in enumerate(pd.Index(df[c].astype(str).unique()))} for c in DET_CAT}
def encode(df,maps):
    d=add_feats(df); X=pd.DataFrame(index=d.index)
    for c in DET_NUM: X[c]=pd.to_numeric(d[c],errors="coerce").fillna(0).astype(float)
    for c in DET_CAT: X[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(int)
    return X
def ylab(df,t):
    s=pd.to_numeric(df["Critical Fraud" if t=="crit2" else "Fraud"],errors="coerce").fillna(0)
    return (s==2).astype(int).values if t=="crit2" else s.round().clip(0,1).astype(int).values
def fit_score(target):
    maps=fit_enc(train); clf=lgb.LGBMClassifier(**LGB)
    clf.fit(encode(train,maps),ylab(train,target),categorical_feature=DET_CAT)
    sv=clf.predict_proba(encode(valid,maps))[:,1]
    st=clf.predict_proba(encode(test ,maps))[:,1]
    return sv,st
sv_f,st_f=fit_score("fraud"); sv_c,st_c=fit_score("crit2")
for nm,y,s in [("Fraud",ylab(test,"fraud"),st_f),("Crit2",ylab(test,"crit2"),st_c)]:
    print("detector %-6s TEST  AUPRC=%.3f  AUROC=%.3f  (prevalence %.4f)"
          %(nm,average_precision_score(y,s),roc_auc_score(y,s),y.mean()))

# ---- 3. DIAGNOSTIC A: country-of-origin coding (so CX2 tier map matches) ----
co=test["Country of Origin"]
print("\n[Country of Origin] dtype=%s  cardinality=%d"%(co.dtype,co.nunique()))
print("  example values:",list(co.astype(str).unique()[:8]))
print("  top 15 by frequency (test):")
print(co.astype(str).value_counts().head(15).to_string())
ind=test["Country of Origin Indicator"]
print("[Country of Origin Indicator] cardinality=%d  values:"%ind.nunique(),
      list(ind.astype(str).unique()[:12]))

# ---- 4. DIAGNOSTIC B: importer clustering / design-effect preview ----
def cluster_report(df,name):
    g=df.groupby(df["Importer ID"].astype(str)).size()
    print("\n[%s] Importer ID: n_importers=%d  n_rows=%d"%(name,len(g),len(df)))
    print("  decls/importer: mean=%.2f median=%.0f p95=%.0f max=%d  singletons=%.1f%%"
          %(g.mean(),g.median(),g.quantile(.95),g.max(),100*(g==1).mean()))
    # crude design-effect preview: deff ~ 1 + (mean cluster size - 1)*ICC; show mean size
    print("  (mean cluster size m=%.2f -> if ICC~0.2, deff~%.1f)"%(g.mean(),1+(g.mean()-1)*0.2))
cluster_report(test,"TEST"); cluster_report(valid,"VALID")
# FP concentration preview: among non-fraud test rows, how concentrated are importers?
nf=test[ylab(test,"fraud")==0]
gnf=nf.groupby(nf["Importer ID"].astype(str)).size().sort_values(ascending=False)
print("\n[non-fraud test] top-1%% importers hold %.1f%% of legitimate decls"
      %(100*gnf.head(max(1,int(0.01*len(gnf)))).sum()/gnf.sum()))

# ---- 5. save tidy CSVs for CX2 (Drive) ----
OUT="/content/drive/MyDrive/Kaggle/gnn_outputs/"
try: os.makedirs(OUT,exist_ok=True)
except Exception: OUT="./"  # falls back to local if Drive not mounted
def tidy(df,sf,sc):
    return pd.DataFrame({"importer_id":df["Importer ID"].astype(str).values,
        "origin":df["Country of Origin"].astype(str).values,
        "origin_ind":df["Country of Origin Indicator"].astype(str).values,
        "fraud":ylab(df,"fraud"),"crit2":ylab(df,"crit2"),
        "score_fraud":sf,"score_crit2":sc})
tidy(test ,st_f,st_c).to_csv(OUT+"customs_audit_test.csv",index=False)
tidy(valid,sv_f,sv_c).to_csv(OUT+"customs_audit_valid.csv",index=False)
print("\n[saved] customs_audit_test.csv + customs_audit_valid.csv ->",OUT)
print("[DONE CX1] paste the printed block back so I can finalize the LDC tier map + CX2.")
