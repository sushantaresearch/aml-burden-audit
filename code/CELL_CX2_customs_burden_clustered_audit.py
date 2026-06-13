# =====================================================================
# CELL_CX2_customs_burden_clustered_audit.py
# Thesis C, step 2: burden ratio R by origin tier (strict UN-LDC vs
# OECD-developed) on the Korea/Jeong customs detector scores, with
# IMPORTER-CLUSTERED bootstrap vs naive (design effect), BH across
# cells, budget-continuous burden curve + AUBC, and a per-origin
# one-vs-rest robustness sweep. Both fraud tracks, both pi-frames.
# Consumes the CSVs written by CX1. Self-contained otherwise.
#
# HONESTY BOUNDARY (unchanged): report R as-is at every budget; never
# manufacture significance by stacking favorable choices; inversions
# and small-n (wide CI) are findings, not failures.
# =====================================================================
import os, numpy as np, pandas as pd

OUT="/content/drive/MyDrive/Kaggle/gnn_outputs/"
if not os.path.exists(OUT+"customs_audit_test.csv"): OUT="./"
te=pd.read_csv(OUT+"customs_audit_test.csv")
va=pd.read_csv(OUT+"customs_audit_valid.csv")
print("loaded test/valid:",te.shape,va.shape)

# ---- origin tier maps (ISO-3166 alpha-2) ----
UN_LDC={"AF","BD","BF","BI","BJ","KH","KM","CF","TD","CD","DJ","ER","ET","GM","GN",
 "GW","HT","KI","LA","LS","LR","MG","MW","ML","MR","MZ","MM","NP","NE","RW","SN",
 "SL","SB","SO","SS","SD","TZ","TL","TG","TV","UG","YE","ZM","AO","BT","ST"}  # 2024 UN list (46)
OECD_DEV={"AU","AT","BE","CA","CL","CZ","DK","EE","FI","FR","DE","GR","HU","IS","IE",
 "IL","IT","JP","KR","LV","LT","LU","NL","NZ","NO","PL","PT","SK","SI","ES","SE",
 "CH","TR","GB","US","CO","CR","MX"}  # OECD members
def tier(o):
    o=str(o).upper()
    if o in UN_LDC: return "LDC"
    if o in OECD_DEV: return "DEV"
    return "OTHER"
for d in (te,va): d["tier"]=d["origin"].map(tier)

BUDGETS=[0.01,0.02,0.05,0.10,0.20]; B=3000; RNG=np.random.default_rng(0)

# ---- LDC estimability check (print counts FIRST) ----
print("\n[tier counts] TEST:");  print(te["tier"].value_counts().to_string())
for lab in ["fraud","crit2"]:
    sub=te[te[lab]==0]  # legitimate only (FP denominator)
    print(" legitimate(%s) by tier:"%lab, sub["tier"].value_counts().to_dict())

# ---- core R machinery ----
def tau_at(scores_valid, b):           # threshold from VALID at (1-b) quantile
    return np.quantile(scores_valid, 1-b)
def fpr(df, lab, scorecol, thr):       # FP rate among legitimate rows of a group
    leg=df[df[lab]==0]
    if len(leg)==0: return np.nan, 0
    return float((leg[scorecol]>=thr).mean()), len(leg)
def burden_R(df_eval, scol, lab, thr, gA="LDC", gB="DEV"):
    a=df_eval[df_eval.tier==gA]; b=df_eval[df_eval.tier==gB]
    fa,na=fpr(a,lab,scol,thr); fb,nb=fpr(b,lab,scol,thr)
    # prevalence per tier (all rows, not just legitimate)
    pa=df_eval[df_eval.tier==gA][lab].mean() if len(a) else np.nan
    pb=df_eval[df_eval.tier==gB][lab].mean() if len(b) else np.nan
    if not (fb>0 and pb>0 and pa>0 and fa==fa and fb==fb): return np.nan,fa,fb,pa,pb
    return (fa/fb)/(pa/pb), fa, fb, pa, pb

def boot_R(df_eval, scol, lab, thr, frameB=False, gA="LDC", gB="DEV", cluster=True):
    # returns array of B bootstrap R's. cluster=True resamples IMPORTERS; else rows.
    sub=df_eval[df_eval.tier.isin([gA,gB])].copy()
    base_pi={gA:sub[sub.tier==gA][lab].mean(), gB:sub[sub.tier==gB][lab].mean()}
    out=[]
    if cluster:
        # group rows by importer within the two tiers
        idx_by_imp=sub.groupby("importer_id").indices
        imps=np.array(list(idx_by_imp.keys()))
        arrs=[sub.index.get_indexer(sub.index[idx_by_imp[k]]) for k in imps] # positional
        pos={k:np.where(sub.index.isin(sub.index[idx_by_imp[k]]))[0] for k in imps}
    sub=sub.reset_index(drop=True)
    rows_by_imp={k:np.where(sub.importer_id.values==k)[0] for k in sub.importer_id.unique()} if cluster else None
    imp_keys=np.array(list(rows_by_imp.keys())) if cluster else None
    n_rows=len(sub)
    for _ in range(B):
        if cluster:
            pick=RNG.choice(imp_keys, size=len(imp_keys), replace=True)
            ridx=np.concatenate([rows_by_imp[k] for k in pick])
        else:
            ridx=RNG.integers(0, n_rows, n_rows)
        bs=sub.iloc[ridx]
        a=bs[bs.tier==gA]; b=bs[bs.tier==gB]
        la=a[a[lab]==0]; lb=b[b[lab]==0]
        if len(lb)==0 or len(la)==0: out.append(np.nan); continue
        fa=(la[scol].values>=thr).mean(); fb=(lb[scol].values>=thr).mean()
        if frameB:
            pa,pb=base_pi[gA],base_pi[gB]               # pi fixed
        else:
            pa=a[lab].mean(); pb=b[lab].mean()          # pi random
        if fb>0 and pa>0 and pb>0:
            out.append((fa/fb)/(pa/pb))
        else: out.append(np.nan)
    return np.array(out, float)

def ci(arr):
    a=arr[~np.isnan(arr)]
    if len(a)<50: return (np.nan,np.nan,np.nan)
    return (np.nanmedian(a), np.percentile(a,2.5), np.percentile(a,97.5))
def p_two_sided(arr, null=1.0):
    a=arr[~np.isnan(arr)]
    if len(a)<50: return np.nan
    p=2*min((a<=null).mean(),(a>=null).mean()); return min(1.0,max(p,1/len(a)))

# ---- run: both labels x budgets, point R + clustered & naive CI + both frames ----
rows=[]
for lab,scol in [("fraud","score_fraud"),("crit2","score_crit2")]:
    for b in BUDGETS:
        thr=tau_at(va[scol].values, b)
        R,fa,fb,pa,pb=burden_R(te,scol,lab,thr)
        bc_A=boot_R(te,scol,lab,thr,frameB=False,cluster=True)
        bc_B=boot_R(te,scol,lab,thr,frameB=True ,cluster=True)
        bn_A=boot_R(te,scol,lab,thr,frameB=False,cluster=False)
        mc,lcA,hcA=ci(bc_A); _,lcB,hcB=ci(bc_B); mn,lnA,hnA=ci(bn_A)
        # design effect ~ (clustered CI width / naive CI width)^2
        deff=((hcA-lcA)/(hnA-lnA))**2 if (hnA-lnA)>0 and hcA==hcA else np.nan
        rows.append(dict(label=lab,budget=b,R=R,fpr_LDC=fa,fpr_DEV=fb,pi_LDC=pa,pi_DEV=pb,
            clus_A_lo=lcA,clus_A_hi=hcA,clus_A_p=p_two_sided(bc_A),
            clus_B_lo=lcB,clus_B_hi=hcB,clus_B_p=p_two_sided(bc_B),
            naive_A_lo=lnA,naive_A_hi=hnA,naive_A_p=p_two_sided(bn_A),
            design_effect=deff))
res=pd.DataFrame(rows)

# ---- BH across the (label x budget) cells, clustered Frame A and Frame B ----
def bh(pvals, a=0.05):
    p=np.array(pvals,float); m=np.sum(~np.isnan(p)); order=np.argsort(np.nan_to_num(p,nan=1))
    sig=np.zeros(len(p),bool); thresh=0
    for rank,i in enumerate(order,1):
        if not np.isnan(p[i]) and p[i]<=rank/m*a: thresh=rank
    for rank,i in enumerate(order,1):
        if rank<=thresh: sig[i]=True
    return sig
res["BH_clusA"]=bh(res["clus_A_p"]); res["BH_clusB"]=bh(res["clus_B_p"])

pd.set_option("display.width",200,"display.max_columns",40)
print("\n===== CUSTOMS BURDEN RATIO  R_{LDC,DEV}  (strict UN-LDC vs OECD) =====")
for _,r in res.iterrows():
    print("%-6s @%4.0f%%  R=%6.3f | clusA[%.2f,%.2f]p=%.3f | clusB[%.2f,%.2f]p=%.3f | naiveA[%.2f,%.2f]p=%.3f | deff=%.2f"%(
        r.label,100*r.budget,r.R,r.clus_A_lo,r.clus_A_hi,r.clus_A_p,
        r.clus_B_lo,r.clus_B_hi,r.clus_B_p,r.naive_A_lo,r.naive_A_hi,r.naive_A_p,r.design_effect))
print("BH(0.05) clustered Frame A sig:",int(res["BH_clusA"].sum()),
      "| Frame B sig:",int(res["BH_clusB"].sum()))

# ---- AUBC: area under the budget-continuous burden curve (trapezoid over budgets) ----
for lab in ["fraud","crit2"]:
    g=res[res.label==lab].sort_values("budget")
    aubc=np.trapz(g.R.values, g.budget.values)/(g.budget.max()-g.budget.min())
    print("AUBC[%s] = %.3f (mean R across budget range; 1.0 = parity)"%(lab,aubc))

# ---- per-origin one-vs-rest robustness sweep (each origin vs all others) ----
def sweep(lab, scol, b=0.10, min_leg=30):
    thr=tau_at(va[scol].values,b); rowsS=[]
    for o in te["origin"].unique():
        grp=te.assign(t=np.where(te.origin==o,"A","B"))
        a=grp[grp.t=="A"]; la=a[a[lab]==0]
        if len(la)<min_leg: continue
        fa=(la[scol].values>=thr).mean(); pa=a[lab].mean()
        b_=grp[grp.t=="B"]; lb=b_[b_[lab]==0]; fb=(lb[scol].values>=thr).mean(); pb=b_[lab].mean()
        if fb>0 and pa>0 and pb>0:
            rowsS.append((o,len(a),(fa/fb)/(pa/pb)))
    s=pd.DataFrame(rowsS,columns=["origin","n","R"]).sort_values("R",ascending=False)
    frac=((np.sign(s.R-1)).abs().mean())  # all defined; sign share
    over=(s.R>1.2).mean(); under=(s.R<0.8).mean()
    print("\n[per-origin sweep %s @%.0f%%] origins with >=%d legit rows: %d"%(lab,100*b,min_leg,len(s)))
    print("  fraction R>1.2: %.2f | R<0.8: %.2f | median R: %.3f"%(over,under,s.R.median()))
    print(s.head(8).to_string(index=False)); print("  ...tail:"); print(s.tail(5).to_string(index=False))
    return s
sw_f=sweep("fraud","score_fraud"); sw_c=sweep("crit2","score_crit2")

res.to_csv(OUT+"customs_burden_results.csv",index=False)
sw_f.to_csv(OUT+"customs_perorigin_fraud.csv",index=False)
sw_c.to_csv(OUT+"customs_perorigin_crit2.csv",index=False)
print("\n[saved] customs_burden_results.csv + per-origin sweeps ->",OUT)
print("[DONE CX2] paste the printed block back.")
