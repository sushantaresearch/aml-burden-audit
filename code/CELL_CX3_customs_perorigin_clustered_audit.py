# =====================================================================
# CELL_CX3_customs_perorigin_clustered_audit.py
# Thesis C, step 3 (publication-grade): PER-ORIGIN burden-ratio audit
# on the Korea/Jeong customs detector scores.
#   PRIMARY  = per-origin one-vs-rest R distribution, importer-CLUSTERED
#              bootstrap CIs (both pi-frames), FP-count guard, BH across
#              origins, at a primary budget; + budget sweep -> per-origin AUBC.
#   SECONDARY= non-OECD vs OECD contrast across budgets, clustered vs naive
#              (design-effect), both frames.
#   Fixes vs CX2: (1) FP guard kills R=0 thin-slice artifacts; (2) AUBC
#   guarded over estimable budgets; (3) sweep now has clustered CIs + BH.
#
# Correct two-sample clustered bootstrap: resample IMPORTERS once over the
# whole sample each iteration (preserves within-importer FP correlation and
# multi-origin importers), then recompute every group's R from that resample.
#
# HONESTY BOUNDARY (unchanged): R reported as-is; "not estimable" when the
# FP base is too thin; inversions are findings; no significance manufactured.
# Consumes customs_audit_test.csv + _valid.csv from CX1. Self-contained.
# =====================================================================
import os, numpy as np, pandas as pd

OUT="/content/drive/MyDrive/Kaggle/gnn_outputs/"
if not os.path.exists(OUT+"customs_audit_test.csv"): OUT="./"
te=pd.read_csv(OUT+"customs_audit_test.csv").reset_index(drop=True)
va=pd.read_csv(OUT+"customs_audit_valid.csv").reset_index(drop=True)
print("loaded test/valid:",te.shape,va.shape)

OECD={"AU","AT","BE","CA","CL","CZ","DK","EE","FI","FR","DE","GR","HU","IS","IE",
 "IL","IT","JP","KR","LV","LT","LU","NL","NZ","NO","PL","PT","SK","SI","ES","SE",
 "CH","TR","GB","US","CO","CR","MX"}
te["oecd"]=np.where(te["origin"].str.upper().isin(OECD),"OECD","NON_OECD")

BUDGETS=[0.01,0.02,0.05,0.10,0.20]; PRIMARY_B=0.10
B=2000; MIN_LEG=30; MIN_FP=5      # guards: focal group needs >=30 legit rows AND >=5 legit FPs
RNG=np.random.default_rng(0)

def tau_at(s_valid,b): return float(np.quantile(s_valid,1-b))

# ---- point R for "mask A vs mask B" (legitimate = label==0; FP = legit & score>=thr) ----
def point_R(df,scol,lab,thr,mA,mB):
    A=df[mA]; Bd=df[mB]; lA=A[A[lab]==0]; lB=Bd[Bd[lab]==0]
    nfpA=int((lA[scol].values>=thr).sum()); nfpB=int((lB[scol].values>=thr).sum())
    if len(lA)<MIN_LEG or nfpA<MIN_FP or len(lB)==0 or nfpB==0: 
        return np.nan,len(lA),nfpA
    fA=nfpA/len(lA); fB=nfpB/len(lB)
    pA=A[lab].mean(); pB=Bd[lab].mean()
    if pA<=0 or pB<=0 or fB<=0: return np.nan,len(lA),nfpA
    return (fA/fB)/(pA/pB),len(lA),nfpA

# ---- one importer-level resample of the WHOLE sample -> row indices ----
rows_by_imp={k:np.where(te["importer_id"].values==k)[0] for k in te["importer_id"].unique()}
imp_keys=np.array(list(rows_by_imp.keys()))
def resample_rows():
    pick=RNG.choice(imp_keys,size=len(imp_keys),replace=True)
    return np.concatenate([rows_by_imp[k] for k in pick])

def clustered_R_dist(scol,lab,thr,groupcol,levels,frameB):
    """returns dict level-> array(R) over B importer-resamples (one-vs-rest per level)."""
    base_pi={lv: te[te[groupcol]==lv][lab].mean() for lv in levels}
    acc={lv:[] for lv in levels}
    gcol=te[groupcol].values; y=te[lab].values; sc=te[scol].values
    for _ in range(B):
        ridx=resample_rows()
        gg=gcol[ridx]; yy=y[ridx]; ss=sc[ridx]
        legit=yy==0; fp=legit&(ss>=thr)
        for lv in levels:
            inA=gg==lv; inB=~inA
            lA=legit&inA; lB=legit&inB
            nlA=lA.sum(); nlB=lB.sum()
            if nlA<MIN_LEG or fp[lA].sum()<MIN_FP or nlB==0 or fp[lB].sum()==0:
                acc[lv].append(np.nan); continue
            fA=fp[lA].sum()/nlA; fB=fp[lB].sum()/nlB
            if frameB: pA,pB=base_pi[lv], None
            if frameB:
                pA=base_pi[lv]
                # rest prevalence fixed too:
                pB=te[te[groupcol]!=lv][lab].mean()
            else:
                pA=yy[inA].mean(); pB=yy[inB].mean()
            acc[lv].append((fA/fB)/(pA/pB) if (pA>0 and pB>0 and fB>0) else np.nan)
    return {lv:np.array(acc[lv],float) for lv in levels}

def summ(arr,null=1.0):
    a=arr[~np.isnan(arr)]
    if len(a)<50: return (np.nan,np.nan,np.nan,np.nan,len(a))
    med=np.median(a); lo,hi=np.percentile(a,[2.5,97.5])
    p=min(1.0,max(2*min((a<=null).mean(),(a>=null).mean()),1/len(a)))
    return (med,lo,hi,p,len(a))

def bh(p,a=0.05):
    p=np.array(p,float); m=np.sum(~np.isnan(p)); order=np.argsort(np.nan_to_num(p,nan=1.0)); k=0
    for rank,i in enumerate(order,1):
        if not np.isnan(p[i]) and p[i]<=rank/m*a: k=rank
    sig=np.zeros(len(p),bool)
    for rank,i in enumerate(order,1):
        if rank<=k: sig[i]=True
    return sig

# ============ PRIMARY: per-origin one-vs-rest at PRIMARY_B ============
print("\n================ PER-ORIGIN BURDEN RATIO  (one-vs-rest, budget %.0f%%) ================"%(100*PRIMARY_B))
for lab,scol in [("fraud","score_fraud"),("crit2","score_crit2")]:
    thr=tau_at(va[scol].values,PRIMARY_B)
    # which origins pass the guard (point estimate first)
    origins=[]
    for o in te["origin"].unique():
        R,nl,nfp=point_R(te,scol,lab,thr,te["origin"]==o,te["origin"]!=o)
        if not np.isnan(R): origins.append(o)
    dA=clustered_R_dist(scol,lab,thr,"origin",origins,frameB=False)
    dB=clustered_R_dist(scol,lab,thr,"origin",origins,frameB=True)
    recs=[]
    for o in origins:
        R,nl,nfp=point_R(te,scol,lab,thr,te["origin"]==o,te["origin"]!=o)
        mA,loA,hiA,pA,_=summ(dA[o]); mB,loB,hiB,pB,_=summ(dB[o])
        recs.append(dict(origin=o,n=int((te["origin"]==o).sum()),legit=nl,nfp=nfp,R=R,
            clusA_lo=loA,clusA_hi=hiA,clusA_p=pA,clusB_lo=loB,clusB_hi=hiB,clusB_p=pB))
    d=pd.DataFrame(recs).sort_values("R",ascending=False)
    d["BH_clusA"]=bh(d["clusA_p"].values); d["BH_clusB"]=bh(d["clusB_p"].values)
    print("\n--- %s : %d estimable origins (>=%d legit, >=%d FP) ---"%(lab,len(d),MIN_LEG,MIN_FP))
    for _,r in d.iterrows():
        star="*" if r.BH_clusB else (" " )
        print(" %s%-3s n=%4d fp=%3d  R=%6.3f | clusA[%.2f,%.2f]p=%.3f | clusB[%.2f,%.2f]p=%.3f"%(
            star,r.origin,r.n,r.nfp,r.R,r.clusA_lo,r.clusA_hi,r.clusA_p,r.clusB_lo,r.clusB_hi,r.clusB_p))
    frac_over=(d.R>1.2).mean(); frac_under=(d.R<0.8).mean()
    print("  median R=%.3f | frac R>1.2: %.2f | frac R<0.8: %.2f | BH-sig (FrameB): %d/%d origins"%(
        d.R.median(),frac_over,frac_under,int(d["BH_clusB"].sum()),len(d)))
    d.to_csv(OUT+"customs_perorigin_%s.csv"%lab,index=False)

# ============ per-origin AUBC across budgets (point estimates, guarded) ============
print("\n================ PER-ORIGIN AUBC (mean R across estimable budgets) ================")
for lab,scol in [("fraud","score_fraud"),("crit2","score_crit2")]:
    rows=[]
    for o in te["origin"].unique():
        Rs=[]; bs=[]
        for b in BUDGETS:
            thr=tau_at(va[scol].values,b)
            R,_,_=point_R(te,scol,lab,thr,te["origin"]==o,te["origin"]!=o)
            if not np.isnan(R): Rs.append(R); bs.append(b)
        if len(Rs)>=3:
            aubc=np.trapezoid(Rs,bs)/(max(bs)-min(bs))
            rows.append((o,len(Rs),round(aubc,3)))
    a=pd.DataFrame(rows,columns=["origin","n_budgets","AUBC"]).sort_values("AUBC",ascending=False)
    print("\n[%s] origins with >=3 estimable budgets: %d | median AUBC=%.3f"%(lab,len(a),a.AUBC.median()))
    print(a.to_string(index=False))

# ============ SECONDARY: non-OECD vs OECD across budgets (clustered vs naive) ============
print("\n================ NON-OECD vs OECD  (clustered vs naive, design effect) ================")
def naive_R_dist(scol,lab,thr,frameB):
    base_pi={"NON_OECD":te[te.oecd=="NON_OECD"][lab].mean(),"OECD":te[te.oecd=="OECD"][lab].mean()}
    y=te[lab].values; sc=te[scol].values; g=te["oecd"].values; n=len(te); out=[]
    for _ in range(B):
        ridx=RNG.integers(0,n,n); gg=g[ridx]; yy=y[ridx]; ss=sc[ridx]
        legit=yy==0; fp=legit&(ss>=thr); inA=gg=="NON_OECD"
        lA=legit&inA; lB=legit&~inA
        if lA.sum()==0 or lB.sum()==0 or fp[lA].sum()==0 or fp[lB].sum()==0: out.append(np.nan); continue
        fA=fp[lA].sum()/lA.sum(); fB=fp[lB].sum()/lB.sum()
        pA,pB=(base_pi["NON_OECD"],base_pi["OECD"]) if frameB else (yy[inA].mean(),yy[~inA].mean())
        out.append((fA/fB)/(pA/pB) if (pA>0 and pB>0 and fB>0) else np.nan)
    return np.array(out,float)
secrows=[]
for lab,scol in [("fraud","score_fraud"),("crit2","score_crit2")]:
    for b in BUDGETS:
        thr=tau_at(va[scol].values,b)
        R,nl,nfp=point_R(te,scol,lab,thr,te.oecd=="NON_OECD",te.oecd=="OECD")
        cA=clustered_R_dist(scol,lab,thr,"oecd",["NON_OECD"],frameB=False)["NON_OECD"]
        cB=clustered_R_dist(scol,lab,thr,"oecd",["NON_OECD"],frameB=True )["NON_OECD"]
        nA=naive_R_dist(scol,lab,thr,frameB=False)
        mcA,lcA,hcA,pcA,_=summ(cA); mcB,lcB,hcB,pcB,_=summ(cB); mnA,lnA,hnA,pnA,_=summ(nA)
        deff=((hcA-lcA)/(hnA-lnA))**2 if (hnA-lnA)>0 and hcA==hcA else np.nan
        secrows.append(dict(label=lab,budget=b,R=R,clusA_lo=lcA,clusA_hi=hcA,clusA_p=pcA,
            clusB_lo=lcB,clusB_hi=hcB,clusB_p=pcB,naive_lo=lnA,naive_hi=hnA,naive_p=pnA,deff=deff))
sec=pd.DataFrame(secrows)
for _,r in sec.iterrows():
    print(" %-6s @%4.0f%% R=%6.3f | clusA[%.2f,%.2f]p=%.3f | clusB[%.2f,%.2f]p=%.3f | naive[%.2f,%.2f]p=%.3f | deff=%.2f"%(
        r.label,100*r.budget,r.R,r.clusA_lo,r.clusA_hi,r.clusA_p,r.clusB_lo,r.clusB_hi,r.clusB_p,r.naive_lo,r.naive_hi,r.naive_p,r.deff))
sec["BH_clusB"]=bh(sec["clusB_p"].values)
print(" BH(0.05) non-OECD/OECD Frame B sig cells:",int(sec["BH_clusB"].sum()),"/",len(sec),
      "| median deff=%.2f"%sec["deff"].median())
sec.to_csv(OUT+"customs_nonoecd_oecd.csv",index=False)
print("\n[saved] per-origin + AUBC + non-OECD/OECD CSVs ->",OUT)
print("[DONE CX3] paste the printed block back; then we lock the headline from these R values.")
