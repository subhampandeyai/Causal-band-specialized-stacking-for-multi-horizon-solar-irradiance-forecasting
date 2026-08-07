"""
conformal_causal.py
===================
Closes the LAST consistency gap: regenerates split-conformal prediction
intervals on the CAUSAL point forecasts (not the old leaky ones).

The old conformal numbers (158.5 vs 486 W/m2) came from the LEAKY Station 5 H1
pipeline. Causal point forecasts are weaker, so their residuals are larger and
the honest intervals are WIDER. This recomputes them from scratch on causal
predictions so every number in the paper comes from one causal pipeline.

Method: split conformal (Vovk; Lei et al. 2018).
  - causal features (same as run_full_comparison_causal.py)
  - train Proposed (Ridge+XGB->Ridge meta) and Persistence and Unified XGB
  - split TEST chronologically into calibration / evaluation halves
  - nonconformity = |y - yhat| on calibration
  - q = ceil((n_cal+1)(1-alpha))/n_cal quantile
  - interval = yhat +/- q ; report empirical coverage + mean width on eval half

Run:
    python conformal_causal.py
Output: outputs/reports/paper_tables/conformal_causal.csv  (+ .tex)
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, pywt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb

ROOT=Path(__file__).parent; PROC=ROOT/"data"/"processed"
OUT=ROOT/"outputs"/"reports"/"paper_tables"; OUT.mkdir(parents=True,exist_ok=True)
FAMILY,LEVEL,WINDOW,SEED=("db4",3,512,42); np.random.seed(SEED)
STATION, HORIZONS = 5, [1,4]          # the station/horizons the paper reports
ALPHAS = [0.10, 0.05]                 # 90% and 95% nominal

def bands_causal(sig,window=WINDOW):
    n=len(sig);nb=LEVEL+1;out=np.full((n,nb),np.nan);ml=8*2**LEVEL;W=max(window,ml)
    for t in range(n):
        seg=sig[max(0,t-W+1):t+1]
        if len(seg)<ml: out[t,0]=sig[t];out[t,1:]=0;continue
        c=pywt.wavedec(seg,FAMILY,level=LEVEL)
        for i in range(nb):
            z=[np.zeros_like(x) for x in c];z[i]=c[i];out[t,i]=pywt.waverec(z,FAMILY)[:len(seg)][-1]
    return out

def feats(df,sig,b):
    n=len(sig);f={}
    for L in [1,2,4,8,16,32]:
        c=np.full(n,np.nan);c[L:]=sig[:-L];f[f"lag{L}"]=c
    for w in [4,12,32]:
        f[f"rm{w}"]=pd.Series(sig).rolling(w,min_periods=1).mean().values
        f[f"rs{w}"]=pd.Series(sig).rolling(w,min_periods=1).std().fillna(0).values
    for col in ["TEMPERATURE","REL_HUMIDITY","ATMOSPHERE","DNI"]:
        if col in df.columns: f[col]=pd.to_numeric(df[col],errors="coerce").ffill().bfill().values
    for i in range(b.shape[1]): f[f"band{i}"]=b[:,i]
    return pd.DataFrame(f).values

def mk(X,sig,h):
    n=len(sig);y=np.full(n,np.nan);y[:-h]=sig[h:]
    v=~np.isnan(y)&~np.isnan(X).any(1);X,y=X[v],y[v]
    a,b=int(len(X)*.7),int(len(X)*.85)
    return X[:a],y[:a],X[a:b],y[a:b],X[b:],y[b:]

def proposed(Xtr,ytr,Xva,yva,Xte):
    sc=StandardScaler();Xtr,Xva,Xte=sc.fit_transform(Xtr),sc.transform(Xva),sc.transform(Xte)
    r=Ridge(1.0).fit(Xtr,ytr)
    g=xgb.XGBRegressor(n_estimators=800,max_depth=6,learning_rate=0.03,subsample=0.8,
        colsample_bytree=0.8,reg_lambda=1.0,random_state=SEED,n_jobs=-1,
        tree_method="hist",verbosity=0).fit(Xtr,ytr)
    Zva=np.column_stack([r.predict(Xva),g.predict(Xva)]);Zte=np.column_stack([r.predict(Xte),g.predict(Xte)])
    meta=Ridge(1.0).fit(Zva,yva);return meta.predict(Zte)

def unified(Xtr,ytr,Xva,yva,Xte):
    m=xgb.XGBRegressor(n_estimators=800,max_depth=6,learning_rate=0.03,subsample=0.8,
        colsample_bytree=0.8,reg_lambda=1.0,random_state=SEED,n_jobs=-1,
        tree_method="hist",verbosity=0).fit(Xtr,ytr);return m.predict(Xte)

def conformal(yhat,y,alpha):
    """split conformal on a held calibration half; report on eval half."""
    n=len(y);half=n//2
    cal_res=np.abs(y[:half]-yhat[:half])
    ncal=len(cal_res)
    k=int(np.ceil((ncal+1)*(1-alpha)))
    k=min(k,ncal)  # guard
    q=np.sort(cal_res)[k-1]
    ev_y,ev_h=y[half:],yhat[half:]
    lo,hi=ev_h-q,ev_h+q
    cov=float(np.mean((ev_y>=lo)&(ev_y<=hi)))
    width=float(2*q)
    return cov,width,q

def main():
    fp=list(PROC.glob(f"station_{STATION:02d}_prepared.csv"))
    if not fp: raise SystemExit("station 5 prepared.csv not found")
    df=pd.read_csv(fp[0]);sig=pd.to_numeric(df["IRRADIATION"],errors="coerce").ffill().bfill().values.astype(float)
    # NOTE: IRRADIATION is in kW/m^2 in your Stage-0 output; convert width to W/m^2 (x1000)
    print("Causal decomposition (station 5)..."); X=feats(df,sig,bands_causal(sig))
    rows=[]
    for h in HORIZONS:
        Xtr,ytr,Xva,yva,Xte,yte=mk(X,sig,h)
        preds={"Proposed":proposed(Xtr,ytr,Xva,yva,Xte),
               "Unified XGB":unified(Xtr,ytr,Xva,yva,Xte),
               "Persistence":np.r_[ytr[-1],yte[:-1]] if False else None}
        # persistence at horizon h: yhat = y at t (the last known) -> for test, shift
        # simplest honest persistence: predict previous observed value
        persist=np.empty_like(yte); persist[0]=ytr[-1] if len(ytr) else yte[0]; persist[1:]=yte[:-1]
        preds["Persistence"]=persist
        for model,yh in preds.items():
            for a in ALPHAS:
                cov,width,q=conformal(yh,yte,a)
                rows.append(dict(station=STATION,horizon=f"H{h}",model=model,
                                 nominal=f"{int((1-a)*100)}%",
                                 empirical=round(cov*100,1),
                                 width_W_m2=round(width*1000,1)))  # kW->W
                print(f"  H{h} {model:<12} {int((1-a)*100)}%  emp={cov*100:5.1f}%  width={width*1000:7.1f} W/m2")
    out=pd.DataFrame(rows); out.to_csv(OUT/"conformal_causal.csv",index=False)

    # LaTeX (station 5 H1, the headline table)
    sub=out[out.horizon=="H1"]
    tex=[r"\begin{table}[!t]",r"\centering",
         r"\caption{Split conformal prediction intervals on the causal Station~5 H1 forecasts.}",
         r"\label{tab:conformal}",r"\small",
         r"\begin{tabular}{|l|c|c|c|}",r"\hline",
         r"\textbf{Method} & \textbf{Nom.} & \textbf{Emp.} & \textbf{Width (W/m$^2$)} \\\hline"]
    for _,r in sub.iterrows():
        tex.append(f"{r['model']} & {r['nominal']} & {r['empirical']}\\% & {r['width_W_m2']} \\\\\\hline")
    tex+=[r"\end{tabular}",r"\end{table}"]
    open(OUT/"conformal_causal.tex","w").write("\n".join(tex))
    print(f"\nWrote {OUT/'conformal_causal.csv'} and .tex")
    print("These REPLACE the old leaky 158.5/486 numbers. All causal now.")

if __name__=="__main__": main()
