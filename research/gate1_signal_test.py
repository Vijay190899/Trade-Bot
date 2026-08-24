import json, glob, os, numpy as np, pandas as pd, math, warnings
warnings.filterwarnings('ignore')

TF='1d'
files=sorted(glob.glob(f'user_data/data/binance/*-{TF}.json'))
frames={}
for f in files:
    sym=os.path.basename(f).replace(f'-{TF}.json','').replace('_','/')
    raw=json.load(open(f))
    d=pd.DataFrame(raw,columns=['ts','open','high','low','close','volume'])
    d['date']=pd.to_datetime(d['ts'],unit='ms',utc=True)
    frames[sym]=d.set_index('date')[['close','volume']]
print(f"loaded {len(frames)} pairs, timeframe {TF}")

close=pd.DataFrame({k:v['close'] for k,v in frames.items()}).sort_index()
vol  =pd.DataFrame({k:v['volume'] for k,v in frames.items()}).sort_index()
close=close.dropna(axis=1,thresh=int(len(close)*0.95))   # keep pairs with full history
print(f"panel: {close.shape[0]} days x {close.shape[1]} assets  "
      f"({close.index.min().date()} -> {close.index.max().date()})")

ret=close.pct_change()

# ---------- feature construction (all strictly causal) ----------
feats={}
for lb in [5,10,20,60]:
    feats[f'mom{lb}']=close.pct_change(lb)
for lb in [2,3]:
    feats[f'rev{lb}']=-close.pct_change(lb)          # short-term reversal
feats['vol20']=ret.rolling(20).std()
feats['volm']=np.log1p(vol).diff(5)
feats['dist_hi']=close/close.rolling(60).max()-1

H=5   # forward horizon (days)
fwd=close.shift(-H)/close-1

def xs(df):                                   # cross-sectional z-score per date
    return df.sub(df.mean(axis=1),axis=0).div(df.std(axis=1).replace(0,np.nan),axis=0)

X_parts={k:xs(v) for k,v in feats.items()}
y=xs(fwd)                                     # relative forward return

rows=[]
for dt in close.index:
    if dt not in y.index: continue
    yy=y.loc[dt]
    xx={k:v.loc[dt] for k,v in X_parts.items() if dt in v.index}
    if not xx: continue
    df=pd.DataFrame(xx); df['y']=yy; df['date']=dt
    rows.append(df.dropna())
panel=pd.concat(rows).reset_index().rename(columns={'index':'asset'})
print(f"panel observations: {len(panel)}")

fcols=[c for c in panel.columns if c not in ('asset','y','date')]
dates=np.sort(panel['date'].unique())
cut=dates[int(len(dates)*0.70)]
tr=panel[panel['date']<cut]; te=panel[panel['date']>=cut]
print(f"train {len(tr)} obs (to {pd.Timestamp(cut).date()}) | test {len(te)} obs\n")

Xtr=tr[fcols].values; ytr=tr['y'].values
Xte=te[fcols].values; yte=te['y'].values

# ridge
lam=1.0
Xtr1=np.column_stack([np.ones(len(Xtr)),Xtr]); Xte1=np.column_stack([np.ones(len(Xte)),Xte])
A=Xtr1.T@Xtr1+lam*np.eye(Xtr1.shape[1]); A[0,0]-=lam
beta=np.linalg.solve(A,Xtr1.T@ytr)
pred=Xte1@beta

ss_res=((yte-pred)**2).sum(); ss_tot=((yte-yte.mean())**2).sum()
r2=1-ss_res/ss_tot
ic=np.corrcoef(pred,yte)[0,1]
n=len(yte)
t=ic*math.sqrt((n-2)/(1-ic*ic)); p=2*(1-0.5*(1+math.erf(abs(t)/math.sqrt(2))))

print("="*68); print(f"GATE 1 RESULT  (forward horizon {H}d, out-of-sample)"); print("="*68)
print(f"  OOS R^2                 : {r2:+.5f}      threshold > 0.02")
print(f"  Information Coefficient : {ic:+.4f}  p={p:.2e}  n={n}")
print(f"  feature loadings        : " + ", ".join(f"{c}={b:+.3f}" for c,b in zip(fcols,beta[1:])))
print()
print(f"  VERDICT: {'PASS' if r2>0.02 else 'FAIL on R^2 threshold'}")
