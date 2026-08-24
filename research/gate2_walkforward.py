import json, glob, os, numpy as np, pandas as pd, math, warnings
warnings.filterwarnings('ignore')
frames={}
for f in sorted(glob.glob('user_data/data/binance/*-1d.json')):
    sym=os.path.basename(f).replace('-1d.json','').replace('_','/')
    raw=json.load(open(f)); d=pd.DataFrame(raw,columns=['ts','o','h','l','close','volume'])
    d['date']=pd.to_datetime(d['ts'],unit='ms',utc=True); frames[sym]=d.set_index('date')[['close','volume']]
close=pd.DataFrame({k:v['close'] for k,v in frames.items()}).sort_index()
vol=pd.DataFrame({k:v['volume'] for k,v in frames.items()}).sort_index()
close=close.dropna(axis=1,thresh=int(len(close)*0.95)); vol=vol[close.columns]; ret=close.pct_change()
feats={}
for lb in [5,10,20,60]: feats[f'mom{lb}']=close.pct_change(lb)
for lb in [2,3]: feats[f'rev{lb}']=-close.pct_change(lb)
feats['vol20']=ret.rolling(20).std(); feats['volm']=np.log1p(vol).diff(5)
feats['dist_hi']=close/close.rolling(60).max()-1
def xs(d): return d.sub(d.mean(axis=1),axis=0).div(d.std(axis=1).replace(0,np.nan),axis=0)
H=5; fwd=close.shift(-H)/close-1
X={k:xs(v) for k,v in feats.items()}; Y=xs(fwd)
rows=[]
for dt in close.index:
    if dt not in Y.index: continue
    df=pd.DataFrame({k:v.loc[dt] for k,v in X.items() if dt in v.index})
    df['y']=Y.loc[dt]; df['date']=dt; rows.append(df.dropna())
panel=pd.concat(rows).reset_index().rename(columns={'index':'asset'})
fcols=[c for c in panel.columns if c not in ('asset','y','date')]
dates=np.sort(panel['date'].unique()); FEE=0.001; TOPN=8

def fit(tr):
    Xt=np.column_stack([np.ones(len(tr)),tr[fcols].values])
    A=Xt.T@Xt+1.0*np.eye(Xt.shape[1]); A[0,0]-=1.0
    return np.linalg.solve(A,Xt.T@tr['y'].values)

def perf(beta,dts):
    eq=1.0;c=[]
    for dt in dts:
        sub=panel[panel['date']==dt]
        if len(sub)<10: continue
        pr=np.column_stack([np.ones(len(sub)),sub[fcols].values])@beta
        s=pd.Series(pr,index=sub['asset'].values).sort_values(ascending=False)
        nxt=dt+pd.Timedelta(days=H)
        fut=close.index[(close.index>dt)&(close.index<=nxt)]
        if len(fut)==0: continue
        rl=(close.loc[fut[-1],s.head(TOPN).index]/close.loc[dt,s.head(TOPN).index]-1).mean()
        rs=-(close.loc[fut[-1],s.tail(TOPN).index]/close.loc[dt,s.tail(TOPN).index]-1).mean()
        eq*=(1+0.5*rl+0.5*rs-2*FEE); c.append(eq)
    if len(c)<5: return None
    a=np.array(c); rr=np.diff(np.r_[1.0,a])/np.r_[1.0,a[:-1]]
    return a[-1]-1, rr.mean()/rr.std()*math.sqrt(365/H) if rr.std() else 0

def bench(dts):
    d0=dts[0]; d1=min(dts[-1]+pd.Timedelta(days=H),close.index[-1])
    ew=(close.loc[d1]/close.loc[d0]-1).mean(); bt=close.loc[d1,'BTC/USDT']/close.loc[d0,'BTC/USDT']-1
    return ew,bt

print("="*82); print(f"WALK-FORWARD  (expanding train, {TOPN}-long/{TOPN}-short, fee {FEE*100}%/side)"); print("="*82)
print(f"{'fold':>4} {'test period':>26} | {'strategy':>10} {'Sharpe':>8} | {'eq-wt':>9} {'BTC':>9} | beat?")
print("-"*82)
NF=6; start=int(len(dates)*0.40); step=(len(dates)-start)//NF
wins=0; tot=0
for i in range(NF):
    te0=start+i*step; te1=min(te0+step,len(dates))
    if te1-te0<20: continue
    tr=panel[panel['date']<dates[te0]]
    if len(tr)<2000: continue
    b=fit(tr); dts=list(dates[te0:te1])[::H]
    r=perf(b,dts)
    if not r: continue
    ew,bt=bench(dts); tot+=1
    beat = r[0]>ew and r[0]>bt
    wins+= 1 if beat else 0
    print(f"{i+1:>4} {str(pd.Timestamp(dates[te0]).date())+' to '+str(pd.Timestamp(dates[te1-1]).date()):>26} | "
          f"{r[0]*100:+9.2f}% {r[1]:+8.2f} | {ew*100:+8.2f}% {bt*100:+8.2f}% | {'YES' if beat else 'no'}")
print("-"*82)
print(f"  beat BOTH benchmarks in {wins}/{tot} folds")
print(f"  GATE 3 threshold: >= 6/8 folds  ->  {'PASS' if tot and wins/tot>=0.75 else 'FAIL'}")
