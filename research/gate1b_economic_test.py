import json, glob, os, numpy as np, pandas as pd, math, warnings
warnings.filterwarnings('ignore')
TF='1d'
frames={}
for f in sorted(glob.glob(f'user_data/data/binance/*-{TF}.json')):
    sym=os.path.basename(f).replace(f'-{TF}.json','').replace('_','/')
    raw=json.load(open(f)); d=pd.DataFrame(raw,columns=['ts','o','h','l','close','volume'])
    d['date']=pd.to_datetime(d['ts'],unit='ms',utc=True); frames[sym]=d.set_index('date')[['close','volume']]
close=pd.DataFrame({k:v['close'] for k,v in frames.items()}).sort_index()
vol=pd.DataFrame({k:v['volume'] for k,v in frames.items()}).sort_index()
close=close.dropna(axis=1,thresh=int(len(close)*0.95))
vol=vol[close.columns]; ret=close.pct_change()

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
dates=np.sort(panel['date'].unique()); cut=dates[int(len(dates)*0.70)]
tr=panel[panel['date']<cut]
Xtr=np.column_stack([np.ones(len(tr)),tr[fcols].values]); ytr=tr['y'].values
A=Xtr.T@Xtr+1.0*np.eye(Xtr.shape[1]); A[0,0]-=1.0
beta=np.linalg.solve(A,Xtr.T@ytr)

FEE=0.001
te_dates=[d for d in dates if d>=cut][::H]          # NON-OVERLAPPING rebalances
print("="*74); print(f"GATE 1b — ECONOMIC TEST (out-of-sample, {len(te_dates)} rebalances every {H}d, fee {FEE*100}%/side)"); print("="*74)

def run(topn, shorting):
    eq=1.0; curve=[]
    for dt in te_dates:
        sub=panel[panel['date']==dt]
        if len(sub)<10: continue
        pred=np.column_stack([np.ones(len(sub)),sub[fcols].values])@beta
        s=pd.Series(pred,index=sub['asset'].values).sort_values(ascending=False)
        longs=s.head(topn).index
        nxt=dt+pd.Timedelta(days=H)
        fut=close.reindex(close.index[(close.index>dt)&(close.index<=nxt)])
        if fut.empty: continue
        try: r_long=(close.loc[fut.index[-1],longs]/close.loc[dt,longs]-1).mean()
        except Exception: continue
        r=r_long-2*FEE
        if shorting:
            shorts=s.tail(topn).index
            r_short=-(close.loc[fut.index[-1],shorts]/close.loc[dt,shorts]-1).mean()
            r=0.5*r_long+0.5*r_short-2*FEE
        eq*=(1+r); curve.append(eq)
    if not curve: return None
    c=np.array(curve); rs=np.diff(np.r_[1.0,c])/np.r_[1.0,c[:-1]]
    yrs=len(te_dates)*H/365
    cagr=c[-1]**(1/yrs)-1
    sh=rs.mean()/rs.std()*math.sqrt(365/H) if rs.std() else 0
    dd=(c/np.maximum.accumulate(c)-1).min()
    return c[-1]-1,cagr,sh,dd

for topn in [3,5,8]:
    for sm,lbl in [(False,'long-only '),(True,'long/short')]:
        r=run(topn,sm)
        if r: print(f"  top{topn} {lbl}: total {r[0]*100:+8.2f}%  CAGR {r[1]*100:+7.2f}%  Sharpe {r[2]:+5.2f}  maxDD {r[3]*100:6.1f}%")

# benchmarks over identical window
d0,d1=te_dates[0],te_dates[-1]+pd.Timedelta(days=H)
d1=min(d1,close.index[-1])
yrs=(d1-d0).days/365
ew=(close.loc[d1]/close.loc[d0]-1).mean()
btc=close.loc[d1,'BTC/USDT']/close.loc[d0,'BTC/USDT']-1
print(f"\n  BENCHMARKS same window ({d0.date()} -> {d1.date()}, {yrs:.2f}y):")
print(f"    equal-weight all 29 : {ew*100:+8.2f}%  CAGR {((1+ew)**(1/yrs)-1)*100:+7.2f}%")
print(f"    BTC buy & hold      : {btc*100:+8.2f}%  CAGR {((1+btc)**(1/yrs)-1)*100:+7.2f}%")
