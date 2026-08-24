import pandas as pd, numpy as np, glob, json, math

# 1. all out-of-sample walk-forward predictions
fs=sorted(glob.glob(r'user_data/models/v3_test/backtesting_predictions/*.feather'))
pred=pd.concat([pd.read_feather(f) for f in fs],ignore_index=True)
pred=pred.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
pred['action']=pred['&-s-close'].round().astype(int)
print(f"predictions: {len(pred)} rows, {pred['date'].min()} -> {pred['date'].max()}")
print("action distribution:", pred['action'].value_counts().sort_index().to_dict())
print("do_predict==1:", int((pred['do_predict']==1).sum()), "/", len(pred))

# 2. actual price
raw=json.load(open(r'user_data/data/binance/ETH_USDT-1h.json'))
px=pd.DataFrame(raw,columns=['ts','open','high','low','close','volume'])
px['date']=pd.to_datetime(px['ts'],unit='ms',utc=True)
px=px[['date','close']]

df=pred.merge(px,on='date',how='inner')
print(f"merged: {len(df)} rows")

# 3. forward returns at several horizons
for h in [1,6,12,24,48]:
    df[f'fwd{h}']=df['close'].shift(-h)/df['close']-1

df=df[df['do_predict']==1].dropna(subset=['fwd24'])
print(f"usable (do_predict=1, has fwd24): {len(df)}\n")

print("="*70)
print("EDGE TEST 1 — do Long_enter signals precede higher returns?")
print("="*70)
sig=df[df['action']==1]; rest=df[df['action']!=1]
for h in [1,6,12,24,48]:
    c=f'fwd{h}'
    a=sig[c].dropna(); b=rest[c].dropna()
    if len(a)<5 or len(b)<5: continue
    # Welch t-test
    va,vb=a.var(ddof=1),b.var(ddof=1)
    se=math.sqrt(va/len(a)+vb/len(b))
    t=(a.mean()-b.mean())/se if se else 0
    dof=(va/len(a)+vb/len(b))**2/((va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1)) if se else 1
    p=2*(1-0.5*(1+math.erf(abs(t)/math.sqrt(2))))
    edge=(a.mean()-b.mean())*100
    print(f"  fwd {h:2d}h: signal {a.mean()*100:+6.3f}%  vs  other {b.mean()*100:+6.3f}%   "
          f"edge {edge:+6.3f}pp  t={t:+5.2f} p={p:.3f} {'SIGNIFICANT' if p<0.05 else 'noise'}")

print()
print("="*70)
print("EDGE TEST 2 — correlation between action and forward return")
print("="*70)
for h in [1,6,12,24,48]:
    c=f'fwd{h}'
    d2=df[['action',c]].dropna()
    # is_long = action==1 treated as directional signal
    x=(d2['action']==1).astype(float); y=d2[c]
    if x.std()==0: continue
    r=np.corrcoef(x,y)[0,1]
    n=len(d2)
    tt=r*math.sqrt((n-2)/(1-r*r)) if abs(r)<1 else 0
    p=2*(1-0.5*(1+math.erf(abs(tt)/math.sqrt(2))))
    print(f"  fwd {h:2d}h: r={r:+.4f}  n={n}  p={p:.3f}  {'SIGNIFICANT' if p<0.05 else 'noise'}")
