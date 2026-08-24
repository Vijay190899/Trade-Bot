import json, numpy as np, pandas as pd, math

def load(p,name):
    raw=json.load(open(p))
    d=pd.DataFrame(raw,columns=['ts','open','high','low','close','volume'])
    d['date']=pd.to_datetime(d['ts'],unit='ms',utc=True)
    return d[['date','close']].rename(columns={'close':name})

eth=load(r'user_data/data/binance/ETH_USDT-1h.json','ETH')
btc=load(r'user_data/data/binance/BTC_USDT-1h.json','BTC')
df=eth.merge(btc,on='date').sort_values('date').reset_index(drop=True)
df['rE']=df['ETH'].pct_change(); df['rB']=df['BTC'].pct_change()
df=df.dropna().reset_index(drop=True)

def pval(r,n):
    if abs(r)>=1 or n<3: return 1.0
    t=r*math.sqrt((n-2)/(1-r*r))
    return 2*(1-0.5*(1+math.erf(abs(t)/math.sqrt(2))))

print("="*76)
print("TEST A — does RELATIVE momentum predict RELATIVE forward returns?")
print("  (seed test for a cross-sectional long/short strategy)")
print("="*76)
print(f"{'lookback':>9} {'horizon':>8} | {'corr':>9} {'p':>9}  verdict")
print("-"*76)
found=False
for lb in [24,72,168]:
    for hz in [24,72,168]:
        past=(df['ETH']/df['ETH'].shift(lb)-1)-(df['BTC']/df['BTC'].shift(lb)-1)
        fwd=(df['ETH'].shift(-hz)/df['ETH']-1)-(df['BTC'].shift(-hz)/df['BTC']-1)
        m=past.notna()&fwd.notna()
        # non-overlapping
        idx=np.arange(len(df))[m][::max(hz,1)]
        a=past.iloc[idx].values; b=fwd.iloc[idx].values
        if len(a)<30: continue
        r=np.corrcoef(a,b)[0,1]; p=pval(r,len(a))
        sig = p<0.05
        found = found or sig
        print(f"{lb:9d} {hz:8d} | {r:+9.4f} {p:9.3f}  {'SIGNIFICANT' if sig else 'noise'}  (n={len(a)})")

print()
print("="*76)
print("TEST B — is ETH-vs-BTC spread mean-reverting? (pairs-trade seed)")
print("="*76)
df['spread']=np.log(df['ETH'])-np.log(df['BTC'])
df['z']=(df['spread']-df['spread'].rolling(168).mean())/df['spread'].rolling(168).std()
d2=df.dropna(subset=['z']).copy()
d2['fwd_spread']=d2['spread'].shift(-72)-d2['spread']
idx=np.arange(len(d2))[::72]
a=d2['z'].iloc[idx].values; b=d2['fwd_spread'].iloc[idx].values
m=~np.isnan(a)&~np.isnan(b); a,b=a[m],b[m]
r=np.corrcoef(a,b)[0,1]; p=pval(r,len(a))
print(f"  z-score vs forward 72h spread change: r={r:+.4f}  p={p:.4f}  n={len(a)}")
print(f"  -> {'MEAN-REVERTING (tradeable)' if (r<-0.1 and p<0.05) else 'no exploitable reversion'}")

print()
print("="*76)
print("Underlying drift check (is there ANY return to harvest?)")
print("="*76)
for c,lbl in [('ETH','ETH'),('BTC','BTC')]:
    tot=df[c].iloc[-1]/df[c].iloc[0]-1
    yrs=len(df)/24/365
    print(f"  {lbl}: total {tot*100:+8.2f}%  over {yrs:.2f}y  =  CAGR {((1+tot)**(1/yrs)-1)*100:+7.2f}%")
