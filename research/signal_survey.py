import json, numpy as np, pandas as pd, math

def load(p):
    raw=json.load(open(p))
    df=pd.DataFrame(raw,columns=['ts','open','high','low','close','volume'])
    df['date']=pd.to_datetime(df['ts'],unit='ms',utc=True)
    return df

eth=load(r'user_data/data/binance/ETH_USDT-1h.json')
eth['ret']=eth['close'].pct_change()
eth['absret']=eth['ret'].abs()
eth=eth.dropna()

def ac(s,lag):
    a=s.iloc[:-lag].values; b=s.iloc[lag:].values
    return np.corrcoef(a,b)[0,1]

def pval(r,n):
    if abs(r)>=1: return 0.0
    t=r*math.sqrt((n-2)/(1-r*r))
    return 2*(1-0.5*(1+math.erf(abs(t)/math.sqrt(2))))

n=len(eth)
print("="*74)
print(f"WHERE IS THE SIGNAL?  ETH/USDT 1h, n={n} candles ({n/24:.0f} days)")
print("="*74)
print(f"\n{'lag':>5} | {'DIRECTION (ret)':>22} | {'VOLATILITY (|ret|)':>24}")
print(f"{'':>5} | {'r':>9} {'p':>11} | {'r':>9} {'p':>13}")
print("-"*74)
for lag in [1,2,6,12,24,48,168]:
    rd=ac(eth['ret'],lag); rv=ac(eth['absret'],lag)
    pd_=pval(rd,n-lag); pv=pval(rv,n-lag)
    fd='' if pd_>=0.05 else '*'
    fv='' if pv>=0.05 else '*'
    print(f"{lag:5d} | {rd:+9.4f} {pd_:10.3f}{fd:1s} | {rv:+9.4f} {pv:12.3g}{fv:1s}")

print("\n  * = statistically significant (p<0.05)")

# variance ratio of predictability: R^2 from simple AR(1)
print("\n" + "="*74)
print("PRACTICAL PREDICTABILITY (out-of-sample R^2, AR model, 70/30 split)")
print("="*74)
def ar_r2(series, lags=24):
    X=np.column_stack([series.shift(i).values for i in range(1,lags+1)])
    y=series.values
    m=~np.isnan(X).any(axis=1)
    X,y=X[m],y[m]
    cut=int(len(y)*0.7)
    Xtr,ytr,Xte,yte=X[:cut],y[:cut],X[cut:],y[cut:]
    Xtr1=np.column_stack([np.ones(len(Xtr)),Xtr]); Xte1=np.column_stack([np.ones(len(Xte)),Xte])
    beta,*_=np.linalg.lstsq(Xtr1,ytr,rcond=None)
    pred=Xte1@beta
    ss_res=((yte-pred)**2).sum(); ss_tot=((yte-yte.mean())**2).sum()
    return 1-ss_res/ss_tot
print(f"  next-hour RETURN   : R^2 = {ar_r2(eth['ret']):+.5f}   <- ~0 or negative = unpredictable")
print(f"  next-hour |RETURN| : R^2 = {ar_r2(eth['absret']):+.5f}   <- volatility")
