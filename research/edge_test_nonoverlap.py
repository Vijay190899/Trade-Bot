import pandas as pd, numpy as np, glob, json, math

fs=sorted(glob.glob(r'user_data/models/v3_test/backtesting_predictions/*.feather'))
pred=pd.concat([pd.read_feather(f) for f in fs],ignore_index=True)
pred=pred.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
pred['action']=pred['&-s-close'].round().astype(int)
raw=json.load(open(r'user_data/data/binance/ETH_USDT-1h.json'))
px=pd.DataFrame(raw,columns=['ts','open','high','low','close','volume'])
px['date']=pd.to_datetime(px['ts'],unit='ms',utc=True)
df=pred.merge(px[['date','close']],on='date',how='inner')

def welch(a,b):
    va,vb=a.var(ddof=1),b.var(ddof=1)
    se=math.sqrt(va/len(a)+vb/len(b))
    if se==0: return 0,1
    t=(a.mean()-b.mean())/se
    p=2*(1-0.5*(1+math.erf(abs(t)/math.sqrt(2))))
    return t,p

print("="*72)
print("NON-OVERLAPPING TEST  (sample every h candles so windows don't overlap)")
print("="*72)
for h in [24,48]:
    df[f'fwd{h}']=df['close'].shift(-h)/df['close']-1
    best=None
    # average across all possible phase offsets to avoid cherry-picking
    ts=[];ps=[];edges=[];ns=[]
    for off in range(h):
        sub=df.iloc[off::h].dropna(subset=[f'fwd{h}'])
        a=sub[sub['action']==1][f'fwd{h}']; b=sub[sub['action']!=1][f'fwd{h}']
        if len(a)<8 or len(b)<8: continue
        t,p=welch(a,b)
        ts.append(t);ps.append(p);edges.append((a.mean()-b.mean())*100);ns.append(len(a))
    if ts:
        print(f"\n  fwd {h}h  ({len(ts)} independent phase-offset samples)")
        print(f"    mean edge      : {np.mean(edges):+.3f} pp")
        print(f"    mean t-stat    : {np.mean(ts):+.2f}")
        print(f"    median p-value : {np.median(ps):.3f}")
        print(f"    signal count/sample: ~{int(np.mean(ns))}")
        frac=sum(1 for p in ps if p<0.05)/len(ps)
        print(f"    fraction of offsets significant: {frac*100:.0f}%  "
              f"({'REAL effect' if frac>0.5 else 'not robust - likely artifact'})")

print()
print("="*72)
print("SANITY: is the signal just 'buys dips in a falling market'?")
print("="*72)
df['ret24_past']=df['close']/df['close'].shift(24)-1
sig=df[df['action']==1]['ret24_past'].dropna()
rest=df[df['action']!=1]['ret24_past'].dropna()
t,p=welch(sig,rest)
print(f"  prior 24h return when signalling: {sig.mean()*100:+.3f}%")
print(f"  prior 24h return otherwise      : {rest.mean()*100:+.3f}%")
print(f"  -> model buys after moves of {sig.mean()*100:+.3f}%  (t={t:+.2f}, p={p:.3f})")
print(f"  market over window: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:+.2f}%")
