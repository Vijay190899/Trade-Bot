import json, numpy as np, pandas as pd

raw=json.load(open(r'user_data/data/binance/ETH_USDT-1h.json'))
df=pd.DataFrame(raw,columns=['ts','open','high','low','close','volume'])
df['date']=pd.to_datetime(df['ts'],unit='ms',utc=True)
df['ret']=df['close'].pct_change()
df=df.dropna().reset_index(drop=True)

# predicted vol = EWMA of |ret| (the signal we just proved is predictable)
df['vol']=df['ret'].ewm(span=72).std()
df['vol_pred']=df['vol'].shift(1)          # strictly causal
df=df.dropna().reset_index(drop=True)

TARGET=df['vol_pred'].median()             # target vol level
MAXLEV=1.0                                  # never lever up (spot, no margin)

df['w_vol']=(TARGET/df['vol_pred']).clip(0,MAXLEV)   # vol-targeted exposure
df['w_bh']=1.0                                        # buy & hold

def stats(w,label,ret):
    r=w*ret
    eq=(1+r).cumprod()
    total=eq.iloc[-1]-1
    ann=(eq.iloc[-1])**(24*365/len(r))-1
    vol=r.std()*np.sqrt(24*365)
    sharpe=ann/vol if vol else 0
    dd=(eq/eq.cummax()-1).min()
    print(f"  {label:<26} total {total*100:+8.2f}%   CAGR {ann*100:+7.2f}%   "
          f"vol {vol*100:5.1f}%   Sharpe {sharpe:+5.2f}   maxDD {dd*100:6.1f}%   avg expo {w.mean()*100:5.1f}%")
    return sharpe

print("="*112)
print(f"VOL-TARGETING TEST  ETH/USDT 1h, {len(df)} candles ({len(df)/24/365:.2f} years), spot long-only, no leverage")
print("="*112)
s_bh=stats(df['w_bh'],"buy & hold",df['ret'])
s_vt=stats(df['w_vol'],"vol-targeted exposure",df['ret'])

# out-of-sample split
cut=int(len(df)*0.7)
print(f"\n  --- out-of-sample (last 30%, {len(df)-cut} candles) ---")
oos=df.iloc[cut:]
stats(oos['w_bh'],"buy & hold (OOS)",oos['ret'])
stats(oos['w_vol'],"vol-targeted (OOS)",oos['ret'])
print(f"\n  Sharpe improvement (full): {s_vt-s_bh:+.2f}  -> {'vol-targeting HELPS' if s_vt>s_bh else 'no benefit'}")
