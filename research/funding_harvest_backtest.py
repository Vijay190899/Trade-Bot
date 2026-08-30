"""
Cash-and-carry (funding harvest) backtest.

Delta-neutral: long N units spot, short N units perpetual.
Income  = funding received by the short leg when funding is positive.
Costs   = fees on all four executions, plus funding PAID when negative.
Risks   = basis divergence, perp-leg liquidation.

Returns are reported on TOTAL DEPLOYED CAPITAL (spot notional + perp margin),
not on notional, because margin is capital that cannot be used elsewhere.

Gates are pre-registered in docs/GATES_FUNDING.md and are NOT relaxed here.

Usage:  python research/funding_harvest_backtest.py
"""
import json
import math
import os
import time

import httpx
import numpy as np
import pandas as pd

DATA_DIR = "user_data/data/binance"
CACHE = "user_data/data/funding_cache.json"

FEE_TAKER = 0.0010   # 0.10% Binance VIP0 taker
FEE_MAKER = 0.0005   # 0.05% Binance VIP0 maker
MAINT_MARGIN = 0.005  # 0.5% maintenance margin on notional


# ---------------------------------------------------------------- data

def fetch_funding(symbol, max_pages=10):
    """Full funding-rate history for a perp symbol (8-hourly)."""
    out, end = [], int(time.time() * 1000)
    for _ in range(max_pages):
        r = httpx.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": 1000, "endTime": end},
            timeout=25,
        )
        d = r.json()
        if not d:
            break
        out = d + out
        end = int(d[0]["fundingTime"]) - 1
        time.sleep(0.08)
    df = pd.DataFrame(out).drop_duplicates("fundingTime")
    df["t"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["mark"] = pd.to_numeric(df["markPrice"], errors="coerce")
    return df.dropna(subset=["rate"]).sort_values("t").reset_index(drop=True)


def load_funding(symbols):
    cache = {}
    if os.path.exists(CACHE):
        try:
            cache = json.load(open(CACHE))
        except Exception:
            cache = {}
    out = {}
    for s in symbols:
        if s in cache:
            df = pd.DataFrame(cache[s])
            df["t"] = pd.to_datetime(df["t"], utc=True, format="ISO8601")
            out[s] = df
        else:
            print(f"    fetching funding history for {s} ...")
            df = fetch_funding(s)
            out[s] = df
            cache[s] = df.assign(t=df["t"].astype(str))[["t", "rate", "mark"]].to_dict("list")
            cache[s] = pd.DataFrame(cache[s]).to_dict("list")
    try:
        json.dump(cache, open(CACHE, "w"))
    except Exception:
        pass
    return out


def load_spot(symbol, timeframe="1h"):
    """Spot OHLCV from the local freqtrade data store."""
    path = os.path.join(DATA_DIR, f"{symbol.replace('USDT', '_USDT')}-{timeframe}.json")
    if not os.path.exists(path):
        return None
    raw = json.load(open(path))
    df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "close", "v"])
    df["t"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["t", "close"]]


def fetch_perp_klines(symbol, start_ms, cache_dir=DATA_DIR):
    """
    Perpetual-future hourly closes.

    Required because the `markPrice` field on the funding endpoint is an
    INSTANTANEOUS snapshot at settlement, while local spot data is an hourly
    BAR CLOSE. Differencing those two produced apparent basis of -10%/+8% on
    violent days - a sampling artifact, not real basis. Sampling both legs
    identically (1h close vs 1h close) measures basis correctly (~+-0.4%).
    """
    cache = os.path.join(cache_dir, f"{symbol}-perp-1h.json")
    if os.path.exists(cache):
        raw = json.load(open(cache))
    else:
        raw, end = [], None
        cur = start_ms
        while True:
            params = {"symbol": symbol, "interval": "1h", "limit": 1500, "startTime": cur}
            r = httpx.get("https://fapi.binance.com/fapi/v1/klines", params=params, timeout=25)
            d = r.json()
            if not isinstance(d, list) or not d:
                break
            raw.extend([[k[0], float(k[4])] for k in d])
            if len(d) < 1500:
                break
            cur = int(d[-1][0]) + 1
            time.sleep(0.1)
        try:
            json.dump(raw, open(cache, "w"))
        except Exception:
            pass
    if not raw:
        return None
    df = pd.DataFrame(raw, columns=["ts", "perp"])
    df["t"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["t", "perp"]]


# ---------------------------------------------------------------- engine

def backtest(symbol, funding, spot, capital=10_000.0, leverage=1.0,
             fee=FEE_TAKER, margin_mode="cross", rebal_trigger=0.35,
             verbose=False):
    """
    Open one delta-neutral position at the start, hold to the end, collecting
    funding every 8h. Models fees, negative funding, basis, and liquidation.

    leverage L: notional per leg S, margin M = S / L, total capital C = S + M
                => S = C * L / (1 + L)

    margin_mode:
      'isolated'  - perp leg has its own margin pool. Liquidates on a large
                    UPWARD move even though the spot leg gains the offsetting
                    amount. Included to show why this mode is unusable.
      'cross'     - total account equity backs the position (Binance Portfolio
                    Margin). Spot gains offset perp losses, so a delta-neutral
                    book is structurally safe. This is how desks run it.
      'rebalance' - isolated margin, but top the perp margin up from the spot
                    leg whenever its margin ratio falls below `rebal_trigger`.
                    Each top-up costs a spot sale fee. Realistic for accounts
                    without portfolio margin.
    """
    df = pd.merge_asof(
        funding.sort_values("t"),
        spot.sort_values("t").rename(columns={"close": "spot"}),
        on="t", direction="nearest", tolerance=pd.Timedelta("1h"),
    ).dropna(subset=["spot", "mark"]).reset_index(drop=True)
    if len(df) < 100:
        return None   # e.g. funding periods with no overlapping spot history

    # Perp hourly close, sampled identically to spot, so basis is measured
    # correctly. Falls back to markPrice only if the fetch fails.
    perp = fetch_perp_klines(symbol, int(df["t"].iloc[0].timestamp() * 1000))
    if perp is not None:
        df = pd.merge_asof(
            df, perp.sort_values("t"),
            on="t", direction="nearest", tolerance=pd.Timedelta("1h"),
        )
        df["perp"] = df["perp"].fillna(df["mark"])
    else:
        df["perp"] = df["mark"]
    df = df.dropna(subset=["spot", "perp", "mark"]).reset_index(drop=True)
    if len(df) < 100:
        return None

    S = capital * leverage / (1.0 + leverage)   # notional of each leg
    M = capital - S                             # margin for the perp short
    units = S / df["spot"].iloc[0]

    entry_spot = df["spot"].iloc[0]
    entry_mark = df["perp"].iloc[0]

    # open both legs
    cash = -fee * S * 2                          # entry fees, both legs
    funding_cum = 0.0
    margin = M
    spot_units = units
    n_rebal = 0
    equity_curve, liquidated, liq_idx = [], False, None

    for i, row in df.iterrows():
        # funding settles every 8h; short receives when rate > 0
        pay = units * row["mark"] * row["rate"]
        funding_cum += pay

        # mark-to-market both legs
        spot_val = spot_units * row["spot"]
        spot_pnl = spot_units * (row["spot"] - entry_spot)
        perp_pnl = units * (entry_mark - row["perp"])     # short
        notional = units * row["perp"]
        maint = MAINT_MARGIN * notional

        if margin_mode == "cross":
            # total equity backs the book; spot gain offsets perp loss
            equity_total = spot_val + margin + perp_pnl + funding_cum + cash
            if equity_total <= maint:
                liquidated, liq_idx = True, i
                equity_curve.append(equity_total - fee * S * 2)
                break

        elif margin_mode == "rebalance":
            # top up the perp margin from the spot leg before it liquidates
            if (margin + perp_pnl) < rebal_trigger * notional and spot_val > 0:
                need = rebal_trigger * notional - (margin + perp_pnl)
                sell = min(need, spot_val * 0.5)
                if sell > 0:
                    sold_units = sell / row["spot"]
                    spot_units -= sold_units
                    margin += sell
                    cash -= fee * sell         # fee on the spot sale
                    n_rebal += 1
            if (margin + perp_pnl + funding_cum) <= maint:
                liquidated, liq_idx = True, i
                equity_curve.append(spot_val + margin + perp_pnl + funding_cum + cash - fee * S * 2)
                break

        else:  # isolated
            if (margin + perp_pnl + funding_cum) <= maint:
                liquidated, liq_idx = True, i
                equity_curve.append(capital + spot_pnl + perp_pnl + funding_cum + cash - fee * S * 2)
                break

        equity = spot_val + margin + perp_pnl + funding_cum + cash \
                 if margin_mode != "isolated" else \
                 capital + spot_pnl + perp_pnl + funding_cum + cash
        equity_curve.append(equity)

    if not liquidated:
        # close both legs
        equity_curve[-1] -= fee * S * 2

    eq = np.maximum(np.array(equity_curve), 0.0)   # exchange liquidates at 0
    years = (df["t"].iloc[len(eq) - 1] - df["t"].iloc[0]).total_seconds() / (365 * 24 * 3600)
    total = eq[-1] / capital - 1
    cagr = (eq[-1] / capital) ** (1 / years) - 1 if years > 0 else 0.0
    rets = np.diff(eq) / eq[:-1]
    sharpe = rets.mean() / rets.std() * math.sqrt(3 * 365) if rets.std() else 0.0
    dd = (eq / np.maximum.accumulate(eq) - 1).min()

    return {
        "symbol": symbol, "years": years, "total": total, "cagr": cagr,
        "sharpe": sharpe, "maxdd": dd, "liquidated": liquidated,
        "funding_cum": funding_cum, "fees": fee * S * 4,
        "pct_pos": (df["rate"] > 0).mean(), "n": len(eq),
        "equity": eq, "t": df["t"].iloc[:len(eq)].values,
        "capital": capital, "notional": S, "margin": M,
        "mode": margin_mode, "n_rebal": n_rebal,
    }


# ---------------------------------------------------------------- report

def main():
    symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", "SOLUSDT", "BNBUSDT"]
    print("Loading funding history ...")
    fund = load_funding(symbols)

    print("\n" + "=" * 106)
    print("GATE F2 / F4 / F5 — net-of-fee carry on TOTAL DEPLOYED CAPITAL")
    print("=" * 106)
    header = (f"{'symbol':<9}{'years':>7}{'CAGR %':>9}{'total %':>9}"
              f"{'Sharpe':>8}{'maxDD %':>9}{'funding':>11}{'fees':>8}{'rebal':>7}  liq")
    results = []
    for mode in ["isolated", "cross", "rebalance"]:
        for lev in [1.0, 2.0]:
            print(f"\n  --- {mode.upper()} margin, leverage {lev:.0f}x "
                  f"(capital = {(1+lev)/lev:.1f}x notional) ---")
            print("  " + header)
            print("  " + "-" * 102)
            for s in symbols:
                spot = load_spot(s)
                if spot is None:
                    continue
                r = backtest(s, fund[s], spot, leverage=lev, margin_mode=mode)
                if not r:
                    continue
                results.append((mode, lev, r))
                print(f"  {r['symbol']:<9}{r['years']:>7.2f}{r['cagr']*100:>9.2f}"
                      f"{r['total']*100:>9.2f}{r['sharpe']:>8.2f}{r['maxdd']*100:>9.2f}"
                      f"{r['funding_cum']:>11.0f}{r['fees']:>8.0f}{r['n_rebal']:>7}"
                      f"  {'YES' if r['liquidated'] else 'no'}")

    # ---- gate evaluation ----
    print("\n" + "=" * 106)
    print("GATE EVALUATION  (thresholds fixed in docs/GATES_FUNDING.md before running)")
    print("=" * 106)
    for mode in ["isolated", "cross", "rebalance"]:
        for lev in [1.0, 2.0]:
            rs = [r for m, l, r in results if m == mode and l == lev]
            if not rs:
                continue
            f2 = [r for r in rs if r["cagr"] > 0.05]
            liq = [r["symbol"] for r in rs if r["liquidated"]]
            f5 = [r for r in rs if r["maxdd"] > -0.10]
            passing = [r["symbol"] for r in rs
                       if r["cagr"] > 0.05 and not r["liquidated"] and r["maxdd"] > -0.10]
            print(f"\n  {mode} @ {lev:.0f}x")
            print(f"    F2 CAGR > 5% on capital : {len(f2)}/{len(rs)}   {[r['symbol'] for r in f2]}")
            print(f"    F4 zero liquidations    : {'PASS' if not liq else 'FAIL -> ' + ','.join(liq)}")
            print(f"    F5 max drawdown < 10%   : {len(f5)}/{len(rs)}")
            print(f"    ==> passing ALL gates   : {passing or 'NONE'}")

    gate_f3(symbols, fund, leverage=1.0)



def walk_forward(symbol, funding, spot, leverage=1.0, margin_mode="cross"):
    """GATE F3 - is carry positive in >=75% of calendar quarters?"""
    # Slice the RAW funding series by quarter; backtest() does its own merge.
    f = funding.copy()
    f["q"] = f["t"].dt.to_period("Q")
    out = []
    for q, g in f.groupby("q"):
        if len(g) < 60:
            continue
        r = backtest(symbol, g.drop(columns=["q"]).reset_index(drop=True), spot,
                     leverage=leverage, margin_mode=margin_mode)
        if r:
            out.append((str(q), r["total"], r["liquidated"]))
    return out


def gate_f3(symbols, fund, leverage=1.0):
    print()
    print("=" * 106)
    print("GATE F3 - WALK-FORWARD BY QUARTER  (threshold: positive in >= 75% of quarters)")
    print("=" * 106)
    for s in symbols:
        spot = load_spot(s)
        if spot is None:
            continue
        res = walk_forward(s, fund[s], spot, leverage=leverage)
        if not res:
            continue
        pos = sum(1 for _, t, _ in res if t > 0)
        liq = sum(1 for _, _, l in res if l)
        pct = pos / len(res) * 100
        marks = " ".join(("+" if t > 0 else "-") for _, t, _ in res)
        verdict = "PASS" if pct >= 75 and liq == 0 else "FAIL"
        print(f"  {s:<9} {pos:>2}/{len(res):<2} quarters positive ({pct:5.1f}%)  liq={liq}  {verdict}   {marks}")

if __name__ == "__main__":
    main()
