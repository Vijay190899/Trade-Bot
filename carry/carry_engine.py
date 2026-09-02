"""
Cash-and-carry (funding harvest) DRY-RUN engine.

Delta-neutral: long N units spot, short N units perpetual, harvesting funding.

SAFETY: this process CANNOT place orders. It holds no API keys and imports no
signed-request client. Every "fill" is simulated against live public prices.
That is deliberate - the strategy is being validated for operational stability
(Gate F6), not traded.

What it does each tick:
  1. Pull live spot price, perp price and funding rate (public endpoints)
  2. Mark both legs to market
  3. Accrue funding at each 8h settlement boundary
  4. Check HEDGE INTEGRITY - the two legs must stay size-matched
  5. Check MARGIN health
  6. Persist state so a restart resumes rather than double-opening
  7. Honour a kill switch file

Run:  python carry/carry_engine.py
"""
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

import httpx

# ----------------------------------------------------------------- config
BOT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BOT_ROOT, "carry", "carry_state.sqlite")
LOG_PATH = os.path.join(BOT_ROOT, "user_data", "logs", "carry_run.log")
KILL_SWITCH = os.path.join(BOT_ROOT, "carry", "STOP")

# Pairs that passed every gate in docs/GATES_FUNDING.md
PAIRS = ["BTCUSDT", "XRPUSDT", "DOGEUSDT", "ETHUSDT"]

CAPITAL_PER_PAIR = 2_500.0     # simulated USDT per pair
LEVERAGE = 1.0                  # M = S; capital = 2x notional
FEE = 0.0010                    # 0.10% taker, charged on every execution

TICK_SECONDS = 300              # 5 minutes

# --- safety thresholds ---
HEDGE_DRIFT_ALERT = 0.01        # 1%  of notional -> warn
HEDGE_DRIFT_FLATTEN = 0.05      # 5%  of notional -> flatten, hedge is broken
MARGIN_RATIO_ALERT = 0.15       # equity/notional below this -> warn
MARGIN_RATIO_FLATTEN = 0.05     # below this -> flatten before liquidation
MAX_NOTIONAL_PER_PAIR = 5_000.0  # hard cap, simulated


def log(msg):
    # LOCAL time, matching freqtrade's log format. The watchdog compares the
    # last log timestamp against local Get-Date; logging UTC here made the
    # engine look permanently stale and caused a restart loop.
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - carry - {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ----------------------------------------------------------------- storage
def db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("""CREATE TABLE IF NOT EXISTS positions(
        symbol TEXT PRIMARY KEY, opened_at TEXT, units REAL,
        entry_spot REAL, entry_perp REAL, capital REAL, notional REAL,
        margin REAL, fees_paid REAL, funding_cum REAL,
        last_funding_ts INTEGER, status TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS equity(
        ts TEXT, symbol TEXT, spot REAL, perp REAL, basis_pct REAL,
        spot_pnl REAL, perp_pnl REAL, funding_cum REAL, equity REAL,
        hedge_drift REAL, margin_ratio REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS events(
        ts TEXT, symbol TEXT, level TEXT, message TEXT)""")
    con.commit()
    return con


def event(con, symbol, level, message):
    con.execute("INSERT INTO events VALUES(?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), symbol, level, message))
    con.commit()
    log(f"[{level}] {symbol}: {message}")


# ----------------------------------------------------------------- market
def market(symbol):
    """Live spot price, perp price, and current funding rate."""
    with httpx.Client(timeout=15) as c:
        spot = float(c.get("https://api.binance.com/api/v3/ticker/price",
                           params={"symbol": symbol}).json()["price"])
        prem = c.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                     params={"symbol": symbol}).json()
    return {
        "spot": spot,
        "perp": float(prem["markPrice"]),
        "rate": float(prem["lastFundingRate"]),
        "next_funding": int(prem["nextFundingTime"]),
    }


# ----------------------------------------------------------------- engine
def open_position(con, symbol, m):
    S = CAPITAL_PER_PAIR * LEVERAGE / (1.0 + LEVERAGE)
    S = min(S, MAX_NOTIONAL_PER_PAIR)
    margin = CAPITAL_PER_PAIR - S
    units = S / m["spot"]
    fees = FEE * S * 2                       # both legs
    con.execute("INSERT OR REPLACE INTO positions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (symbol, datetime.now(timezone.utc).isoformat(), units,
                 m["spot"], m["perp"], CAPITAL_PER_PAIR, S, margin, fees, 0.0,
                 0, "OPEN"))
    con.commit()
    event(con, symbol, "INFO",
          f"opened DRY-RUN hedge: {units:.6f} units, notional ${S:,.0f}, "
          f"margin ${margin:,.0f}, spot {m['spot']:.4f} perp {m['perp']:.4f}, fees ${fees:.2f}")


def flatten(con, symbol, reason):
    row = con.execute("SELECT notional FROM positions WHERE symbol=?", (symbol,)).fetchone()
    if row:
        con.execute("UPDATE positions SET status=?, fees_paid=fees_paid+? WHERE symbol=?",
                    ("FLAT", FEE * row[0] * 2, symbol))
        con.commit()
    event(con, symbol, "CRITICAL", f"FLATTENED - {reason}")


def tick(con, symbol):
    row = con.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
    m = market(symbol)

    if row is None:
        open_position(con, symbol, m)
        return
    (_, _, units, entry_spot, entry_perp, capital, notional,
     margin, fees_paid, funding_cum, last_fts, status) = row
    if status != "OPEN":
        return

    # --- accrue funding once per settlement boundary ---
    # nextFundingTime moves forward after each settlement; when it changes we
    # know a settlement occurred, so credit exactly one payment.
    if last_fts and m["next_funding"] != last_fts:
        pay = units * m["perp"] * m["rate"]
        funding_cum += pay
        con.execute("UPDATE positions SET funding_cum=?, last_funding_ts=? WHERE symbol=?",
                    (funding_cum, m["next_funding"], symbol))
        con.commit()
        log(f"  {symbol}: funding settled {pay:+.4f} USDT (rate {m['rate']*100:+.4f}%)")
    elif not last_fts:
        con.execute("UPDATE positions SET last_funding_ts=? WHERE symbol=?",
                    (m["next_funding"], symbol))
        con.commit()

    # --- mark to market ---
    spot_val = units * m["spot"]
    spot_pnl = units * (m["spot"] - entry_spot)
    perp_pnl = units * (entry_perp - m["perp"])          # short
    equity = capital + spot_pnl + perp_pnl + funding_cum - fees_paid
    basis_pct = (m["perp"] - m["spot"]) / m["spot"] * 100

    # --- HEDGE INTEGRITY: the two legs must remain size-matched ---
    long_notional = units * m["spot"]
    short_notional = units * m["perp"]
    drift = abs(long_notional - short_notional) / max(long_notional, 1e-9)

    # --- margin health (cross-margin: whole equity backs the book) ---
    margin_ratio = equity / max(short_notional, 1e-9)

    con.execute("INSERT INTO equity VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), symbol, m["spot"], m["perp"],
                 basis_pct, spot_pnl, perp_pnl, funding_cum, equity, drift, margin_ratio))
    con.commit()

    if drift >= HEDGE_DRIFT_FLATTEN:
        flatten(con, symbol, f"hedge drift {drift*100:.2f}% >= {HEDGE_DRIFT_FLATTEN*100:.0f}%")
    elif drift >= HEDGE_DRIFT_ALERT:
        event(con, symbol, "WARN", f"hedge drift {drift*100:.2f}%")

    if margin_ratio <= MARGIN_RATIO_FLATTEN:
        flatten(con, symbol, f"margin ratio {margin_ratio*100:.2f}% <= {MARGIN_RATIO_FLATTEN*100:.0f}%")
    elif margin_ratio <= MARGIN_RATIO_ALERT:
        event(con, symbol, "WARN", f"margin ratio {margin_ratio*100:.2f}%")


def main():
    log("=" * 78)
    log("Cash-and-carry DRY-RUN engine starting (Gate F6 stability clock)")
    log(f"  pairs={PAIRS}  capital/pair=${CAPITAL_PER_PAIR:,.0f}  leverage={LEVERAGE}x")
    log("  THIS PROCESS CANNOT PLACE ORDERS - no API keys, no signed client")
    log("=" * 78)
    con = db()
    while True:
        if os.path.exists(KILL_SWITCH):
            for s in PAIRS:
                flatten(con, s, "kill switch engaged")
            log("kill switch present - exiting")
            return
        for s in PAIRS:
            try:
                tick(con, s)
            except Exception as e:
                log(f"  ERROR {s}: {type(e).__name__}: {e}")
        # heartbeat, so the watchdog can detect staleness the same way as the bots.
        # Take the LATEST row per symbol - each pair is written at a slightly
        # different timestamp, so a single MAX(ts) matches only one of them.
        tot = con.execute("""
            SELECT COUNT(*), SUM(equity), SUM(funding_cum) FROM equity e
            WHERE ts = (SELECT MAX(ts) FROM equity WHERE symbol = e.symbol)
        """).fetchone()
        log(f"heartbeat. pairs={tot[0]} equity={tot[1] or 0:,.2f} "
            f"funding_cum={tot[2] or 0:+,.4f} USDT")
        time.sleep(TICK_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("interrupted")
        sys.exit(0)
