"""Qoherenz Trading Bot — Dashboard v6"""

import logging
import os
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import httpx
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

BOT_ROOT  = Path("V:/Antigravity/Trade tool/bot")
USERDATA  = BOT_ROOT / "user_data"
MEMORY_DB = USERDATA / "models" / "trade_memory.db"
RL_DB     = BOT_ROOT / "tradesv3.rl.sqlite"
GRID_DB   = BOT_ROOT / "tradesv3.grid.sqlite"
LOG_FILE  = USERDATA / "logs" / "dry_run.log"
GRID_LOG  = USERDATA / "logs" / "grid_run.log"
MODEL_DIR = USERDATA / "models" / "antigravity_rl_v1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Qoherenz Dashboard")


def _db_query(db_path: Path, sql: str, params: tuple = ()) -> List[Dict]:
    if not db_path.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("DB query failed (%s): %s", db_path.name, e)
        return []


def _proc_stats(pid: int) -> Dict:
    try:
        p = psutil.Process(pid)
        with p.oneshot():
            mem_mb = p.memory_info().rss / 1024 / 1024
            create_time = datetime.fromtimestamp(p.create_time(), tz=timezone.utc)
            uptime_s = (datetime.now(tz=timezone.utc) - create_time).total_seconds()
        return {"pid": pid, "mem_mb": round(mem_mb, 1), "uptime_s": int(uptime_s)}
    except Exception:
        return {}


def _bot_status_for(strategy_fragment: str, _log_file: Path = None) -> Dict:
    pid = None
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            if "freqtrade" in cmd and "trade" in cmd and strategy_fragment in cmd:
                pid = proc.info["pid"]
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    stats = _proc_stats(pid) if pid else {}
    return {"running": pid is not None, "pid": pid, **stats}


def _rl_retrain_info() -> Dict:
    try:
        files = list(MODEL_DIR.glob("**/*.zip")) + list(MODEL_DIR.glob("**/*.pkl"))
        if not files:
            return {}
        latest = max(files, key=os.path.getmtime)
        last_mtime = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
        hours_since = (datetime.now(tz=timezone.utc) - last_mtime).total_seconds() / 3600
        return {
            "last_retrain": last_mtime.strftime("%Y-%m-%d %H:%M UTC"),
            "hours_since": round(hours_since, 1),
            "next_retrain_h": round(max(0.0, 8.0 - hours_since), 1),
        }
    except Exception:
        return {}


@app.get("/api/status")
async def api_status():
    rl   = _bot_status_for("AntigravityStrategy",     LOG_FILE)
    grid = _bot_status_for("AntigravityGridStrategy", GRID_LOG)
    return {"rl": rl, "grid": grid}


@app.get("/api/logs")
async def api_logs(mode: str = "rl"):
    log_path = GRID_LOG if mode == "grid" else LOG_FILE
    if not log_path.exists():
        return {"lines": [f"No log file yet for {mode} bot."]}
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return {"lines": [l.rstrip() for l in lines[-80:]]}
    except Exception as e:
        return {"lines": [str(e)]}


@app.get("/api/price")
async def api_price():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get("https://api.binance.com/api/v3/ticker/24hr?symbol=ETHUSDT")
            d = r.json()
        return {
            "price": float(d["lastPrice"]),
            "change_pct": float(d["priceChangePercent"]),
            "high_24h": float(d["highPrice"]),
            "low_24h": float(d["lowPrice"]),
            "volume_24h": float(d["quoteVolume"]),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/indicators")
async def api_indicators(interval: str = "1h"):
    safe_interval = interval if interval in ("1m","5m","15m","30m","1h","4h","1d") else "1h"
    limit = 220 if safe_interval == "1h" else 80
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": "ETHUSDT", "interval": safe_interval, "limit": limit}
            )
        raw = r.json()
        closes = [float(c[4]) for c in raw]
        highs  = [float(c[2]) for c in raw]
        lows   = [float(c[3]) for c in raw]
        rsi       = _compute_rsi(closes, 14)
        bb_pos, _ = _compute_bb(closes, 20, 2)
        bb_width  = _compute_bb_width_series(closes, 20, 2)
        adx       = _compute_adx(highs, lows, closes, 14)
        if safe_interval == "1h" and len(closes) >= 200:
            sma200 = sum(closes[-200:]) / 200
            bb_width_ma = sum([_compute_bb_width_series(closes[:i+1], 20, 2)
                               for i in range(max(len(closes)-20, 0), len(closes))]) / 20
            if bb_width > bb_width_ma * 1.5:
                regime = "high-volatility"
            elif adx > 25:
                regime = "bull-trending" if closes[-1] > sma200 else "bear-trending"
            else:
                regime = "ranging"
        else:
            if bb_width > 0.08:
                regime = "high-volatility"
            elif adx > 25:
                regime = "bull-trending"
            else:
                regime = "ranging"
        return {
            "rsi": round(rsi, 1), "bb_position": round(bb_pos, 3),
            "bb_width": round(bb_width, 4), "adx": round(adx, 1), "regime": regime,
        }
    except Exception as e:
        return {"error": str(e)}


def _compute_rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    deltas = [closes[i]-closes[i-1] for i in range(1,len(closes))]
    gains  = [d if d>0 else 0 for d in deltas]
    losses = [-d if d<0 else 0 for d in deltas]
    ag = sum(gains[:period])/period; al = sum(losses[:period])/period
    for i in range(period, len(deltas)):
        ag = (ag*(period-1)+gains[i])/period; al = (al*(period-1)+losses[i])/period
    return 100.0 if al==0 else 100-(100/(1+ag/al))

def _compute_bb(closes, period=20, std_dev=2.0):
    if len(closes)<period: return 0.5,0.04
    w=closes[-period:]; ma=sum(w)/period
    std=(sum((x-ma)**2 for x in w)/period)**.5
    upper=ma+std_dev*std; lower=ma-std_dev*std
    return (closes[-1]-lower)/(upper-lower) if upper!=lower else 0.5, (upper-lower)/ma if ma else 0.0

def _compute_bb_width_series(closes, period=20, std_dev=2.0):
    _,bw=_compute_bb(closes,period,std_dev); return bw

def _compute_adx(highs, lows, closes, period=14):
    if len(closes)<period+1: return 20.0
    tr_l,pdm_l,ndm_l=[],[],[]
    for i in range(1,len(closes)):
        h,l,pc=highs[i],lows[i],closes[i-1]
        tr_l.append(max(h-l,abs(h-pc),abs(l-pc)))
        pdm_l.append(max(highs[i]-highs[i-1],0) if highs[i]-highs[i-1]>lows[i-1]-lows[i] else 0)
        ndm_l.append(max(lows[i-1]-lows[i],0) if lows[i-1]-lows[i]>highs[i]-highs[i-1] else 0)
    def smooth(lst,p):
        s=sum(lst[:p]); r=[s]
        for v in lst[p:]: s=s-s/p+v; r.append(s)
        return r
    tr_s=smooth(tr_l,period); pdm_s=smooth(pdm_l,period); ndm_s=smooth(ndm_l,period)
    dx=[]
    for t,p,n in zip(tr_s,pdm_s,ndm_s):
        pdi=100*p/t if t else 0; ndi=100*n/t if t else 0
        dx.append(100*abs(pdi-ndi)/(pdi+ndi) if pdi+ndi else 0)
    return sum(dx[-period:])/period if dx else 20.0


@app.get("/api/trades")
async def api_trades(mode: str = "rl"):
    db       = GRID_DB if mode == "grid" else RL_DB
    strategy = "AntigravityGridStrategy" if mode == "grid" else "AntigravityStrategy"
    open_trades = _db_query(db, """
        SELECT id,pair,open_rate,open_date,stake_amount,max_rate,stop_loss_pct,enter_tag
        FROM trades WHERE is_open=1 AND strategy=? ORDER BY open_date DESC
    """, (strategy,))
    closed_trades = _db_query(db, """
        SELECT id,pair,open_rate,close_rate,open_date,close_date,
               close_profit AS profit_ratio, close_profit_abs AS profit_abs,
               exit_reason,enter_tag
        FROM trades WHERE is_open=0 AND strategy=? ORDER BY close_date DESC LIMIT 50
    """, (strategy,))
    stats_row = _db_query(db, """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN close_profit>0 THEN 1 ELSE 0 END) AS wins,
               SUM(close_profit_abs) AS total_pnl,
               AVG(close_profit_abs) AS avg_profit,
               MIN(close_profit_abs) AS worst_trade,
               SUM(CASE WHEN date(close_date)=date('now') THEN close_profit_abs ELSE 0 END) AS daily_pnl
        FROM trades WHERE is_open=0 AND strategy=?
    """, (strategy,))
    s=stats_row[0] if stats_row else {}
    total=s.get("total") or 0; wins=s.get("wins") or 0
    pnl=s.get("total_pnl") or 0; avg=s.get("avg_profit") or 0
    worst=s.get("worst_trade") or 0; daily=s.get("daily_pnl") or 0
    return {
        "open": open_trades, "closed": closed_trades,
        "stats": {
            "total_closed": total,
            "win_rate": round(wins/total*100,1) if total else None,
            "total_pnl": round(pnl,4), "avg_profit": round(avg,4),
            "max_dd": round(worst,4), "daily_pnl": round(daily,4),
        }
    }


@app.get("/api/memory")
async def api_memory():
    regime_rows = _db_query(MEMORY_DB, """
        SELECT regime,trade_count,win_rate,avg_profit_pct,avg_hold_hours,updated_at
        FROM regime_stats ORDER BY trade_count DESC
    """)
    summary = _db_query(MEMORY_DB, """
        SELECT COUNT(*) AS total, AVG(CASE WHEN profit_pct>0 THEN 1.0 ELSE 0.0 END) AS win_rate,
               SUM(profit_abs) AS total_pnl FROM trades
    """)
    recent = _db_query(MEMORY_DB, """
        SELECT AVG(CASE WHEN profit_pct<0 THEN 1.0 ELSE 0.0 END) AS loss_rate
        FROM trades WHERE exit_time>datetime('now','-7 days')
    """)
    return {
        "regime_stats": regime_rows,
        "summary": summary[0] if summary else {},
        "recent_loss_rate": (recent[0].get("loss_rate") or 0.0) if recent else 0.0,
    }


@app.get("/api/equity")
async def api_equity(mode: str = "rl"):
    db       = GRID_DB if mode == "grid" else RL_DB
    strategy = "AntigravityGridStrategy" if mode == "grid" else "AntigravityStrategy"
    trades = _db_query(db, """
        SELECT close_date, close_profit_abs AS profit_abs FROM trades
        WHERE is_open=0 AND strategy=? ORDER BY close_date ASC
    """, (strategy,))
    balance = 100.0
    points  = [{"t": "start", "balance": balance, "profit_abs": 0}]
    for t in trades:
        pa = t.get("profit_abs") or 0
        balance += pa
        points.append({"t": t.get("close_date",""), "balance": round(balance,4), "profit_abs": round(pa,4)})
    return {"points": points}


@app.get("/api/rl")
async def api_rl():
    return _rl_retrain_info()


@app.post("/api/bot/stop")
async def api_bot_stop(mode: str = "rl"):
    strategy = "AntigravityGridStrategy" if mode == "grid" else "AntigravityStrategy"
    info = _bot_status_for(strategy)
    if not info["running"]:
        return {"ok": False, "msg": f"{mode.upper()} bot is not running"}
    try:
        psutil.Process(info["pid"]).terminate()
        return {"ok": True, "msg": f"Stopped {mode.upper()} bot (PID {info['pid']})"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/bot/start")
async def api_bot_start(mode: str = "rl"):
    bat = str(BOT_ROOT / ("run_grid.bat" if mode == "grid" else "run_dryrun.bat"))
    try:
        subprocess.Popen([bat], creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=str(BOT_ROOT))
        return {"ok": True, "msg": f"{mode.upper()} bot launch initiated"}
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# HTML — Qoherenz v6  |  Phase 1: Foundation + Topbar + Stat Row
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qoherenz Trading Bot</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
/* ── TOKENS ─────────────────────────────────────── */
:root {
  --bg:  #06060B;
  --s1:  #0B0B12;
  --s2:  #101018;
  --s3:  #16161F;
  --s4:  #1D1D28;
  --bd:  rgba(255,255,255,.07);
  --bd2: rgba(255,255,255,.04);
  --bd3: rgba(255,255,255,.13);

  --t1: #F0F0FF;
  --t2: #C8C0A0;
  --t3: #B89228;
  --t4: #6B5515;

  --ac:  #39FF14;
  --ac2: rgba(57,255,20,.07);
  --ac3: rgba(57,255,20,.22);

  --dn:  #FF5C5C;
  --dn2: rgba(255,92,92,.07);
  --dn3: rgba(255,92,92,.22);

  --wn:  #FFB347;
  --wn2: rgba(255,179,71,.07);
  --wn3: rgba(255,179,71,.22);

  --fn: 'Plus Jakarta Sans', -apple-system, sans-serif;
  --mo: 'JetBrains Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--t2);
  font-family: var(--fn);
  -webkit-font-smoothing: antialiased;
  display: flex; flex-direction: column;
  height: 100vh; overflow: hidden;
  position: relative;
}

/* Ambient background glow — stays behind content */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 70% 55% at 12% 10%, rgba(57,255,20,.045) 0%, transparent 65%),
    radial-gradient(ellipse 55% 45% at 88% 88%, rgba(255,92,92,.035) 0%, transparent 65%);
  background-size: 180% 180%;
  animation: bgDrift 30s ease-in-out infinite alternate;
}
@keyframes bgDrift {
  0%   { background-position: 0% 0%,   100% 100%; }
  50%  { background-position: 50% 20%,  50% 80%;  }
  100% { background-position: 100% 100%, 0%   0%;  }
}

.topbar, .stat-row, .main-scroll { position: relative; z-index: 1; }

/* ── SCROLLBARS ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bd3); border-radius: 2px; }

/* ── UTILITY ── */
.mono { font-family: var(--mo); font-variant-numeric: lining-nums tabular-nums; }
.up   { color: var(--ac); }
.dn   { color: var(--dn); }
.wn   { color: var(--wn); }

.badge {
  display: inline-flex; align-items: center;
  padding: 2px 7px; border-radius: 3px;
  font-family: var(--fn); font-size: 9px; font-weight: 800;
  letter-spacing: .08em; text-transform: uppercase; border: 1px solid;
}
.b-up { color: var(--ac); background: var(--ac2); border-color: var(--ac3); }
.b-dn { color: var(--dn); background: var(--dn2); border-color: var(--dn3); }
.b-wn { color: var(--wn); background: var(--wn2); border-color: var(--wn3); }
.b-nu { color: var(--t3); background: var(--s2);  border-color: var(--bd);  }

/* ── TOOLTIP ── */
.help {
  display: inline-flex; align-items: center; justify-content: center;
  width: 14px; height: 14px; border-radius: 50%;
  border: 1px solid var(--t4); color: var(--t4);
  font-family: var(--fn); font-size: 8px; font-weight: 700;
  cursor: help; position: relative; flex-shrink: 0;
  transition: border-color .12s, color .12s;
}
.help:hover { border-color: var(--t2); color: var(--t2); }
.help[data-tip]::after {
  content: attr(data-tip);
  position: absolute; bottom: calc(100% + 8px); left: 50%;
  transform: translateX(-50%);
  min-width: 210px; max-width: 290px;
  background: var(--s3); border: 1px solid var(--bd3);
  color: var(--t1); font-family: var(--fn);
  font-size: 11px; font-weight: 400; line-height: 1.65;
  padding: 10px 13px; border-radius: 6px;
  opacity: 0; pointer-events: none; transition: opacity .15s; z-index: 999;
  white-space: normal; text-transform: none; letter-spacing: 0;
}
.help.tip-dn[data-tip]::after { bottom: auto; top: calc(100% + 8px); }
.help:hover[data-tip]::after  { opacity: 1; }

/* ── PRICE FLASH ── */
@keyframes flashUp { 0%{color:var(--ac)} 100%{color:var(--t1)} }
@keyframes flashDn { 0%{color:var(--dn)} 100%{color:var(--t1)} }
.flash-up { animation: flashUp .6s ease-out; }
.flash-dn { animation: flashDn .6s ease-out; }

/* ======================================================
   TOPBAR — 56px
====================================================== */
.topbar {
  height: 56px; flex-shrink: 0;
  display: grid; grid-template-columns: 1fr auto 1fr;
  align-items: center; padding: 0 28px;
  background: var(--s1); border-bottom: 1px solid var(--bd);
}

.brand {
  font-size: 11px; font-weight: 800;
  letter-spacing: .2em; text-transform: uppercase;
  color: var(--t1);
}

.bot-tabs { display: flex; gap: 4px; }

.bot-tab {
  display: flex; align-items: center; gap: 9px;
  height: 36px; padding: 0 15px;
  background: transparent; border: 1px solid var(--bd);
  border-radius: 6px; cursor: pointer;
  font-family: var(--fn); font-size: 13px; font-weight: 500;
  color: var(--t3); transition: all .15s; user-select: none;
}
.bot-tab:hover { color: var(--t2); border-color: var(--bd3); background: rgba(255,255,255,.02); }
.bot-tab[aria-selected="true"] {
  color: var(--ac); border-color: var(--ac3); background: var(--ac2); font-weight: 700;
}

.tab-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--t4); flex-shrink: 0; transition: background .2s;
}
.bot-tab[aria-selected="true"] .tab-dot { background: var(--ac); box-shadow: 0 0 5px var(--ac); }

.tab-st {
  font-family: var(--fn); font-size: 8px; font-weight: 800;
  letter-spacing: .1em; text-transform: uppercase;
  padding: 2px 6px; border-radius: 3px; border: 1px solid;
}
.tab-st.live { color: var(--ac); border-color: var(--ac3); background: var(--ac2); }
.tab-st.off  { color: var(--t4); border-color: var(--t4);  background: transparent; }

.top-actions { display: flex; align-items: center; justify-content: flex-end; gap: 8px; }

.tick-badge {
  font-family: var(--mo); font-size: 10px; color: var(--t3);
  padding: 4px 10px; border-radius: 4px; border: 1px solid var(--bd);
  white-space: nowrap; transition: all .2s;
}
.tick-badge.live  { color: var(--ac); border-color: var(--ac3); background: var(--ac2); }
.tick-badge.stale { color: var(--dn); border-color: var(--dn3); background: var(--dn2); }

.ctrl-btn {
  height: 34px; padding: 0 18px; border-radius: 5px;
  font-family: var(--fn); font-size: 11px; font-weight: 800;
  letter-spacing: .08em; text-transform: uppercase;
  cursor: pointer; border: 1px solid; transition: all .15s;
}
.btn-start { color: var(--ac); border-color: var(--ac3); background: var(--ac2); }
.btn-start:hover { background: rgba(57,255,20,.14); }
.btn-start:disabled { opacity: .3; pointer-events: none; }
.btn-stop  { color: var(--t3); border-color: var(--bd); background: transparent; }
.btn-stop:hover { color: var(--dn); border-color: var(--dn3); background: var(--dn2); }
.btn-stop.armed   { color: var(--dn); border-color: var(--dn3); background: var(--dn2); animation: breathe 2.5s ease-in-out infinite; }
.btn-stop.confirm { color: var(--t1); border-color: var(--dn); background: rgba(255,92,92,.18); animation: none; }
.btn-stop.firing  { opacity: .35; pointer-events: none; }
@keyframes breathe { 0%,100%{opacity:1} 50%{opacity:.45} }

/* ======================================================
   STAT ROW — 96px
====================================================== */
.stat-row {
  height: 96px; flex-shrink: 0;
  display: grid; grid-template-columns: repeat(5, 1fr);
  background: var(--s1); border-bottom: 1px solid var(--bd);
}
.stat-card {
  padding: 18px 24px;
  border-right: 1px solid var(--bd);
  display: flex; flex-direction: column; justify-content: center; gap: 5px;
  transition: background .15s;
}
.stat-card:last-child { border-right: none; }
.stat-card:hover { background: rgba(255,255,255,.018); }

.stat-lbl {
  font-size: 9px; font-weight: 700; letter-spacing: .13em;
  text-transform: uppercase; color: var(--t3);
  display: flex; align-items: center; gap: 5px;
}
.stat-val {
  font-family: var(--mo); font-size: 24px; font-weight: 600;
  letter-spacing: -.02em; line-height: 1;
  color: var(--t1); font-variant-numeric: lining-nums tabular-nums;
  transition: color .3s;
}
.stat-sub {
  font-family: var(--mo); font-size: 11px; font-weight: 500;
  font-variant-numeric: tabular-nums;
}

/* ======================================================
   MAIN SCROLL
====================================================== */
.main-scroll { flex: 1; overflow-y: auto; min-height: 0; }

/* Shared section shell */
.sec { border-bottom: 1px solid var(--bd); }

.sec-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 11px 24px;
  background: var(--s1); border-bottom: 1px solid var(--bd);
}
.sec-title {
  font-size: 10px; font-weight: 700; letter-spacing: .13em;
  text-transform: uppercase; color: var(--t3);
  display: flex; align-items: center; gap: 7px;
}
.sec-meta { font-family: var(--mo); font-size: 11px; color: var(--t3); }

/* ======================================================
   EQUITY CHART — 250px
====================================================== */
.eq-wrap {
  height: 250px; position: relative; background: var(--bg);
}
#eq-chart { width: 100%; height: 100%; display: block; }
.eq-overlay {
  position: absolute; top: 14px; right: 20px; pointer-events: none;
  font-family: var(--mo); font-size: 11px; color: var(--t3);
  font-variant-numeric: tabular-nums;
}

/* ======================================================
   MID ROW — Market Conditions | RL Training
====================================================== */
.mid-row {
  display: grid; grid-template-columns: 1fr 340px;
}
.mid-col { display: flex; flex-direction: column; }
.mid-col:first-child { border-right: 1px solid var(--bd); }

.ind-list { padding: 16px 24px; display: flex; flex-direction: column; gap: 0; }
.cond-row {
  display: flex; align-items: flex-start; gap: 12px;
  padding: 11px 0; border-bottom: 1px solid var(--bd2);
}
.cond-row:last-child { border-bottom: none; }
.cond-mk {
  flex-shrink: 0; width: 18px; height: 18px; border-radius: 3px;
  display: flex; align-items: center; justify-content: center;
  font-size: 8px; font-weight: 800; margin-top: 1px;
}
.mk-y { background: var(--ac2); color: var(--ac); border: 1px solid var(--ac3); }
.mk-n { background: var(--dn2); color: var(--dn); border: 1px solid var(--dn3); }
.cond-title { font-size: 13px; font-weight: 600; color: var(--t1); }
.cond-desc  { font-family: var(--mo); font-size: 10px; color: var(--t3); margin-top: 2px; }

.ind-divider { height: 1px; background: var(--bd); margin: 12px 0; }

.ind-bar-row {
  display: flex; align-items: center; gap: 10px; padding: 7px 0;
}
.ind-bar-name { font-size: 13px; font-weight: 500; color: var(--t2); flex: 1; }
.ind-bar-note { font-size: 10px; color: var(--t3); min-width: 60px; text-align: right; }
.ind-bar-val  { font-family: var(--mo); font-size: 13px; font-weight: 600; min-width: 44px; text-align: right; font-variant-numeric: tabular-nums; }
.ind-track    { width: 72px; height: 3px; background: var(--s3); border-radius: 2px; overflow: hidden; flex-shrink: 0; }
.ind-fill     { height: 100%; border-radius: 2px; transition: width .5s ease; }

/* RL cells */
.rl-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--bd); margin: 16px 24px; border-radius: 6px; overflow: hidden; }
.rl-cell { background: var(--s2); padding: 14px 16px; }
.rl-cl   { font-size: 9px; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; color: var(--t3); margin-bottom: 6px; }
.rl-cv   { font-family: var(--mo); font-size: 22px; font-weight: 600; color: var(--t1); line-height: 1; font-variant-numeric: tabular-nums; }
.rl-cv.sm { font-size: 12px; }

/* ======================================================
   TRADES ROW — Open | Closed
====================================================== */
.trades-row {
  display: grid; grid-template-columns: 340px 1fr;
}
.trade-col { display: flex; flex-direction: column; max-height: 300px; }
.trade-col:first-child { border-right: 1px solid var(--bd); }
.trade-col-body { overflow-y: auto; flex: 1; }

/* Open position cards */
.pos-card {
  padding: 13px 24px; border-bottom: 1px solid var(--bd2); transition: background .12s;
}
.pos-card:last-child { border-bottom: none; }
.pos-card:hover { background: rgba(255,255,255,.02); }
.pos-top  { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.pos-pair { font-size: 14px; font-weight: 700; color: var(--t1); }
.pos-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 5px 10px; }
.pf       { display: flex; flex-direction: column; gap: 2px; }
.pf-l     { font-size: 9px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--t3); }
.pf-v     { font-family: var(--mo); font-size: 11px; color: var(--t2); font-variant-numeric: tabular-nums; }

/* Trade table */
.tbl-wrap { overflow-x: auto; }
table.tbl { border-collapse: collapse; width: 100%; }
table.tbl th {
  padding: 8px 16px; font-size: 9px; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase; color: var(--t3);
  text-align: left; border-bottom: 1px solid var(--bd);
  background: var(--s1); white-space: nowrap;
  position: sticky; top: 0; z-index: 2;
}
table.tbl th.r { text-align: right; }
table.tbl td {
  padding: 9px 16px; font-size: 12px; color: var(--t2);
  border-bottom: 1px solid var(--bd2); white-space: nowrap;
}
table.tbl td.r  { text-align: right; }
table.tbl td.mc { font-family: var(--mo); font-size: 10px; color: var(--t3); }
table.tbl td.mv { font-family: var(--mo); font-variant-numeric: tabular-nums; }
table.tbl tr:hover td { background: rgba(255,255,255,.02); }

/* Empty state */
.empty { display: flex; align-items: center; justify-content: center; padding: 30px 20px; }
.empty span { font-size: 12px; color: var(--t4); }

/* Toast */
#toasts { position: fixed; bottom: 20px; right: 20px; z-index: 9999; display: flex; flex-direction: column; gap: 6px; pointer-events: none; }
.toast { padding: 10px 14px; border-radius: 6px; border: 1px solid; font-size: 13px; font-weight: 600; opacity: 0; transform: translateX(10px); transition: all .2s; pointer-events: auto; max-width: 280px; }
.toast.show { opacity: 1; transform: translateX(0); }
.t-ok  { color: var(--ac); background: var(--ac2); border-color: var(--ac3); }
.t-err { color: var(--dn); background: var(--dn2); border-color: var(--dn3); }
.t-inf { color: var(--t2); background: var(--s2);  border-color: var(--bd);  }

/* ── RESPONSIVE ── */
@media (max-width: 1100px) {
  .stat-row { grid-template-columns: repeat(3,1fr); height: auto; }
  .stat-card { border-bottom: 1px solid var(--bd); }
  .mid-row { grid-template-columns: 1fr; }
  .mid-col:first-child { border-right: none; border-bottom: 1px solid var(--bd); }
}
@media (max-width: 800px) {
  body { height: auto; overflow: auto; }
  .main-scroll { overflow: visible; }
  .trades-row { grid-template-columns: 1fr; }
  .trade-col:first-child { border-right: none; border-bottom: 1px solid var(--bd); }
}
</style>
</head>
<body>

<!-- ── TOPBAR ── -->
<div class="topbar">
  <div class="brand">Qoherenz Trading Bot</div>

  <div class="bot-tabs">
    <button class="bot-tab" id="tab-rl" aria-selected="true" onclick="setView('rl')">
      <span class="tab-dot"></span>
      <span>RL Bot</span>
      <span class="tab-st off" id="rl-st">OFFLINE</span>
    </button>
    <button class="bot-tab" id="tab-grid" aria-selected="false" onclick="setView('grid')">
      <span class="tab-dot"></span>
      <span>Grid / DCA</span>
      <span class="tab-st off" id="gr-st">OFFLINE</span>
    </button>
  </div>

  <div class="top-actions">
    <span class="tick-badge" id="tb-fresh">—</span>
    <button class="ctrl-btn btn-start" id="start-btn" onclick="handleStart()">Start</button>
    <button class="ctrl-btn btn-stop"  id="stop-btn"  onclick="handleStop()">Stop</button>
  </div>
</div>

<!-- ── STAT ROW ── -->
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-lbl">ETH / USDT <span class="help tip-dn" data-tip="Live spot price from Binance 24hr ticker. Percentage shows 24-hour change.">?</span></div>
    <div class="stat-val" id="s-price">—</div>
    <div class="stat-sub" id="s-chg">—</div>
  </div>
  <div class="stat-card">
    <div class="stat-lbl">Wallet Balance <span class="help tip-dn" data-tip="Starting balance of $100 plus all realised profits and losses from closed trades.">?</span></div>
    <div class="stat-val" id="s-wallet">—</div>
    <div class="stat-sub t3">base + realised P&amp;L</div>
  </div>
  <div class="stat-card">
    <div class="stat-lbl">Realised P&amp;L <span class="help tip-dn" data-tip="Total money earned or lost from all closed trades. Green is profit, red is loss.">?</span></div>
    <div class="stat-val" id="s-pnl">—</div>
    <div class="stat-sub" id="s-pnl-sub">from closed trades</div>
  </div>
  <div class="stat-card">
    <div class="stat-lbl">Open Trades</div>
    <div class="stat-val" id="s-open">—</div>
    <div class="stat-sub t3">active positions</div>
  </div>
  <div class="stat-card">
    <div class="stat-lbl">Closed Trades</div>
    <div class="stat-val" id="s-closed">—</div>
    <div class="stat-sub" id="s-wr">win rate —</div>
  </div>
</div>

<!-- ── MAIN SCROLL ── -->
<div class="main-scroll">

  <!-- EQUITY CURVE -->
  <div class="sec">
    <div class="sec-head">
      <span class="sec-title">Equity Curve <span class="help" data-tip="Running balance over every closed trade. Green line = overall profitable, red line = overall at a loss. Each point is one closed trade.">?</span></span>
      <span class="sec-meta" id="eq-meta">—</span>
    </div>
    <div class="eq-wrap">
      <canvas id="eq-chart"></canvas>
      <div class="eq-overlay" id="eq-overlay">—</div>
    </div>
  </div>

  <!-- MID: MARKET CONDITIONS + RL TRAINING -->
  <div class="mid-row sec">

    <div class="mid-col sec">
      <div class="sec-head">
        <span class="sec-title">
          Market Conditions
          <span class="help" data-tip="Entry criteria checked before the bot opens a trade. Y (green) = condition met. N (red) = blocked. Bot needs the regime gate plus at least one price condition to enter.">?</span>
        </span>
        <span class="sec-meta" id="mc-meta">—</span>
      </div>
      <div class="ind-list" id="ind-body">
        <div class="empty"><span>Loading indicators…</span></div>
      </div>
    </div>

    <div class="mid-col sec">
      <div class="sec-head">
        <span class="sec-title">RL Training <span class="help" data-tip="The RL model retrains every 8 hours on new price data. Hours Since shows how stale the current model is. Next Retrain turns red when overdue.">?</span></span>
        <span class="sec-meta" id="rl-meta">8h cycle</span>
      </div>
      <div id="rl-body">
        <div class="empty"><span>Loading…</span></div>
      </div>
    </div>

  </div>

  <!-- TRADES ROW -->
  <div class="trades-row sec">

    <div class="trade-col">
      <div class="sec-head">
        <span class="sec-title">Open Positions <span class="help tip-dn" data-tip="Currently active trades. P&L is unrealised and changes with the live ETH price.">?</span></span>
        <span class="sec-meta" id="open-cnt">0 active</span>
      </div>
      <div class="trade-col-body" id="open-body">
        <div class="empty"><span>No open positions</span></div>
      </div>
    </div>

    <div class="trade-col">
      <div class="sec-head">
        <span class="sec-title">Closed Trades <span class="help tip-dn" data-tip="All completed trades ordered by close date. P% is the percentage return, P&L is absolute USDT earned or lost.">?</span></span>
        <span class="sec-meta" id="closed-cnt">—</span>
      </div>
      <div class="trade-col-body" id="closed-body">
        <div class="empty"><span>No closed trades yet</span></div>
      </div>
    </div>

  </div>

</div><!-- /main-scroll -->

<div id="toasts"></div>

<script>
/* ── STATE ── */
let view = 'rl', chart = null, lastPrice = null;
let stopPending = false, stopTimer = null;
let lastOk = null, refreshTimer = null, freshTimer = null;

const $   = id => document.getElementById(id);
const fmt  = (n, d=2) => (n==null||isNaN(+n)) ? '—' : Number(n).toFixed(d);
const fmtM = n => n==null ? '—' : Number(n).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const fmtP = (n, d=2) => n==null ? '—' : (n>=0?'+':'')+fmt(n,d);
const pcls = n => n>0 ? 'up' : n<0 ? 'dn' : 'wn';

/* ── TOAST ── */
function toast(msg, type='info') {
  const el = document.createElement('div');
  el.className = 'toast t-'+(type==='ok'?'ok':type==='err'?'err':'inf');
  el.textContent = msg;
  $('toasts').appendChild(el);
  requestAnimationFrame(()=>requestAnimationFrame(()=>el.classList.add('show')));
  setTimeout(()=>{ el.classList.remove('show'); setTimeout(()=>el.remove(),220); }, 3500);
}

/* ── CHART ── */
function initChart() {
  if (chart) { chart.destroy(); chart = null; }
  const ctx = $('eq-chart').getContext('2d');
  const g = ctx.createLinearGradient(0,0,0,250);
  g.addColorStop(0,'rgba(57,255,20,.14)'); g.addColorStop(1,'rgba(57,255,20,0)');
  chart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{
      data: [], borderColor: '#39FF14', backgroundColor: g,
      borderWidth: 1.5, pointRadius: 0, pointHoverRadius: 4,
      pointHoverBackgroundColor: '#39FF14',
      fill: 'origin', tension: 0.3,
    }]},
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 380 },
      plugins: { legend: { display: false },
        tooltip: {
          backgroundColor: '#101018', titleColor: '#52526A', bodyColor: '#F0F0FF',
          borderColor: 'rgba(255,255,255,.07)', borderWidth: 1, padding: 10, cornerRadius: 5,
          callbacks: { label: c => ' $'+c.parsed.y.toFixed(4)+' USDT' }
        }
      },
      scales: {
        x: { ticks: { color: '#52526A', maxTicksLimit: 8, font: { size: 9, family: 'JetBrains Mono' } },
             grid: { color: 'rgba(255,255,255,.04)' }, border: { color: 'rgba(255,255,255,.04)' } },
        y: { position: 'right',
             ticks: { color: '#52526A', font: { size: 9, family: 'JetBrains Mono' }, callback: v=>'$'+v.toFixed(2) },
             grid: { color: 'rgba(255,255,255,.04)' }, border: { color: 'rgba(255,255,255,.04)' } }
      }
    }
  });
}

/* ── FRESHNESS ── */
function fmtAge(ms) {
  if (ms===null) return '—';
  const s=Math.floor(ms/1000);
  if (s<60) return 'LIVE  '+s+'s';
  const m=Math.floor(s/60); return m<10?m+'m '+s%60+'s':'STALE  '+m+'m';
}
function updateFreshness() {
  const el=$('tb-fresh'); if(!el) return;
  const age = lastOk===null?null:Date.now()-lastOk;
  el.textContent = fmtAge(age);
  el.className = 'tick-badge'+(age!==null&&age<30000?' live':age!==null&&age>=120000?' stale':'');
}

/* ── STATUS ── */
async function updStatus() {
  const d = await fetch('/api/status').then(r=>r.json()).catch(()=>null);
  if (!d) return;
  const apply = (tabId, stId, info, isActive) => {
    const tab=$('tab-'+tabId), st=$(stId); if(!tab||!st) return;
    const on = info&&info.running;
    st.textContent = on?'LIVE':'OFFLINE';
    st.className = 'tab-st '+(on?'live':'off');
    /* glowing dot only when live */
    const dot = tab.querySelector('.tab-dot');
    if (dot) dot.style.cssText = on?'background:var(--ac);box-shadow:0 0 5px var(--ac)':'';
    if (isActive) {
      const sb=$('stop-btn');
      if (on&&!stopPending) sb.classList.add('armed');
      else if (!on) sb.classList.remove('armed','confirm');
    }
  };
  apply('rl',  'rl-st', d.rl,   view==='rl');
  apply('grid','gr-st', d.grid, view==='grid');
}

/* ── PRICE ── */
async function updPrice() {
  const d = await fetch('/api/price').then(r=>r.json()).catch(()=>null);
  if (!d||d.error) return;
  const newP = +d.price, chg = d.change_pct!=null ? +d.change_pct : 0;
  const pe=$('s-price'), ce=$('s-chg');
  if (pe) {
    if (lastPrice!==null && newP!==lastPrice) {
      const dir = newP>lastPrice?'up':'dn';
      pe.classList.remove('flash-up','flash-dn');
      void pe.offsetWidth;
      pe.classList.add('flash-'+dir);
    }
    pe.textContent = '$'+fmtM(newP);
  }
  if (ce) {
    ce.textContent = (chg>=0?'+':'')+chg.toFixed(2)+'%  24h';
    ce.className = 'stat-sub '+(chg>0?'up':chg<0?'dn':'wn');
  }
  lastPrice = newP;
}

/* ── TRADES ── */
async function updTrades() {
  const d = await fetch('/api/trades?mode='+view).then(r=>r.json()).catch(()=>null);
  if (!d) return;
  const st=d.stats||{}, open=d.open||[], closed=d.closed||[];
  const {total_pnl:pnl=0, total_closed:tc=0, win_rate:wr} = st;

  /* stat row */
  const wallet = 100 + (pnl||0);
  const we=$('s-wallet');
  if (we) { we.textContent='$'+fmtM(wallet); we.className='stat-val '+(wallet>100?'up':wallet<100?'dn':''); }
  const pe=$('s-pnl');
  if (pe) { pe.textContent=fmtP(pnl,4)+' USDT'; pe.className='stat-val '+pcls(pnl||0); }
  const ps=$('s-pnl-sub');
  if (ps) ps.textContent=tc+' trades closed';
  const oe=$('s-open'); if (oe) oe.textContent=open.length+'';
  const ce=$('s-closed'); if (ce) ce.textContent=tc+'';
  const we2=$('s-wr');
  if (we2) { we2.textContent = wr!=null ? 'win rate '+wr+'%' : 'win rate —'; we2.className='stat-sub '+(wr!=null?(wr>50?'up':wr<50?'dn':'wn'):''); }
  if ($('open-cnt')) $('open-cnt').textContent=open.length+' active';
  if ($('closed-cnt')) $('closed-cnt').textContent=tc+' total';

  /* open positions */
  const ob=$('open-body');
  if (ob) {
    if (!open.length) { ob.innerHTML='<div class="empty"><span>No open positions</span></div>'; }
    else ob.innerHTML=open.map(t=>{
      const cur=lastPrice||t.open_rate;
      const pct=(cur-t.open_rate)/t.open_rate*100;
      const ms=Date.now()-new Date(t.open_date+'Z');
      const h=Math.floor(ms/3600000), m=Math.floor((ms%3600000)/60000);
      return `<div class="pos-card">
        <div class="pos-top">
          <span class="pos-pair">${t.pair}</span>
          <span class="badge ${pct>=0?'b-up':'b-dn'}">${fmtP(pct)}%</span>
        </div>
        <div class="pos-grid">
          <div class="pf"><span class="pf-l">Entry</span><span class="pf-v">$${fmt(t.open_rate,2)}</span></div>
          <div class="pf"><span class="pf-l">Current</span><span class="pf-v ${pcls(pct)}">$${fmt(cur,2)}</span></div>
          <div class="pf"><span class="pf-l">Size</span><span class="pf-v">$${fmt(t.stake_amount,2)}</span></div>
          <div class="pf"><span class="pf-l">Opened</span><span class="pf-v">${(t.open_date||'').slice(0,10)}</span></div>
          <div class="pf"><span class="pf-l">Age</span><span class="pf-v">${h}h ${m}m</span></div>
          <div class="pf"><span class="pf-l">Tag</span><span class="pf-v">${t.enter_tag||'—'}</span></div>
        </div>
      </div>`;
    }).join('');
  }

  /* closed trades */
  const cb=$('closed-body');
  if (cb) {
    if (!closed.length) { cb.innerHTML='<div class="empty"><span>No closed trades yet</span></div>'; }
    else {
      const rCls = r=>r==='roi'?'b-up':r==='stop_loss'?'b-dn':r==='trailing_stop'?'b-wn':'b-nu';
      cb.innerHTML=`<div class="tbl-wrap"><table class="tbl">
        <thead><tr>
          <th>Pair</th><th>Opened</th><th>Closed</th>
          <th class="r">P%</th><th class="r">P&amp;L</th><th>Result</th>
        </tr></thead>
        <tbody>${closed.map(t=>{
          const pr=(t.profit_ratio||0)*100, pa=t.profit_abs||0;
          return `<tr>
            <td style="font-weight:700;color:var(--t1)">${t.pair}</td>
            <td class="mc">${(t.open_date||'').slice(0,10)}</td>
            <td class="mc">${(t.close_date||'').slice(0,10)}</td>
            <td class="mv r ${pcls(pr)}">${fmtP(pr)}%</td>
            <td class="mv r ${pcls(pa)}">${fmtP(pa,4)}</td>
            <td><span class="badge ${rCls(t.exit_reason)}">${t.exit_reason||'—'}</span></td>
          </tr>`;
        }).join('')}</tbody>
      </table></div>`;
    }
  }
}

/* ── EQUITY ── */
async function updEquity() {
  const d = await fetch('/api/equity?mode='+view).then(r=>r.json()).catch(()=>null);
  if (!d||!chart) return;
  const pts=d.points||[];
  if (pts.length<=1) {
    $('eq-overlay').textContent='No trades yet';
    chart.data.labels=[]; chart.data.datasets[0].data=[];
    chart.update('none'); return;
  }
  const last=pts[pts.length-1].balance, first=pts[0].balance, diff=last-first;
  const lc = diff>0?'#39FF14':diff<0?'#FF5C5C':'#FFB347';
  const fr = diff>0?[57,255,20]:diff<0?[255,92,92]:[255,179,71];
  const ctx2=$('eq-chart').getContext('2d');
  const g2=ctx2.createLinearGradient(0,0,0,250);
  g2.addColorStop(0,`rgba(${fr[0]},${fr[1]},${fr[2]},.13)`);
  g2.addColorStop(1,`rgba(${fr[0]},${fr[1]},${fr[2]},0)`);
  chart.data.datasets[0].borderColor=lc;
  chart.data.datasets[0].backgroundColor=g2;
  chart.data.datasets[0].pointHoverBackgroundColor=lc;
  chart.data.labels=pts.map((p,i)=>i===0?'Start':p.t.slice(5,10));
  chart.data.datasets[0].data=pts.map(p=>p.balance);
  chart.update('active');
  $('eq-overlay').textContent=(diff>=0?'+':'')+diff.toFixed(4)+' USDT · '+(pts.length-1)+' trades';
  $('eq-meta').textContent='$'+fmtM(last)+' balance';
}

/* ── INDICATORS / MARKET CONDITIONS ── */
async function updIndicators() {
  const tf = view==='grid'?'15m':'1h';
  const d = await fetch('/api/indicators?interval='+tf).then(r=>r.json()).catch(()=>null);
  if (!d||d.error) { return; }
  const {rsi, bb_position:bbp, bb_width:bbw, adx, regime} = d;

  const rmap = {
    'ranging':         ['b-wn','Ranging'],
    'bull-trending':   ['b-up','Bull Trend'],
    'bear-trending':   ['b-dn','Bear Trend'],
    'high-volatility': ['b-dn','High Vol'],
  };
  const [rc,rl]=(rmap[regime]||['b-nu','Unknown']);
  $('mc-meta').innerHTML=`<span class="badge ${rc}">${rl}</span>`;

  let conds;
  if (view==='grid') {
    const m1=adx<28&&rsi<29&&rsi>22&&bbp<0.357;
    const m2=adx>=20&&adx<50&&rsi<48&&rsi>28&&bbp<0.55;
    const m3=bbw<0.022;
    conds=[
      [m1,'Mode 1 — Ranging',       `ADX ${fmt(adx,1)} below 28  ·  RSI ${fmt(rsi,1)} at 22–29  ·  BBPos below 0.357`],
      [m2,'Mode 2 — Trend Pullback',`ADX 20–50  ·  RSI ${fmt(rsi,1)} at 28–48  ·  BBPos below 0.55`],
      [m3,'Mode 3 — Squeeze',       `BB Width ${fmt(bbw,4)}  below 0.022 (squeeze active)`],
    ];
  } else {
    const r1=regime!=='high-volatility'&&regime!=='bear-trending';
    const r2=+rsi<45; const r3=+bbp<0.5;
    conds=[
      [r1,'Regime gate', `${regime} — ${r1?'entry allowed':'entry blocked'}`],
      [r2,'RSI gate',    `RSI ${fmt(rsi,1)} — ${r2?'below 45':'above 45, blocked'}`],
      [r3,'BB Position', `BBPos ${fmt(bbp,3)} — ${r3?'lower half, ok':'upper half, blocked'}`],
    ];
  }

  const condHtml = conds.map(([ok,t,desc])=>`
    <div class="cond-row">
      <div class="cond-mk ${ok?'mk-y':'mk-n'}">${ok?'Y':'N'}</div>
      <div>
        <div class="cond-title">${t}</div>
        <div class="cond-desc">${desc}</div>
      </div>
    </div>`).join('');

  const mkBar=(name,val,pct,col,note)=>`
    <div class="ind-bar-row">
      <span class="ind-bar-name">${name}</span>
      <span class="ind-bar-note">${note}</span>
      <div class="ind-track"><div class="ind-fill" style="width:${Math.min(100,Math.max(0,pct))}%;background:${col}"></div></div>
      <span class="ind-bar-val" style="color:${col}">${val}</span>
    </div>`;

  const bars=[
    ['RSI 14', fmt(rsi,1), +rsi, +rsi<30?'var(--ac)':+rsi>70?'var(--dn)':'var(--t3)', +rsi<30?'Oversold':+rsi>70?'Overbought':'Neutral'],
    ['BB Pos',  fmt(bbp,3), +bbp*100, +bbp<0.35?'var(--ac)':+bbp>0.8?'var(--dn)':'var(--t3)', +bbp<0.35?'Lower':+bbp>0.8?'Upper':'Mid'],
    ['BB Width',fmt(bbw,4), Math.min(+bbw*800,100), +bbw>0.08?'var(--dn)':+bbw<0.022?'var(--wn)':'var(--t3)', +bbw>0.08?'Volatile':+bbw<0.022?'Squeeze':'Normal'],
    ['ADX 14',  fmt(adx,1), Math.min(+adx*2,100), +adx>40?'var(--wn)':+adx>25?'var(--ac)':'var(--t3)', +adx>40?'Strong':+adx>25?'Trending':'Weak'],
  ];

  $('ind-body').innerHTML = condHtml
    + `<div class="ind-divider"></div>`
    + bars.map(b=>mkBar(...b)).join('');
}

/* ── RL TRAINING ── */
async function updRL() {
  const rl=$('rl-body'); if(!rl) return;
  if (view!=='rl') {
    rl.innerHTML='<div class="empty"><span>RL info shown for RL Bot only</span></div>';
    $('rl-meta').textContent='—';
    return;
  }
  const d = await fetch('/api/rl').then(r=>r.json()).catch(()=>null);
  if (!d||!d.last_retrain) {
    rl.innerHTML=`<div class="rl-grid">
      <div class="rl-cell"><div class="rl-cl">Last Retrain</div><div class="rl-cv sm" style="color:var(--t3)">—</div></div>
      <div class="rl-cell"><div class="rl-cl">Hours Since</div><div class="rl-cv" style="color:var(--t3)">—</div></div>
      <div class="rl-cell"><div class="rl-cl">Next Retrain</div><div class="rl-cv" style="color:var(--t3)">—</div></div>
      <div class="rl-cell"><div class="rl-cl">Cycle</div><div class="rl-cv">8h</div></div>
    </div>`;
    $('rl-meta').textContent='no model';
    return;
  }
  const nh=+d.next_retrain_h;
  const nc=nh<1?'var(--dn)':nh<3?'var(--wn)':'var(--ac)';
  rl.innerHTML=`<div class="rl-grid">
    <div class="rl-cell"><div class="rl-cl">Last Retrain</div><div class="rl-cv sm">${d.last_retrain}</div></div>
    <div class="rl-cell"><div class="rl-cl">Hours Since</div><div class="rl-cv">${d.hours_since}h</div></div>
    <div class="rl-cell"><div class="rl-cl">Next Retrain</div><div class="rl-cv" style="color:${nc}">${nh}h</div></div>
    <div class="rl-cell"><div class="rl-cl">Cycle</div><div class="rl-cv">8h</div></div>
  </div>`;
  $('rl-meta').textContent='next in '+nh+'h';
}

/* ── VIEW SWITCH ── */
function setView(m) {
  view=m;
  document.querySelectorAll('.bot-tab').forEach(el=>el.setAttribute('aria-selected','false'));
  $('tab-'+m).setAttribute('aria-selected','true');
  refreshFocused();
}

/* ── STOP / START ── */
function handleStop() {
  const b=$('stop-btn');
  if (stopPending) { clearTimeout(stopTimer); stopPending=false; resetStop(); doStop(); return; }
  stopPending=true;
  b.textContent='CONFIRM'; b.className='ctrl-btn btn-stop confirm';
  stopTimer=setTimeout(()=>{ stopPending=false; resetStop(); },4000);
}
function resetStop() { const b=$('stop-btn'); b.textContent='Stop'; b.className='ctrl-btn btn-stop armed'; }
async function doStop() {
  const b=$('stop-btn');
  b.textContent='…'; b.className='ctrl-btn btn-stop firing';
  const d=await fetch('/api/bot/stop?mode='+view,{method:'POST'}).then(r=>r.json()).catch(()=>null);
  toast(d?d.msg:'Request failed', d?.ok?'ok':'err');
  setTimeout(()=>{ b.textContent='Stop'; b.className='ctrl-btn btn-stop'; updStatus(); },800);
}
async function handleStart() {
  const b=$('start-btn'); b.disabled=true; b.textContent='…';
  const d=await fetch('/api/bot/start?mode='+view,{method:'POST'}).then(r=>r.json()).catch(()=>null);
  toast(d?d.msg:'Request failed', d?.ok?'ok':'err');
  setTimeout(()=>{ b.disabled=false; b.textContent='Start'; updStatus(); },1200);
}

/* ── REFRESH ── */
async function refreshFocused() {
  await Promise.allSettled([updTrades(),updEquity(),updIndicators(),updRL()]);
}
async function refreshAll() {
  await Promise.allSettled([updStatus(),updPrice(),refreshFocused()]);
  lastOk=Date.now(); updateFreshness();
}

document.addEventListener('DOMContentLoaded',()=>{
  initChart();
  setView('rl');
  refreshAll();
  refreshTimer=setInterval(refreshAll,5000);
  freshTimer  =setInterval(updateFreshness,1000);
  document.addEventListener('visibilitychange',()=>{
    if (document.hidden) {
      clearInterval(refreshTimer); refreshTimer=null;
      clearInterval(freshTimer);   freshTimer=null;
    } else {
      refreshAll();
      if (!refreshTimer) refreshTimer=setInterval(refreshAll,5000);
      if (!freshTimer)   freshTimer  =setInterval(updateFreshness,1000);
    }
  });
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=HTML)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8899, log_level="warning")
