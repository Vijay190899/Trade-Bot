"""
Gate F6 status report — operational stability of the cash-and-carry dry run.

F6 threshold: 30 consecutive days with zero unplanned outages.

Reports uptime, gaps in the heartbeat, accrued funding, hedge integrity and
margin health, so progress toward the gate is measurable rather than assumed.

Run:  python carry/f6_status.py
"""
import os
import sqlite3
from datetime import datetime, timedelta

BOT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BOT_ROOT, "carry", "carry_state.sqlite")
LOG = os.path.join(BOT_ROOT, "user_data", "logs", "carry_run.log")

TICK_MIN = 5
GAP_TOLERANCE_MIN = 20      # > 4 missed ticks counts as an outage
F6_DAYS = 30


def heartbeats():
    if not os.path.exists(LOG):
        return []
    out = []
    with open(LOG, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "heartbeat" in line:
                try:
                    out.append(datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    pass
    return sorted(out)


def main():
    hb = heartbeats()
    print("=" * 74)
    print("GATE F6 — OPERATIONAL STABILITY  (threshold: 30 consecutive days)")
    print("=" * 74)
    if not hb:
        print("  no heartbeats recorded yet")
        return

    start, last = hb[0], hb[-1]
    age = (datetime.now() - last).total_seconds() / 60
    elapsed = (last - start).total_seconds() / 86400

    gaps = []
    for a, b in zip(hb, hb[1:]):
        mins = (b - a).total_seconds() / 60
        if mins > GAP_TOLERANCE_MIN:
            gaps.append((a, b, mins))

    # longest clean run
    streak_start, best, best_start, best_end = start, timedelta(0), start, start
    for a, b, _ in gaps:
        if a - streak_start > best:
            best, best_start, best_end = a - streak_start, streak_start, a
        streak_start = b
    if last - streak_start > best:
        best, best_start, best_end = last - streak_start, streak_start, last

    print(f"  first heartbeat : {start:%Y-%m-%d %H:%M}")
    print(f"  last heartbeat  : {last:%Y-%m-%d %H:%M}  ({age:.1f} min ago)")
    print(f"  status          : {'RUNNING' if age < GAP_TOLERANCE_MIN else 'STALE / DOWN'}")
    print(f"  elapsed         : {elapsed:.2f} days")
    print(f"  heartbeats      : {len(hb)}")
    print(f"  outages (>{GAP_TOLERANCE_MIN}m) : {len(gaps)}")
    for a, b, m in gaps[-5:]:
        print(f"      {a:%Y-%m-%d %H:%M} -> {b:%Y-%m-%d %H:%M}  ({m:.0f} min)")

    days = best.total_seconds() / 86400
    print(f"\n  longest clean run: {days:.2f} days  ({best_start:%Y-%m-%d %H:%M} -> {best_end:%Y-%m-%d %H:%M})")
    print(f"  F6 progress      : {days:.2f} / {F6_DAYS} days  ({days/F6_DAYS*100:.1f}%)")
    print(f"  F6 verdict       : {'PASS' if days >= F6_DAYS else 'IN PROGRESS'}")

    if not os.path.exists(DB):
        return
    con = sqlite3.connect(DB)
    print("\n" + "=" * 74)
    print("POSITION STATE")
    print("=" * 74)
    rows = con.execute("""
        SELECT symbol, spot, perp, basis_pct, funding_cum, equity,
               hedge_drift, margin_ratio
        FROM equity e
        WHERE ts = (SELECT MAX(ts) FROM equity WHERE symbol = e.symbol)
        ORDER BY symbol
    """).fetchall()
    print(f"  {'sym':<9}{'basis%':>8}{'funding':>10}{'equity':>11}{'drift%':>8}{'margin%':>9}")
    tot_eq = tot_f = 0.0
    for r in rows:
        print(f"  {r[0]:<9}{r[3]:>8.3f}{r[4]:>10.4f}{r[5]:>11.2f}{r[6]*100:>8.3f}{r[7]*100:>9.1f}")
        tot_eq += r[5]
        tot_f += r[4]
    if rows:
        deployed = 2500.0 * len(rows)
        pnl = tot_eq - deployed
        print(f"  {'TOTAL':<9}{'':>8}{tot_f:>10.4f}{tot_eq:>11.2f}")
        print(f"\n  deployed {deployed:,.0f} USDT   P&L {pnl:+,.2f} ({pnl/deployed*100:+.3f}%)")
        if elapsed > 0.5:
            print(f"  annualised (naive): {(pnl/deployed)*(365/elapsed)*100:+.2f}%")

    ev = con.execute("SELECT ts,symbol,level,message FROM events "
                     "WHERE level IN ('WARN','CRITICAL') ORDER BY ts DESC LIMIT 8").fetchall()
    print("\n" + "=" * 74)
    print(f"SAFETY EVENTS  ({len(ev)} recent warn/critical)")
    print("=" * 74)
    if not ev:
        print("  none — hedge integrity and margin within limits")
    for t, s, l, m in ev:
        print(f"  {t[:19]}  [{l}] {s}: {m}")


if __name__ == "__main__":
    main()
