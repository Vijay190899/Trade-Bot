"""
TradeMemory — persistent experience store for the self-improving RL loop.

Every completed trade is written to a SQLite database on V: drive.
The AntigravityRLModel reads this database during each retraining cycle
to shape the PPO reward function based on what actually happened in live
(or dry-run) trading — wins are reinforced, loss patterns are penalised.

Flow:
  strategy.confirm_trade_entry() → cache_entry(pair, time, features)
  strategy.confirm_trade_exit()  → store_trade(trade, features, profit)
                                       ↓
                              trade_memory.db (SQLite, V: drive)
                                       ↓  (every 8 h retraining)
  AntigravityRLModel.fit()     → reads regime stats
                               → adjusts reward weights
                               → retrains PPO on OHLCV simulation
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DB_PATH = Path("V:/Antigravity/Trade tool/bot/user_data/models/trade_memory.db")


class TradeMemory:
    """
    Thread-safe SQLite-backed trade experience store.
    Uses WAL mode so the strategy writer and the RL model reader
    never block each other.
    """

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        # In-memory cache for entry features (pair+open_time → dict)
        self._entry_cache: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS trades (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id         TEXT    NOT NULL,
                    pair             TEXT    NOT NULL,
                    side             TEXT    NOT NULL DEFAULT 'long',
                    entry_time       TEXT    NOT NULL,
                    exit_time        TEXT    NOT NULL,
                    entry_price      REAL    NOT NULL,
                    exit_price       REAL    NOT NULL,
                    profit_pct       REAL    NOT NULL,
                    profit_abs       REAL    NOT NULL,
                    exit_reason      TEXT,
                    hold_hours       REAL,
                    regime           TEXT,
                    rsi_entry        REAL,
                    bb_pos_entry     REAL,
                    adx_entry        REAL,
                    ml_score_entry   REAL,
                    features_json    TEXT,
                    created_at       TEXT    DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_pair_exit  ON trades (pair, exit_time);
                CREATE INDEX IF NOT EXISTS idx_regime     ON trades (regime);

                CREATE TABLE IF NOT EXISTS regime_stats (
                    regime           TEXT PRIMARY KEY,
                    trade_count      INTEGER DEFAULT 0,
                    win_rate         REAL    DEFAULT 0.5,
                    avg_profit_pct   REAL    DEFAULT 0.0,
                    avg_hold_hours   REAL    DEFAULT 0.0,
                    updated_at       TEXT
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Entry caching (pair + open_time → features dict)
    # ------------------------------------------------------------------

    def cache_entry(self, pair: str, open_time: datetime, features: dict) -> None:
        key = f"{pair}|{open_time.isoformat()}"
        self._entry_cache[key] = {**features, "cached_at": datetime.now(timezone.utc).isoformat()}

    def pop_entry(self, pair: str, open_time: datetime) -> dict:
        key = f"{pair}|{open_time.isoformat()}"
        return self._entry_cache.pop(key, {})

    # ------------------------------------------------------------------
    # Write a completed trade
    # ------------------------------------------------------------------

    def store_trade(
        self,
        trade_id: str,
        pair: str,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
        exit_price: float,
        profit_pct: float,
        profit_abs: float,
        exit_reason: str,
        entry_features: dict,
    ) -> None:
        hold_hours = (exit_time - entry_time).total_seconds() / 3600.0
        regime = entry_features.get("regime", "unknown")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    trade_id, pair, side,
                    entry_time, exit_time,
                    entry_price, exit_price,
                    profit_pct, profit_abs,
                    exit_reason, hold_hours, regime,
                    rsi_entry, bb_pos_entry, adx_entry, ml_score_entry,
                    features_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    str(trade_id), pair, "long",
                    entry_time.isoformat(), exit_time.isoformat(),
                    entry_price, exit_price,
                    profit_pct, profit_abs,
                    exit_reason, round(hold_hours, 2), regime,
                    entry_features.get("rsi14"),
                    entry_features.get("bb_position"),
                    entry_features.get("adx14"),
                    entry_features.get("ml_score"),
                    json.dumps({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                for k, v in entry_features.items()}),
                ),
            )
        self._refresh_regime_stats(regime)
        logger.info(
            "TradeMemory: stored trade %s | %s | profit=%.2f%%",
            trade_id, pair, profit_pct * 100,
        )

    # ------------------------------------------------------------------
    # Regime statistics (read by RL reward function)
    # ------------------------------------------------------------------

    def _refresh_regime_stats(self, regime: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO regime_stats
                    (regime, trade_count, win_rate, avg_profit_pct, avg_hold_hours, updated_at)
                SELECT
                    regime,
                    COUNT(*)                                             AS trade_count,
                    AVG(CASE WHEN profit_pct > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
                    AVG(profit_pct)                                      AS avg_profit_pct,
                    AVG(hold_hours)                                      AS avg_hold_hours,
                    datetime('now')
                FROM trades
                WHERE regime = ?
                GROUP BY regime
                """,
                (regime,),
            )

    def get_regime_stats(self) -> Dict[str, dict]:
        """Return all regime performance stats (used by RL reward shaper)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM regime_stats").fetchall()
        return {
            r["regime"]: {
                "trade_count":    r["trade_count"],
                "win_rate":       r["win_rate"],
                "avg_profit_pct": r["avg_profit_pct"],
                "avg_hold_hours": r["avg_hold_hours"],
            }
            for r in rows
        }

    def get_regime_win_rate(self, regime: str) -> float:
        """Return win rate for a specific regime, default 0.5 if no data."""
        stats = self.get_regime_stats()
        return stats.get(regime, {}).get("win_rate", 0.5)

    # ------------------------------------------------------------------
    # Penalty signal for the RL reward function
    # ------------------------------------------------------------------

    def recent_loss_penalty(self, days: int = 7) -> float:
        """
        Returns a scalar in [0.0, 1.0] indicating how bad the recent period
        was. 0.0 = all wins, 1.0 = all losses. Used to scale entry penalties
        in the RL reward function during retraining.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT AVG(CASE WHEN profit_pct < 0 THEN 1.0 ELSE 0.0 END)
                FROM trades WHERE exit_time > ?
                """,
                (cutoff,),
            ).fetchone()
        val = row[0] if row and row[0] is not None else 0.0
        return float(val)

    def similar_entry_win_rate(
        self, rsi: float, bb_pos: float, regime: str, tolerance: float = 0.10
    ) -> float:
        """
        Win rate of past trades entered in similar conditions.
        Used to adjust entry reward in the RL environment.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT AVG(CASE WHEN profit_pct > 0 THEN 1.0 ELSE 0.0 END)
                FROM trades
                WHERE regime = ?
                  AND ABS(rsi_entry - ?) < 15
                  AND ABS(bb_pos_entry - ?) < ?
                """,
                (regime, rsi, bb_pos, tolerance),
            ).fetchone()
        val = row[0] if row and row[0] is not None else None
        return float(val) if val is not None else 0.5  # 0.5 = no info yet

    # ------------------------------------------------------------------
    # Bulk read for RL training
    # ------------------------------------------------------------------

    def to_dataframe(self, days: int = 90) -> pd.DataFrame:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            return pd.read_sql(
                "SELECT * FROM trades WHERE exit_time > ? ORDER BY exit_time",
                conn,
                params=(cutoff,),
            )

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                                             AS total,
                    AVG(CASE WHEN profit_pct > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
                    AVG(profit_pct)                                      AS avg_pct,
                    SUM(profit_abs)                                      AS total_eur
                FROM trades
                """
            ).fetchone()
        if not row or row["total"] == 0:
            return {"message": "No trades recorded yet."}
        return {
            "total_trades":    row["total"],
            "win_rate":        f"{row['win_rate'] * 100:.1f}%",
            "avg_profit":      f"{row['avg_pct'] * 100:.3f}%",
            "total_profit_eur": f"€{row['total_eur']:.4f}",
        }

    def regime_report(self) -> str:
        stats = self.get_regime_stats()
        if not stats:
            return "No regime data yet."
        lines = ["Regime Performance:"]
        for regime, s in sorted(stats.items()):
            lines.append(
                f"  {regime:<20} trades={s['trade_count']:>4}  "
                f"win={s['win_rate']*100:>5.1f}%  "
                f"avg={s['avg_profit_pct']*100:>+6.3f}%"
            )
        return "\n".join(lines)
