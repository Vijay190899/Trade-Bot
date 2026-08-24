"""
AntigravityGridStrategy — DCA Multi-Mode Day Trading
15m timeframe, 3 signal modes, 3-level DCA grid via position adjustment.

Mode 1 — Ranging Mean Reversion  (ADX < 20): buy oversold dips near lower BB
Mode 2 — Trend Pullback          (ADX 20-40): buy EMA20 touches in uptrend
Mode 3 — Squeeze Breakout        (BB squeeze): enter when compression releases upward

DCA grid:
  L1 — initial entry (25 USDT)
  L2 — add 25 USDT if trade falls -1.5%
  L3 — add 25 USDT if trade falls -3.0% (averaged entry now much lower)

No FreqAI — starts immediately, no GPU, no training.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

logger = logging.getLogger(__name__)

_MEM_DIR = Path("V:/Antigravity/Trade tool/bot/user_data/freqaimodels")
sys.path.insert(0, str(_MEM_DIR))
try:
    from trade_memory import TradeMemory
    _trade_memory = TradeMemory()
    logger.info("TradeMemory loaded for grid strategy.")
except Exception as _e:
    logger.warning("TradeMemory unavailable (%s)", _e)
    _trade_memory = None


class AntigravityGridStrategy(IStrategy):

    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "15m"
    process_only_new_candles = True
    startup_candle_count = 60

    # DCA: initial + 2 adjustments = 3 entries max
    position_adjustment_enable = True
    max_entry_position_adjustment = 2

    # Tight ROI for 15m — take profit quickly
    minimal_roi = {
        "0":   0.015,   # 1.5% — momentum spike exit
        "30":  0.008,   # 0.8% after 30 min
        "60":  0.005,   # 0.5% after 1h
        "180": 0.003,   # 0.3% after 3h
        "480": 0.0,     # Break-even after 8h — don't hold losers
    }

    stoploss = -0.015
    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = True

    use_entry_signal = True
    entry_profit_only = False
    ignore_roi_if_entry_signal = False

    # Tunable thresholds — defaults from hyperopt 2026-05-26 (300 epochs, SortinoLoss, 27W/6L backtest)
    buy_rsi_ranging  = IntParameter(28, 50,   default=29,   space="buy",  optimize=True)
    buy_bb_pos       = DecimalParameter(0.10, 0.45, default=0.357, space="buy",  optimize=True)
    buy_rsi_trend    = IntParameter(30, 58,   default=39,   space="buy",  optimize=True)
    sell_rsi_exit    = IntParameter(62, 80,   default=62,   space="sell", optimize=True)
    sell_bb_exit     = DecimalParameter(0.70, 0.95, default=0.888, space="sell", optimize=True)

    @property
    def protections(self):
        return [
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 96,   # 24h on 15m
                "trade_limit": 5,
                "stop_duration_candles": 8,
                "max_allowed_drawdown": 0.08,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,
                "trade_limit": 4,
                "stop_duration_candles": 4,
                "only_per_pair": False,
            },
        ]

    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        # RSI
        rsi = pta.rsi(dataframe["close"], length=14)
        dataframe["rsi"] = rsi if rsi is not None else pd.Series(50.0, index=dataframe.index)

        # Bollinger Bands
        bb = pta.bbands(dataframe["close"], length=20, std=2.0)
        if bb is not None:
            dataframe["bb_upper"] = bb["BBU_20_2.0"]
            dataframe["bb_mid"]   = bb["BBM_20_2.0"]
            dataframe["bb_lower"] = bb["BBL_20_2.0"]
            bw = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"].replace(0, np.nan)
            dataframe["bb_width"] = bw.fillna(0)
            bp = (dataframe["close"] - dataframe["bb_lower"]) / (
                (dataframe["bb_upper"] - dataframe["bb_lower"]).replace(0, np.nan))
            dataframe["bb_position"] = bp.clip(0, 1).fillna(0.5)
        else:
            dataframe["bb_upper"]   = dataframe["close"]
            dataframe["bb_mid"]     = dataframe["close"]
            dataframe["bb_lower"]   = dataframe["close"]
            dataframe["bb_width"]   = 0.0
            dataframe["bb_position"] = 0.5

        # ADX
        adx_df = pta.adx(dataframe["high"], dataframe["low"], dataframe["close"], length=14)
        if adx_df is not None and "ADX_14" in adx_df.columns:
            dataframe["adx"] = adx_df["ADX_14"].fillna(20)
        else:
            dataframe["adx"] = 20.0

        # EMA
        ema20 = pta.ema(dataframe["close"], length=20)
        ema50 = pta.ema(dataframe["close"], length=50)
        dataframe["ema20"] = ema20 if ema20 is not None else dataframe["close"]
        dataframe["ema50"] = ema50 if ema50 is not None else dataframe["close"]

        # Volume moving average
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean().fillna(dataframe["volume"])

        # MACD
        macd_df = pta.macd(dataframe["close"])
        if macd_df is not None and "MACD_12_26_9" in macd_df.columns:
            dataframe["macd"]        = macd_df["MACD_12_26_9"].fillna(0)
            dataframe["macd_signal"] = macd_df["MACDs_12_26_9"].fillna(0)
        else:
            dataframe["macd"]        = 0.0
            dataframe["macd_signal"] = 0.0

        # Squeeze detection (BB compression then release)
        dataframe["bb_squeeze"]   = dataframe["bb_width"] < 0.020
        dataframe["prior_squeeze"] = dataframe["bb_squeeze"].rolling(6).sum() >= 4
        dataframe["bb_expanding"] = (
            (dataframe["bb_width"] > dataframe["bb_width"].shift(1)) &
            (dataframe["bb_width"].shift(1) > dataframe["bb_width"].shift(2))
        )

        # RSI momentum (falling into entry)
        dataframe["rsi_falling"] = dataframe["rsi"] < dataframe["rsi"].shift(2)

        return dataframe

    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "enter_long"] = 0
        dataframe.loc[:, "enter_tag"]  = ""

        # ── Mode 1: Ranging Mean Reversion ────────────────────────────
        # ADX not strong (< 28), price near lower BB, RSI oversold but not crashing
        # Raised from 20→28 so it fires during moderate ranges, not only dead-calm markets
        mode1 = (
            (dataframe["adx"] < 28) &
            (dataframe["rsi"] < self.buy_rsi_ranging.value) &
            (dataframe["rsi"] > 22) &
            (dataframe["bb_position"] < self.buy_bb_pos.value) &
            (dataframe["rsi_falling"]) &
            (dataframe["close"] > dataframe["bb_lower"]) &
            (dataframe["volume"] > dataframe["volume_ma"] * 0.7)
        )
        dataframe.loc[mode1, "enter_tag"]  = "ranging_reversion"
        dataframe.loc[mode1, "enter_long"] = 1

        # ── Mode 2: Trend Pullback ────────────────────────────────────
        # ADX moderate, price pulling back to EMA20.
        # EMA50 requirement relaxed to ±2% so it fires when ETH is near (not strictly above) EMA50.
        mode2 = (
            (dataframe["adx"] >= 20) &
            (dataframe["adx"] < 50) &
            (dataframe["close"] > dataframe["ema50"] * 0.98) &
            (dataframe["rsi"] < self.buy_rsi_trend.value) &
            (dataframe["rsi"] > 28) &
            (dataframe["close"] <= dataframe["ema20"] * 1.010) &
            (dataframe["close"] >= dataframe["ema20"] * 0.990) &
            (dataframe["bb_position"] < 0.55) &
            (dataframe["enter_long"] == 0)
        )
        dataframe.loc[mode2, "enter_tag"]  = "trend_pullback"
        dataframe.loc[mode2, "enter_long"] = 1

        # ── Mode 3: Squeeze Breakout ──────────────────────────────────
        # Compression releases upward with volume and MACD confirmation
        mode3 = (
            (dataframe["prior_squeeze"]) &
            (dataframe["bb_expanding"]) &
            (dataframe["bb_width"] > 0.022) &
            (dataframe["rsi"] > 50) &
            (dataframe["rsi"] < 70) &
            (dataframe["macd"] > dataframe["macd_signal"]) &
            (dataframe["volume"] > dataframe["volume_ma"] * 1.15) &
            (dataframe["enter_long"] == 0)
        )
        dataframe.loc[mode3, "enter_tag"]  = "squeeze_breakout"
        dataframe.loc[mode3, "enter_long"] = 1

        return dataframe

    # ------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "exit_long"] = 0

        # Exit on overbought conditions
        overbought = (
            (dataframe["rsi"] > self.sell_rsi_exit.value) &
            (dataframe["bb_position"] > self.sell_bb_exit.value)
        )
        dataframe.loc[overbought, "exit_long"] = 1

        return dataframe

    # ------------------------------------------------------------------
    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Optional[Union[float, Tuple[Optional[float], str]]]:
        """
        DCA grid — add capital at deeper dip levels.
        L2: triggered at -1.5% loss (averaged entry improves)
        L3: triggered at -3.0% loss (final safety net buy)
        """
        if min_stake is None:
            return None

        count = trade.nr_of_successful_entries
        stake = min(25.0, max_stake)

        if stake < min_stake:
            return None

        if count == 1 and current_profit < -0.015:
            logger.info("DCA L2 %s profit=%.2f%%", trade.pair, current_profit * 100)
            return stake, "dca_l2"

        if count == 2 and current_profit < -0.030:
            logger.info("DCA L3 %s profit=%.2f%%", trade.pair, current_profit * 100)
            return stake, "dca_l3"

        return None

    # ------------------------------------------------------------------
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str,
                           amount: float, rate: float, time_in_force: str,
                           exit_reason: str, current_time: datetime, **kwargs) -> bool:
        """Log closed trades to TradeMemory."""
        if _trade_memory is None:
            return True
        try:
            profit_pct = trade.calc_profit_ratio(rate)
            hold_hours = (current_time - trade.open_date_utc).total_seconds() / 3600
            _trade_memory.store_trade(
                trade_id=str(trade.id),
                pair=pair,
                entry_time=str(trade.open_date_utc),
                exit_time=str(current_time),
                entry_price=trade.open_rate,
                exit_price=rate,
                profit_pct=profit_pct,
                profit_abs=trade.calc_profit(rate),
                exit_reason=exit_reason,
                hold_hours=hold_hours,
                regime=self._get_current_regime(pair),
            )
        except Exception as exc:
            logger.warning("TradeMemory log failed: %s", exc)
        return True

    def _get_current_regime(self, pair: str) -> str:
        try:
            df = self.dp.get_pair_dataframe(pair, self.timeframe)
            if df is None or df.empty:
                return "unknown"
            last = df.iloc[-1]
            if last.get("bb_width", 0) > 0.08:
                return "high-volatility"
            if last.get("adx", 0) > 25:
                return "trending"
            return "ranging"
        except Exception:
            return "unknown"
