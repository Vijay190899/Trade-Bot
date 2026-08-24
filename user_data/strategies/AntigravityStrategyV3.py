"""
Antigravity Trading Strategy  — v2 (Self-Improving RL Edition)

4-stage Ruflo-inspired pipeline:
  Stage 1 — Regime detection      (market-analyst equivalent)
  Stage 2 — FreqAI ML + RL        (trading-strategist + RL policy)
  Stage 3 — Circuit-breaker gate  (risk-analyst, via protections)
  Stage 4 — Mean-reversion entry  (execute)

Self-improvement loop (new in v2):
  Every completed trade → TradeMemory (SQLite, V: drive)
  Every 8 h            → AntigravityRLModel reads TradeMemory
                       → recomputes reward weights per regime
                       → retrains PPO on updated rewards
                       → bot avoids patterns that lost, reinforces winners

Exchange : Kraken  (BTC/EUR)
Capital  : 20 EUR wallet, max 2 open trades × 9 EUR stake
GPU      : PPO + LightGBM via CUDA (RTX 2070)
"""

import logging
import sys
from datetime import timezone
from functools import reduce
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy
from pandas import DataFrame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TradeMemory — import from freqaimodels directory
# ---------------------------------------------------------------------------
_MEM_DIR = Path("V:/Antigravity/Trade tool/bot/user_data/freqaimodels")
sys.path.insert(0, str(_MEM_DIR))
try:
    from trade_memory import TradeMemory
    _trade_memory = TradeMemory()
    logger.info("TradeMemory loaded — self-improvement loop active.")
except Exception as _e:
    logger.warning("TradeMemory unavailable (%s) — trade logging disabled.", _e)
    _trade_memory = None


class AntigravityStrategyV3(IStrategy):

    INTERFACE_VERSION = 3
    can_short = False

    # ------------------------------------------------------------------
    # ROI table (keys = minutes)
    # ------------------------------------------------------------------
    # ROI ladder must stay ABOVE trailing_stop_positive_offset (0.025), otherwise
    # ROI fires first and the trailing stop is dead code. The old ladder decayed
    # to 0.5% at 12h while stoploss risked 3%, giving a 0.42:1 payoff ratio and
    # a NEGATIVE expectancy despite a 70% win rate. Winners now run until the
    # trailing stop takes them out 1.5% off the peak.
    minimal_roi = {
        "0":    0.080,   # immediate spike: 8%
        "360":  0.045,   # 6 h:  4.5%
        "1440": 0.030,   # 24 h: 3.0%
        "2880": 0.025,   # 48 h: 2.5% (floor = trailing offset)
    }

    stoploss = -0.03
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    timeframe = "1h"
    process_only_new_candles = True
    use_entry_signal = True
    entry_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count = 80

    # ------------------------------------------------------------------
    # Hyperopt parameters
    # ------------------------------------------------------------------
    buy_rsi_max      = IntParameter(20,   40,  default=32,  space="buy",  optimize=True)
    buy_bb_pos_max   = DecimalParameter(0.05, 0.35, default=0.20, space="buy",  optimize=True)
    buy_min_ml_score = DecimalParameter(0.00, 0.02, default=0.005, space="buy",  optimize=True)
    sell_rsi_min     = IntParameter(60,   85,  default=68,  space="sell", optimize=True)
    sell_bb_pos_min  = DecimalParameter(0.65, 0.95, default=0.80, space="sell", optimize=True)

    # ------------------------------------------------------------------
    # Per-pair feature cache — updated each candle, read at trade entry
    # ------------------------------------------------------------------
    _latest_features: dict = {}   # pair → {rsi14, bb_position, adx14, regime, ml_score}

    @property
    def protections(self):
        return [
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 1,
                "stop_duration_candles": 12,
                "max_allowed_drawdown": 0.10,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 72,
                "trade_limit": 2,
                "stop_duration_candles": 24,
                "only_per_pair": False,
            },
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 2,
            },
        ]

    # ------------------------------------------------------------------
    # FreqAI — feature engineering
    # ------------------------------------------------------------------

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        rsi = pta.rsi(dataframe["close"], length=period)
        dataframe[f"%-rsi-period_{period}"] = rsi if rsi is not None else 50.0

        atr = pta.atr(dataframe["high"], dataframe["low"], dataframe["close"], length=period)
        if atr is None:
            atr = pd.Series(0.0, index=dataframe.index)
        dataframe[f"%-atr_pct-period_{period}"] = (atr / dataframe["close"]).fillna(0)

        adx_df = pta.adx(dataframe["high"], dataframe["low"], dataframe["close"], length=period)
        dataframe[f"%-adx-period_{period}"] = adx_df[f"ADX_{period}"] if adx_df is not None else 0.0

        bb = pta.bbands(dataframe["close"], length=period, std=2.0)
        if bb is not None:
            upper = bb[f"BBU_{period}_2.0"]
            lower = bb[f"BBL_{period}_2.0"]
            mid   = bb[f"BBM_{period}_2.0"]
        else:
            upper = lower = mid = dataframe["close"]
        bb_range = upper - lower
        dataframe[f"%-bb_width-period_{period}"]    = np.where(mid > 0, bb_range / mid, 0.0)
        dataframe[f"%-bb_position-period_{period}"] = np.where(bb_range > 0, (dataframe["close"] - lower) / bb_range, 0.5)

        sma = pta.sma(dataframe["close"], length=period)
        if sma is None:
            sma = dataframe["close"]
        dataframe[f"%-sma-period_{period}"]          = sma
        dataframe[f"%-close_vs_sma-period_{period}"] = np.where(sma > 0, (dataframe["close"] - sma) / sma, 0.0)

        ema = pta.ema(dataframe["close"], length=period)
        if ema is None:
            ema = dataframe["close"]
        dataframe[f"%-ema-period_{period}"]          = ema
        dataframe[f"%-close_vs_ema-period_{period}"] = np.where(ema > 0, (dataframe["close"] - ema) / ema, 0.0)

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        macd_df = pta.macd(dataframe["close"])
        dataframe["%-macd"]        = macd_df["MACD_12_26_9"]
        dataframe["%-macd_signal"] = macd_df["MACDs_12_26_9"]
        dataframe["%-macd_hist"]   = macd_df["MACDh_12_26_9"]

        vol_mean = dataframe["volume"].rolling(20).mean()
        vol_std  = dataframe["volume"].rolling(20).std()
        dataframe["%-volume_zscore"] = np.where(vol_std > 0, (dataframe["volume"] - vol_mean) / vol_std, 0.0)
        dataframe["%-volume_change"] = dataframe["volume"].pct_change().fillna(0)

        ohlc_max = dataframe[["open", "close"]].max(axis=1)
        ohlc_min = dataframe[["open", "close"]].min(axis=1)
        dataframe["%-candle_body"]  = np.where(dataframe["open"] > 0, (dataframe["close"] - dataframe["open"]) / dataframe["open"], 0.0)
        dataframe["%-upper_wick"]   = np.where(dataframe["open"] > 0, (dataframe["high"] - ohlc_max) / dataframe["open"], 0.0)
        dataframe["%-lower_wick"]   = np.where(dataframe["open"] > 0, (ohlc_min - dataframe["low"]) / dataframe["open"], 0.0)

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        for lag in [1, 4, 12, 24]:
            dataframe[f"%-pct_change_{lag}"] = dataframe["close"].pct_change(lag).fillna(0)

        dataframe["%-hour_sin"] = np.sin(2 * np.pi * dataframe["date"].dt.hour / 24)
        dataframe["%-hour_cos"] = np.cos(2 * np.pi * dataframe["date"].dt.hour / 24)
        dataframe["%-dow_sin"]  = np.sin(2 * np.pi * dataframe["date"].dt.dayofweek / 7)
        dataframe["%-dow_cos"]  = np.cos(2 * np.pi * dataframe["date"].dt.dayofweek / 7)
        # Required by FreqAI RL — used to build prices_train/prices_test in the Gym env
        dataframe["%-raw_open"]   = dataframe["open"]
        dataframe["%-raw_high"]   = dataframe["high"]
        dataframe["%-raw_low"]    = dataframe["low"]
        dataframe["%-raw_close"]  = dataframe["close"]
        dataframe["%-raw_volume"] = dataframe["volume"]

        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        # Regression target: 24-candle forward return (used in LightGBM mode)
        # In RL mode this column is ignored — RL learns its own value function
        dataframe["&-s-close"] = dataframe["close"].shift(-24) / dataframe["close"] - 1
        return dataframe

    # ------------------------------------------------------------------
    # Stage 1 — Regime detection
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_regime(
        adx: pd.Series, bb_width: pd.Series, close: pd.Series, sma200: pd.Series
    ) -> pd.Series:
        bb_width_ma = bb_width.rolling(20).mean()
        regime = pd.Series("ranging", index=adx.index, dtype=object)
        trending_mask = adx > 25
        regime[trending_mask] = np.where(
            close[trending_mask] > sma200[trending_mask], "bull-trending", "bear-trending"
        )
        regime[bb_width > bb_width_ma * 1.5] = "high-volatility"
        return regime

    # ------------------------------------------------------------------
    # populate_indicators — Stage 1 + FreqAI (Stage 2) + feature cache
    # ------------------------------------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Stage 2: FreqAI / RL inference
        dataframe = self.freqai.start(dataframe, metadata, self)

        # Stage 1: regime detection indicators
        adx14_df = pta.adx(dataframe["high"], dataframe["low"], dataframe["close"], length=14)
        dataframe["adx14"] = adx14_df["ADX_14"]

        bb20 = pta.bbands(dataframe["close"], length=20, std=2.0)
        dataframe["bb_upper"]    = bb20["BBU_20_2.0"]
        dataframe["bb_lower"]    = bb20["BBL_20_2.0"]
        dataframe["bb_mid"]      = bb20["BBM_20_2.0"]
        bb_range                  = dataframe["bb_upper"] - dataframe["bb_lower"]
        dataframe["bb_width"]    = np.where(dataframe["bb_mid"] > 0, bb_range / dataframe["bb_mid"], 0.0)
        dataframe["bb_position"] = np.where(bb_range > 0, (dataframe["close"] - dataframe["bb_lower"]) / bb_range, 0.5)

        dataframe["rsi14"]  = pta.rsi(dataframe["close"], length=14)
        dataframe["sma200"] = pta.sma(dataframe["close"], length=200)

        dataframe["regime"] = self._classify_regime(
            dataframe["adx14"], dataframe["bb_width"], dataframe["close"], dataframe["sma200"]
        )

        # ── Cache latest features for TradeMemory entry logging ──────────
        if not dataframe.empty:
            last = dataframe.iloc[-1]
            ml_score = float(last.get("&-s-close", 0.0)) if "&-s-close" in dataframe.columns else 0.0
            self._latest_features[metadata["pair"]] = {
                "regime":      str(last.get("regime", "unknown")),
                "rsi14":       float(last.get("rsi14",      50.0)),
                "bb_position": float(last.get("bb_position", 0.5)),
                "adx14":       float(last.get("adx14",       0.0)),
                "bb_width":    float(last.get("bb_width",    0.0)),
                "ml_score":    ml_score,
            }

        return dataframe

    # ------------------------------------------------------------------
    # Stage 3 + 4 — Entry / exit signals
    # ------------------------------------------------------------------

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # FreqAI gate — RL mode: &-s-close holds integer actions (1=Long_enter)
        #               LightGBM mode: &-s-close holds continuous predicted return
        is_rl = "RL" in self.config.get("freqaimodel", "")
        if is_rl:
            conditions.append(dataframe["&-s-close"] == 1)  # Long_enter action
            # In RL mode DI_threshold is deactivated so do_predict is unreliable at startup;
            # trust the PPO action directly — do_predict=1 is kept for LightGBM mode only.
        else:
            conditions.append(dataframe["&-s-close"] > self.buy_min_ml_score.value)
            conditions.append(dataframe["do_predict"] == 1)

        # Only block extreme volatility — PPO already learned regime context from
        # 430 features including ADX, SMA200, BB width. The bear-trending gate was
        # blocking the exact dips the PPO was trained to buy.
        conditions.append(dataframe["regime"] != "high-volatility")

        # Volume sanity check only — RSI/BB gates removed: PPO already incorporates
        # those 430+ features, adding hard thresholds created a structural misalignment
        # where the two gates never fired simultaneously (PPO said buy at RSI 38-48,
        # hard gate required RSI < 32 — they never overlapped).
        conditions.append(dataframe["volume"] > 0)

        dataframe.loc[reduce(lambda a, b: a & b, conditions), "enter_long"] = 1

        # DEBUG — log last candle signal state every candle
        if not dataframe.empty:
            try:
                last = dataframe.iloc[-1]
                action_raw = last.get("&-s-close", float("nan"))
                action_val = float(action_raw) if not pd.isna(action_raw) else -9.0
                do_pred = last.get("do_predict", -1)
                regime_val = last.get("regime", "N/A")
                enter_raw = last.get("enter_long", 0)
                enter = 0 if pd.isna(enter_raw) else int(enter_raw)
                logger.info(
                    "SIGNAL %s | action=%.1f | do_predict=%s | regime=%s | enter_long=%d | close=%.2f",
                    metadata.get("pair", "?"),
                    action_val, do_pred, regime_val, enter,
                    float(last.get("close", 0)),
                )
            except Exception as _dbg_exc:
                logger.warning("SIGNAL debug log failed: %s", _dbg_exc)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(dataframe["bb_position"] > self.sell_bb_pos_min.value)
        conditions.append(dataframe["rsi14"] > self.sell_rsi_min.value)

        is_rl = "RL" in self.config.get("freqaimodel", "")
        if is_rl:
            ml_exit = (dataframe["&-s-close"] == 2) & (dataframe["do_predict"] == 1)
        else:
            ml_exit = (dataframe["&-s-close"] < -0.005) & (dataframe["do_predict"] == 1)

        dataframe.loc[reduce(lambda a, b: a & b, conditions) | ml_exit, "exit_long"] = 1
        return dataframe

    def custom_stoploss(self, current_time, current_rate: float,
                        current_profit: float, trade, **kwargs) -> float:
        if current_profit > 0.01:
            return -0.03     # tighten to 3% trail once in profit
        return self.stoploss  # hard -3% (stoploss = -0.03)

    # ------------------------------------------------------------------
    # Self-improvement hooks — feed TradeMemory
    # ------------------------------------------------------------------

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """Cache entry conditions so confirm_trade_exit() can log them."""
        if _trade_memory is not None:
            features = self._latest_features.get(pair, {})
            _trade_memory.cache_entry(pair, current_time, features)
            logger.debug(
                "TradeMemory: cached entry features for %s | regime=%s | rsi=%.1f | bb_pos=%.2f",
                pair,
                features.get("regime", "?"),
                features.get("rsi14", 0),
                features.get("bb_position", 0),
            )
        return True  # always allow entry

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time,
        **kwargs,
    ) -> bool:
        """
        Log the completed trade to TradeMemory.
        Called right before Freqtrade places the exit order — profit is known.
        This is the event that drives self-improvement on the next retrain.
        """
        if _trade_memory is not None:
            try:
                profit_pct = trade.calc_profit_ratio(rate)
                profit_abs = trade.calc_profit(rate)

                # Retrieve the entry features we cached at open
                open_dt = trade.open_date
                if open_dt.tzinfo is None:
                    open_dt = open_dt.replace(tzinfo=timezone.utc)
                entry_features = _trade_memory.pop_entry(pair, open_dt)

                # If cache miss (e.g. bot restarted), use whatever we have now
                if not entry_features:
                    entry_features = self._latest_features.get(pair, {})

                close_dt = current_time
                if hasattr(close_dt, "tzinfo") and close_dt.tzinfo is None:
                    close_dt = close_dt.replace(tzinfo=timezone.utc)

                _trade_memory.store_trade(
                    trade_id=str(trade.id),
                    pair=pair,
                    entry_time=open_dt,
                    exit_time=close_dt,
                    entry_price=float(trade.open_rate),
                    exit_price=float(rate),
                    profit_pct=float(profit_pct),
                    profit_abs=float(profit_abs),
                    exit_reason=exit_reason,
                    entry_features=entry_features,
                )

                sign = "+" if profit_pct >= 0 else ""
                logger.info(
                    "TradeMemory: logged trade #%s %s | %s%.2f%% | reason=%s | regime=%s",
                    trade.id, pair,
                    sign, profit_pct * 100,
                    exit_reason,
                    entry_features.get("regime", "?"),
                )

            except Exception as exc:
                # Never block an exit due to logging error
                logger.warning("TradeMemory logging error (non-fatal): %s", exc)

        return True  # always allow exit
