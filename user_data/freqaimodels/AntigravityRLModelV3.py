"""
AntigravityRLModel — Self-improving PPO agent for the Antigravity trading bot.

Architecture:
  • Extends Freqtrade's built-in ReinforcementLearner (Stable-Baselines3 PPO)
  • Inner MyRLEnv overrides calculate_reward() with trade-memory feedback
  • Every retraining cycle (every 8 h live):
      1. Reads TradeMemory SQLite for regime win-rates & recent loss rate
      2. Recomputes reward weights from real trade history
      3. Retrains PPO on OHLCV simulation with updated rewards
      4. New policy is deployed — bot is literally smarter than last cycle

Self-improvement loop:
  bad regime  → low win-rate in TradeMemory
              → higher entry penalty in MyRLEnv.calculate_reward()
              → PPO learns to avoid that regime
              → fewer losses in next live session
              → win-rate in TradeMemory improves
              → penalty relaxes → bot cautiously re-enters when conditions improve
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TradeMemory import — graceful fallback if file is missing
# ---------------------------------------------------------------------------
_MEM_PATH = Path("V:/Antigravity/Trade tool/bot/user_data/freqaimodels/trade_memory.py")
sys.path.insert(0, str(_MEM_PATH.parent))
try:
    from trade_memory import TradeMemory
    _MEMORY = TradeMemory()
except Exception as _e:
    logger.warning("TradeMemory unavailable (%s) — reward shaping disabled.", _e)
    _MEMORY = None

# ---------------------------------------------------------------------------
# Freqtrade RL imports (Freqtrade 2025.6 API)
# ---------------------------------------------------------------------------
try:
    from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
    from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv
    from freqtrade.freqai.RL.BaseEnvironment import Positions
except ImportError as exc:
    raise ImportError(
        "Freqtrade RL dependencies missing. "
        "Run: pip install 'freqtrade[freqai-rl]' stable-baselines3 gymnasium"
    ) from exc


# ===========================================================================
# Custom RL Model
# ===========================================================================

class AntigravityRLModelV3(ReinforcementLearner):
    """
    Self-improving PPO trading model for the Antigravity bot.

    How it improves over time
    ─────────────────────────
    1. Live (or dry-run) trades are logged to TradeMemory (SQLite).
    2. Every `live_retrain_hours` (default 8 h), Freqtrade calls fit().
    3. fit() reads TradeMemory → updates model_reward_parameters in rl_config:
         • regime_win_rates  — per-regime success rates
         • recent_loss_rate  — how bad the last 7 days were
    4. These weights are injected into MyRLEnv.calculate_reward() via rl_config.
    5. PPO retrains on OHLCV simulation with updated rewards.
    6. Bad entry conditions get larger penalties → model avoids them.
    7. Good entry conditions stay rewarded → model reinforces them.
    """

    class MyRLEnv(Base5ActionRLEnv):
        """
        Custom Gym environment with TradeMemory-informed reward shaping.

        Reward design:
        ┌───────────────────────────────────────────────────────────┐
        │ Profitable exit (> fee threshold)   → large positive      │
        │ Stop-loss exit                       → large negative      │
        │ Entry in bad regime / overbought     → entry penalty       │
        │ Holding too long without progress    → small decay penalty │
        │ Good early exit (cut loss)           → small positive      │
        └───────────────────────────────────────────────────────────┘
        """

        # Minimum profit to consider an exit truly "successful" (above Kraken fees)
        # Raised from 0.005: with the old 0.5% bar the agent was rewarded for
        # penny scalps that lost money after the 3% stoploss risk was priced in
        # (0.42:1 payoff, negative expectancy). Exits below 1.5% now count as
        # churn, matching the ROI ladder's 2.5% floor.
        _FEE_THRESHOLD = 0.015   # 1.5%
        _LOSS_THRESHOLD = -0.03  # -3% → encourage early exit before hard stop

        # Reward terms are heterogeneous in scale (invalid=-2, idle=-0.0005,
        # profitable exit up to ~pnl*750). Nothing normalises this for PPO, so
        # a rare large outlier (e.g. a gap-down stop-out) can dominate a whole
        # batch. Clip to a fixed band so the value function sees a consistent
        # scale regardless of which branch fired.
        # Was 5.0 combined with a *5.0 amplifier below, which saturated at
        # 0.67% profit - a 1% win and an 8% win produced an identical reward,
        # destroying any gradient toward larger profits. Amplifier removed and
        # band widened so the full realistic P&L range stays distinguishable.
        _REWARD_CLIP = 10.0

        # Flat per-exit cost, roughly fee-equivalent (0.2% round trip -> ~0.3
        # reward units at the 1.5x multiplier). Makes sub-fee scalps net
        # negative WITHOUT ever letting a loss outscore a gain.
        _EXIT_COST = 0.25

        def calculate_reward(self, action: int) -> float:  # noqa: C901
            return float(np.clip(self._calculate_reward_raw(action), -self._REWARD_CLIP, self._REWARD_CLIP))

        def _calculate_reward_raw(self, action: int) -> float:  # noqa: C901
            if not self._is_valid(action):
                self.tensorboard_log("invalid", category="actions")
                return -2

            reward_params = self.rl_config.get("model_reward_parameters", {})
            regime_win_rates: dict = reward_params.get("regime_win_rates", {})
            recent_loss_rate: float = float(reward_params.get("recent_loss_rate", 0.0))

            pnl = self.get_unrealized_profit()
            factor = 100.0
            max_trade_duration = self.rl_config.get("max_trade_duration_candles", 300)

            # ── End of episode ───────────────────────────────────────────────
            if self._current_tick == self._end_tick:
                return -1.0 if pnl < 0 else 0.0

            # ── Currently LONG ───────────────────────────────────────────────
            if self._position == Positions.Long:
                trade_duration = (
                    self._current_tick - self._last_trade_tick
                    if self._last_trade_tick is not None else 0
                )

                if action == Actions.Long_exit.value:
                    # MONOTONIC in pnl. The previous branch structure paid +0.3
                    # for ANY small loss but -0.1 for a small win, so a -2.0%
                    # loss outscored a +1.0% win by 0.4. The agent farmed that:
                    # enter -> exit at a small loss -> collect +0.3, repeatedly.
                    # Backtest showed reward climbing to +19 while the account
                    # lost 23%. One formula now, strictly increasing in pnl, so
                    # no outcome can ever be beaten by a worse one.
                    if pnl > 0:
                        multiplier = 1.5 if trade_duration <= max_trade_duration else 0.5
                    else:
                        multiplier = 2.0  # losses weighted harder than gains
                    return float(pnl * factor * multiplier - self._EXIT_COST)

                if pnl < self._LOSS_THRESHOLD:
                    return -0.4  # push agent to exit deeply losing position

                if trade_duration > 0 and trade_duration % 12 == 0:
                    return -0.05  # small time-decay every 12 candles

                return 0.0  # neutral hold

            # ── Currently NEUTRAL ────────────────────────────────────────────
            if self._position == Positions.Neutral:
                if action == Actions.Long_enter.value:
                    regime = self._infer_regime()
                    win_rate = regime_win_rates.get(regime, 0.5)
                    regime_penalty = max(0.0, (0.5 - win_rate) * 2.0)
                    global_penalty = recent_loss_rate * 0.5
                    entry_penalty = regime_penalty + global_penalty

                    entry_reward = self._mean_reversion_entry_score()
                    net = entry_reward - entry_penalty
                    return float(np.clip(net, -1.0, 1.0))

                if action == Actions.Neutral.value:
                    # Was -1/tick: over a ~590-candle episode that punished
                    # correctly waiting for a setup by up to -590, dwarfing the
                    # best possible trade reward (~+18.75). The reward-maximising
                    # policy was therefore "always be in a trade" regardless of
                    # quality. Use a tiny per-tick cost instead — enough to still
                    # prefer trading over permanent inaction, far too small to
                    # outweigh waiting for a genuinely good entry.
                    return -float(reward_params.get("idle_penalty", 0.0005))

                return 0.0

            return 0.0

        # ── Helpers ──────────────────────────────────────────────────────────

        def _infer_regime(self) -> str:
            """Infer market regime from feature columns at current tick."""
            try:
                idx = self._current_tick
                if idx >= len(self.signal_features):
                    return "unknown"

                adx = 0.0
                bb_w = 0.0
                bb_w_ma = 0.0

                for col in self.signal_features.columns:
                    col_lower = col.lower()
                    if "adx" in col_lower and adx == 0.0:
                        val = self.signal_features[col].iloc[idx]
                        adx = float(val) if not np.isnan(val) else 0.0
                    if "bb_width" in col_lower:
                        series = self.signal_features[col]
                        bb_w = float(series.iloc[idx]) if not np.isnan(series.iloc[idx]) else 0.0
                        ma_val = series.rolling(20).mean().iloc[idx]
                        bb_w_ma = float(ma_val) if not np.isnan(ma_val) else bb_w

                if bb_w_ma > 0 and bb_w > bb_w_ma * 1.5:
                    return "high-volatility"
                if adx > 25:
                    return "bull-trending"
                return "ranging"
            except Exception:
                return "unknown"

        def _mean_reversion_entry_score(self) -> float:
            """Return 0.0–0.3 based on how well entry conditions align."""
            try:
                idx = self._current_tick
                if idx >= len(self.signal_features):
                    return 0.0

                rsi = 50.0
                bb_pos = 0.5

                for col in self.signal_features.columns:
                    col_lower = col.lower()
                    if "rsi" in col_lower:
                        val = self.signal_features[col].iloc[idx]
                        if not np.isnan(val):
                            rsi = float(val)
                        break

                for col in self.signal_features.columns:
                    col_lower = col.lower()
                    if "bb_position" in col_lower:
                        val = self.signal_features[col].iloc[idx]
                        if not np.isnan(val):
                            bb_pos = float(val)
                        break

                rsi_score = max(0.0, (40.0 - rsi) / 40.0)
                bb_score = max(0.0, (0.3 - bb_pos) / 0.3)
                return float(np.clip((rsi_score + bb_score) / 2.0 * 0.3, 0.0, 0.3))
            except Exception:
                return 0.0

        # ── Spot-only: mask out Short actions ───────────────────────────────
        # Base5ActionRLEnv never checks can_short itself (only Base3ActionRLEnv
        # does), so on a spot-only bot the agent was free to "learn" a
        # Short_enter/Short_exit policy that can never execute live — wasted
        # exploration budget on 2 of 5 actions. Disable them at both gates.

        def _is_valid(self, action: int) -> bool:
            if action in (Actions.Short_enter.value, Actions.Short_exit.value):
                return False
            return super()._is_valid(action)

        def is_tradesignal(self, action: int) -> bool:
            if action in (Actions.Short_enter.value, Actions.Short_exit.value):
                return False
            return super().is_tradesignal(action)

        # ── Distinguish truncation from termination ─────────────────────────
        # Base5ActionRLEnv.step() marks both "ran out of candles" and "blew
        # through max_training_drawdown_pct" as the same terminal signal
        # (truncated hardcoded False). PPO's GAE bootstrapping treats every
        # episode end as a true terminal state (value=0) either way, so a
        # policy that simply reached the end of the data got the same value
        # target as one that genuinely failed. Split them properly.
        def step(self, action: int):
            self._done = False
            self._current_tick += 1

            end_of_data = self._current_tick == self._end_tick
            if end_of_data:
                self._done = True

            self._update_unrealized_total_profit()
            step_reward = self.calculate_reward(action)
            self.total_reward += step_reward
            self.tensorboard_log(self.actions._member_names_[action], category="actions")

            trade_type = None
            if self.is_tradesignal(action):
                if action == Actions.Neutral.value:
                    self._position = Positions.Neutral
                    trade_type = "neutral"
                    self._last_trade_tick = None
                elif action == Actions.Long_enter.value:
                    self._position = Positions.Long
                    trade_type = "enter_long"
                    self._last_trade_tick = self._current_tick
                elif action == Actions.Long_exit.value:
                    self._update_total_profit()
                    self._position = Positions.Neutral
                    trade_type = "exit_long"
                    self._last_trade_tick = None

                if trade_type is not None:
                    self.trade_history.append({
                        "price": self.current_price(),
                        "index": self._current_tick,
                        "type": trade_type,
                        "profit": self.get_unrealized_profit(),
                    })

            drawdown_breach = (
                self._total_profit < self.max_drawdown
                or self._total_unrealized_profit < self.max_drawdown
            )
            if drawdown_breach:
                self._done = True

            terminated = drawdown_breach
            truncated = end_of_data and not drawdown_breach

            self._position_history.append(self._position)

            info = dict(
                tick=self._current_tick,
                action=action,
                total_reward=self.total_reward,
                total_profit=self._total_profit,
                position=self._position.value,
                trade_duration=self.get_trade_duration(),
                current_profit_pct=self.get_unrealized_profit(),
            )

            observation = self._get_observation()
            self._update_history(info)

            return observation, step_reward, terminated, truncated, info

    # ── Model-level methods ──────────────────────────────────────────────────

    # PPO parameters accepted by SB3 PPO constructor (keeps config merging safe)
    _PPO_PARAMS = {"learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
                   "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
                   "device", "verbose", "seed", "target_kl"}

    def fit(self, data_dictionary: Dict[str, Any], dk: Any, **kwargs) -> Any:
        """Inject TradeMemory reward weights, then train PPO with clean params."""
        self._inject_memory_reward_weights()

        # Strip LightGBM-only params (n_estimators etc.) before passing to PPO.
        # When config_backtest.json + config_rl.json are deep-merged, LightGBM
        # params survive in model_training_parameters and crash PPO's constructor.
        original = self.freqai_info.get("model_training_parameters", {})
        ppo_only = {k: v for k, v in original.items() if k in self._PPO_PARAMS}
        # Force SB3/PyTorch device name — LightGBM uses "gpu", PyTorch uses "cuda"
        ppo_only["device"] = "cuda"
        ppo_only.setdefault("verbose", 0)
        self.freqai_info["model_training_parameters"] = ppo_only
        try:
            result = super().fit(data_dictionary, dk, **kwargs)
        finally:
            self.freqai_info["model_training_parameters"] = original
        return result

    def _inject_memory_reward_weights(self) -> None:
        """Read TradeMemory and update rl_config reward params for next training."""
        if _MEMORY is None:
            logger.debug("TradeMemory not available — using default reward params.")
            return

        try:
            regime_stats = _MEMORY.get_regime_stats()
            recent_loss = _MEMORY.recent_loss_penalty(days=7)
            regime_win_rates = {r: s["win_rate"] for r, s in regime_stats.items()}

            # Update directly in freqai_info so BaseEnvironment picks it up via config
            rl_cfg = self.freqai_info.get("rl_config", {})
            reward_params = rl_cfg.setdefault("model_reward_parameters", {})
            reward_params["regime_win_rates"] = regime_win_rates
            reward_params["recent_loss_rate"] = recent_loss

            summary = _MEMORY.summary()
            logger.info(
                "TradeMemory -> reward update | trades=%s | win_rate=%s | "
                "recent_loss_rate=%.2f | regimes=%s",
                summary.get("total_trades", 0),
                summary.get("win_rate", "N/A"),
                recent_loss,
                list(regime_win_rates.keys()),
            )
            logger.info("\n%s", _MEMORY.regime_report())
        except Exception as exc:
            logger.warning("Failed to read TradeMemory for reward shaping: %s", exc)
