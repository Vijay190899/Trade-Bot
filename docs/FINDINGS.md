# Detailed Research Findings

Full evidence log for the validation of the Qoherenz RL trading system.
Every number here is reproducible with the scripts in [`../research/`](../research/).

---

## Part 1 — Does the RL layer learn?

### 1.1 Reward trend across 286 live retrain cycles

Extracted from TensorBoard event files written by every live retrain.

| Period | Cycles | `ep_rew_mean` slope | `eval/mean_reward` slope |
|---|---|---|---|
| Pre-fix | 255 | -0.161 | -0.0023 |
| Post-fix | 31 | +1.156 | -0.305 |

Significance test on the post-fix window:

| Metric | Pearson r | p-value | Verdict |
|---|---|---|---|
| `ep_rew_mean` | +0.055 | 0.765 | not significant |
| `eval/mean_reward` | -0.040 | 0.829 | not significant |

Standard deviation of reward across cycles was **236**, dwarfing any drift.
**Conclusion: no learning signal.**

Reproduce: `research/rl_reward_trend.py`

### 1.2 Do the model's predictions predict anything?

3,576 out-of-sample walk-forward predictions merged with realised forward returns.

Naive test (overlapping windows) appeared to show a significant *negative* edge at 48h:

| Horizon | Edge | p |
|---|---|---|
| 48h | -1.275 pp | 0.000 |

**This was an artifact.** Consecutive hourly 48h-forward returns share 47 of 48 hours, so observations are not independent. Correcting by sampling non-overlapping windows across every phase offset:

| Horizon | Mean edge | Mean t | Median p | Offsets significant |
|---|---|---|---|---|
| 24h | -0.140 pp | -0.25 | 0.738 | 0 / 18 |
| 48h | -1.007 pp | -1.17 | 0.234 | 0 / 7 |

**Conclusion: zero predictive power, in either direction.**

Reproduce: `research/edge_test_predictions.py`, `research/edge_test_nonoverlap.py`

### 1.3 What the model actually learned

| | Prior 24h return |
|---|---|
| When model signals buy | **-0.715%** |
| Otherwise | +0.070% |

t = -5.54, p < 0.001 — robust (point-in-time, no overlap problem).

The model learned exactly one behaviour: **buy dips**. Over the test window the market fell 3.45%, so it spent five months catching falling knives. This explains the 70% win rate paired with a negative expectancy — dips bounce slightly (small wins) until they do not (large losses).

---

## Part 2 — Engineering defects found

### 2.1 Reward inversion (critical)

Original `calculate_reward()` branch structure:

```python
if pnl > FEE_THRESHOLD:            return pnl * 100 * multiplier
if LOSS_THRESHOLD < pnl < 0:       return 0.3     # "cut loss early"
if 0 <= pnl <= FEE_THRESHOLD:      return -0.1    # "churn penalty"
return pnl * 100 * 2.0                            # stopped out
```

Evaluated across the PnL range:

| PnL | Reward | Note |
|---|---|---|
| +1.0% | **-0.10** | small win *penalised* |
| +0.1% | **-0.10** | |
| -0.5% | **+0.30** | small loss *rewarded* |
| -2.0% | **+0.30** | large loss *rewarded* |

A -2.0% loss outscored a +1.0% win by 0.40. The agent farmed this: enter, exit at a small loss, collect +0.30, repeat.

Observed consequence in backtest (V2 config): **reward climbed to +19.01 while the account lost 22.96%.** Reward and PnL became anti-correlated — the definitive signature of a misaligned objective.

**Fix (V3):** one monotonic formula, strictly increasing in PnL.

```python
multiplier = (1.5 if fast else 0.5) if pnl > 0 else 2.0
return pnl * 100 * multiplier - EXIT_COST
```

| PnL | V2 | V3 |
|---|---|---|
| +2.0% | 3.00 | 2.75 |
| +1.0% | **-0.10** | 1.25 |
| -0.5% | **+0.30** | -1.25 |
| -2.0% | **+0.30** | -4.25 |

Break-even lands at 0.17% (fee level). No outcome can be beaten by a worse one.

Verification that the fix worked: V3 reward = **-16.29** while losing 5.39% — reward and PnL now move together.

### 2.2 Dead action space

`Base5ActionRLEnv` never checks `can_short`, unlike `Base3ActionRLEnv`. On a spot-only bot the agent was free to learn a short-selling policy it could never execute.

Overriding `_is_valid()` alone does **not** fix this — it only penalises selection (-2 reward) rather than preventing it, because `action_masks()` is ignored unless `model_type` is `MaskablePPO`.

| Config | Short actions | Invalid actions |
|---|---|---|
| Baseline (PPO) | 18.0% | 28.0% |
| `_is_valid` override only (PPO) | 15.9% | 22.8% |
| **MaskablePPO** | **0.0%** | **0.0%** |

### 2.3 Curse of dimensionality

Feature count reconstruction (matches the logged 433 exactly):

```
 9 indicators x 3 periods            = 27
 + 8 basic indicators                = 35   per (timeframe, pair)
 x 2 timeframes x 2 pairs            = 140
 x (1 + 2 shifted candles)           = 420
 + 13 standard features              = 433
```

Sample count: 30 days x 24 candles = 720, minus 24 label lookahead, minus 15% test split = **590**.

| | Before | After |
|---|---|---|
| Features | 433 | 153 |
| Samples | 590 | 1,813 |
| **Ratio** | **1.36 : 1** | **11.9 : 1** |

Rule of thumb requires 10-50:1. Fixed by dropping `include_corr_pairlist` (BTC/USDT correlates >0.99 with the traded asset — pure redundancy) and halving shifted candles, plus raising `train_period_days` 30 → 90.

Note: `DI_threshold` and `use_SVM_to_remove_outliers` in the config are **silently disabled by Freqtrade for all RL models** (`BaseReinforcementLearningModel.unset_outlier_removal`). They were never doing anything.

### 2.4 Terminated vs truncated conflation

`Base5ActionRLEnv.step()` hardcodes `truncated = False`, so "ran out of candles" and "breached max drawdown" are both reported as terminal. PPO's GAE bootstrapping then assigns value 0 to both, mixing genuine failure signal with an artifact of dataset length. Fixed in V3 by splitting the two.

---

## Part 3 — Exit geometry

Measured over 47 managed trades (excluding one outlier, see 3.1):

| | Value |
|---|---|
| Win rate | 70.2% |
| Average win | +0.82% (+0.37 USDT) |
| Average loss | -1.94% (-0.87 USDT) |
| Payoff ratio | 0.42 : 1 |
| Break-even win rate | **70.3%** |
| Expectancy | **-0.0008 USDT/trade** |

The ROI ladder decayed to 0.5% at 12h while the stoploss risked 3%. Worse, ROI fired *below* the trailing stop's 2.5% activation threshold, making `trailing_stop` dead code that never once executed.

### 3.1 The outlier that hid the problem

Headline P&L showed +12.26 USDT. Decomposed:

| | |
|---|---|
| Trades 1-47 | **-0.04 USDT** |
| Trade 48 alone | **+12.30 USDT** (+27.4%) |

Trade 48 opened 2026-08-07, closed 2026-08-22. The log shows a **10-day gap (Aug 8 → Aug 18)** — the bot was crashed for most of the hold. ETH moved 1909.28 → 2436.87, i.e. **+27.63% buy-and-hold**; the trade returned **+27.38%**.

The entire profit was an artifact of downtime during a rally, not a decision. Over its full lifetime the bot returned +12.26% while buy-and-hold returned +20.79% — **underperforming the market by 8.5 points.**

---

## Part 4 — Where signal actually lives

ETH/USDT 1h, 23,175 candles (966 days).

### 4.1 Direction vs volatility

| Lag | Direction autocorr | p | Volatility autocorr | p |
|---|---|---|---|---|
| 1 | +0.0052 | 0.427 | **+0.2234** | ~0 |
| 6 | -0.0006 | 0.923 | **+0.1274** | ~0 |
| 24 | -0.0172 | 0.009 | **+0.1224** | ~0 |
| 168 | -0.0031 | 0.635 | **+0.1106** | ~0 |

Out-of-sample AR model R-squared:

| Target | R-squared |
|---|---|
| Next-hour return | **-0.0066** |
| Next-hour absolute return | **+0.108** |

Volatility clustering is strong and persistent. Direction is not predictable.

### 4.2 Volatility targeting does not create alpha

| Strategy | Total | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|---|
| Buy and hold | +6.35% | +2.35% | 66.1% | +0.04 | -69.1% |
| Vol-targeted | -13.85% | -5.48% | 54.0% | -0.10 | -63.8% |

Sharpe **fell** by 0.14. Vol targeting improves risk-adjusted returns only when there is positive drift to harvest; ETH had a 2.35% CAGR over 2.65 years. It did reduce drawdown and volatility, which is valuable for risk control, but it is not an alpha source.

### 4.3 The universe mattered more than the algorithm

Over the identical 2.65-year window:

| Asset | Total return | CAGR |
|---|---|---|
| ETH | +5.86% | **+2.17%** |
| BTC | +81.06% | **+25.16%** |

The bot traded the one asset with no drift while the asset in the same data folder compounded 25% a year. No amount of model sophistication recovers a 23-point CAGR gap caused by universe selection.

---

## Part 5 — Gate results, 29-pair universe

Panel: 29 pairs, 1,331 daily bars (2023-01-01 → 2026-08-23), 36,714 observations.
Features: momentum (5/10/20/60d), short-term reversal (2/3d), 20d volatility, volume change, distance from 60d high. All cross-sectionally z-scored, strictly causal.

### Gate 1 — statistical

| Metric | Value | Threshold | Result |
|---|---|---|---|
| OOS R-squared | +0.00277 | > 0.02 | **FAIL** |
| Information Coefficient | +0.0636 | — | p = 2.19e-11 |

The R-squared threshold was **miscalibrated**. Returns are ~97% noise; R-squared of 0.02 for cross-sectional return prediction would be world-class. Production equity strategies routinely operate on IC of 0.02-0.05, and this is 0.064.

Rather than relax the threshold post hoc (which defeats the purpose of pre-registration), the decision was escalated to a **harder, independent economic test.**

### Gate 1b — economic

Out-of-sample, 76 non-overlapping 5-day rebalances, 0.1% fee per side.

| Strategy | Total | CAGR | Sharpe | Max DD |
|---|---|---|---|---|
| top-3 long/short | -32.42% | -31.36% | -0.54 | -47.9% |
| top-5 long/short | -23.27% | -22.46% | -0.63 | -38.4% |
| **top-8 long/short** | **-18.86%** | -18.19% | -0.78 | -31.3% |
| top-8 long-only | -60.98% | -59.50% | -1.51 | -68.6% |
| *equal-weight (29)* | *-60.89%* | *-59.42%* | | |
| *BTC buy and hold* | *-39.74%* | *-38.52%* | | |

Long/short beat both benchmarks by a wide margin, but **every configuration lost money.**

### Gate 2 — walk-forward across regimes

Expanding-window training, 8-long / 8-short, 0.1% fee per side.

| Fold | Test period | Strategy | Equal-wt | BTC | Beat both |
|---|---|---|---|---|---|
| 1 | 2024-07 → 11 | -4.19% | **+62.00%** | +42.78% | no |
| 2 | 2024-11 → 2025-03 | -2.64% | -40.88% | -15.52% | yes |
| 3 | 2025-03 → 08 | -5.79% | **+19.60%** | +39.13% | no |
| 4 | 2025-08 → 12 | -9.04% | -29.12% | -18.24% | yes |
| 5 | 2025-12 → 2026-04 | -7.12% | -29.21% | -16.17% | yes |
| 6 | 2026-04 → 08 | -0.80% | -15.34% | -5.08% | yes |

**4/6 folds beat benchmarks — threshold was 6/8. FAIL.**

Critically: the strategy **lost money in all six folds.** It only "wins" when markets fall hard, and badly underperforms in every rally. That is a chronic loser that loses less than a collapsing benchmark, not a profitable strategy.

---

## Overall conclusion

The signal is statistically real (IC 0.064, p = 2e-11) but too weak to survive 0.2% round-trip transaction costs. The RL machinery — masking, dimensionality, reward monotonicity, training budget — was defective in three genuine ways, all now fixed and verified. None of the fixes produced profit, because there was no exploitable edge underneath.

**The pipeline stopped the project before any capital was deployed. That is the system working as designed.**

### What would need to change

1. **Different input data.** Classical TA indicators on liquid pairs are the most-arbitraged signals in existence. Order-flow imbalance, funding rates, basis, or on-chain flows carry structural premia that TA does not.
2. **Different market structure.** Long-only spot on a single asset cannot profit in a downtrend. Shorting, market-neutral construction, or genuine cash-timing (which requires directional edge) are the only routes to "profitable in all market conditions."
3. **Lower cost base.** At 0.2% round trip, an IC of 0.064 is consumed by fees. Maker rebates, longer holding periods, or a venue with lower fees change the arithmetic materially.
