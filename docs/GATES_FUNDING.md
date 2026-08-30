# Funding-Harvest Strategy — Pre-Registered Gates

**Registered before any backtest was run.** Thresholds may not be relaxed after
seeing results. If a threshold proves miscalibrated, that must be stated
explicitly and re-validated against a *harder* independent test.

## Strategy under test

Delta-neutral cash-and-carry:
- **Long** N units spot
- **Short** N units perpetual future (same asset)
- Price exposure nets to approximately zero
- Income = funding payments received by the short leg when funding is positive
- Costs = taker/maker fees on all four executions (2 legs x open/close)
- Risks = negative funding, basis divergence, perp-leg liquidation, exchange risk

## Gates

| Gate | Test | Pass threshold | On failure |
|---|---|---|---|
| **F1** | Gross carry positive over full history | > 0% annualised, > 70% of periods positive | STOP |
| **F2** | Net of all fees and financing | **> 5% annualised on TOTAL deployed capital** | STOP — edge consumed by costs |
| **F3** | Walk-forward across regimes | positive in **>= 75% of quarters** | Reduce universe to pairs that pass; STOP if none |
| **F4** | Liquidation safety | **zero liquidations** at chosen leverage over full history | Reduce leverage and re-test |
| **F5** | Drawdown | **max drawdown < 10%** on total capital | Reduce leverage / add pairs |
| **F6** | Operational stability | **30 consecutive days** dry-run, zero unplanned outages | Fix reliability before any live consideration |

## Explicit accounting rules

1. **Return is measured on total deployed capital**, not on notional. Margin
   posted for the perp leg is capital that cannot be used elsewhere. Quoting
   funding yield on notional alone would overstate returns by the leverage
   factor.
2. **Fees applied to every execution**: entry spot, entry perp, exit spot,
   exit perp. Default 0.05% maker / 0.10% taker (Binance VIP0).
3. **Negative funding periods are paid, not skipped.** No look-ahead filtering
   of "good" periods.
4. **Liquidation is terminal for that position** — modelled as a forced close
   at the liquidation price with fees, not as a paper loss.
5. **Basis risk is modelled**: entry and exit occur at real spot and mark
   prices, so spot-perp convergence/divergence affects P&L.

## Leverage definition

- `S` = notional of each leg
- `M` = margin posted for the perp short
- Total capital deployed `C = S + M`
- Leverage `L = S / M`
- Liquidation of the short occurs at roughly a `+1/L` upward price move
  (isolated margin, ignoring maintenance-margin buffer)

At `L = 1` (M = S) capital deployed is 2x notional, so effective yield on
capital is roughly half the headline funding yield. This is the honest number.

---

# RESULTS (run 2026-08-24)

Data: 2.66-2.82 years of overlap (funding history vs local spot history),
6 pairs, fees 0.10% taker on all four executions.

## F2 / F4 / F5 — cross margin (Binance Portfolio Margin)

| Pair | Leverage 1x CAGR | maxDD | Leverage 2x CAGR | maxDD | Liquidations |
|---|---|---|---|---|---|
| BTCUSDT | **+6.16%** | -0.33% | **+8.08%** | -0.41% | none |
| ETHUSDT | +4.74% | -0.25% | **+6.24%** | -0.32% | none |
| XRPUSDT | **+6.50%** | -1.30% | **+8.51%** | -1.64% | none |
| DOGEUSDT | **+9.46%** | -0.41% | **+12.30%** | -0.52% | none |
| SOLUSDT | **+10.66%** | -3.16% | **+13.83%** | -3.87% | none |
| BNBUSDT | -0.91% | -6.98% | -1.21% | -9.27% | none |

## F3 — walk-forward by quarter (cross, 1x)

| Pair | Quarters positive | Verdict |
|---|---|---|
| DOGEUSDT | 11/12 (91.7%) | PASS |
| BTCUSDT | 9/11 (81.8%) | PASS |
| ETHUSDT | 9/11 (81.8%) | PASS |
| XRPUSDT | 9/12 (75.0%) | PASS |
| SOLUSDT | 7/12 (58.3%) | FAIL |
| BNBUSDT | 4/12 (33.3%) | FAIL |

## Verdict

**Pairs passing ALL of F2/F3/F4/F5: BTC, XRP, DOGE at 1x; plus ETH at 2x.**
SOL fails F3 (inconsistent), BNB fails F2 and F3 (funding is structurally
negative — persistent short bias in that market).

## Margin mode is not optional

| Mode | Outcome |
|---|---|
| **cross** (portfolio margin) | zero liquidations, drawdown < 3.2% |
| isolated | **liquidated on 5-6 of 6 pairs** — a +100% spot move liquidates the short even though the spot leg gains the offsetting amount |
| naive rebalance | **liquidated on 4 of 6** — funding the short's margin by selling the spot leg progressively unwinds the hedge, converting it into a naked short. Death spiral in a bull market. |

**Portfolio Margin is a hard requirement.** Without it this strategy is not
merely worse, it is ruinous.

## Caveats that materially limit confidence

1. **Sharpe of ~12 is an artifact.** The model assumes a perfectly matched
   hedge that never needs re-balancing for drift. Real cash-and-carry desks
   report Sharpe 2-4. Treat the RETURN estimate as credible and the SHARPE
   as meaningless.
2. **Not modelled:** slippage, hedge-drift rebalancing costs, discrete
   contract sizes, funding-rate changes mid-period, API/exchange downtime,
   withdrawal freezes, exchange insolvency.
3. **Failures are correlated.** The negative quarters cluster in the SAME
   periods across BTC/ETH/XRP. Funding regime is market-wide, so holding
   multiple pairs does NOT diversify this risk.
4. **Only ~2.8 years** of overlapping history, one broad market cycle.
5. **F6 (30-day operational stability) is UNTESTED** and current
   infrastructure has not survived a week without an outage.
