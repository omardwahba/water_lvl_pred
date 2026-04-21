# Experiment Observations Log
**Project:** Water Level Prediction — Online Learning Recovery After Leak Fix
**Updated:** 2026-04-21

---

## Background

The original `main` branch had a **data leakage bug** in `RollingNormTimeSeriesDataset`:
future `y_raw` values were used to compute the H_bar normalization range, giving the model
a hidden hint about the magnitude of what it was predicting.

After the fix (`after_leak_fix` branch), results collapsed. This log tracks the recovery effort.

---

## Leak (Pre-Fix) Baseline — DO NOT USE AS TARGET

| Metric | Value |
|--------|-------|
| RMSE   | 10.20 |
| MAE    | 2.50  |
| R²     | 0.32  |

> These numbers were **inflated by the leak**. Beating them is not a valid goal.
> The real signal is whether post-fix results are physically reasonable.

---

## Phase 1 — Normalization Experiments

**Setup:** Adam (lr=1e-3, defaults), MLP, batch_size=1, lookback=48h, horizon=24h

### Why normalization matters here
After the leak fix, `y_scaled = (y_future - h_bar_min_lookback) / h_bar_range_lookback`.
If the next 24h contains a flood spike that exceeds the 48h lookback max,
`y_scaled` can reach values like 10, 50, or higher → MSE loss explodes → model destroyed.

### Results

| Method | RMSE | MAE | R² | Notes |
|---|---|---|---|---|
| **RollingNorm + clip** | **5.0762** | **1.4829** | **0.7961** | **WINNER** |
| Percentile (5th–95th) | 6.1968 | 1.8588 | 0.6936 | Decent, but range still too narrow for floods |
| ZScore | 306.82 | 83.98 | −711.22 | **FAILED — catastrophic** |

### What failed and why

**ZScore (no clip) — catastrophic failure:**
- Theory said z-score would bound `y_scaled` to ~[−3, 3] but this only holds if the future
  distribution matches the lookback.
- During calm periods, lookback std is tiny. A future spike → `y_scaled = spike / tiny_std` → still explodes.
- Same root problem as raw min-max without clip: unbounded `y_scaled`.
- Fix: add `torch.clamp(y_scaled, -5, 5)` to `ZScoreNormTimeSeriesDataset` (not yet tested).

**Percentile — acceptable but not optimal:**
- Using 5th–95th quantiles of the lookback makes the range slightly wider than strict min-max
  so it's less sensitive to one-sample outliers in the lookback window.
- Still clips `y_scaled` to `[-5, 5]` for safety.
- Loses ~1.1 RMSE vs RollingNorm+clip — likely because the percentile range is still
  narrower than the actual flood exceedance.

### Key finding
> **RollingNorm + clip is the correct baseline.** Post-fix RMSE=5.08, R²=0.796 is
> *genuinely better* than the leaking baseline (RMSE=10.20, R²=0.32), confirming the
> leak was hurting, not helping, the model's real-world performance.

---

## Phase 2a — Optimizer Comparison

**Setup:** RollingNorm+clip, MLP, batch_size=1, max_grad_norm=1.0

| Method | RMSE | MAE | R² | Notes |
|---|---|---|---|---|
| Adam default | 5.0762 | 1.4829 | 0.7961 | Phase 1 reference |
| **Adam-WD** | **5.2012** | **1.3342** | **0.7953** | Best MAE — WD reduces spike overfitting |
| RMSprop | 6.7286 | 1.7934 | 0.6575 | Worse than Adam |
| SGD+Momentum | 6.8855 | 1.7932 | 0.6413 | Worst — too slow to adapt online |

### What failed and why

**RMSprop:** Per-parameter LR adaptation helps with gradient scale variance but
hurt here — the `alpha=0.99` decay is too slow for a batch_size=1 online stream
where each step is a single 24h window.

**SGD+Momentum:** Momentum carries stale gradient direction from past windows.
In a non-stationary stream (river floods, seasonal patterns) this causes the model
to chase old directions instead of adapting. Nesterov lookahead didn't help enough.

### Key finding
> **Adam is the right optimizer family.** RMSprop and SGD are dropped.
> Adam-WD trades a tiny RMSE increase (+0.12) for a meaningful MAE improvement (−0.15),
> suggesting weight decay reduces large individual errors (peaks) at the cost of a slight
> average bias. Whether this trade-off is desirable depends on use case
> (flood warning → minimize peak errors → prefer Adam-WD).

---

## Phase 2b — Adam Hyperparameter Ablation + FTRL

**Setup:** RollingNorm+clip, MLP, batch_size=1, max_grad_norm=1.0
**Status:** Running / results pending

### Configs being tested

**Adam variants:**
| Config | Key change | Hypothesis |
|---|---|---|
| Adam-default | baseline | — |
| Adam-WD | lr=1e-4, wd=1e-4 | L2 reg reduces spike overfitting |
| Adam-b1=0.7 | beta1=0.7 | Faster gradient forgetting after drift |
| Adam-b1=0.5 | beta1=0.5 | Aggressive forgetting — heavy drift |
| Adam-b2=0.9 | beta2=0.9 | Faster step-size adaptation to variance |
| Adam-eps=1e-4 | eps=1e-4 | Conservative steps on calm periods |
| Adam-amsgrad | amsgrad=True | Monotone LR — long-term stability |
| Adam-b1=0.5-amsgrad | beta1=0.5, b2=0.9, eps=1e-4, amsgrad | Combined fast forgetting + stable steps |

**FTRL variants (custom PyTorch implementation — `cstm_models/ftrl.py`):**
| Config | Key change | Hypothesis |
|---|---|---|
| FTRL-default | alpha=0.01 | Online-theory-optimal base |
| FTRL-l2=0.01 | lambda2=0.01 | L2 stability |
| FTRL-l1+l2 | lambda1=0.001, lambda2=0.01 | Sparse + stable weights |
| FTRL-alpha=0.1 | alpha=0.1 | Higher LR — faster adaptation |

---

## Running Summary — What Works, What Doesn't

| Approach | Verdict | Reason |
|---|---|---|
| Per-window min-max + y_clip | ✅ Works | Bounded y_scaled, no leak, stable loss |
| ZScore without clip | ❌ Fails | y_scaled unbounded during calm-lookback + flood-future |
| Percentile norm | ⚠️ Acceptable | Slightly worse than min-max+clip, more robust to lookback outliers |
| Adam (default) | ✅ Best so far | Fast adaptation, per-param LR |
| Adam + weight decay | ✅ Best MAE | Reduces large individual errors; slight RMSE cost |
| RMSprop | ❌ Worse | alpha=0.99 too slow for online stream |
| SGD + Momentum | ❌ Worst | Stale momentum kills online adaptation |
| FTRL | 🔄 Pending | Best theoretical fit — results awaited |
| Gradient clipping (max_norm=1.0) | ✅ Added | Prevents parameter explosion post-spike |
| Amnesia strategy | ✅ Available | Resets optimizer on loss spike — still valid complement to FTRL |

---

## Next Steps

- [ ] Get Phase 2b results → identify winning Adam config
- [ ] Run peak detection analysis on all_test_results
- [ ] ZScore + clip (fix the ZScore failure cheaply)
- [ ] Phase 3: ADWIN drift detection integrated with amnesia
- [ ] Phase 4: Full ablation comparison table + flood-event zoom plots
