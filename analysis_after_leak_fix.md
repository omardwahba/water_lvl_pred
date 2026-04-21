# Analysis: Why Results Dropped After Leak Fix

## Main Points (Quick Overview)

| Component | What it does |
|---|---|
| `RollingNormTimeSeriesDataset` | Per-window min-max norm (only lookback for X and y) |
| `ZScoreNormTimeSeriesDataset` | Per-window z-score norm (added in this branch) |
| `create_sequences` | Non-overlapping sliding window — steps by `horizon`, not by 1 |
| `train_model_offline` | Global min-max, batch training, early stopping |
| `train_model_online` | Sequential batch_size=1, Adam, + amnesia strategy |
| `Amnesia Strategy` | Resets optimizer state on loss spikes |

---

## Detailed Analysis: Why Results Dropped After Leak Fix

### What the Leak Actually Was

In the `main` branch, `RollingNormTimeSeriesDataset` computed H_bar normalization stats using **both** lookback AND future y:

```python
# OLD (main branch) — LEAKING
y_min = torch.min(y_raw)   # ← future data used here
y_max = torch.max(y_raw)
h_bar_min = torch.min(min_vals[h_bar_index], y_min)
h_bar_max = torch.max(max_vals[h_bar_index], y_max)
```

```python
# NEW (after_leak_fix) — clean
h_bar_min = min_vals[self.h_bar_index]   # only from 48h lookback
h_bar_range = range_vals[self.h_bar_index]
```

### Root Cause of the Drop

**1. The old "good" results were an artifact — the model got a hidden hint about future magnitude.**
When future y is included in the normalization range, `y_scaled` is always guaranteed to be within `[0,1]`. The model was essentially told "how extreme the future values are" through the normalization scale, making its job trivially easier.

**2. After the fix, `y_scaled` can go far outside `[0,1]` — the model was never trained to handle this.**
If the next 24 hours has a flood spike that exceeds the 48h lookback max:
```
y_scaled = (y_raw - h_bar_min) / h_bar_range  →  could be 5, 10, or even 100
```
The loss on this single sample = MSE(pred≈0.5, target=10) = **~90** — the model blows up.

**3. This is exactly the spike seen: loss = ~194 billion at batch 1900.**
That is not a bug — it is a real flood event or regime change that the lookback window did not capture. The normalization range was narrow, but the future was extreme.

**4. Min-max norm is fragile for non-stationary time series in online mode.**
With `batch_size=1`, each bad sample is a full update. One extreme event can destroy what the model learned across hundreds of batches.

**5. The Adam optimizer carries momentum from the past — wrong in a non-stationary setting.**
After a spike, Adam's `m` (1st moment) and `v` (2nd moment) accumulators are corrupted with the extreme gradient. Even with amnesia (optimizer state reset), Adam re-accumulates stale info quickly.

---

## What the Supervisor Tried

| Fix | What it does | Why it is insufficient |
|---|---|---|
| **Z-score norm** | Centers on 0, unit variance → y_scaled stays ~[−3,3] even on extremes | Better, but std from 48h window can still be tiny during calm periods → z-score still explodes |
| **Gradient clipping** | Caps gradient norm via `clip_grad_norm_` | Prevents parameter explosion *after* the loss spike, but does not prevent the bad loss signal itself |

Neither fixes the root: the **normalization reference window is too narrow** for min-max to handle distribution shift.

---

## Alternative Leak Fixes Worth Exploring

These were not tried and address the root cause differently:

| Approach | Idea | Tradeoff |
|---|---|---|
| **Global y normalization** | Compute y min/max from entire training set; use per-window for X only | Simple, stable — but needs full training data, offline only |
| **Running/incremental stats** | Maintain exponential moving mean+std that updates each batch | True online, no leak, adapts to drift — adds complexity |
| **Percentile-based norm** | Use 5th–95th percentile of lookback (not strict min/max) | Less sensitive to lookback outliers — still can extrapolate |
| **Clip y_scaled** | After computing y_scaled, clip to e.g. `[-5, 5]` | Quick hack — prevents explosive loss but truncates extreme events |
| **Log transform first** | Apply `log1p` before normalization | Compresses large spikes naturally — good for river data |

---

## On the Suggested Optimizers

| Optimizer | Fit for this problem | Key issue |
|---|---|---|
| **SGD + Momentum** | Better than Adam for non-stationary | Momentum carries stale gradient direction — slow to adapt after drift |
| **FTRL** | Best theoretical fit — designed for online learning | Not in PyTorch natively; `river` library has it |
| **RMSprop** | Good — adapts per-parameter LR, handles gradient scale variance | Slightly less aggressive than Adam on non-stationary |
| **Adagrad** | **Bad here** — LR monotonically decays to 0 → model stops adapting | Designed for sparse/convex problems, not streaming drift |
| **ADWIN** | Not an optimizer — it is a **drift detector** | Should trigger optimizer reset or model re-init when concept drift is detected; pairs well with amnesia strategy |

**Recommended order to try:** RMSprop → SGD+Momentum (with LR schedule) → FTRL (via `river` library) → ADWIN as drift signal to trigger amnesia.

---

## Summary

The leak fix is **correct** but it **exposed a design mismatch**: online min-max normalization with a 48h lookback is too volatile for a river that can flood. The results dropped because the previous performance was inflated by the leak. The real bottlenecks now are:

1. **Normalization instability** — primary issue, not optimizer choice
2. **Optimizer** — Adam is acceptable but RMSprop may be more stable
3. **ADWIN** — solid addition to make amnesia drift-aware rather than reactive to loss spikes only
