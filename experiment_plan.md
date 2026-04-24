# Experiment Plan — After Leak Fix Recovery

> **Goal:** Recover (and surpass) pre-leak-fix performance using legitimate techniques.
> **Baseline to beat:** online-ADAM | RMSE=10.20 | MAE=2.50 | R²=0.32 *(this was inflated by the leak)*

---

## Phase 1 — Fix the Normalization Root Cause

The optimizer doesn't matter if `y_scaled` is exploding. Fix this first.

- [x] **1.1 Percentile-based min-max** — `PercentileNormTimeSeriesDataset` added to `util_fun.py`
  - Uses `torch.quantile(x_raw, 0.05/0.95)` per feature; y_scaled clamped to [-5, 5]

- [x] **1.2 Clip y_scaled** — `torch.clamp(y_scaled, -5.0, 5.0)` added to `RollingNormTimeSeriesDataset`
  - Also applied in `PercentileNormTimeSeriesDataset`

- [x] **Leak fix** — `RollingNormTimeSeriesDataset` now uses only lookback stats for y (no future y_raw)

- [x] **ZScore dataset** — `ZScoreNormTimeSeriesDataset` added — **FAILED** (RMSE=306, R²=-711): unbounded y_scaled when lookback std is tiny during calm periods; needs clamp
- [ ] **ZScore + clip** — add `torch.clamp(y_scaled, -5, 5)` to `ZScoreNormTimeSeriesDataset` and retest

- [x] **Gradient clipping** — `max_grad_norm` param added to `train_model_online`; pass e.g. `max_grad_norm=1.0`

- [ ] **1.3 Log1p transform** — apply `torch.log1p` to H_bar before normalization, invert with `torch.expm1` after
  - Compresses flood spikes naturally — good for river data
  - Test on both `RollingNorm` and `ZScore` variants

- [ ] **1.4 Running/incremental stats** — maintain EMA of mean+std that updates each batch
  - True online normalization: `mu = alpha*x + (1-alpha)*mu`
  - No look-ahead, adapts to drift — most principled fix for online setting

- [ ] **1.5 Global y normalization + per-window X** — compute y stats from full training set only
  - Eliminates the per-window instability for y entirely
  - Only viable for the offline training phase, not pure online

---

## Phase 2 — Optimizer Experiments

Run each optimizer with the **best normalization from Phase 1**. Keep all other params fixed.

- [ ] **2.1 RMSprop** — `torch.optim.RMSprop(lr=1e-3, alpha=0.99)`
  - Adapts LR per parameter, handles gradient scale variance
  - Most promising drop-in replacement for Adam

- [ ] **2.2 SGD + Momentum** — `torch.optim.SGD(lr=1e-2, momentum=0.9)`
  - More stable than Adam on non-stationary sequences
  - Try with and without `nesterov=True`
  - May need LR warmup — start at `1e-3`, ramp to `1e-2` over first 100 batches

- [ ] **2.3 SGD + Momentum + LR Schedule** — add `ReduceLROnPlateau` or cosine annealing
  - Decays LR when loss plateaus — helps after regime shifts

- [x] **2.4 FTRL** — custom PyTorch `Optimizer` implemented in `cstm_models/ftrl.py`
  - Full FTRL-Proximal (McMahan et al. 2013) — per-coordinate adaptive LR + L1/L2
  - Exported via `cstm_models.FTRL`; tested in Phase 2b ablation loop alongside Adam

- [ ] **2.5 Adam with lower LR + weight decay** — `torch.optim.Adam(lr=1e-4, weight_decay=1e-4)`
  - Regularization may help stability without changing optimizer family
  - Compare directly against current Adam baseline

---

## Phase 3 — Drift Detection (ADWIN)

Upgrade the amnesia strategy to be drift-aware rather than loss-spike-reactive.

- [ ] **3.1 ADWIN-U integration** — use `river.drift.ADWIN` on the batch loss stream
  - Feed each batch loss into `adwin.update(loss)` — triggers on `adwin.drift_detected`
  - When drift detected: reset optimizer state (like amnesia) + optionally reset model weights
  - Compare: ADWIN trigger vs. current threshold-based amnesia trigger

- [ ] **3.2 ADWIN + RMSprop** — combine best optimizer from Phase 2 with ADWIN reset
  - RMSprop + ADWIN is the most promising combination based on analysis

- [ ] **3.3 ADWIN on residuals** — instead of raw loss, feed `|y_true - y_pred|` into ADWIN
  - More meaningful signal: actual prediction error, not training loss

---

## Phase 4 — Ablation & Comparison

- [ ] **4.1 Full comparison table** — run all winning configs side-by-side
  - Metrics: RMSE, MAE, R² on test set
  - Track: number of amnesia/ADWIN triggers, worst-case batch loss

- [ ] **4.2 Flood event zoom-in** — isolate the batch-1900 spike period
  - Plot `y_true` vs `y_pred` around the flood event for each config
  - Key question: does the model recover faster with ADWIN vs. amnesia?

- [ ] **4.3 Offline baseline re-run** — confirm offline models are unaffected by leak fix
  - Offline uses global normalization — should be unchanged

---

## Priority Order

```text
1.2 (clip y_scaled)  ← fastest win, try today
1.1 (percentile norm)
2.1 (RMSprop)
3.1 (ADWIN-U)
1.4 (running stats)  ← most effort, most correct
2.4 (FTRL)           ← most effort on optimizer side
```

---

## Notes

- All experiments run on `online_learning.ipynb`
- Fixed seed for reproducibility: `torch.manual_seed(42)`
- Do **not** try Adagrad — LR decays to 0, kills online adaptation
- ADWIN requires: `pip install river`
- FTRL requires: `pip install river`
