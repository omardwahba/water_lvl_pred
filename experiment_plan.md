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

## Phase 5 — Peak Threshold Sweep (winner vs solution) — DONE 2026-07-18

Evaluate the winning optimizer (FTRL-default) under the fixed pipeline ("the solution":
`scale_floor` q0.15 + no-clip/tripwire + honest ground truth) across peak thresholds,
counting peaks detected at each threshold.

- [x] **5.1 Threshold sweep Q50→Q100** — `peak_threshold_sweep.py` →
  `results/testing/online_fixed/peak_threshold_sweep.csv`; figures via
  `plot_peak_sweep.py` → `results/figures/peak_sweep_{summary,timeline}.png`
  - Detection 97% @Q50 → 70% @Q90 → 50–58% @Q97–98; peak-magnitude bias
    (undershoot) grows monotonically −6 @Q50 → −25 to −33 raw units at Q97+.
    Undershoot persists on the UNCENSORED pipeline → not a clipping artifact.
  - Events split at temporal data gaps (test set is non-contiguous). Data discovery:
    the split is BY YEAR — test = 2008 + 2014, train = 2009–13, 2015, 2018–19
    (same station, disjoint years → no temporal overlap between train and test).
  - **REINTERPRETED 2026-07-18 (persistence audit, `baselines.py`):** the detection
    curve is NOT model skill. Model POD equals persistence POD at every threshold
    (model-only detections = 0); advance-warning POD — credit only for exceedance
    forecasts issued while the river was still below threshold — is **0% at every
    threshold**. The models collapsed onto persistence (predicted deviation = 4.0%
    of observed; FTRL skill vs persistence only +0.018, RMSprop/Adam negative).
    The curve measures the event definition (windows already at/near threshold when
    issued), and the growing undershoot is the signature of a persistence predictor,
    not of a miscalibrated learner. Sweep now reports FAR/CSI/rising/AW POD plus a
    persistence column so this is visible in the artifact itself.

---

## Phase 6 — Loss Re-weighting (raw-unit loss) — DONE 2026-07-18

Tests the diagnosed cause of the persistence collapse (`normalization_deep_dive.md`
section 5): scaled-space MSE equals `raw_error^2 / scale^2`, discounting flood-window
errors ~270x. Treatment computes the loss in raw units (weights scaled loss by `scale^2`).

- [x] **6.0 Baselines** — `baselines.py` → `results/testing/baselines_skill.csv`
  - Persistence: RMSE 4.881, R²/NSE 0.820. Climatology: RMSE 12.33, R² −0.149.
  - FTRL +0.018 skill vs persistence; RMSprop −0.365; Adam −0.626 (worse than
    doing nothing). **Aggregate R² is nearly all persistence** — this table must
    accompany every future results table.

- [x] **6.1 loss_space ablation** — `loss_reweight_experiment.py` (2 loss spaces ×
  {FTRL, Adam} × 5 seeds; control regression to RMSE 4.836 PASSED) →
  `results/testing/online_reweighted/` (see `PHASE6_RESULTS.md` there)

  | config | RMSE | skill vs persist | flood skill | dev ratio | Q99 peak bias |
  |---|---|---|---|---|---|
  | scaled / FTRL (control) | 4.835 | +0.019 | +0.011 | 0.041 | −24.9 |
  | **raw / FTRL** | **4.782** | **+0.040** | **+0.023** | **0.087** | −25.1 |
  | scaled / Adam | 6.699 | −0.886 | −1.133 | 0.557 | +7.5 |
  | raw / Adam | 5.523 | −0.284 | −0.358 | 0.457 | −9.1 |

  - **Verdict: mechanism CONFIRMED, fix INSUFFICIENT.** Raw-unit loss doubles the
    predicted-deviation ratio (0.041 → 0.087) and doubles skill vs persistence,
    improving every metric for both optimizers — causal evidence that the
    `1/scale^2` weighting drives the collapse. But 0.087 is still ~9% of real
    movement: advance-warning POD stays 0%, peak bias unchanged (−25). Adam moves
    further off persistence (dev ratio 0.46–0.56) but is far worse than doing
    nothing — leaving persistence without learning dynamics produces noise.
  - FTRL deterministic (std ~0.001 over 5 seeds); Adam seed-sensitive (±0.3 RMSE).

- [ ] **6.2 Next levers (untested)** — intermediate weighting `scale^p`, p ∈ [0.5, 2];
  multi-epoch/replay (only ~2.5k online updates may be too few to escape the
  attractor); direct-level or rate-of-change targets (removes the persistence zero
  point); asymmetric/quantile loss for peak magnitude specifically.

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
