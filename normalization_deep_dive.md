# Normalization Deep Dive -- How the Fixed Pipeline Scales Its Data

> Companion doc to `util_fun.py` (`RollingNormTimeSeriesDataset`), `rerun_fixed_pipeline.py`,
> and `experiment_plan.md` Phases 5-6. All numbers measured on the one-station dataset
> (train = years 2009-13, 2015, 2018-19; test = held-out years 2008 + 2014). 2026-07-18.
> ASCII-only by convention: H_bar means the level feature, ^2 means squared.

**One-sentence version:** every sample is re-expressed relative to *its own recent
past* -- inputs and target are centered on the current level and scaled by recent
variability (floored) -- so the model never sees absolute water levels at all; it
lives entirely in the language of *"how far from now, in units of recent movement."*

---

## 1. Two normalization schemes live inside every sample

A sample's input is 48 hours x 4 features (Q_bar discharge, P_bar precip, T_bar temp,
H_bar level). The two groups are treated differently:

**Scheme A -- exogenous features (Q_bar, P_bar, T_bar): per-window min-max.**

```python
x_scaled = (x_raw - min_vals) / (max_vals - min_vals + eps)
```

Each feature is squashed into [0, 1] using *that window's own* min/max. The model
learns "precipitation is at the top of its recent range" -- pure *shape* information;
absolute magnitudes are discarded. Lossy but harmless: these are inputs only,
nothing is ever reconstructed from them.

**Scheme B -- the water level H_bar, in both X and y: last-value centering +
floored-std scaling.**

```python
h_bar_ref   = h_bar_lookback[-1]            # the level RIGHT NOW (hour 48)
h_bar_scale = max(std(lookback), floor) + eps  # recent variability, floored
x_scaled[:, H] = (h_bar_lookback - ref) / scale   # the 48 input levels
y_scaled       = (y_raw - ref) / scale            # the 24 future levels
```

H_bar gets the careful treatment because it is an input *and* the prediction target,
and predictions must be invertible to raw units. Min-max was tried for it and
rejected (see the class docstring): anchoring to the window *minimum* shifted flood
predictions systematically downward -- the lookback before a flood is calm, so its
min is low, and "within recent range" is exactly wrong during a flood. Last-value
centering makes `y_scaled = 0` mean "level stays where it is now" -- persistence --
regardless of season or regime.

## 2. Why per-window at all -- the online-learning constraint

Offline pipelines normalize with *global* statistics (`train_model_offline` does:
min/max over the whole training set). A streaming system cannot: the future's range
is unknown at deployment, and a fixed global scale goes stale as the river drifts.
Per-window normalization is **self-renewing** -- every prediction carries its own
calibration, computed only from the 48 hours the model can legitimately see.

This is also why the clipping pathology (see `diagnostics/clip_proof.py`) was
specific to the online setting: the offline path never divides by a 48-sample
statistic, so it never had the degenerate-denominator problem.

## 3. The scale, step by step, with real windows

| window | last value (ref) | lookback std | scale used | situation |
|---|---|---|---|---|
| calm typical | 154.00 | 0.56 | 0.56 | median window |
| dead flat | 150.00 | 0.0000 | **0.0947** (floor) | stagnant / quantized sensor |
| pre-flood 2008 | 169.95 | 9.24 | 9.24 | already-agitated river |

- **Calm:** river drifts to 154.80 over 24 h -> `y_scaled = 0.8/0.56 ~= +1.4`.
- **Flat:** std is exactly 0 (13.7% of train windows are below 0.05). Without the
  floor, scale = eps = 1e-6 and a mundane 0.5-unit rise became `y_scaled = 500,000`
  -- the original defect. With the floor (**0.094705** = q0.15 of train-window
  stds), the same rise is `y_scaled ~= 5.3`: large, informative, learnable.
- **Flood:** river surges to 301.67 -> `y_scaled ~= +14.3` -- genuinely extreme, and
  delivered to the model at full magnitude (no clip) instead of censored to +5.

The floor deliberately does *not* touch the calm or flood windows (their stds exceed
0.0947). It intervenes only in the degenerate bottom ~15% -- which is why q0.15 was
chosen: one step above where the flat-window lump ends (q0.10 evaluated to 0.006,
still inside the lump). See `compute_scale_floor` and `diagnostics/floor_sweep.py`.

## 4. The inverse transform -- and the asymmetry behind the old bug

```python
y_pred_true = y_pred_scaled * scale + ref     # legitimate: model output, translated
y_truth     = y_raw                           # NEVER reconstructed -- taken verbatim
```

Un-scaling the *prediction* is a change of units -- always valid. Un-scaling the
*target* is valid only if nothing modified the target after scaling; the legacy +-5
clamp broke that, so reconstructed "truth" differed from the gauge on 8.9% of test
hours and inflated reported metrics by 16-40% RMSE. The fixed pipeline eliminates
the round-trip: raw observations ride along in the dataset 5-tuple and are used
directly as ground truth.

## 5. The deepest consequence: the normalization IS a loss weighting

The model minimizes MSE **in scaled space**:

```
loss = (y_pred_scaled - y_scaled)^2  =  (y_pred_raw - y_raw)^2 / scale^2
```

A raw-unit error is divided by **scale^2**, so the objective weights errors
*inversely to recent variability*:

- 1-unit raw error in a **calm** window (scale 0.56) -> loss ~= 3.2
- 1-unit raw error in the **2008 flood** window (scale 9.24) -> loss ~= 0.012 --
  **270x less**

The normalization that makes online learning possible also tells the optimizer that
flood-magnitude errors barely matter. The model is being *rational* when it
undershoots the 2008 peak by 66 raw units: in its training currency, that error is
cheap.

**Empirically confirmed (persistence audit, 2026-07-18):** the trained models have
collapsed onto the persistence solution this weighting favors. Predicted deviation
from the reference level is **4.0%** of observed deviation overall and **3.5%** on
Q90-peak windows; FTRL-default's skill vs persistence is only +0.018 (persistence
alone: RMSE 4.881, R2 0.820), and RMSprop/Adam are *worse than no model* (-0.36 /
-0.63). Event "detection" in the Phase 5.1 sweep is exactly matched by persistence
(model-only detections = 0 at every threshold). This also explains why the peak
undershoot survived the removal of clipping (peak bias -33 at Q99.5,
`peak_threshold_sweep.py`): it was never primarily the clamp.

**The tested lever (Phase 6, `loss_reweight_experiment.py`):** compute the training
loss in raw units -- `criterion(y_pred_scaled * scale, y_scaled * scale)` --
i.e. weight scaled-space loss by scale^2, restoring full weight to flood errors.
See `experiment_plan.md` Phase 6 for results.

## 6. `y_scaled = 0` is the persistence forecast

Because of last-value centering, the zero of target space *is* "no change":

- An untrained model outputting zeros already implements hydrology's strongest
  naive baseline (persistence).
- Everything the model learns is a *correction to persistence* -- which is why
  skill-vs-persistence (`baselines.py`) is the purest measure of what the
  parameters contribute.
- The step-shaped prediction curves in `results/figures/peak_sweep_timeline.png`
  are this structure made visible: each 24-h block is one window's predicted
  deviation profile added to that window's `ref`.

## 7. Leakage audit -- what information enters each sample

| Quantity | Source | Future data? |
|---|---|---|
| min/max for Q_bar, P_bar, T_bar | this window's 48-h lookback | no |
| `ref` (center) | last lookback hour | no |
| `std` (scale) | lookback only | no |
| `floor` | **train-set** window stds, one fixed constant | no (train-derived) |
| `y_raw` in the tuple | the future -- used only as target/metric, never as input | not leaked into inputs |

Discipline rule: the floor's quantile is chosen from train statistics and is never
tuned against test metrics (that would be model selection on the test set).

## 8. Known limitations (honest edges)

1. **48 samples is a noisy std estimate** -- autocorrelation makes the window look
   calmer than the process; the floor bounds the damage but doesn't fix the
   estimator.
2. **The scale is backward-looking by design** -- the window before a flood is calm,
   so flood targets are huge in scaled units: informative, but the source of the
   heteroscedastic loss weighting in section 5.
3. **Exogenous features lose magnitude entirely** -- min-max keeps only shape;
   "heaviest rainfall in years" and "drizzle that is locally the week's max" look
   identical in P_bar. A magnitude-preserving alternative (train-global scaling for
   P_bar) is an untested experiment.

## 9. Open data-hygiene issue: windows spanning time gaps

`create_sequences` slices **by position**, so windows silently span temporal gaps:
**181 / 2,461 train windows (7.4%)** and **18 / 728 test windows (2.5%)** contain a
jump > 1 h inside their 72-h span (train years are internally gappy -- 185 gaps --
not just the test's 2008->2014 boundary). For those windows the "48-h lookback"
mixes non-adjacent hours as if contiguous, so `ref` and `std` describe a fictional
timeline. Most gaps are a few hours (mild); the year-boundary ones are severe.

Status: **known, unfixed.** The clean fix is a gap-aware sequence builder (drop or
rebuild windows spanning a gap beyond a tolerance) and a re-run of the headline
models to measure the metric shift. Same class of bug as the event-counter fix
(`find_events` in `util_fun.py` splits runs at gaps), one level deeper.
