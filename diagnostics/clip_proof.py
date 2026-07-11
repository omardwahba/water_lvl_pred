"""
Diagnostic: measure what y_clip censoring did to the LEGACY pipeline.

Documents the defect fixed on 2026-07-11 (see rerun_fixed_pipeline.py):
- Part A: rebuild the legacy dataset (scale_floor=None, y_clip=5.0) and measure
  where the clamp bound, how far the reconstructed "truth" drifted from the real
  gauge, and the worst-censored windows (they are the flood peaks).
- Part B: audit the pre-fix exported CSVs in results/testing/predictions_vs_true/
  by joining their stored "True" column to the actual observed gauge, then
  recompute RMSE/MAE/R2 both ways -> direct measurement of the metric inflation.

Run from anywhere: python diagnostics/clip_proof.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
import util_fun as uf

LOOKBACK, HORIZON, H_BAR_INDEX = 48, 24, 3

test_df = pd.read_csv("dataset/one_station_test_data.csv", parse_dates=[0], index_col=0)
x_test, y_test = uf.create_sequences(test_df.values.astype(np.float32), LOOKBACK, HORIZON, H_BAR_INDEX)

# LEGACY configuration — the one that produced all pre-fix results
ds = uf.RollingNormTimeSeriesDataset(x_test, y_test, H_BAR_INDEX, y_clip=5.0, scale_floor=None)

print("=" * 78)
print("PART A — legacy dataset (scale_floor=None, y_clip=5.0): where the clamp bound")
print("=" * 78)

n = len(ds)
clip_hits = np.zeros((n, HORIZON), dtype=bool)
distort_raw = np.zeros((n, HORIZON), dtype=np.float32)   # |truth error| in raw units
scales = np.zeros(n, dtype=np.float32)
refs = np.zeros(n, dtype=np.float32)

for i in range(n):
    _, y_scaled_clipped, y_raw, ref, scale = ds[i]
    y_scaled_unclipped = (y_raw - ref) / scale
    clip_hits[i] = (y_scaled_unclipped.abs() > 5.0).numpy()
    # what the legacy pipeline reconstructed and reported as "truth":
    y_truth_reconstructed = y_scaled_clipped * scale + ref
    distort_raw[i] = (y_raw - y_truth_reconstructed).abs().numpy()
    scales[i], refs[i] = scale.item(), ref.item()

print(f"windows: {n}, horizon elements: {n * HORIZON}")
print(f"clamp bound on {clip_hits.mean():.3%} of elements, "
      f"{clip_hits.any(axis=1).mean():.3%} of windows")
d = distort_raw[clip_hits]
print(f"truth distortion where it bound (raw units): "
      f"mean {d.mean():.3f} | median {np.median(d):.3f} | max {d.max():.3f}")

print("\n--- Worst-censored windows (they are the flood peaks) ---")
worst = np.argsort(distort_raw.max(axis=1))[::-1][:3]
for w in worst:
    t_end = w * HORIZON + LOOKBACK
    j = int(distort_raw[w].argmax())
    y_raw_v = float(y_test[w, j])
    ts = test_df.index[t_end + j]
    print(f"\n window #{w}  target hour {ts}")
    print(f"   lookback H_bar: last={refs[w]:.2f}, scale={scales[w]:.6f}")
    print(f"   ACTUAL future level:           {y_raw_v:.2f}")
    print(f"   unclipped y_scaled:            {(y_raw_v - refs[w]) / scales[w]:,.1f}")
    print(f"   legacy stored-as-'truth':      {refs[w] + np.sign(y_raw_v - refs[w]) * 5 * scales[w]:.2f}")
    print(f"   truth error injected:          {distort_raw[w, j]:.2f} raw units")

print()
print("=" * 78)
print("PART B — pre-fix exported CSVs vs the ACTUAL observed gauge")
print("=" * 78)

obs = test_df["H_bar"]

def metrics(y_true, y_pred):
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r2 = 1 - float(np.sum(err ** 2)) / float(np.sum((y_true - y_true.mean()) ** 2))
    return rmse, mae, r2

for name in ["FTRL-default", "Adam-default", "RollingNorm_clip", "RMSprop"]:
    path = f"results/testing/predictions_vs_true/online_{name}.csv"
    if not os.path.exists(path):
        print(f"{name}: {path} missing, skipped")
        continue
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df["observed"] = df["Timestamp"].map(obs)
    assert df["observed"].notna().all(), "timestamp join failed"

    diff = (df["True"] - df["observed"]).abs()
    frac_bad = float((diff > 1e-3).mean())
    rmse_s, mae_s, r2_s = metrics(df["True"].values, df["Prediction"].values)      # as reported
    rmse_r, mae_r, r2_r = metrics(df["observed"].values, df["Prediction"].values)  # honest

    print(f"\n### {name}  ({len(df)} rows)")
    print(f"  stored 'True' != observed gauge on {frac_bad:.2%} of rows "
          f"(mean gap {diff[diff > 1e-3].mean():.3f}, max {diff.max():.2f} raw units)")
    print(f"  metrics vs stored (clipped) truth : RMSE {rmse_s:.3f} | MAE {mae_s:.3f} | R2 {r2_s:.4f}")
    print(f"  metrics vs ACTUAL observed river  : RMSE {rmse_r:.3f} | MAE {mae_r:.3f} | R2 {r2_r:.4f}")
    print(f"  -> RMSE understated by {rmse_r - rmse_s:+.3f} ({(rmse_r / rmse_s - 1) * 100:+.1f}%), "
          f"R2 overstated by {r2_s - r2_r:+.4f}")
