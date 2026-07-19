"""
Phase 5.1 — Peak Threshold Sweep (experiment_plan.md).

For each percentile threshold Q50..Q100 (train-derived), find every flood event in
the observed gauge along the online run (contiguous hours >= threshold, min 3 h)
and check whether the model's prediction also reached the threshold during the
event. Reports observed count, detected count, detection rate, and mean peak
bias (pred_max - true_max within events; negative = undershoot).

Primary subject: FTRL-default (winner) under the fixed pipeline; RMSprop and
Adam-default included for comparison. Predictions come from the exported CSVs of
rerun_fixed_pipeline.py — no retraining. Truth column of those CSVs is verified
equal to the raw gauge (see rerun_fixed_pipeline.py assertion).

Run from anywhere: python peak_threshold_sweep.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd

FIXED_DIR = "results/testing/online_fixed"
MODELS = ["FTRL-default", "RMSprop", "Adam-default"]
QUANTILES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
             0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 1.00]
MIN_RUN_HOURS = 3   # an event must persist >= 3 h (filters single-sample noise)

train_h = pd.read_csv("dataset/one_station_train_data.csv", index_col=0)["H_bar"].values

def find_events(mask, min_len=MIN_RUN_HOURS):
    """Contiguous True-runs of mask as (start, end_exclusive), length >= min_len."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return [(s, e) for s, e in edges.reshape(-1, 2) if e - s >= min_len]

frames = {}
for m in MODELS:
    path = f"{FIXED_DIR}/online_fixed_{m}.csv"
    if not os.path.exists(path):
        print(f"{m}: {path} missing — run rerun_fixed_pipeline.py first"); continue
    frames[m] = pd.read_csv(path, parse_dates=["Timestamp"]).sort_values("Timestamp")

ref = frames["FTRL-default"]
y_true = ref["True"].values  # verified == raw gauge

rows = []
for q in QUANTILES:
    thr = float(np.quantile(train_h, q))
    events = find_events(y_true >= thr)
    row = {"quantile": q, "threshold": round(thr, 2), "peaks_observed": len(events)}
    for m, df in frames.items():
        y_pred = df["Prediction"].values
        detected = sum(1 for s, e in events if y_pred[s:e].max() >= thr)
        bias = (np.mean([y_pred[s:e].max() - y_true[s:e].max() for s, e in events])
                if events else np.nan)
        row[f"detected_{m}"] = detected
        row[f"rate_{m}"] = detected / len(events) if events else np.nan
        row[f"peak_bias_{m}"] = round(bias, 2) if events else np.nan
    rows.append(row)

out = pd.DataFrame(rows)
os.makedirs(FIXED_DIR, exist_ok=True)
out_path = f"{FIXED_DIR}/peak_threshold_sweep.csv"
out.to_csv(out_path, index=False)

# --- Report: full table for the winner, then detection-rate comparison ---
w = "FTRL-default"
print(f"\nPhase 5.1 — peak threshold sweep | winner = {w} | fixed pipeline")
print(f"event = contiguous hours with observed >= threshold, min {MIN_RUN_HOURS} h")
print(f"\n{'Q':>6} {'thr':>8} {'observed':>9} {'detected':>9} {'rate':>7} {'peak bias':>10}")
print("-" * 55)
for r in rows:
    rate = f"{r[f'rate_{w}']:.0%}" if r["peaks_observed"] else "  —"
    bias = f"{r[f'peak_bias_{w}']:+.2f}" if r["peaks_observed"] else "   —"
    print(f"{r['quantile']:>6} {r['threshold']:>8.2f} {r['peaks_observed']:>9} "
          f"{r[f'detected_{w}']:>9} {rate:>7} {bias:>10}")

print(f"\nDetection-rate comparison (observed peaks in parentheses):")
print(f"{'Q':>6} " + " ".join(f"{m:>14}" for m in frames))
for r in rows:
    if not r["peaks_observed"]:
        continue
    cells = " ".join(f"{r[f'rate_{m}']:>13.0%} " for m in frames)
    print(f"{r['quantile']:>6} {cells}  ({r['peaks_observed']})")

print(f"\nExported: {out_path}")
