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

import util_fun as uf

LOOKBACK, HORIZON, H_BAR_INDEX = 48, 24, 3
FIXED_DIR = "results/testing/online_fixed"
MODELS = ["FTRL-default", "RMSprop", "Adam-default"]
QUANTILES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
             0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 1.00]
MIN_RUN_HOURS = 3   # an event must persist >= 3 h (filters single-sample noise)

train_h = pd.read_csv("dataset/one_station_train_data.csv", index_col=0)["H_bar"].values

# Persistence reference level per hour: the issuing window's last lookback value.
# Needed for rising-limb / advance-warning POD and the persistence baseline column.
test_df = pd.read_csv("dataset/one_station_test_data.csv", parse_dates=[0], index_col=0)
x_test, _ = uf.create_sequences(test_df.values.astype(np.float32),
                                LOOKBACK, HORIZON, H_BAR_INDEX)
ref_levels = np.repeat(x_test[:, -1, H_BAR_INDEX].astype(np.float64), HORIZON)

find_events = uf.find_events   # gap-aware, shared implementation

frames = {}
for m in MODELS:
    path = f"{FIXED_DIR}/online_fixed_{m}.csv"
    if not os.path.exists(path):
        print(f"{m}: {path} missing — run rerun_fixed_pipeline.py first"); continue
    frames[m] = pd.read_csv(path, parse_dates=["Timestamp"]).sort_values("Timestamp")

ref = frames["FTRL-default"]
y_true = ref["True"].values  # verified == raw gauge
t_ref = ref["Timestamp"].values

assert len(ref_levels) == len(y_true), "reference-level grid mismatch"

rows = []
for q in QUANTILES:
    thr = float(np.quantile(train_h, q))
    events = find_events(y_true >= thr, times=t_ref, min_len=MIN_RUN_HOURS)
    row = {"quantile": q, "threshold": round(thr, 2), "peaks_observed": len(events)}

    # Persistence baseline on the same event metric: predicting "no change" still
    # "detects" any event whose issuing window was already at/above threshold.
    row["detected_persistence"] = sum(1 for s, e in events
                                      if ref_levels[s:e].max() >= thr)
    row["rate_persistence"] = (row["detected_persistence"] / len(events)
                               if events else np.nan)

    for m, df in frames.items():
        y_pred = df["Prediction"].values
        pm = uf.peak_event_metrics(y_true, y_pred, t_ref, thr, ref_levels,
                                   horizon=HORIZON, min_len=MIN_RUN_HOURS)
        row[f"detected_{m}"] = pm["hits"]
        row[f"rate_{m}"] = pm["pod"]
        row[f"rising_pod_{m}"] = pm["rising_pod"]
        row[f"aw_pod_{m}"] = pm["aw_pod"]          # advance warning: the honest one
        row[f"far_{m}"] = pm["far"]
        row[f"csi_{m}"] = pm["csi"]
        row[f"peak_bias_{m}"] = (round(pm["peak_bias"], 2)
                                 if not np.isnan(pm["peak_bias"]) else np.nan)
    rows.append(row)

out = pd.DataFrame(rows)
os.makedirs(FIXED_DIR, exist_ok=True)
out_path = f"{FIXED_DIR}/peak_threshold_sweep.csv"
out.to_csv(out_path, index=False)

# --- Report: full table for the winner, incl. baseline + falsifiable columns ---
w = "FTRL-default"
print(f"\nPhase 5.1 -- peak threshold sweep | winner = {w} | fixed pipeline")
print(f"event = contiguous hours with observed >= threshold, min {MIN_RUN_HOURS} h "
      f"(split at data gaps)")
print("POD = any-hour detection (inflated by already-in-flood windows); "
      "rising = events that began below thr;")
print("AW = advance warning (forecast issued while still below thr) -- "
      "the only genuine-forecast column")
print(f"\n{'Q':>6} {'thr':>8} {'obs':>5} {'POD':>6} {'persist':>8} {'rising':>7} "
      f"{'AW':>6} {'FAR':>6} {'CSI':>6} {'peak bias':>10}")
print("-" * 78)
for r in rows:
    if not r["peaks_observed"]:
        print(f"{r['quantile']:>6} {r['threshold']:>8.2f} {0:>5}   -- no events --")
        continue
    print(f"{r['quantile']:>6} {r['threshold']:>8.2f} {r['peaks_observed']:>5} "
          f"{r[f'rate_{w}']:>6.0%} {r['rate_persistence']:>8.0%} "
          f"{r[f'rising_pod_{w}']:>7.0%} {r[f'aw_pod_{w}']:>6.0%} "
          f"{r[f'far_{w}']:>6.2f} {r[f'csi_{w}']:>6.2f} {r[f'peak_bias_{w}']:>+10.2f}")

print("\nPOD comparison -- models vs persistence baseline (events in parentheses):")
print(f"{'Q':>6} " + " ".join(f"{m:>14}" for m in frames) + f"{'persistence':>14}")
for r in rows:
    if not r["peaks_observed"]:
        continue
    cells = " ".join(f"{r[f'rate_{m}']:>13.0%} " for m in frames)
    print(f"{r['quantile']:>6} {cells}{r['rate_persistence']:>13.0%}   "
          f"({r['peaks_observed']})")

print(f"\nExported: {out_path}")
