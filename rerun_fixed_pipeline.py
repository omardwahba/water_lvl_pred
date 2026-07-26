"""
Re-run the headline online models under the fixed pipeline:
  - scale_floor (train-derived) so flat lookback windows can't explode y_scaled
  - y_clip demoted to a pure tripwire (500 — above any genuine dynamics, 0% binding)
  - ground truth in metrics = raw observations (never the clipped target)
  - max_grad_norm=1.0 for every run (removes the Phase 1 vs Phase 2 confound)

Outputs metrics to stdout and prediction CSVs to results/testing/online_fixed/.
Old (pre-fix) numbers are recomputed from the existing exported CSVs so the
comparison table is self-contained and auditable.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import util_fun as uf
from cstm_models import build_mlp
from cstm_models.ftrl import FTRL

LOOKBACK, HORIZON, N_FEATURES, BATCH_SIZE, H_BAR_INDEX = 48, 24, 4, 1, 3
# Train max |y_scaled| with the floor is ~122; genuine flood onsets reach 10-80 sigma,
# so the tripwire must sit far above real dynamics — it may only catch data corruption.
Y_CLIP_TRIPWIRE = 500.0
FLOOR_QUANTILE = 0.15  # q0.10 was still ~0 (14% of train windows are near-flat)
OUT_DIR = "results/testing/online_fixed"
OLD_DIR = "results/testing/predictions_vs_true"

# --- Data (timestamps are ISO) ---
train_df = pd.read_csv("dataset/one_station_train_data.csv", parse_dates=[0], index_col=0)
test_df = pd.read_csv("dataset/one_station_test_data.csv", parse_dates=[0], index_col=0)

x_train, y_train = uf.create_sequences(train_df.values.astype(np.float32), LOOKBACK, HORIZON, H_BAR_INDEX)
x_test, y_test = uf.create_sequences(test_df.values.astype(np.float32), LOOKBACK, HORIZON, H_BAR_INDEX)

# --- Fixed-pipeline datasets: train-derived floor, tripwire clip ---
floor = uf.compute_scale_floor(x_train, H_BAR_INDEX, quantile=FLOOR_QUANTILE)
train_ds = uf.RollingNormTimeSeriesDataset(x_train, y_train, H_BAR_INDEX, y_clip=Y_CLIP_TRIPWIRE, scale_floor=floor)
test_ds = uf.RollingNormTimeSeriesDataset(x_test, y_test, H_BAR_INDEX, y_clip=Y_CLIP_TRIPWIRE, scale_floor=floor)

bind_train = uf.report_y_clip_binding(train_ds, "train")
bind_test = uf.report_y_clip_binding(test_ds, "test")
if max(bind_train, bind_test) > 0:
    print("WARNING: tripwire bound — inputs wilder than anything in train; investigate.")

# --- Models (same seed/configs as the original experiments) ---
def make_adam_default():
    return build_mlp(LOOKBACK * N_FEATURES, HORIZON)  # Adam lr=1e-3, MSE

def make_rmsprop():
    model, _, crit = build_mlp(LOOKBACK * N_FEATURES, HORIZON)
    return model, optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99), crit

def make_ftrl_default():
    model, _, crit = build_mlp(LOOKBACK * N_FEATURES, HORIZON)
    return model, FTRL(model.parameters(), alpha=0.01, beta=1.0, lambda1=0.0, lambda2=0.0), crit

MODELS = [
    ("Adam-default", make_adam_default),
    ("RMSprop", make_rmsprop),
    ("FTRL-default", make_ftrl_default),
]

def metrics(y_true, y_pred):
    return (root_mean_squared_error(y_true, y_pred),
            mean_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred))

os.makedirs(OUT_DIR, exist_ok=True)
obs = test_df["H_bar"]
results = {}

for label, factory in MODELS:
    print(f"\n=== {label} (fixed pipeline) ===")
    torch.manual_seed(42)
    model, optimizer, criterion = factory()
    uf.train_model_online(model, optimizer, criterion, train_ds, train_df,
                          LOOKBACK, HORIZON, BATCH_SIZE, silent=True)
    y_true, y_pred, ts = uf.model_evaluate_with_norm(model, criterion, test_ds,
                                                     test_df.index, LOOKBACK, HORIZON, BATCH_SIZE)

    # Honest-truth assertion: exported truth must equal the observed gauge
    gauge = pd.Series(pd.to_datetime(ts)).map(obs).values
    max_diff = np.abs(y_true - gauge).max()
    assert max_diff < 1e-3, f"ground truth deviates from gauge by {max_diff}"
    print(f"truth == observed gauge (max diff {max_diff:.2e}) OK")

    results[label] = metrics(y_true, y_pred)
    uf.export_results_to_csv({label: (y_true, y_pred, ts)}, f"{OUT_DIR}/online_fixed")

# --- Old numbers recomputed from the existing exported CSVs ---
def old_metrics(csv_name):
    path = f"{OLD_DIR}/online_{csv_name}.csv"
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df["observed"] = df["Timestamp"].map(obs)
    reported = metrics(df["True"].values, df["Prediction"].values)  # vs clipped truth
    honest = metrics(df["observed"].values, df["Prediction"].values)  # vs real gauge
    return reported, honest

print("\n" + "=" * 94)
print(f"{'model':<14} | {'old reported (clipped truth)':>28} | {'old honest (real gauge)':>26} | {'FIXED pipeline':>18}")
print("-" * 94)
for label, _ in MODELS:
    reported, honest = old_metrics(label)
    fmt = lambda m: f"RMSE {m[0]:.3f} R2 {m[2]:.3f}" if m else "n/a"
    print(f"{label:<14} | {fmt(reported):>28} | {fmt(honest):>26} | {fmt(results[label]):>18}")
print("=" * 94)
print("All three columns' predictions differ (old runs used the clipped/no-floor datasets);")
print("only the FIXED column is trained AND scored on the corrected pipeline.")
