"""
Phase 6 -- Loss re-weighting experiment (experiment_plan.md).

Tests the predicted fix for the persistence collapse: training loss in RAW units
(scaled-space MSE weighted by scale^2) restores full weight to flood-window
errors, which the per-window normalization implicitly discounts ~270x
(normalization_deep_dive.md section 5).

Grid: loss_space {scaled control, raw treatment} x {FTRL-default, Adam-default}
x seeds [0, 1, 2, 7, 42]. All on the fixed pipeline (scale_floor q0.15,
y_clip=None, honest ground truth, max_grad_norm=1.0).

Per run: aggregate RMSE/MAE/R2, skill vs persistence (aggregate + flood@Q99),
predicted-deviation ratio (the collapse diagnostic; control ~0.04), and event
metrics at Q90/Q99 (rising-limb POD, advance-warning POD + lead, peak bias).

Regression: control FTRL @ seed 42 must reproduce RMSE 4.836 (+-0.01).
Outputs: per-run CSVs + reweighted_summary.csv in results/testing/online_reweighted/.
Run from anywhere: python loss_reweight_experiment.py
"""
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import util_fun as uf
from cstm_models import build_mlp
from cstm_models.ftrl import FTRL

LOOKBACK, HORIZON, N_FEATURES, BATCH_SIZE, H = 48, 24, 4, 1, 3
SEEDS = [0, 1, 2, 7, 42]
OUT = "results/testing/online_reweighted"
os.makedirs(OUT, exist_ok=True)

train_df = pd.read_csv("dataset/one_station_train_data.csv", parse_dates=[0], index_col=0)
test_df = pd.read_csv("dataset/one_station_test_data.csv", parse_dates=[0], index_col=0)
x_train, y_train = uf.create_sequences(train_df.values.astype(np.float32), LOOKBACK, HORIZON, H)
x_test, y_test = uf.create_sequences(test_df.values.astype(np.float32), LOOKBACK, HORIZON, H)

floor = uf.compute_scale_floor(x_train, H, quantile=0.15)
train_ds = uf.RollingNormTimeSeriesDataset(x_train, y_train, H, y_clip=None, scale_floor=floor)
test_ds = uf.RollingNormTimeSeriesDataset(x_test, y_test, H, y_clip=None, scale_floor=floor)

# --- Fixed evaluation-grid quantities ---
n_win = len(x_test)
y_true_grid = y_test.flatten().astype(np.float64)
refs = x_test[:, -1, H].astype(np.float64)
ref_levels = np.repeat(refs, HORIZON)                # persistence value per hour
y_persist = ref_levels.copy()
grid_ts = np.concatenate(
    [test_df.index[i * HORIZON + LOOKBACK: i * HORIZON + LOOKBACK + HORIZON].values
     for i in range(n_win)])

train_h = train_df["H_bar"].values
q90 = float(np.quantile(train_h, 0.90))
q99 = float(np.quantile(train_h, 0.99))
flood = y_true_grid >= q99
mse_persist = float(np.mean((y_true_grid - y_persist) ** 2))
mse_persist_flood = float(np.mean((y_true_grid[flood] - y_persist[flood]) ** 2))

obs_dev = np.abs(y_true_grid - ref_levels)           # observed movement
q90_win = y_test.max(axis=1) >= q90
obs_dev_q90 = np.abs((y_test - refs[:, None]))[q90_win]

OPTIMIZERS = {
    "FTRL-default": lambda m: FTRL(m.parameters(), alpha=0.01, beta=1.0, lambda1=0.0, lambda2=0.0),
    "Adam-default": lambda m: torch.optim.Adam(m.parameters(), lr=1e-3),
}

def evaluate(y_pred):
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = float(np.mean((y_true_grid - y_pred) ** 2))
    mse_f = float(np.mean((y_true_grid[flood] - y_pred[flood]) ** 2))
    dev = np.abs(y_pred - ref_levels)
    dev_win = np.abs(y_pred.reshape(n_win, HORIZON) - refs[:, None])
    out = dict(
        rmse=root_mean_squared_error(y_true_grid, y_pred),
        mae=mean_absolute_error(y_true_grid, y_pred),
        r2=r2_score(y_true_grid, y_pred),
        skill_persist=1 - mse / mse_persist,
        flood_skill_persist=1 - mse_f / mse_persist_flood,
        dev_ratio=float(dev.mean() / obs_dev.mean()),
        dev_ratio_q90win=float(dev_win[q90_win].mean() / obs_dev_q90.mean()),
    )
    for tag, thr in (("q90", q90), ("q99", q99)):
        m = uf.peak_event_metrics(y_true_grid, y_pred, grid_ts, thr, ref_levels, HORIZON)
        out.update({f"{tag}_pod": m["pod"], f"{tag}_rising_pod": m["rising_pod"],
                    f"{tag}_aw_pod": m["aw_pod"], f"{tag}_lead_h": m["mean_lead_h"],
                    f"{tag}_peak_bias": m["peak_bias"], f"{tag}_far": m["far"]})
    return out

rows = []
total = len(["scaled", "raw"]) * len(OPTIMIZERS) * len(SEEDS)
run_i = 0
for loss_space in ["scaled", "raw"]:
    for opt_name, make_opt in OPTIMIZERS.items():
        for seed in SEEDS:
            run_i += 1
            label = f"{loss_space}_{opt_name}_s{seed}"
            print(f"[{run_i}/{total}] {label} ...", flush=True)
            try:
                torch.manual_seed(seed)
                model, _, crit = build_mlp(LOOKBACK * N_FEATURES, HORIZON)
                opt = make_opt(model)
                uf.train_model_online(model, opt, crit, train_ds, train_df,
                                      LOOKBACK, HORIZON, BATCH_SIZE,
                                      silent=True, loss_space=loss_space)
                y_t, y_p, ts = uf.model_evaluate_with_norm(
                    model, crit, test_ds, test_df.index, LOOKBACK, HORIZON, BATCH_SIZE)
                assert np.abs(np.asarray(y_t, dtype=np.float64) - y_true_grid).max() < 1e-3

                r = dict(loss_space=loss_space, optimizer=opt_name, seed=seed)
                r.update(evaluate(y_p))
                rows.append(r)

                if loss_space == "scaled" and opt_name == "FTRL-default" and seed == 42:
                    assert abs(r["rmse"] - 4.836) < 0.01, \
                        f"CONTROL REGRESSION: rmse {r['rmse']:.4f} != 4.836"
                    print("   control regression PASSED")

                uf.export_results_to_csv({label: (y_t, y_p, ts)}, f"{OUT}/reweighted")
            except Exception:
                print(f"   RUN FAILED: {label}")
                traceback.print_exc()

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/reweighted_runs.csv", index=False)
summary = df.groupby(["loss_space", "optimizer"]).agg(["mean", "std"]).round(4)
summary.to_csv(f"{OUT}/reweighted_summary.csv")

print("\n" + "=" * 100)
print("SUMMARY (mean +- std over seeds)")
print("=" * 100)
key_cols = ["rmse", "skill_persist", "flood_skill_persist", "dev_ratio",
            "q99_aw_pod", "q99_peak_bias", "q99_far"]
g = df.groupby(["loss_space", "optimizer"])
hdr = f"{'config':<28}" + "".join(f"{c:>20}" for c in key_cols)
print(hdr)
print("-" * len(hdr))
for (ls, on), sub in g:
    cells = "".join(f"{sub[c].mean():>12.4f}+-{sub[c].std():>5.3f}" for c in key_cols)
    print(f"{ls + ' / ' + on:<28}{cells}")

print("\nVERDICT INPUTS: control dev_ratio ~0.04 = persistence collapse.")
print("Confirmation requires: raw dev_ratio >> 0.04 AND flood_skill_persist > 0 AND q99_aw_pod > 0.")
print(f"Exported: {OUT}/reweighted_runs.csv, {OUT}/reweighted_summary.csv")
