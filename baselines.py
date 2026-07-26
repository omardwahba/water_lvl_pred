"""
Zero-learning baselines + skill scores (Experiment 1 / plan step 1).

Baselines:
  - Persistence: y_hat(t+h) = H_bar(t) for h = 1..24 ("level stays where it is").
    The RollingNorm target is (future - current)/scale, so y_scaled = 0 IS
    persistence -- skill vs persistence measures exactly what learning added.
  - Climatology: mean TRAIN-years level per (day-of-year, hour); falls back to
    the day-of-year mean, then the global train mean, for combos missing due to
    train gaps.

Skill score: SS = 1 - MSE_model / MSE_baseline (1 = perfect, 0 = no better,
< 0 = worse than the baseline). The R2 reported everywhere equals NSE
(Nash-Sutcliffe Efficiency): the skill score vs the predict-the-test-mean baseline.

Safety: asserts model CSVs align with the reconstructed grid and reproduces the
known headline numbers (regression net). Reads existing artifacts, writes only
results/testing/baselines_skill.csv. Run from anywhere: python baselines.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import util_fun as uf

LOOKBACK, HORIZON, H_BAR_INDEX = 48, 24, 3
MODEL_DIRS = {  # dir -> filename prefix
    "results/testing/online_fixed": "online_fixed_",
    "results/testing/online_reweighted": "reweighted_",
}
KNOWN = {  # regression net: name -> (rmse, r2), tolerance 0.01
    "Persistence": (4.881, 0.8197),
    "FTRL-default": (4.836, 0.8231),
}

train_df = pd.read_csv("dataset/one_station_train_data.csv", parse_dates=[0], index_col=0)
test_df = pd.read_csv("dataset/one_station_test_data.csv", parse_dates=[0], index_col=0)
vals = test_df.values.astype(np.float32)
x_test, y_test = uf.create_sequences(vals, LOOKBACK, HORIZON, H_BAR_INDEX)

# --- Ground truth + timestamps on the exact evaluation grid ---
n_win = len(x_test)
y_true = y_test.flatten().astype(np.float64)
ts = pd.DatetimeIndex(np.concatenate(
    [test_df.index[i * HORIZON + LOOKBACK: i * HORIZON + LOOKBACK + HORIZON] for i in range(n_win)]))

# --- Baseline 1: persistence ---
refs = x_test[:, -1, H_BAR_INDEX]
y_persist = np.repeat(refs[:, None], HORIZON, axis=1).flatten().astype(np.float64)

# --- Baseline 2: climatology (train years only) ---
clim = train_df.groupby([train_df.index.dayofyear, train_df.index.hour])["H_bar"].mean()
clim_doy = train_df.groupby(train_df.index.dayofyear)["H_bar"].mean()
y_clim = clim.reindex(list(zip(ts.dayofyear, ts.hour))).values
missing = np.isnan(y_clim)
if missing.any():
    y_clim[missing] = clim_doy.reindex(ts.dayofyear[missing]).values
y_clim = np.where(np.isnan(y_clim), train_df["H_bar"].mean(), y_clim)

def metrics(y_hat):
    return (root_mean_squared_error(y_true, y_hat),
            mean_absolute_error(y_true, y_hat),
            r2_score(y_true, y_hat))

mse = lambda y_hat: float(np.mean((y_true - y_hat) ** 2))
mse_persist, mse_clim = mse(y_persist), mse(y_clim)

q99 = float(np.quantile(train_df["H_bar"].values, 0.99))
flood = y_true >= q99
mse_p_flood = float(np.mean((y_true[flood] - y_persist[flood]) ** 2))

def flood_split(y_hat):
    rc = root_mean_squared_error(y_true[~flood], y_hat[~flood])
    rf = root_mean_squared_error(y_true[flood], y_hat[flood])
    sf = 1 - float(np.mean((y_true[flood] - y_hat[flood]) ** 2)) / mse_p_flood
    return rc, rf, sf

rows = []
def add_row(name, y_hat, source):
    rmse, mae, r2 = metrics(y_hat)
    sp = 1 - mse(y_hat) / mse_persist
    sc = 1 - mse(y_hat) / mse_clim
    rc, rf, sf = flood_split(y_hat)
    rows.append(dict(predictor=name, rmse=round(rmse, 4), mae=round(mae, 4),
                     r2_nse=round(r2, 4), skill_vs_persistence=round(sp, 4),
                     skill_vs_climatology=round(sc, 4), rmse_calm=round(rc, 4),
                     rmse_flood=round(rf, 4), flood_skill_vs_persistence=round(sf, 4),
                     source=source))
    if name in KNOWN:
        kr, k2 = KNOWN[name]
        assert abs(rmse - kr) < 0.01 and abs(r2 - k2) < 0.01, \
            f"REGRESSION: {name} rmse {rmse:.3f} r2 {r2:.4f} != known {kr}/{k2}"

add_row("Persistence", y_persist, "computed")
add_row("Climatology", y_clim, "computed")

# --- Models: every exported CSV in the known dirs, alignment-asserted ---
for d, prefix in MODEL_DIRS.items():
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if not (f.startswith(prefix) and f.endswith(".csv")):
            continue
        df = pd.read_csv(os.path.join(d, f))
        if not {"Timestamp", "True", "Prediction"} <= set(df.columns):
            continue          # metrics/summary file, not a prediction export
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp")
        if len(df) != len(y_true):
            print(f"skip {f}: {len(df)} rows != grid {len(y_true)}")
            continue
        assert (df["Timestamp"].values == ts.values).all(), f"{f}: timestamp grid mismatch"
        assert np.abs(df["True"].values - y_true).max() < 1e-3, f"{f}: truth != gauge"
        add_row(f[len(prefix):-4], df["Prediction"].values.astype(np.float64), d)

out = pd.DataFrame(rows)
os.makedirs("results/testing", exist_ok=True)
out.to_csv("results/testing/baselines_skill.csv", index=False)

print(f"\nflood threshold = train Q99 = {q99:.2f} ({flood.mean():.2%} of test hours)")
print(f"\n{'predictor':<22} {'RMSE':>7} {'R2/NSE':>8} {'sk-persist':>11} {'sk-clim':>8} "
      f"{'RMSE flood':>11} {'flood-skill':>12}")
print("-" * 84)
for r in rows:
    print(f"{r['predictor']:<22} {r['rmse']:>7.3f} {r['r2_nse']:>8.4f} "
          f"{r['skill_vs_persistence']:>11.4f} {r['skill_vs_climatology']:>8.4f} "
          f"{r['rmse_flood']:>11.3f} {r['flood_skill_vs_persistence']:>12.4f}")
print("\nAll alignment + regression assertions passed.")
print("Exported: results/testing/baselines_skill.csv")
