"""
Diagnostic: size the scale floor from TRAIN data.

Evidence behind the fixed-pipeline choices (scale_floor = q0.15, y_clip = 500):
- sweeps floor candidates and reports binding at clip=10 plus max/p99.9 |y_scaled|
- separates two populations: degenerate windows (std ~ 0, fixed by the floor) vs
  healthy windows whose GENUINE moves exceed 10 sigma (flood onsets — must never
  be clipped, which is why the tripwire sits at 500, not 10)
- floors >= 0.2 start inflating healthy-window denominators (distorting real
  dynamics), so bigger is not better

Train-only analysis — the test set plays no role in choosing the floor.
Run from anywhere: python diagnostics/floor_sweep.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import util_fun as uf

LOOKBACK, HORIZON, H = 48, 24, 3

train = np.loadtxt("dataset/one_station_train_data.csv", delimiter=",", skiprows=1,
                   usecols=(1, 2, 3, 4), dtype=np.float32)
x, y = uf.create_sequences(train, LOOKBACK, HORIZON, H)
stds = x[:, :, H].std(axis=1, ddof=1)   # ddof=1 matches torch.std
ref = x[:, -1, H]
move = np.abs(y - ref[:, None])         # raw |future - now|

print("train window-std quantiles:")
for q in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    print(f"  q{q:.2f}: {np.quantile(stds, q):.4f}")
print(f"  fraction of windows with std < 0.05: {(stds < 0.05).mean():.2%}")

print("\nfloor sweep (train, binding measured at clip=10):")
print(f"{'floor':>8} {'bind%':>8} {'bind% healthy-only*':>20} {'max|y_s|':>10} {'p99.9':>8}")
healthy = stds >= 0.1
for floor in [None, 0.006, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    s = stds if floor is None else np.maximum(stds, floor)
    ys = move / (s[:, None] + 1e-6)
    print(f"{str(floor):>8} {(ys > 10).mean():8.3%} {(ys[healthy] > 10).mean():20.3%} "
          f"{ys.max():10.1f} {np.quantile(ys, 0.999):8.1f}")
print("* windows with std >= 0.1 — their binding is genuine dynamics, not degeneracy;")
print("  a well-sized floor leaves this column at its no-floor value (~1.0%).")

s = np.maximum(stds, 0.15)
ys = move / (s[:, None] + 1e-6)
mask = ys > 10
q99 = np.quantile(train[:, H], 0.99)
print(f"\nwith floor=0.15: {mask.mean():.3%} of elements exceed 10-sigma — GENUINE moves:")
print(f"  their raw levels: mean {y[mask].mean():.1f}, {(y[mask] >= q99).mean():.1%} >= train Q99 ({q99:.1f})")
print(f"  their raw |moves|: mean {move[mask].mean():.2f}, max {move[mask].max():.2f}")
print(f"  max |y_scaled| overall: {ys.max():.1f}  -> tripwire at 500 never binds on real data")
