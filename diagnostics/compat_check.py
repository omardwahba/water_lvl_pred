"""
Regression check: the fixed dataset class must reproduce legacy outputs exactly
when scale_floor=None and y_clip=5.0 (only the added y_raw element differs).

Guards the backward-compatibility promise of the 2026-07-11 pipeline fix: old
notebook cells re-run on the new util_fun.py produce the same tensors as before.

Run from anywhere: python diagnostics/compat_check.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader
import util_fun as uf

LOOKBACK, HORIZON, H = 48, 24, 3

test_vals = np.loadtxt("dataset/one_station_test_data.csv", delimiter=",", skiprows=1,
                       usecols=(1, 2, 3, 4), dtype=np.float32)
x, y = uf.create_sequences(test_vals, LOOKBACK, HORIZON, H)
ds = uf.RollingNormTimeSeriesDataset(x, y, H, y_clip=5.0, scale_floor=None)


def legacy_item(idx):
    """The pre-fix __getitem__ logic, replicated verbatim."""
    x_raw = torch.tensor(x[idx])
    y_raw = torch.tensor(y[idx])
    mn, _ = torch.min(x_raw, dim=0)
    mx, _ = torch.max(x_raw, dim=0)
    rng = mx - mn + 1e-6
    xs = (x_raw - mn) / rng
    hb = x_raw[:, H]
    ref = hb[-1]
    scale = hb.std() + 1e-6
    xs[:, H] = (hb - ref) / scale
    ys = torch.clamp((y_raw - ref) / scale, -5.0, 5.0)
    return xs, ys, ref, scale


rng_idx = np.random.default_rng(0).choice(len(ds), 50, replace=False)
for i in rng_idx:
    xs_n, ys_n, y_raw_n, ref_n, scale_n = ds[int(i)]
    xs_o, ys_o, ref_o, scale_o = legacy_item(int(i))
    assert torch.equal(xs_n, xs_o), f"x mismatch at {i}"
    assert torch.equal(ys_n, ys_o), f"y mismatch at {i}"
    assert torch.equal(ref_n, ref_o) and torch.equal(scale_n, scale_o), f"ref/scale mismatch at {i}"
    assert torch.equal(y_raw_n, torch.tensor(y[int(i)])), f"y_raw mismatch at {i}"
print(f"backward-compat OK: {len(rng_idx)} random samples identical to legacy logic (plus y_raw)")

b = next(iter(DataLoader(ds, batch_size=4)))
print("DataLoader 5-tuple batch shapes:", [tuple(t.shape) for t in b])
