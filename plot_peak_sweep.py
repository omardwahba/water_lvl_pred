"""
Figures for Phase 5.1 — peak threshold sweep (peak_threshold_sweep.py).

Fig 1  results/figures/peak_sweep_summary.png
       (a) events observed vs detected per threshold quantile
       (b) peak-magnitude bias (undershoot) per quantile
Fig 2  results/figures/peak_sweep_timeline.png
       (a) test hydrograph (held-out years 2008 | 2014, broken axis) with Q90
           events shaded detected / missed
       (b) zoom on the largest flood: observed vs predicted, undershoot annotated

Run from anywhere: python plot_peak_sweep.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# palette (validated: #2a78d6 + #008300 pass all light-surface gates)
SURFACE, INK, SEC, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, GREEN, CRIT = "#2a78d6", "#008300", "#d03b3b"

MIN_RUN = 3
sweep = pd.read_csv("results/testing/online_fixed/peak_threshold_sweep.csv")
sweep = sweep[sweep["peaks_observed"] > 0].reset_index(drop=True)   # drop empty Q100
pred_df = pd.read_csv("results/testing/online_fixed/online_fixed_FTRL-default.csv",
                      parse_dates=["Timestamp"]).sort_values("Timestamp")
t = pred_df["Timestamp"].values
y_true, y_pred = pred_df["True"].values, pred_df["Prediction"].values
train_h = pd.read_csv("dataset/one_station_train_data.csv", index_col=0)["H_bar"].values


def find_events(mask, times, min_len=MIN_RUN):
    """True-runs (start, end_exclusive) >= min_len, split at temporal gaps > 1 h
    (test set = years 2008 + 2014 plus short sensor gaps — positional adjacency
    alone would merge events across real time jumps)."""
    gap_after = np.flatnonzero(np.diff(times) > np.timedelta64(1, "h"))
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    out = []
    for s, e in edges.reshape(-1, 2):
        cuts = [g + 1 for g in gap_after if s <= g < e - 1]
        for a, b in zip([s] + cuts, cuts + [e]):
            if b - a >= min_len:
                out.append((a, b))
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


FOOT = ("Event = ≥3 consecutive hours above threshold (split at data gaps) · FTRL-default, fixed pipeline "
        "(no clip, honest truth), seed 42 · held-out test years 2008 & 2014")

# ════════════════════════ Fig 1 — sweep summary ════════════════════════
qlabels = [f"Q{q * 100:g}" for q in sweep["quantile"]]
x = np.arange(len(sweep))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 7.4), sharex=True,
                               gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.28})
fig.patch.set_facecolor(SURFACE)

style(ax1)
ax1.bar(x, sweep["peaks_observed"], width=0.62, color=GRID,
        edgecolor=BASE, linewidth=0.8, label="Missed")
ax1.bar(x, sweep["detected_FTRL-default"], width=0.62, color=BLUE, label="Detected")
for i in [0, 8, 12, 14]:                      # selective % labels: Q50, Q90, Q98, Q99.5
    r = sweep.iloc[i]
    ax1.annotate(f"{r['rate_FTRL-default']:.0%}",
                 (x[i], r["peaks_observed"] + 0.9),
                 ha="center", fontsize=9, color=SEC)
ax1.set_ylabel("flood events", fontsize=10, color=SEC)
ax1.set_title("Peaks observed vs detected across severity thresholds",
              fontsize=12.5, color=INK, loc="left", pad=10)
ax1.legend(frameon=False, fontsize=9.5, labelcolor=SEC, loc="upper right")

style(ax2)
ax2.axhline(0, color=BASE, linewidth=1, linestyle="--")
ax2.plot(x, sweep["peak_bias_FTRL-default"], color=BLUE, linewidth=2,
         marker="o", markersize=6, markerfacecolor=BLUE, markeredgecolor=SURFACE,
         markeredgewidth=1.2)
last = sweep["peak_bias_FTRL-default"].iloc[-1]
ax2.annotate(f"{last:+.1f} raw units", (x[-1], last), xytext=(-12, 6),
             textcoords="offset points", ha="right", fontsize=9.5, color=SEC)
ax2.set_ylabel("peak bias (pred max − obs max)", fontsize=10, color=SEC)
ax2.set_title("Peak-magnitude undershoot grows with severity",
              fontsize=12.5, color=INK, loc="left", pad=10)
ax2.set_xticks(x, qlabels, rotation=45, ha="right")

fig.text(0.005, 0.005, FOOT, fontsize=8, color=MUTED)
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/peak_sweep_summary.png", dpi=200,
            bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)

# ════════════════════════ Fig 2 — timeline demo ════════════════════════
q90 = float(np.quantile(train_h, 0.90))
q995 = float(np.quantile(train_h, 0.995))
ev90 = find_events(y_true >= q90, t)
det90 = [(s, e, y_pred[s:e].max() >= q90) for s, e in ev90]
n_det = sum(1 for *_, d in det90 if d)

fig = plt.figure(figsize=(10.5, 7.6))
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.4, wspace=0.05)
axL = fig.add_subplot(gs[0, 0])
axR = fig.add_subplot(gs[0, 1], sharey=axL)
ax2 = fig.add_subplot(gs[1, :])

pad = np.timedelta64(36, "h")
for ax, (y0, y1) in ((axL, (2008, 2009)), (axR, (2014, 2015))):
    style(ax)
    lo = np.datetime64(f"{y0}-01-01")
    hi = np.datetime64(f"{y1}-01-01")
    m = (t >= lo) & (t < hi)
    ax.plot(t[m], y_true[m], color=SEC, linewidth=0.8)
    for s, e, d in det90:
        if lo <= t[s] < hi:
            ax.axvspan(t[s], max(t[e - 1], t[s] + pad),
                       color=BLUE if d else CRIT, alpha=0.30, linewidth=0)
    ax.axhline(q90, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlim(lo, hi - np.timedelta64(2, "h"))   # avoid duplicate Jan tick at the seam
    ax.xaxis.set_major_locator(mdates.MonthLocator([1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.annotate(str(y0), (0.02, 0.9), xycoords="axes fraction",
                fontsize=11, fontweight="bold", color=SEC)
axR.tick_params(labelleft=False)
axL.annotate(f"Q90 = {q90:.1f}", (np.datetime64("2008-11-01"), q90), xytext=(0, 5),
             textcoords="offset points", fontsize=8.5, color=MUTED)
axL.set_ylabel("H̄ (raw units)", fontsize=10, color=SEC)
axL.set_title(f"Observed level in the held-out years — flood events above Q90: "
              f"{n_det}/{len(ev90)} detected", fontsize=12.5, color=INK, loc="left", pad=10)
handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.30),
           plt.Rectangle((0, 0), 1, 1, color=CRIT, alpha=0.30)]
axR.legend(handles, [f"detected ({n_det})", f"missed ({len(ev90) - n_det})"],
           frameon=False, fontsize=9.5, labelcolor=SEC, loc="upper right")

# zoom: the extreme event with the largest observed peak (July 2008 flood)
ev995 = find_events(y_true >= q995, t)
s, e = max(ev995, key=lambda se: y_true[se[0]:se[1]].max())
lo_i, hi_i = max(0, s - 96), min(len(t), e + 144)
obs_pk, prd_pk = y_true[s:e].max(), y_pred[s:e].max()
pk_i = s + int(np.argmax(y_true[s:e]))

style(ax2)
ax2.plot(t[lo_i:hi_i], y_true[lo_i:hi_i], color=GREEN, linewidth=2, label="Observed")
ax2.plot(t[lo_i:hi_i], y_pred[lo_i:hi_i], color=BLUE, linewidth=2, label="FTRL prediction")
ax2.axhline(q995, color=MUTED, linewidth=1, linestyle="--")
ax2.annotate(f"Q99.5 = {q995:.1f}", (t[lo_i + 8], q995), xytext=(0, 5),
             textcoords="offset points", fontsize=8.5, color=MUTED)
ax2.annotate("", (t[pk_i], prd_pk), xytext=(t[pk_i], obs_pk),
             arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.2))
ax2.annotate(f"undershoot {prd_pk - obs_pk:+.1f}", (t[pk_i], (obs_pk + prd_pk) / 2),
             xytext=(10, 0), textcoords="offset points", fontsize=9.5, color=INK)
ax2.annotate("Observed", (t[hi_i - 1], y_true[hi_i - 1]), xytext=(6, 4),
             textcoords="offset points", fontsize=9, color=GREEN, fontweight="bold")
ax2.annotate("Prediction", (t[hi_i - 1], y_pred[hi_i - 1]), xytext=(6, -12),
             textcoords="offset points", fontsize=9, color=BLUE, fontweight="bold")
ax2.set_title(f"Zoom: largest flood ({pd.Timestamp(t[pk_i]):%B %Y}) — "
              "the model sees the event but undershoots its magnitude",
              fontsize=12.5, color=INK, loc="left", pad=10)
ax2.set_ylabel("H̄ (raw units)", fontsize=10, color=SEC)
ax2.legend(frameon=False, fontsize=9.5, labelcolor=SEC, loc="upper right")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

fig.text(0.005, 0.005, FOOT + " · event spans widened to ≥36 h for visibility",
         fontsize=8, color=MUTED)
fig.savefig("results/figures/peak_sweep_timeline.png", dpi=200,
            bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)

print("saved results/figures/peak_sweep_summary.png")
print("saved results/figures/peak_sweep_timeline.png")
