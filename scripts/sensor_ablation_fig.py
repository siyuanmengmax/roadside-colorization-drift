#!/usr/bin/env python3
"""Sensor-count ablation figure (vehicle F1 with block-bootstrap 95% CIs vs sensor count, label-free protocol)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = (Path(__file__).resolve().parents[1] / "outputs")
data = json.loads((OUT / "sensor_ablation_prf.json").read_text())

CONDS = ["clear", "rain_med", "fog", "night"]
LABELS = {"clear": "Clear", "rain_med": "Rain", "fog": "Fog", "night": "Night"}
CONFIGS = ["1", "2", "4"]
X = [1, 2, 4]
COLORS = {"clear": "#4C72B0", "rain_med": "#55A868", "fog": "#8172B2", "night": "#C44E52"}
MARKERS = {"clear": "o", "rain_med": "s", "fog": "^", "night": "D"}

fig, ax = plt.subplots(figsize=(5.2, 4))
for cond in CONDS:
    v = [data[cond][c]["vehicle"] for c in CONFIGS]
    y = [d["F1"] for d in v]
    err = [[d["F1"] - d["F1_CI95"][0] for d in v], [d["F1_CI95"][1] - d["F1"] for d in v]]
    xo = [x + 0.06 * (CONDS.index(cond) - 1.5) for x in X]
    ax.errorbar(xo, y, yerr=err, marker=MARKERS[cond], color=COLORS[cond], label=LABELS[cond],
                linewidth=1.8, markersize=7, capsize=3)

ax.set_xticks(X)
ax.set_xlabel("Number of LiDAR sensors")
ax.set_ylabel(r"Vehicle $F_1$ (label-free, 3 m BEV)")
ax.set_xlim(0.7, 4.3)
ax.set_ylim(0, 0.6)
ax.legend(frameon=False, loc="upper left")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "sensor_ablation.pdf", bbox_inches='tight')
fig.savefig(OUT / "sensor_ablation.png", bbox_inches='tight', dpi=200)
print("wrote", OUT / "sensor_ablation.pdf")
