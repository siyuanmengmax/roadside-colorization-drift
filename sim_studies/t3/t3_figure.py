#!/usr/bin/env python3
"""T3 figure for the revision: colorization error vs translational drift,
4 condition lines with linear fit + std bands. Reads t3_sweep_<tag>.json."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VISTA = Path(__file__).resolve().parents[2]
tag = sys.argv[1] if len(sys.argv) > 1 else "transonly"
d = json.loads((VISTA / f"outputs/sim_studies/t3/t3_sweep_{tag}.json").read_text())
CM = [0, 1, 2, 3, 5, 7, 10, 15]
colors = {"clear": "#1f77b4", "rain_med": "#2ca02c", "fog": "#7f4fbf", "night": "#333333"}

fig, ax = plt.subplots(figsize=(6, 4.2))
for cond, r in d.items():
    y = [r["mean_color_diff"][str(float(c))] for c in CM]
    e = [r["std_color_diff"][str(float(c))] for c in CM]
    c = colors.get(cond, None)
    ax.errorbar(CM, y, yerr=e, fmt="o", ms=4, color=c, capsize=2, alpha=0.9,
                label=f"{cond} (R²={r['fit']['R2']:.2f}, slope={r['fit']['slope']:.2f})")
    xs = np.linspace(0, 15, 50)
    ax.plot(xs, r["fit"]["slope"] * xs + r["fit"]["intercept"], "-", color=c, lw=1, alpha=0.6)
ax.set_xlabel("Translational calibration drift (cm)")
ax.set_ylabel("Mean colorization error (RGB units/point)")
ax.set_title("Colorization error vs translational drift (8 magnitudes, 200 frames/cond)")
ax.legend(fontsize=7, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
outp = VISTA / "outputs/sim_studies/t3/t3_drift_sweep.pdf"
fig.savefig(outp); fig.savefig(outp.with_suffix(".png"), dpi=110)
print("wrote", outp)
