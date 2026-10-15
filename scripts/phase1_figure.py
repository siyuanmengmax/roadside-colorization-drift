#!/usr/bin/env python3
"""Drift-law figures from the matched-protocol sweep (drift_sweep_matched.json):
  phase1_sim_vs_real_drift.pdf  error vs drift, CARLA x4 + TUMTraf-I + V2X-Real + DAIR-V2X-I, with fits of
                                E_inf*d/(d+d_half) from drift_model_compare.json
  town05_vs_town10hd_drift.pdf  Town10HD vs Town05 (clear)
  pixel_axis.pdf                (a) per-point error vs per-point projection shift (pooled over drift levels);
                                (b) 5-unit budget vs per-point error at one pixel, every site
DAIR-V2X-I groups come from drift_sweep_dair.json / drift_model_compare_dair.json.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V = Path(__file__).resolve().parents[1]
SW = json.loads((V / "outputs/drift_sweep_matched.json").read_text())
FITS = json.loads((V / "outputs/drift_model_compare.json").read_text())
SW.update(json.loads((V / "outputs/drift_sweep_dair.json").read_text()))
FITS.update(json.loads((V / "outputs/drift_model_compare_dair.json").read_text()))
DAIR = [("real_dair_W", "DAIR-V2X-I W", "teal"), ("real_dair_Y1", "DAIR-V2X-I Y1", "darkgoldenrod"),
        ("real_dair_Y2", "DAIR-V2X-I Y2", "goldenrod"), ("real_dair_Y3", "DAIR-V2X-I Y3", "olive"),
        ("real_dair_Ylow", "DAIR-V2X-I Y, low light", "grey")]
D = np.linspace(0, 15, 200)
YL = "Mean colorization error (8-bit units per channel)"


def draw(ax, key, label, color, lw=1.2, ms=3.5, err=False, z=3, ls="-", mk="o"):
    s, h = SW[key], FITS[key]["hyperbolic"]
    ax.plot(D, h["E_inf"] * D / (D + h["d_half_cm"]), ls, color=color, lw=lw, alpha=0.85, zorder=z)
    lab = f"{label} ($d_{{1/2}}$={h['d_half_cm']:.1f} cm)"
    if err:
        ax.errorbar(s["nominal_cm"], s["mean_E"], yerr=s["std_E"], fmt="o", color=color, ms=ms, capsize=2, zorder=z + 1, label=lab)
    else:
        ax.plot(s["nominal_cm"], s["mean_E"], mk, color=color, ms=ms, zorder=z + 1, label=lab)


fig, ax = plt.subplots(figsize=(6.4, 5.2))
for key, name, col in [("sim_clear", "CARLA clear", "tab:blue"), ("sim_rain_med", "CARLA rain", "tab:orange"),
                       ("sim_fog", "CARLA fog", "tab:green"), ("sim_night", "CARLA night", "tab:purple")]:
    draw(ax, key, name, col)
draw(ax, "real_tumtraf_day", "TUMTraf-I, day", "crimson", lw=2.2, ms=5, z=5)
draw(ax, "real_tumtraf_s04", "TUMTraf-I, rain at night", "lightcoral", lw=2.2, ms=5, z=5)
draw(ax, "real_v2xreal", "V2X-Real", "black", lw=2.2, ms=5, z=5)
for key, name, col in DAIR:
    draw(ax, key, name, col, lw=1.6, ms=4, z=4, ls="--", mk="s")
ax.set_xlabel("Translational calibration drift (cm)"); ax.set_ylabel(YL)
ax.grid(alpha=0.25); ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(V / f"outputs/phase1_sim_vs_real_drift.{ext}", dpi=120)

fig, ax = plt.subplots(figsize=(6.0, 4.2))
draw(ax, "sim_clear", "Town10HD (original)", "steelblue", lw=1.8, ms=5, err=True)
draw(ax, "sim_town05_clear", "Town05 (second intersection)", "darkorange", lw=1.8, ms=5, err=True)
ax.set_xlabel("Translational calibration drift (cm)"); ax.set_ylabel(YL)
ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(V / f"outputs/town05_vs_town10hd_drift.{ext}", dpi=120)

fig, (ax, bx) = plt.subplots(2, 1, figsize=(4.4, 6.4), gridspec_kw={"height_ratios": [1.1, 1]})
for key, name, col, ls in [("sim_clear", "CARLA Town10HD clear", "tab:blue", "-"), ("sim_town05_clear", "CARLA Town05 clear", "tab:cyan", "-"),
                           ("real_tumtraf_day", "TUMTraf-I, day", "crimson", "-"), ("real_tumtraf_s04", "TUMTraf-I, rain at night", "lightcoral", "-"),
                           ("real_v2xreal", "V2X-Real", "black", "-"),
                           ("real_dair_W", "DAIR-V2X-I W", "teal", "--"), ("real_dair_Y1", "DAIR-V2X-I Y1", "darkgoldenrod", "--"),
                           ("real_dair_Ylow", "DAIR-V2X-I Y, low light", "grey", "--")]:
    s = SW[key]; e = np.array(s["bins_px_edges"]); n = np.array(s["bins_n"])
    mid = np.sqrt(np.maximum(e[:-1], 1e-9) * e[1:])
    ok = (n > 2000) & (e[:-1] >= 0.2) & (e[1:] <= 25)
    ax.plot(mid[ok], np.array(s["bins_mean_E"], float)[ok], ls, marker="o", ms=2.5, color=col, label=name)
ax.set_xscale("log"); ax.set_xlabel("Per-point projection displacement (pixels)"); ax.set_ylabel("Mean colorization error")
ax.set_title("(a)", fontsize=9, loc="left"); ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=7, loc="upper left")
# (b) E(1 cm) predicted from one zero-drift frame (budget_from_single_frame.py), leave-one-group-out:
# marker = mean prediction over the group's frames, horizontal bar = 10th-90th percentile of single-frame
# predictions, y = measured mean error at 1 cm
Z = np.load(V / "outputs/budget_from_single_frame_frames.npz", allow_pickle=True)
zg, zx, zE1 = Z["group"], Z["G"] * Z["px_avg"], Z["E"][:, 1]
LAB = {"real_tumtraf_day": ("TUMTraf-I day", "crimson", "o"), "real_tumtraf_s04": ("TUMTraf-I rain/night", "lightcoral", "o"),
       "real_v2xreal": ("V2X-Real", "black", "o")}
LAB.update({k: (n.replace("DAIR-V2X-I ", ""), c, "s") for k, n, c in DAIR})
OFF = {"real_dair_W": (-18, 4), "real_dair_Y1": (-2, 7), "real_dair_Y2": (8, 1), "real_dair_Y3": (5, -12),
       "real_tumtraf_s04": (4, -12), "real_v2xreal": (-10, 7)}
for key in dict.fromkeys(zg.tolist()):
    m = zg == key
    k_ = float((zx[~m] * zE1[~m]).sum() / (zx[~m] ** 2).sum())
    pr = k_ * zx[m]; me = float(zE1[m].mean()); lo, hi = np.percentile(pr, [10, 90])
    name, col, mk = LAB[key]
    bx.errorbar(pr.mean(), me, xerr=[[pr.mean() - lo], [hi - pr.mean()]], fmt=mk, color=col, ms=6, capsize=2, lw=0.9)
    bx.annotate(name, (pr.mean(), me), fontsize=7, xytext=OFF.get(key, (5, 3)), textcoords="offset points")
lim = [0.3, 6.5]
bx.plot(lim, lim, "-", color="grey", lw=0.8); bx.fill_between(lim, [0.8 * x for x in lim], [1.2 * x for x in lim], color="grey", alpha=0.12, lw=0)
bx.set_xscale("log"); bx.set_yscale("log"); bx.set_xlim(lim); bx.set_ylim(lim)
from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
for a_ in (bx.xaxis, bx.yaxis):
    a_.set_major_locator(FixedLocator([0.5, 1, 2, 4])); a_.set_minor_locator(NullLocator()); a_.set_major_formatter(FormatStrFormatter("%g"))
bx.set_xlabel("Predicted error at 1 cm from one zero-drift frame"); bx.set_ylabel("Measured error at 1 cm")
bx.set_title("(b)", fontsize=9, loc="left"); bx.grid(alpha=0.25, which="both")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(V / f"outputs/pixel_axis.{ext}", dpi=120)
print("wrote 3 figures")
