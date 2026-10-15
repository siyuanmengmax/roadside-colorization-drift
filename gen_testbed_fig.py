#!/usr/bin/env python3
"""Generate the CARLA testbed schematic figure."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_COLOR = {"ne": "#29B6F6", "nw": "#FF7043", "sw": "#FFF176", "se": "#66BB6A"}
SENSOR_LABEL = {"ne": "NE", "nw": "NW", "sw": "SW", "se": "SE"}
SENSOR_POS   = {"ne": (25, 22), "nw": (-25, 22), "sw": (-25, -22), "se": (25, -22)}
SCALE_M = 50   # half-width of scene in metres

fig, ax = plt.subplots(figsize=(5.5, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_aspect("equal")
ax.set_xlim(-SCALE_M, SCALE_M)
ax.set_ylim(-SCALE_M * 1.05, SCALE_M * 0.95)

# Light quadrant fills
for (xs, ys), color in [
    ((-SCALE_M, 0), "#FF7043"),
    ((0, SCALE_M), "#29B6F6"),
    ((-SCALE_M, 0), "#FFF176"),
    ((0, SCALE_M), "#66BB6A"),
]:
    pass  # handled per-corner below

quad_defs = [
    (0, SCALE_M, 0, SCALE_M, "#29B6F6"),   # NE
    (-SCALE_M, 0, 0, SCALE_M, "#FF7043"),   # NW
    (-SCALE_M, 0, -SCALE_M, 0, "#FFF176"),  # SW
    (0, SCALE_M, -SCALE_M, 0, "#66BB6A"),   # SE
]
for x0, x1, y0, y1, c in quad_defs:
    ax.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color=c, alpha=0.07, zorder=0)

# Grid lines (road)
for v in [-2, 2]:
    ax.axvline(v, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(v, color="#888888", linewidth=0.8, linestyle="--", zorder=1)

# Center crosshair
ax.plot(0, 0, "+", color="#666666", markersize=14, markeredgewidth=1.2, zorder=2)

# Coverage circles (schematic 100 m range — shown as radius ~45 m within plot)
RANGE_R = 45
for corner, (cx, cy) in SENSOR_POS.items():
    circ_fill = mpatches.Circle((cx, cy), RANGE_R,
                                 linewidth=0, facecolor=SENSOR_COLOR[corner],
                                 alpha=0.08, zorder=1)
    ax.add_patch(circ_fill)
    circ_edge = mpatches.Circle((cx, cy), RANGE_R,
                                 linewidth=0.9, edgecolor=SENSOR_COLOR[corner],
                                 facecolor="none", linestyle="--", zorder=2)
    ax.add_patch(circ_edge)

# Sensors and cameras
CAM_OFFSETS = [(0, 3), (3, 0), (0, -3), (-3, 0)]
for corner, (cx, cy) in SENSOR_POS.items():
    # LiDAR square
    sq = mpatches.FancyBboxPatch((cx - 1.5, cy - 1.5), 3, 3,
                                  boxstyle="square,pad=0.2",
                                  facecolor=SENSOR_COLOR[corner],
                                  edgecolor="white", linewidth=0.8, zorder=4)
    ax.add_patch(sq)
    # Label
    lx = cx - 7 if cx > 0 else cx + 2
    ax.text(cx - 4 if cx > 0 else cx + 2, cy + 4,
            SENSOR_LABEL[corner], color=SENSOR_COLOR[corner],
            fontsize=11, fontweight="bold", zorder=5)
    # Camera dots
    for dx, dy in CAM_OFFSETS:
        ax.plot(cx + dx, cy + dy, "o", color="white",
                markersize=4, markeredgecolor="#888888",
                markeredgewidth=0.5, zorder=5)

# Scale bar
SB_X, SB_Y = -SCALE_M + 2, -SCALE_M + 4
ax.plot([SB_X, SB_X + 10], [SB_Y, SB_Y], color="black", linewidth=2, zorder=5)
ax.text(SB_X + 5, SB_Y + 2, "10 m", ha="center", fontsize=8, color="black")

# Legend
legend_elems = [
    mpatches.Patch(facecolor="#29B6F6", edgecolor="white", label="LiDAR sensor (OS-1-128)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="#888888", markersize=6, label="RGB camera (×4 per corner)"),
    mpatches.Patch(facecolor="#29B6F6", edgecolor="#29B6F6", alpha=0.2,
                   label="100 m range (schematic)"),
]
ax.legend(handles=legend_elems, loc="lower right", fontsize=7,
          framealpha=0.9, edgecolor="#cccccc")

ax.set_title("CARLA Multi-LiDAR Roadside Testbed (Town10HD)", fontsize=11)
ax.axis("off")

fig.tight_layout(pad=0.5)
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"testbed.{ext}", dpi=300, bbox_inches="tight",
                facecolor="white")
print("Saved testbed.pdf / testbed.png")
