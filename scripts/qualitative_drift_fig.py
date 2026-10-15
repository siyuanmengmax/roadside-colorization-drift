#!/usr/bin/env python3
"""
Qualitative colorization-drift figure on REAL TUMTraf-I data, pure translational drift only
(same drift model as Table driftlaw / phase1_drift.py -- no coupled rotation, unlike the
earlier simulation-only figure). Three panels on a representative frame:
  (a) born-fused reference colorization (clean extrinsic)
  (b) late-fused, mild drift (3 cm) -- per-point |RGB| error heatmap
  (c) late-fused, severe drift (15 cm) -- per-point |RGB| error heatmap
Numbers in the caption are computed live from this exact frame, so they never drift out of
sync with what's plotted.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tumtraf_adapter import list_frames, match_image, load_calib, label_for, read_pcd_ascii, colorize, project

SEQ = "a9_dataset_r02_s01"
CAM = "s110_camera_basler_south2_8mm"
DRIFT_CM = [3.0, 15.0]
SEED = 0
OUT = (Path(__file__).resolve().parents[1] / "outputs")


def pick_frame():
    for pcd in list_frames(SEQ):
        lbl = label_for(pcd, SEQ)
        if lbl.exists():
            return pcd, lbl
    raise RuntimeError("no labeled frame found")


def main():
    pcd, lbl = pick_frame()
    img = cv2.imread(str(match_image(pcd, SEQ, CAM)))
    pts = read_pcd_ascii(pcd)[:, :3]
    K, T_clean = load_calib(lbl, CAM)
    H, W = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    u, v, z, valid = project(pts, K, T_clean, W, H)
    clean_rgb, clean_valid = colorize(pts, img, K, T_clean)

    rng = np.random.RandomState(SEED)
    dir3 = rng.normal(0, 1, 3); dir3 /= (np.linalg.norm(dir3) + 1e-9)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    dimmed = (img_rgb.astype(np.float32) * 0.35 + 255 * 0.65 * 0.0).astype(np.uint8)  # darken for contrast
    dimmed = (img_rgb.astype(np.float32) * 0.35).astype(np.uint8)

    # panel (a): born-fused reference
    ax = axes[0]
    ax.imshow(dimmed)
    m = clean_valid
    ax.scatter(u[m], v[m], c=clean_rgb[m].astype(np.float32) / 255.0, s=2.5, marker='.', linewidths=0)
    ax.set_title("(a) Born-fused reference\n(zero drift)")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

    mean_errs = []
    vmax = 0
    diffs_cache = []
    for cm in DRIFT_CM:
        Td = T_clean.copy()
        Td[:3, 3] = Td[:3, 3] + dir3 * (cm / 100.0)
        drgb, dvalid = colorize(pts, img, K, Td)
        both = clean_valid & dvalid
        diff = np.abs(drgb[both].astype(np.float32) - clean_rgb[both].astype(np.float32)).mean(axis=1)
        diffs_cache.append((both, diff))
        mean_errs.append(float(diff.mean()))
        vmax = max(vmax, np.percentile(diff, 95))

    for i, (cm, (both, diff)) in enumerate(zip(DRIFT_CM, diffs_cache)):
        ax = axes[i + 1]
        ax.imshow(dimmed)
        sc = ax.scatter(u[both], v[both], c=diff, cmap='viridis', vmin=0, vmax=vmax, s=2.5,
                        marker='.', linewidths=0)
        tag = "mild" if cm == min(DRIFT_CM) else "severe"
        ax.set_title(f"({'b' if i == 0 else 'c'}) Late-fused, {tag} drift ({cm:.0f}\\,cm)\n"
                     f"mean $|\\Delta\\mathrm{{RGB}}|={mean_errs[i]:.1f}$")
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

    cbar = fig.colorbar(sc, ax=axes[1:], fraction=0.025, pad=0.01)
    cbar.set_label("per-point $|\\Delta\\mathrm{RGB}|$ (0-255 scale)")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "qualitative_drift.pdf", bbox_inches='tight', dpi=200)
    fig.savefig(OUT / "qualitative_drift.png", bbox_inches='tight', dpi=200)
    print("frame:", pcd.name)
    print("mean errors (RGB units) at", DRIFT_CM, "cm:", [round(e, 2) for e in mean_errs])
    print("wrote", OUT / "qualitative_drift.pdf")


if __name__ == "__main__":
    main()
