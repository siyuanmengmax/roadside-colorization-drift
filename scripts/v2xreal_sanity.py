#!/usr/bin/env python3
"""V2X-Real calibration sanity: project real infra LiDAR onto the real infra camera
image and overlay, testing both extrinsic conventions (T and inv T) same as
phase0_sanity.py did for TUMTraf-I. Whichever lands points on the scene content is
correct. Saves an overlay PNG to eyeball."""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tumtraf_adapter import project
from v2xreal_adapter import list_infra_frames, read_bin, load_calib, match_image

CAM = "cam1"
OUT = (Path(__file__).resolve().parents[1] / "outputs")


def main():
    frames = list_infra_frames()
    bin_path = frames[len(frames) // 2]  # a middle frame
    img_path = match_image(bin_path, CAM)
    pts = read_bin(bin_path)[:, :3]
    K, T_extrinsic = load_calib(bin_path, CAM)
    img = cv2.imread(str(img_path)); H, W = img.shape[:2]
    print(f"frame {bin_path.relative_to(bin_path.parents[3])}\n img {img_path.name} ({W}x{H})  pts={len(pts)}")

    conventions = {"T (extrinsic as cam_from_lidar)": T_extrinsic,
                   "invT (extrinsic as lidar_from_cam)": np.linalg.inv(T_extrinsic)}
    best = None
    for name, Tcl in conventions.items():
        u, v, z, valid = project(pts, K, Tcl, W, H)
        frac = valid.mean()
        print(f"  {name}: inlier frac = {frac:.3f}  ({valid.sum()} pts)")
        if best is None or frac > best[1]:
            best = (name, frac, Tcl, (u, v, z, valid))

    name, frac, Tcl, (u, v, z, valid) = best
    print(f"=> using convention: {name}")

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(img[:, :, ::-1])
    sc = ax.scatter(u[valid], v[valid], c=np.clip(z[valid], 0, 80), s=2, cmap="jet", alpha=0.6)
    plt.colorbar(sc, ax=ax, label="depth (m)", fraction=0.03)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(f"V2X-Real calib sanity — infra {CAM}\n{name}, inlier {frac:.2f}")
    ax.axis("off")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "v2xreal_calib_overlay.png", dpi=90, bbox_inches="tight")
    print("wrote", OUT / "v2xreal_calib_overlay.png")


if __name__ == "__main__":
    main()
