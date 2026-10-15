#!/usr/bin/env python3
"""Phase 0 calibration sanity: project real LiDAR onto the real camera image and
overlay, for both extrinsic conventions (T and inv T). Whichever lands points on
the scene content is correct. Saves an overlay PNG to eyeball."""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tumtraf_adapter import (list_frames, match_image, load_calib, load_boxes,
                             label_for, read_pcd_ascii, project)

SEQ = "a9_dataset_r02_s01"
CAM = "s110_camera_basler_south2_8mm"
OUT = (Path(__file__).resolve().parents[1] / "outputs")


def main():
    frames = list_frames(SEQ)
    pcd = frames[len(frames) // 2]                       # a middle frame
    lbl = label_for(pcd, SEQ)
    img_path = match_image(pcd, SEQ, CAM)
    pts = read_pcd_ascii(pcd)[:, :3]
    K, T_cam_wrt_lidar = load_calib(lbl, CAM)
    img = cv2.imread(str(img_path)); H, W = img.shape[:2]
    boxes = load_boxes(lbl)
    print(f"frame {pcd.name}\n img {img_path.name} ({W}x{H})  pts={len(pts)}  boxes={len(boxes)}")

    conventions = {"invT (cam_from_lidar=inv(pose))": np.linalg.inv(T_cam_wrt_lidar),
                   "T (pose as cam_from_lidar)": T_cam_wrt_lidar}
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
    # project box centers
    bc = np.array([b["center"] for b in boxes])
    if len(bc):
        bu, bv, bz, bvalid = project(bc, K, Tcl, W, H)
        ax.scatter(bu[bvalid], bv[bvalid], marker="s", s=80, facecolors="none",
                   edgecolors="lime", linewidths=2, label="GT box centers")
        ax.legend(loc="upper right")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(f"TUMTraf-I calib sanity — {SEQ} south LiDAR -> {CAM}\n{name}, inlier {frac:.2f}")
    ax.axis("off")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "phase0_calib_overlay.png", dpi=90, bbox_inches="tight")
    print("wrote", OUT / "phase0_calib_overlay.png")


if __name__ == "__main__":
    main()