#!/usr/bin/env python3
"""
Phase 1 (C2): real-world drift-law on TUMTraf-I (south2 camera).

Late-fused colorization on real Ouster+Basler; inject translational calibration drift on
the REAL extrinsic and sweep {0,1,2,3,5,7,10,15} cm. Colorization error = mean |drifted_rgb -
clean_rgb| over points colored in both. Same protocol as CARLA T3 (same seed/frame -> drift
direction fixed, magnitude scales). Tests whether the (saturating) sim law holds on real data.
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tumtraf_adapter import (list_frames, match_image, load_calib, label_for,
                             read_pcd_ascii, colorize, ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim_studies/t3"))
from t3_sweep import fit_stats   # reuse linear+quad fit / R2 / CI / AIC

CM = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
SEQS = ["a9_dataset_r02_s01", "a9_dataset_r02_s02", "a9_dataset_r02_s03", "a9_dataset_r02_s04"]
CAM = "s110_camera_basler_south2_8mm"
STRIDE = 6
OUT = (Path(__file__).resolve().parents[1] / "outputs")


def frame_curve(pts, img, K, T_clean, seed):
    clean_rgb, clean_valid = colorize(pts, img, K, T_clean)
    rng = np.random.RandomState(seed)
    dir3 = rng.normal(0, 1, 3); dir3 /= (np.linalg.norm(dir3) + 1e-9)   # fixed unit drift dir
    out = {}
    for cm in CM:
        Td = T_clean.copy()
        Td[:3, 3] = Td[:3, 3] + dir3 * (cm / 100.0)                     # translational drift
        drgb, dvalid = colorize(pts, img, K, Td)
        both = clean_valid & dvalid
        if both.sum() < 20:
            out[cm] = np.nan
        else:
            out[cm] = float(np.abs(drgb[both].astype(float) - clean_rgb[both].astype(float)).mean())
    return out


def main():
    frames = []
    for s in SEQS:
        frames += list_frames(s)[::STRIDE]
    print(f"{len(frames)} frames (south LiDAR, {CAM}, stride {STRIDE})")
    accum = {cm: [] for cm in CM}
    for i, pcd in enumerate(frames):
        seq = pcd.parts[pcd.parts.index("tumtraf_i") + 1]
        lbl = label_for(pcd, seq)
        if not lbl.exists():
            continue
        img = cv2.imread(str(match_image(pcd, seq, CAM)))
        pts = read_pcd_ascii(pcd)[:, :3]
        K, T = load_calib(lbl, CAM)
        fc = frame_curve(pts, img, K, T, seed=i)
        for cm in CM:
            if not np.isnan(fc[cm]):
                accum[cm].append(fc[cm])
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(frames)}]", flush=True)

    mean = {cm: round(float(np.mean(accum[cm])), 4) for cm in CM}
    std = {cm: round(float(np.std(accum[cm])), 4) for cm in CM}
    fit = fit_stats(CM, [mean[cm] for cm in CM])
    res = {"dataset": "TUMTraf-I", "camera": CAM, "n_frames": len(accum[CM[0]]),
           "mean_color_diff": mean, "std_color_diff": std, "fit": fit}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase1_real_drift.json").write_text(json.dumps(res, indent=2))
    print("\ncurve:", mean)
    print(f"fit: slope={fit['slope']} R2={fit['R2']} CI={fit['slope_95CI']} "
          f"dAIC(lin-quad)={fit['dAIC_lin_minus_quad']}")
    print("wrote", OUT / "phase1_real_drift.json")


if __name__ == "__main__":
    main()