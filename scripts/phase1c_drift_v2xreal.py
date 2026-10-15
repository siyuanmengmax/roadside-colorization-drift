#!/usr/bin/env python3
"""
Phase 1c (C2 cross-validation): real-world drift-law on V2X-Real infra (UCLA smart
intersection). Second real site -- distinct location/country from TUMTraf-I's Munich A9 site.

Same protocol as phase1_drift.py (TUMTraf-I): late-fused colorization on real
LiDAR+camera; inject translational calibration drift and sweep {0,1,2,3,5,7,10,15} cm;
colorization error = mean |drifted_rgb - clean_rgb| over points colored in both.
Reuses frame_curve() from phase1_drift.py verbatim (dataset-agnostic) and fit_stats()
from the simulated T3 sweep (sim_studies/t3), so the fitted curve is directly comparable.
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2xreal_adapter import list_infra_frames, read_bin, load_calib, match_image
from phase1_drift import frame_curve
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim_studies/t3"))
from t3_sweep import fit_stats  # reuse linear+quad fit / R2 / CI / AIC

CM = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
CAM = "cam1"
STRIDE = 25  # 9092 infra frames / 25 ~= 364, matched to TUMTraf-I's n_frames=361
OUT = (Path(__file__).resolve().parents[1] / "outputs")


def main():
    frames = list_infra_frames()[::STRIDE]
    print(f"{len(frames)} frames (V2X-Real infra, {CAM}, stride {STRIDE})")
    accum = {cm: [] for cm in CM}
    used = 0
    for i, bin_path in enumerate(frames):
        img_path = match_image(bin_path, CAM)
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        pts = read_bin(bin_path)[:, :3]
        K, T = load_calib(bin_path, CAM)
        fc = frame_curve(pts, img, K, T, seed=i)
        ok = any(not np.isnan(fc[cm]) for cm in CM)
        if not ok:
            continue
        used += 1
        for cm in CM:
            if not np.isnan(fc[cm]):
                accum[cm].append(fc[cm])
        if used % 40 == 0:
            print(f"  [{used}/{len(frames)}]", flush=True)

    mean = {cm: round(float(np.mean(accum[cm])), 4) for cm in CM}
    std = {cm: round(float(np.std(accum[cm])), 4) for cm in CM}
    fit = fit_stats(CM, [mean[cm] for cm in CM])
    res = {"dataset": "V2X-Real (UCLA infra)", "camera": CAM, "n_frames": used,
           "mean_color_diff": mean, "std_color_diff": std, "fit": fit}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase1c_v2xreal_drift.json").write_text(json.dumps(res, indent=2))
    print("\ncurve:", mean)
    print(f"fit: slope={fit['slope']} R2={fit['R2']} CI={fit['slope_95CI']} "
          f"dAIC(lin-quad)={fit['dAIC_lin_minus_quad']}")
    print("wrote", OUT / "phase1c_v2xreal_drift.json")


if __name__ == "__main__":
    main()
