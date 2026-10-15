#!/usr/bin/env python3
"""
T3: extended calibration-drift sweep (mDsR-W3). Extend colorization-error vs
translational-drift from 3 to 8 magnitudes and test linearity.

Magnitudes: {0,1,2,3,5,7,10,15} cm, rotation coupled at 0.2 deg/cm (reproduces the
paper's 5 cm / 1.0 deg anchor; 0 cm == born-fused -> intercept ~0). Same seed per
frame/corner across magnitudes => drift DIRECTION fixed, magnitude scales linearly
(controlled sweep). Metric = mean |late_rgb - born_rgb| over points colored in both
(identical to Exp. 4). 200 frames/condition (stride 10).

Fits per condition: linear + quadratic, reports slope, intercept, R^2, 95% CI on
slope, and Delta AIC/BIC (linear vs quadratic).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

VISTA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VISTA))
from born_fused import born_fused_360, late_fused_360, load_frame_data, CORNERS

CM = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
DEG_PER_CM = 0.2   # overridable via --deg_per_cm (0 = pure translation)
CONDITIONS = ["clear", "rain_med", "fog", "night"]


def frame_diffs(frame_dir, seed):
    """Return {cm: mean_color_diff} for one frame (mean over corners)."""
    clouds, images = load_frame_data(frame_dir)
    if not clouds or not images:
        return None
    per_cm = {cm: [] for cm in CM}
    for corner in CORNERS:
        if corner not in clouds or corner not in images:
            continue
        pts, img = clouds[corner], images[corner]
        if not isinstance(img, dict):
            return None  # need 360 cameras
        born = born_fused_360(pts, img)
        born_hc = born[:, 7].astype(bool)
        born_rgb = born[:, 4:7].astype(float)
        for cm in CM:
            rng = np.random.RandomState(seed + CORNERS.index(corner))
            late = late_fused_360(pts, img, cm, cm * frame_diffs.deg_per_cm, rng)
            both = late[:, 7].astype(bool) & born_hc
            if both.any():
                d = np.abs(late[both, 4:7].astype(float) - born_rgb[both]).mean()
                per_cm[cm].append(float(d))
    return {cm: (float(np.mean(v)) if v else np.nan) for cm, v in per_cm.items()}


def fit_stats(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    # linear
    b1, b0 = np.polyfit(x, y, 1)
    yhat = b1 * x + b0
    rss1 = float(((y - yhat) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss1 / tss if tss > 0 else float("nan")
    Sxx = float(((x - x.mean()) ** 2).sum())
    se_slope = np.sqrt(rss1 / (n - 2) / Sxx) if n > 2 and Sxx > 0 else float("nan")
    t975 = 2.447  # df=6
    ci = (b1 - t975 * se_slope, b1 + t975 * se_slope)
    # quadratic
    q2, q1, q0 = np.polyfit(x, y, 2)
    rss2 = float(((y - (q2 * x * x + q1 * x + q0)) ** 2).sum())

    def aic(rss, k):
        return n * np.log(rss / n) + 2 * k if rss > 0 else -np.inf

    def bic(rss, k):
        return n * np.log(rss / n) + k * np.log(n) if rss > 0 else -np.inf

    return {
        "slope": round(b1, 4), "intercept": round(b0, 4), "R2": round(r2, 5),
        "slope_95CI": [round(ci[0], 4), round(ci[1], 4)],
        "quad_coef": round(float(q2), 5),
        "dAIC_lin_minus_quad": round(aic(rss1, 2) - aic(rss2, 3), 3),
        "dBIC_lin_minus_quad": round(bic(rss1, 2) - bic(rss2, 3), 3),
        "n_points": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--deg_per_cm", type=float, default=DEG_PER_CM,
                    help="rotation coupling (0 = pure translation)")
    ap.add_argument("--tag", default="coupled")
    a = ap.parse_args()
    frame_diffs.deg_per_cm = a.deg_per_cm
    out = VISTA / "outputs/sim_studies/t3"; out.mkdir(parents=True, exist_ok=True)

    results = {}
    for cond in a.conditions:
        cond_dir = VISTA / "carla_data" / cond
        frames = sorted(cond_dir.glob("frame_*"))[::a.stride]
        accum = {cm: [] for cm in CM}
        for i, fd in enumerate(frames):
            fdif = frame_diffs(fd, seed=i)
            if fdif is None:
                continue
            for cm in CM:
                if not np.isnan(fdif[cm]):
                    accum[cm].append(fdif[cm])
            if (i + 1) % 50 == 0:
                print(f"  {cond} [{i+1}/{len(frames)}]", flush=True)
        mean_diff = {cm: round(float(np.mean(accum[cm])), 4) for cm in CM}
        std_diff = {cm: round(float(np.std(accum[cm])), 4) for cm in CM}
        fit = fit_stats(CM, [mean_diff[cm] for cm in CM])
        results[cond] = {"mean_color_diff": mean_diff, "std_color_diff": std_diff,
                         "n_frames": len(accum[CM[0]]), "fit": fit}
        print(f"[{cond}] slope={fit['slope']} R2={fit['R2']} "
              f"CI={fit['slope_95CI']} dAIC(lin-quad)={fit['dAIC_lin_minus_quad']}", flush=True)

    (out / f"t3_sweep_{a.tag}.json").write_text(json.dumps(results, indent=2))
    print("wrote", out / f"t3_sweep_{a.tag}.json")


if __name__ == "__main__":
    main()
