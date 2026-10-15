#!/usr/bin/env python3
"""
Frame-level block bootstrap for the drift-law parameters.

Resamples blocks of 10 consecutive sampled frames (all sensors of a frame stay together), recomputes
the mean curve, refits E_inf*d/(d+d_half), and reads the 5- and 10-unit budgets from the resampled
curve. Reports percentile 95% CIs for E_inf, d_half and the budgets, and for the contrasts
Town05 - Town10HD (clear) and TUMTraf-I / V2X-Real, and between the DAIR-V2X-I groups.

  python drift_law_bootstrap.py -> outputs/drift_law_bootstrap.json
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

V = (Path(__file__).resolve().parents[1] / "outputs")
CM = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0])
SETS = ["sim_clear", "sim_rain_med", "sim_fog", "sim_night", "sim_town05_clear", "real_tumtraf", "real_tumtraf_day", "real_tumtraf_s04", "real_v2xreal",
        "real_dair_W", "real_dair_Y1", "real_dair_Y2", "real_dair_Y3", "real_dair_Ylow"]
REC = lambda n: V / (f"drift_sweep_dair_records_{n}.npz" if n.startswith("real_dair_") else f"drift_sweep_matched_records_{n}.npz")
BLOCK, NBOOT = 10, 1000
f = lambda d, E, h: E * d / (d + h)


def fit(e):
    try:
        p, _ = curve_fit(f, CM, e, p0=[e[-1] * 1.5, 8.0], maxfev=5000)
        return p
    except Exception:
        return np.array([np.nan, np.nan])


def budget(e, b):
    return float(np.interp(b, e, CM)) if e[-1] >= b else np.inf


def ci(x):
    x = np.asarray(x, float)
    return [round(float(np.nanpercentile(x, q)), 2) for q in (2.5, 97.5)]


boot, out = {}, {}
for name in SETS:
    z = np.load(REC(name))
    E, fr = z["E"], z["frame"]
    frames = np.unique(fr)
    per_frame = np.array([np.nanmean(E[fr == k], axis=0) for k in frames])     # one curve per frame
    nb = int(np.ceil(len(frames) / BLOCK))
    blocks = [np.arange(i * BLOCK, min((i + 1) * BLOCK, len(frames))) for i in range(nb)]
    rng = np.random.RandomState(0)
    P, B5, B10 = [], [], []
    for _ in range(NBOOT):
        idx = np.concatenate([blocks[b] for b in rng.randint(0, nb, nb)])
        e = np.nanmean(per_frame[idx], axis=0)
        P.append(fit(e)); B5.append(budget(e, 5)); B10.append(budget(e, 10))
    P = np.array(P); boot[name] = P; boot[name, "b5"] = np.minimum(B5, 99)
    e0 = np.nanmean(per_frame, axis=0); p0 = fit(e0)
    fin = lambda b: [v if np.isfinite(v) else None for v in ci(np.where(np.isfinite(b), b, 1e9))]
    out[name] = {"n_frames": int(len(frames)), "E_inf": round(float(p0[0]), 2), "E_inf_CI95": ci(P[:, 0]),
                 "d_half": round(float(p0[1]), 2), "d_half_CI95": ci(P[:, 1]),
                 "budget5": round(budget(e0, 5), 2), "budget5_CI95": ci(np.minimum(B5, 99)),
                 "budget10": (round(budget(e0, 10), 2) if np.isfinite(budget(e0, 10)) else None),
                 "budget10_CI95": ci(np.minimum(B10, 99)), "frac_boot_budget10_reached": round(float(np.isfinite(B10).mean()), 3)}
    print(name, out[name])
a, b = boot["sim_town05_clear"], boot["sim_clear"]
out["town05_minus_town10hd"] = {"E_inf_diff_CI95": ci(a[:, 0] - b[:, 0]), "d_half_diff_CI95": ci(a[:, 1] - b[:, 1])}
a, b = boot["real_tumtraf"], boot["real_v2xreal"]
out["tumtraf_over_v2xreal"] = {"d_half_ratio_CI95": ci(a[:, 1] / b[:, 1]), "E_inf_diff_CI95": ci(a[:, 0] - b[:, 0])}
a, b = boot["real_tumtraf_day"], boot["real_v2xreal"]
out["tumtraf_day_over_v2xreal"] = {"d_half_ratio_CI95": ci(a[:, 1] / b[:, 1]), "E_inf_diff_CI95": ci(a[:, 0] - b[:, 0])}
print("tumtraf_day_over_v2xreal", out["tumtraf_day_over_v2xreal"])
for a_, b_ in (("real_dair_Y1", "real_dair_Y2"), ("real_dair_Y1", "real_dair_Y3"), ("real_dair_Y2", "real_dair_Y3"),
               ("real_dair_W", "real_dair_Y1"), ("real_dair_W", "real_tumtraf"), ("real_dair_W", "real_v2xreal")):
    out[f"{a_}_minus_{b_}"] = {"E_inf_diff_CI95": ci(boot[a_][:, 0] - boot[b_][:, 0]),
                               "d_half_diff_CI95": ci(boot[a_][:, 1] - boot[b_][:, 1]),
                               "budget5_diff_CI95": ci(boot[a_, "b5"] - boot[b_, "b5"])}
    print(a_, b_, out[f"{a_}_minus_{b_}"])
print(out["town05_minus_town10hd"], out["tumtraf_over_v2xreal"])
(V / "drift_law_bootstrap.json").write_text(json.dumps(out, indent=1))
