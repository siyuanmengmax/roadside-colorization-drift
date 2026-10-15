#!/usr/bin/env python3
"""
Sensitivity of the detection block-bootstrap intervals to the block length.

Per-frame TP and FN counts stay autocorrelated for tens of frames (traffic changes slowly), so we
recompute the main F1 intervals and paired F1-difference intervals with non-overlapping blocks of
25, 50 (paper), and 100 frames, 1000 resamples each, from the saved per-frame counts.

  python block_length_sensitivity.py -> journal/output/block_length_sensitivity.json
"""
import json
from pathlib import Path
import numpy as np
from prf_utils import prf, boot_sums

OUT = Path(__file__).resolve().parents[1] / "outputs"
LENGTHS = [25, 50, 100]


def draws_for(n, L, seed=0):
    rng = np.random.RandomState(seed); nblk = int(np.ceil(n / L))
    blocks = [list(range(i * L, min((i + 1) * L, n))) for i in range(nblk)]
    return [[i for b in rng.randint(0, nblk, nblk) for i in blocks[b]] for _ in range(1000)]


def ci_f1(c, d):
    f = prf(boot_sums(c, d))[2]; return [float(np.nanpercentile(f, q)) for q in (2.5, 97.5)]


def ci_df1(a, b, d):
    f = prf(boot_sums(a, d))[2] - prf(boot_sums(b, d))[2]; return [float(np.nanpercentile(f, q)) for q in (2.5, 97.5)]


def acf_lag(x, thr=0.1):
    x = np.asarray(x, float) - np.mean(x); v = (x * x).mean()
    for k in range(1, len(x) // 2):
        if (x[:-k] * x[k:]).mean() / v < thr:
            return k
    return None


cases = {}
for c in ["clear", "rain_med", "fog", "night"]:
    z = np.load(OUT / f"sensor_ablation_counts_{c}.npz")
    cases[f"sim_{c}_4_vehicle"] = ("single", z["4_vehicle"])
    cases[f"sim_{c}_2minus1_vehicle"] = ("paired", z["2_vehicle"], z["1_vehicle"])
    cases[f"sim_{c}_4minus2_vehicle"] = ("paired", z["4_vehicle"], z["2_vehicle"])
for seq in ["s03", "s04"]:
    for mask in ["nonoracle", "oracle"]:
        z = np.load(OUT / f"tumtraf_counts_{seq}_gtfilt_v3_{mask}.npz")
        cases[f"{seq}_{mask}_uniform_vehicle"] = ("single", z["uniform_vehicle"])
        for cl in (["vehicle", "pedestrian"] if seq == "s03" else ["vehicle"]):
            cases[f"{seq}_{mask}_signal_minus_uniform6_{cl}"] = ("paired", z[f"signal_qa_{cl}"], z[f"uniform6_{cl}"])
            cases[f"{seq}_{mask}_norm_minus_uniform_{cl}"] = ("paired", z[f"signal_norm_{cl}"], z[f"uniform_{cl}"])

res = {}
for name, spec in cases.items():
    n = len(spec[1]); r = {"n_frames": n, "acf_lag_TP": acf_lag(spec[1][:, 0]), "acf_lag_ngt": acf_lag(spec[1][:, 2])}
    for L in LENGTHS:
        d = draws_for(n, L)
        ci = ci_f1(spec[1], d) if spec[0] == "single" else ci_df1(spec[1], spec[2], d)
        r[f"L{L}"] = [round(x, 4) for x in ci]
        if spec[0] == "paired":
            r[f"L{L}_excl0"] = bool(ci[0] > 0 or ci[1] < 0)
    w = {L: r[f"L{L}"][1] - r[f"L{L}"][0] for L in LENGTHS}
    r["width_ratio_100_over_50"] = round(w[100] / w[50], 3) if w[50] > 0 else None
    r["width_ratio_25_over_50"] = round(w[25] / w[50], 3) if w[50] > 0 else None
    res[name] = r
    print(f"{name:45s} n={n} lagTP={r['acf_lag_TP']} L25={r['L25']} L50={r['L50']} L100={r['L100']} "
          f"ratio100={r['width_ratio_100_over_50']}" + (f" excl0 {[r[f'L{L}_excl0'] for L in LENGTHS]}" if spec[0] == "paired" else ""))
(OUT / "block_length_sensitivity.json").write_text(json.dumps(res, indent=1))
