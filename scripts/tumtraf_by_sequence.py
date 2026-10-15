#!/usr/bin/env python3
"""
TUMTraf-I drift law split by sequence (s01-s03 daytime, s04 rain at night), from the existing
matched-protocol records (drift_sweep_matched_records_real_tumtraf.npz): frame index i is the
position in the job list of pixel_displacement_sweep.jobs_for("real_tumtraf"), i.e. the
concatenation of list_frames(s)[::6] over s01..s04. Fits E_inf*d/(d+d_half) per sequence with the
same frame-block bootstrap as drift_law_bootstrap.py and reads the 5- and 10-unit budgets.

  python tumtraf_by_sequence.py -> outputs/tumtraf_by_sequence.json
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

VISTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISTA)); sys.path.insert(0, str(VISTA / "scripts"))
from tumtraf_adapter import list_frames

V = VISTA / "outputs"
CM = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0])
SEQS = ["a9_dataset_r02_s01", "a9_dataset_r02_s02", "a9_dataset_r02_s03", "a9_dataset_r02_s04"]
f = lambda d, E, h: E * d / (d + h)


def fit(e):
    try:
        return curve_fit(f, CM, e, p0=[e[-1] * 1.5, 8.0], maxfev=5000)[0]
    except Exception:
        return np.array([np.nan, np.nan])


budget = lambda e, b: float(np.interp(b, e, CM)) if e[-1] >= b else np.inf
ci = lambda x: [round(float(np.nanpercentile(x, q)), 2) for q in (2.5, 97.5)]

seq_of = []
for s in SEQS:
    seq_of += [s[-3:]] * len(list_frames(s)[::6])
z = np.load(V / "drift_sweep_matched_records_real_tumtraf.npz")
E, fr = z["E"], z["frame"]
groups = {"s01-s03 (day)": ["s01", "s02", "s03"], "s04 (rain, night)": ["s04"], "s01": ["s01"], "s02": ["s02"], "s03": ["s03"]}
out = {}
for g, ss in groups.items():
    m = np.array([seq_of[k] in ss for k in fr])
    pf = E[m]
    nb = int(np.ceil(len(pf) / 10)); blocks = [np.arange(i * 10, min((i + 1) * 10, len(pf))) for i in range(nb)]
    rng = np.random.RandomState(0); P, B5, B10 = [], [], []
    for _ in range(1000):
        e = np.nanmean(pf[np.concatenate([blocks[b] for b in rng.randint(0, nb, nb)])], axis=0)
        P.append(fit(e)); B5.append(budget(e, 5)); B10.append(budget(e, 10))
    P = np.array(P); e0 = np.nanmean(pf, axis=0); p0 = fit(e0)
    r2 = 1 - ((f(CM, *p0) - e0) ** 2).sum() / ((e0 - e0.mean()) ** 2).sum()
    out[g] = {"n_frames": int(m.sum()), "E": np.round(e0, 3).tolist(), "E_inf": round(float(p0[0]), 2), "E_inf_CI95": ci(P[:, 0]),
              "d_half": round(float(p0[1]), 2), "d_half_CI95": ci(P[:, 1]), "R2": round(float(r2), 4),
              "budget5": round(budget(e0, 5), 2), "budget5_CI95": ci(np.minimum(B5, 99)),
              "budget10": round(budget(e0, 10), 2) if np.isfinite(budget(e0, 10)) else None}
    print(g, out[g])
(V / "tumtraf_by_sequence.json").write_text(json.dumps(out, indent=1))
