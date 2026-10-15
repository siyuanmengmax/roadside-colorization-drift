#!/usr/bin/env python3
"""
Model comparison for the drift -> colorization-error curves (8 drift levels, per-dataset means).

Fits, per dataset: linear with intercept (the model in the paper), linear through the origin,
exponential saturation E_inf(1-exp(-d/d0)), hyperbolic saturation E_inf d/(d+d0), and a power law
k d^gamma. All but the first satisfy E(0)=0 by construction. Reports RMSE, AIC relative to the
linear-with-intercept fit, and the drift at which the MEASURED curve (piecewise-linear
interpolation) reaches a 5/10/20 RGB-unit budget.

  python drift_model_compare.py   ->  outputs/drift_model_compare.json
  python drift_model_compare.py drift_sweep_dair.json drift_model_compare_dair.json   (other sweep files)
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

VISTA = Path(__file__).resolve().parents[1]
# Matched protocol for every dataset (unit direction x d; zero-range points removed on real data):
# scripts/pixel_displacement_sweep.py --protocol unit -> drift_sweep_matched.json
IN, OUT = (sys.argv[1:3] if len(sys.argv) > 2 else ("drift_sweep_matched.json", "drift_model_compare.json"))
SW = json.loads((VISTA / "outputs" / IN).read_text())
SETS = {k: dict(zip(map(str, v["nominal_cm"]), v["mean_E"])) for k, v in SW.items()}

MODELS = {
    "linear": (lambda d, a, b: a * d + b, [1, 1]),
    "linear_origin": (lambda d, a: a * d, [1]),
    "exp_sat": (lambda d, E, d0: E * (1 - np.exp(-d / d0)), [20, 5]),
    "hyperbolic": (lambda d, E, d0: E * d / (d + d0), [30, 10]),
    "power": (lambda d, k, g: k * np.power(d, g), [2, 0.6]),
}


def aic(rss, n, k):
    return n * np.log(rss / n) + 2 * k


def budget(d, e, emax):
    """Drift at which the measured curve first reaches emax (None if not reached by max d)."""
    for i in range(1, len(d)):
        if e[i] >= emax:
            return float(d[i - 1] + (emax - e[i - 1]) * (d[i] - d[i - 1]) / (e[i] - e[i - 1]))
    return None


out = {}
for name, m in SETS.items():
    d = np.array(sorted(float(k) for k in m)); e = np.array([m[str(x)] for x in d])
    res = {"E": dict(zip(map(str, d), e.tolist())), "models": {}}
    for mn, (f, p0) in MODELS.items():
        p, _ = curve_fit(f, d, e, p0=p0, maxfev=20000)
        rss = float(((f(d, *p) - e) ** 2).sum())
        res["models"][mn] = {"params": [round(float(x), 4) for x in p],
                             "rmse": round(float(np.sqrt(rss / len(d))), 3),
                             "R2": round(1 - rss / float(((e - e.mean()) ** 2).sum()), 4),
                             "aic": round(float(aic(rss, len(d), len(p))), 2)}
    # hyperbolic parameters with 95% CIs from the fit covariance (t, n-2 dof)
    f, p0 = MODELS["hyperbolic"]
    ph, cov = curve_fit(f, d, e, p0=p0, maxfev=20000)
    se = np.sqrt(np.diag(cov)); t = 2.447
    s0 = ph[0] / ph[1]
    se_s0 = s0 * np.sqrt((se[0] / ph[0]) ** 2 + (se[1] / ph[1]) ** 2 - 2 * cov[0, 1] / (ph[0] * ph[1]))
    res["hyperbolic"] = {
        "E_inf": round(float(ph[0]), 2), "E_inf_CI95": [round(float(ph[0] - t * se[0]), 2), round(float(ph[0] + t * se[0]), 2)],
        "d_half_cm": round(float(ph[1]), 2), "d_half_CI95": [round(float(ph[1] - t * se[1]), 2), round(float(ph[1] + t * se[1]), 2)],
        "initial_slope": round(float(s0), 2), "initial_slope_CI95": [round(float(s0 - t * se_s0), 2), round(float(s0 + t * se_s0), 2)],
        "R2": res["models"]["hyperbolic"]["R2"], "max_abs_resid": round(float(np.abs(f(d, *ph) - e).max()), 3)}
    print("  hyperbolic:", res["hyperbolic"])
    a0 = res["models"]["linear"]["aic"]
    for mn in MODELS:
        res["models"][mn]["dAIC_vs_linear"] = round(res["models"][mn]["aic"] - a0, 2)
    lin = res["models"]["linear"]["params"]
    res["budget_cm_measured"] = {str(b): budget(d, e, b) for b in (5, 10, 20)}
    res["budget_cm_linear_fit"] = {str(b): round((b - lin[1]) / lin[0], 2) for b in (5, 10, 20)}
    res["ratio_E15_over_E1"] = round(float(e[-1] / e[1]), 2)
    # pixel-displacement axis: same law fitted against the mean projection shift
    px = np.array(SW[name]["mean_px"]); fh = MODELS["hyperbolic"][0]
    pp, _ = curve_fit(fh, px, e, p0=[e.max() * 1.5, np.median(px)], maxfev=20000)
    edges = np.array(SW[name]["bins_px_edges"]); k1 = int(np.searchsorted(edges, 1.0, side="right") - 1)
    res["pixel_axis"] = {"px_per_cm": round(float(px[1]), 3), "mrad_per_cm": round(float(px[1] / SW[name]["fx_px"] * 1000), 3),
                         "p_half_px": round(float(pp[1]), 2), "E_inf": round(float(pp[0]), 2),
                         "R2": round(1 - float(((fh(px, *pp) - e) ** 2).sum()) / float(((e - e.mean()) ** 2).sum()), 4),
                         "median_depth_m": SW[name]["median_depth_m"], "fx_px": SW[name]["fx_px"],
                         "E_at_1px_per_point": SW[name]["bins_mean_E"][k1], "bin_1px": [float(edges[k1]), float(edges[k1 + 1])]}
    print("  pixel axis:", res["pixel_axis"])
    out[name] = res
    print(f"\n{name}: E={np.round(e, 2).tolist()}")
    for mn, r in res["models"].items():
        print(f"  {mn:14s} params={r['params']} rmse={r['rmse']} R2={r['R2']} dAIC={r['dAIC_vs_linear']}")
    print("  budget measured:", {k: (round(v, 1) if v else None) for k, v in res["budget_cm_measured"].items()},
          "linear-fit:", res["budget_cm_linear_fit"], "E15/E1:", res["ratio_E15_over_E1"])
(VISTA / "outputs" / OUT).write_text(json.dumps(out, indent=2))
