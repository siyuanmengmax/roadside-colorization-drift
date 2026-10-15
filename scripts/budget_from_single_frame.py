#!/usr/bin/env python3
"""
How well does the zero-drift image statistic G predict a site's drift sensitivity and budget
out of sample, and from how many frames?

Per frame (same frames as image_budget_predictor.py) we have G_lidar under the nominal extrinsic,
the frame's pixel displacement per cm averaged over 64 random drift directions (px_avg, computed
from that frame's points and calibration alone), and its measured drift curve. The predictor is
E1_hat = k * G * px_avg, so it uses one zero-drift frame and nothing else.

  1. Leave-one-group-out (LOGO): k is refit on the frames of the other seven groups and used to
     predict the held-out group's mean E(1 cm).
  2. Single-frame and N-frame prediction: draw N frames of a group, average their E1_hat (LOGO k),
     and compare with the group's measured mean E(1 cm). Reported as the median and 90th
     percentile of |relative error| over 2000 draws.
  3. Budget estimator: the small-drift extrapolation x = 5 / E1_hat underestimates the budget
     because the curve is concave. A one-parameter correction b = x (1 + c x) is fitted LOGO
     on the groups whose 5-unit budget is reached, and the predicted budget is compared with the
     measured one, for the group mean and for single frames.

  python budget_from_single_frame.py  ->  outputs/budget_from_single_frame.json
                                         (+ budget_from_single_frame_frames.npz, per-frame table)
"""
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

VISTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISTA)); sys.path.insert(0, str(VISTA / "scripts"))
OUT = VISTA / "outputs"
CM = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0])
BUDGET = 5.0
NS = [1, 3, 5, 10, 20]
RNG = np.random.default_rng(0)


def budget(e, b=BUDGET):
    return float(np.interp(b, e, CM)) if e[-1] >= b else np.inf


def frame_stats(job):
    """(frame, G_lidar, px_avg): px_avg = mean projection shift (pixels) of the in-image points for a
    1-cm camera-frame translation, averaged over 64 random unit directions."""
    from image_budget_predictor import load, stats
    st = stats(job)
    r = load(job)
    if st is None or r is None:
        return None
    img, K, T, xyz = r
    H, W = img.shape[:2]
    pc = (T @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvw = (K @ pc.T).T; u = uvw[:, 0] / uvw[:, 2]; v = uvw[:, 1] / uvw[:, 2]
    ok = (pc[:, 2] > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pc = pc[ok]; uv = np.c_[u[ok], v[ok]]
    dirs = np.random.default_rng(1).normal(size=(64, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    shifts = []
    for d in dirs:
        q = (K @ (pc + 0.01 * d).T).T
        shifts.append(np.linalg.norm(q[:, :2] / q[:, 2:3] - uv, axis=1).mean())
    return st[0], st[1], float(np.mean(shifts))


def per_frame_table():
    cache = OUT / "budget_from_single_frame_frames.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    from image_budget_predictor import GROUPS
    from pixel_displacement_sweep import jobs_for
    rows = {"group": [], "G": [], "px1": [], "px_avg": [], "E": []}
    with mp.get_context("spawn").Pool(16) as pool:
        for g, rec in GROUPS.items():
            _, jobs = jobs_for(g)
            st = {k: (a, b) for k, a, b in (x for x in pool.map(frame_stats, jobs, chunksize=2) if x)}
            z = np.load(OUT / rec)
            for i, k in enumerate(z["frame"]):
                if k in st:
                    rows["group"].append(g); rows["G"].append(st[k][0]); rows["px_avg"].append(st[k][1])
                    rows["px1"].append(z["px"][i, 1]); rows["E"].append(z["E"][i])
            print(g, sum(1 for x in rows["group"] if x == g), flush=True)
    t = {k: np.array(v) for k, v in rows.items()}
    np.savez(cache, **t)
    return t


def fit_k(x, y):
    return float((x * y).sum() / (x * x).sum())


def main():
    t = per_frame_table()
    grp, x, E = t["group"], t["G"] * t["px_avg"], t["E"]
    E1 = E[:, 1]
    groups = list(dict.fromkeys(grp.tolist()))
    meas = {g: E[grp == g].mean(axis=0) for g in groups}
    res = {"n_frames": int(len(grp)), "k_all": round(fit_k(x, E1), 4), "groups": {},
           "px_avg_vs_px1_group_means": {g: [round(float(t["px_avg"][grp == g].mean()), 3),
                                              round(float(t["px1"][grp == g].mean()), 3)] for g in dict.fromkeys(grp.tolist())}}

    # 1-2: LOGO k, group-mean and N-frame prediction of the group's mean E(1 cm)
    k_logo = {g: fit_k(x[grp != g], E1[grp != g]) for g in groups}
    for g in groups:
        m = grp == g; e1m = float(meas[g][1]); pred = k_logo[g] * x[m]
        r = {"n": int(m.sum()), "k_logo": round(k_logo[g], 4), "E1_meas": round(e1m, 3),
             "E1_pred_mean_logo": round(float(pred.mean()), 3),
             "relerr_mean_logo": round(float(pred.mean() / e1m - 1), 3)}
        for n in NS:
            if n > m.sum():
                continue
            draws = np.array([pred[RNG.choice(m.sum(), n, replace=False)].mean() for _ in range(2000)])
            rel = np.abs(draws / e1m - 1)
            r[f"N{n}_absrel_median"] = round(float(np.median(rel)), 3)
            r[f"N{n}_absrel_p90"] = round(float(np.percentile(rel, 90)), 3)
        res["groups"][g] = r

    # 3: budget estimator b = x (1 + c x), x = 5 / E1_hat, c fitted LOGO on reached groups
    bmeas = {g: budget(meas[g]) for g in groups}
    reached = [g for g in groups if np.isfinite(bmeas[g])]

    def fit_c(gs):
        # least squares over groups of (b - x) = c x^2, with x from the LOGO group-mean prediction
        xs = np.array([BUDGET / res["groups"][h]["E1_pred_mean_logo"] for h in gs])
        bs = np.array([bmeas[h] for h in gs])
        return float(((bs - xs) * xs ** 2).sum() / (xs ** 4).sum())

    for g in groups:
        c = fit_c([h for h in reached if h != g])
        m = grp == g
        xg = BUDGET / res["groups"][g]["E1_pred_mean_logo"]
        r = res["groups"][g]
        r.update({"budget_meas": round(bmeas[g], 2) if np.isfinite(bmeas[g]) else None,
                  "c_logo": round(c, 4), "budget_linear_from_mean": round(xg, 2),
                  "budget_pred_from_mean": round(xg * (1 + c * xg), 2)})
        if np.isfinite(bmeas[g]):
            r["budget_relerr_from_mean"] = round(xg * (1 + c * xg) / bmeas[g] - 1, 3)
            pred = k_logo[g] * x[m]
            for n in NS:
                if n > m.sum():
                    continue
                xs = np.array([BUDGET / pred[RNG.choice(m.sum(), n, replace=False)].mean() for _ in range(2000)])
                rel = np.abs(xs * (1 + c * xs) / bmeas[g] - 1)
                r[f"budget_N{n}_absrel_median"] = round(float(np.median(rel)), 3)
                r[f"budget_N{n}_absrel_p90"] = round(float(np.percentile(rel, 90)), 3)
    # ranking of measured budgets by LOGO-predicted budgets (s04 not reached -> rank last)
    from scipy.stats import spearmanr
    bp = [res["groups"][g]["budget_pred_from_mean"] for g in groups]
    bm = [bmeas[g] if np.isfinite(bmeas[g]) else 99.0 for g in groups]
    res["spearman_budget_pred_vs_meas"] = round(float(spearmanr(bp, bm).correlation), 3)
    (OUT / "budget_from_single_frame.json").write_text(json.dumps(res, indent=1))
    for g, r in res["groups"].items():
        print(g, {k: v for k, v in r.items() if not k.startswith("budget_N") or "N1_" in k or "N10_" in k})
    print("spearman", res["spearman_budget_pred_vs_meas"], "k_all", res["k_all"])


if __name__ == "__main__":
    main()
