"""Pooled precision / recall / F1 with block-bootstrap CIs from per-frame (TP, FP, n_gt) counts.

The training-free operator's confidence does not rank detections, so we score its full detection
set at one operating point instead of integrating a PR curve.
"""
import numpy as np


def counts(recs_cell):
    """list of (tp_scores, fp_scores, n_gt) per frame -> (n_frames, 3) int array."""
    return np.array([(len(t), len(f), g) for t, f, g in recs_cell], np.int64)


def prf(c):
    """c: (..., 3) summed counts -> P, R, F1 (nan when undefined)."""
    tp, fp, ng = c[..., 0].astype(float), c[..., 1].astype(float), c[..., 2].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = tp / (tp + fp); r = tp / ng; f = 2 * tp / (2 * tp + fp + (ng - tp))
    return p, r, f


def boot_sums(c, draws):
    """c: (n_frames, 3); draws: list of index lists -> (n_boot, 3) summed counts."""
    return np.stack([c[d].sum(0) for d in draws])


def summarize(c, draws):
    p, r, f = prf(c.sum(0)); bp, br, bf = prf(boot_sums(c, draws))
    ci = lambda x: [round(float(np.nanpercentile(x, q)), 4) for q in (2.5, 97.5)]
    return {"P": round(float(p), 4), "R": round(float(r), 4), "F1": round(float(f), 4),
            "P_CI95": ci(bp), "R_CI95": ci(br), "F1_CI95": ci(bf)}


def paired(c_hi, c_lo, draws):
    out = {}
    s_hi, s_lo = boot_sums(c_hi, draws), boot_sums(c_lo, draws)
    for k, a, b, x, y in zip(("P", "R", "F1"), prf(c_hi.sum(0)), prf(c_lo.sum(0)), prf(s_hi), prf(s_lo)):
        ci = [round(float(np.nanpercentile(x - y, q)), 4) for q in (2.5, 97.5)]
        out[f"delta_{k}"] = round(float(a - b), 4); out[f"delta_{k}_CI95"] = ci
        out[f"delta_{k}_excl0"] = bool(ci[0] > 0 or ci[1] < 0)
    return out
