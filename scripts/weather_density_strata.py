#!/usr/bin/env python3
"""
Is the weather ordering of simulated detection a weather effect or a traffic-density effect?

The four CARLA conditions are separate recordings; their 400-frame test windows differ in traffic
(9.6-15.4 vehicles per frame). This script recomputes the label-free four-sensor detections of
Table "baselines" per frame and reports vehicle recall per condition WITHIN strata of equal
per-frame vehicle count, plus pooled recall after reweighting every condition to a common
density distribution.

  python weather_density_strata.py -> outputs/weather_density_strata.json
"""
import json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

VISTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISTA / "scripts"))
import sensor_ablation_ci as S

S.SENSOR_CONFIGS = {"4": None}
BINS = [(0, 8), (9, 11), (12, 14), (15, 17), (18, 99)]


def work(job):
    S.SENSOR_CONFIGS = {"4": None}
    tp, fp, n = S.process(job)["4"]["vehicle"]
    return len(tp), len(fp), n


def main():
    t0 = time.time(); out = {}
    per = {}
    with mp.get_context("spawn").Pool(16) as pool:
        for cond in S.CONDITIONS:
            cdir = VISTA / "carla_data" / cond
            frames = [str(cdir / f"frame_{i:06d}") for i in range(1600, 2000) if (cdir / f"frame_{i:06d}").is_dir()]
            per[cond] = np.array(pool.map(work, [(cond, f) for f in frames], chunksize=4))
            a = per[cond]
            print(f"[{time.time()-t0:4.0f}s] {cond}: recall {a[:,0].sum()/a[:,2].sum():.3f} veh/frame {a[:,2].mean():.1f}", flush=True)
    # common density distribution = pooled over conditions
    allgt = np.concatenate([a[:, 2] for a in per.values()])
    w = {b: float(((allgt >= b[0]) & (allgt <= b[1])).mean()) for b in BINS}
    for cond, a in per.items():
        res = {"veh_per_frame": round(float(a[:, 2].mean()), 2), "recall": round(float(a[:, 0].sum() / a[:, 2].sum()), 4),
               "precision": round(float(a[:, 0].sum() / max(a[:, 0].sum() + a[:, 1].sum(), 1)), 4), "strata": {}}
        num = den = 0.0
        for b in BINS:
            m = (a[:, 2] >= b[0]) & (a[:, 2] <= b[1])
            if m.sum() >= 10:
                r = float(a[m, 0].sum() / a[m, 2].sum())
                res["strata"][f"{b[0]}-{b[1]}"] = {"frames": int(m.sum()), "recall": round(r, 4)}
                num += w[b] * r; den += w[b]
            else:
                res["strata"][f"{b[0]}-{b[1]}"] = {"frames": int(m.sum()), "recall": None}
        res["recall_density_standardized"] = round(num / den, 4) if den else None
        res["standardization_weight_covered"] = round(den, 3)
        out[cond] = res
        print(cond, json.dumps(res))
    out["_weights"] = {f"{b[0]}-{b[1]}": round(v, 3) for b, v in w.items()}
    (VISTA / "outputs/weather_density_strata.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
