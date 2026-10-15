#!/usr/bin/env python3
"""Theme A difficulty stats (CPU only) from frame_stats.json, on the TEST split.
Characterizes WHY AP is low: object range + points/object + fraction near-invisible."""
import os
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("CARLA_PCDET_DATA", "data/carla_pcdet"))


def stats_for(cond):
    root = ROOT / cond
    fs = json.loads((root / "frame_stats.json").read_text())
    test_ids = set((root / "ImageSets" / "test.txt").read_text().split())
    per = {"vehicle": {"rng": [], "pts": []}, "pedestrian": {"rng": [], "pts": []}}
    for idx, objs in fs.items():
        if idx not in test_ids:
            continue
        for o in objs:
            c = "pedestrian" if o["class"].lower().startswith("ped") else "vehicle"
            per[c]["rng"].append(o["range"]); per[c]["pts"].append(o["n_pts"])
    out = {"condition": cond, "n_test_frames": len(test_ids)}
    for c in ("vehicle", "pedestrian"):
        rng = np.array(per[c]["rng"]); pts = np.array(per[c]["pts"])
        n = len(rng)
        out[c] = {
            "n_gt": int(n),
            "median_range_m": round(float(np.median(rng)), 1),
            "p90_range_m": round(float(np.percentile(rng, 90)), 1),
            "median_pts": int(np.median(pts)),
            "mean_pts": round(float(pts.mean()), 1),
            "frac_lt5_pts": round(float((pts < 5).mean()), 3),
            "frac_0_pts": round(float((pts == 0).mean()), 3),
        }
    return out


if __name__ == "__main__":
    conds = sys.argv[1:] or ["clear", "fog"]
    res = [stats_for(c) for c in conds]
    print(json.dumps(res, indent=2))
    out = (Path(__file__).resolve().parents[2] / "outputs/sim_studies/t1/theme_a_stats.json")
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)
