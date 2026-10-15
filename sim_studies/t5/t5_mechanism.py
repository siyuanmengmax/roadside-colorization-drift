#!/usr/bin/env python3
"""
T5 (Theme E): explain why 2 sensors saturate 4-sensor performance under adverse
weather but not clear (mDsR-W2).

Hypothesis: adverse weather truncates effective LiDAR range -> distant objects leave
the detectable set -> the 4-sensor cross-corner advantage (which lives at long range/
occlusion) vanishes -> 2 sensors saturate a smaller achievable ceiling.

Test: per-range-bin AP_vehicle for 4-sensor (all) vs 2-sensor (NE+SW, the paper's config),
clear vs fog. Plus return-distance and GT object-range distributions.
Oracle dynamic mask (matches the paper's Exp. 2 ablation). Test split (400 frames).
"""
import json, sys
from pathlib import Path
import numpy as np

VISTA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VISTA))
from carla_adapter import process_frame
from carla_detection import detect_frame, match_detections, compute_ap, gt_class, det_class

CONDS = ["clear", "fog"]
CONFIGS = {"4sensor": None, "2sensor_NE+SW": ["ne", "sw"]}
BINS = [(0, 20), (20, 40), (40, 999)]
BIN_LABELS = ["0-20m", "20-40m", ">40m"]


def obj_range(o):
    cx, cy, _ = o["bbox"]["center"]
    return float(o.get("distance_from_center", np.hypot(cx, cy)))


def main():
    results = {}
    for cond in CONDS:
        cond_dir = VISTA / "carla_data" / cond
        ids = [f"{i:06d}" for i in range(1600, 2000) if (cond_dir / f"frame_{i:06d}").is_dir()]

        # accumulators: [config][bin] -> tp,fp,ngt (vehicle only, paper metric)
        acc = {cfg: {b: {"tp": [], "fp": [], "ngt": 0} for b in BIN_LABELS} for cfg in CONFIGS}
        # stats
        max_ret = []                       # per-frame 95th-pct return distance
        gt_by_bin = {b: 0 for b in BIN_LABELS}
        det_by_bin = {b: 0 for b in BIN_LABELS}   # GT with >=5 pts (detectable set)

        for idx in ids:
            fd = cond_dir / f"frame_{idx}"
            pts, cols = process_frame(fd)
            rng_col = pts[:, cols.index("range")]
            max_ret.append(float(np.percentile(rng_col, 95)))
            gt = json.loads((fd / "gt_boxes.json").read_text())
            veh = [o for o in gt["objects"] if gt_class(o["class"]) == "vehicle"]
            # detectable-set stats via oracle dynamic mask point counts per object
            for o in veh:
                r = obj_range(o)
                for (lo, hi), bl in zip(BINS, BIN_LABELS):
                    if lo <= r < hi:
                        gt_by_bin[bl] += 1
                        break

            for cfg, sensors in CONFIGS.items():
                dets = detect_frame(pts, cols, "uniform", sensors=sensors)
                dets_v = [d for d in dets if det_class(d["label"]) == "vehicle"]
                for (lo, hi), bl in zip(BINS, BIN_LABELS):
                    gt_bin = [o for o in veh if lo <= obj_range(o) < hi]
                    det_bin = [d for d in dets_v if lo <= np.hypot(d["cx"], d["cy"]) < hi]
                    tp_s, fp_s, ng = match_detections(det_bin, gt_bin, 3.0)
                    acc[cfg][bl]["tp"].extend(tp_s); acc[cfg][bl]["fp"].extend(fp_s)
                    acc[cfg][bl]["ngt"] += ng

        ap = {cfg: {bl: round(compute_ap(acc[cfg][bl]["tp"], acc[cfg][bl]["fp"], acc[cfg][bl]["ngt"]), 4)
                    for bl in BIN_LABELS} for cfg in CONFIGS}
        gap = {bl: round(ap["4sensor"][bl] - ap["2sensor_NE+SW"][bl], 4) for bl in BIN_LABELS}
        results[cond] = {
            "AP_vehicle_by_bin": ap,
            "gap_4minus2_by_bin": gap,
            "gt_vehicle_by_bin": gt_by_bin,
            "gt_frac_by_bin": {bl: round(gt_by_bin[bl] / max(1, sum(gt_by_bin.values())), 3) for bl in BIN_LABELS},
            "return_dist_p95_median_m": round(float(np.median(max_ret)), 1),
            "n_frames": len(ids),
        }
        print(f"[{cond}] p95 return {results[cond]['return_dist_p95_median_m']}m  "
              f"gap4-2 {gap}  gt_frac {results[cond]['gt_frac_by_bin']}", flush=True)

    out = VISTA / "outputs/sim_studies/t5"; out.mkdir(parents=True, exist_ok=True)
    (out / "t5_mechanism.json").write_text(json.dumps(results, indent=2))
    print("wrote", out / "t5_mechanism.json")


if __name__ == "__main__":
    main()
