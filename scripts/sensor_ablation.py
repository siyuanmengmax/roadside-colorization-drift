#!/usr/bin/env python3
"""
T-ITS sensor-count ablation, re-run under the paper's CURRENT non-oracle protocol
(background-subtraction foreground, 3m BEV match + 11-pt AP) -- NOT the oracle-mask
protocol used for the original conference-version ablation, which is not comparable to the
rest of this paper's numbers.

Configs: 1 sensor (NE), 2 sensors (NE+SW diagonal), 4 sensors (all) x 4 weather conditions.
Mode: uniform (canonical baseline mode used in Table baselines).

  python sensor_ablation.py
"""
import json, sys, time
from pathlib import Path
import numpy as np

VISTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISTA))
from carla_adapter import process_frame
from carla_detection import detect_frame, match_detections, compute_ap, gt_class, det_class
sys.path.insert(0, str(VISTA / "sim_studies/t2"))
from bg_subtract import build_bg, foreground_mask

CONDITIONS = ["rain_med", "fog", "night"]
SENSOR_CONFIGS = {"1 (NE)": ["ne"], "2 (NE+SW)": ["ne", "sw"], "4 (all)": None}
MODE = "uniform"
N_BG = 200
DIST_THRESH = 3.0


def run_condition_config(condition, sensors):
    bgp = VISTA / "work/bg_models" / f"{condition}_n{N_BG}.npz"
    if bgp.exists():
        z = np.load(bgp); models = {k: z[k] for k in z.files}
    else:
        models = build_bg(condition, N_BG)

    cond_dir = VISTA / "carla_data" / condition
    # test split = contiguous frames 1600-1999, same convention as sim_studies/t2/t2_grid.py
    ids = [f"{i:06d}" for i in range(1600, 2000) if (cond_dir / f"frame_{i:06d}").is_dir()]
    classes = ["vehicle", "pedestrian"]
    tp = {c: [] for c in classes}; fp = {c: [] for c in classes}; n_gt = {c: 0 for c in classes}
    dm = "dynamic_mask"

    for idx in ids:
        fd = cond_dir / f"frame_{idx}"
        pts, cols = process_frame(fd)
        pts[:, cols.index(dm)] = foreground_mask(fd, models, base_threshold=1.0).astype(np.float32)
        gt = json.loads((fd / "gt_boxes.json").read_text())
        dets = detect_frame(pts, cols, MODE, sensors=sensors)
        for cls in classes:
            gt_cls = [o for o in gt["objects"] if gt_class(o["class"]) == cls]
            det_cls = [d for d in dets if det_class(d["label"]) == cls]
            tp_s, fp_s, ng = match_detections(det_cls, gt_cls, dist_threshold=DIST_THRESH)
            tp[cls].extend(tp_s); fp[cls].extend(fp_s); n_gt[cls] += ng

    ap_res = {c: compute_ap(tp[c], fp[c], n_gt[c]) for c in classes}
    return ap_res


def main():
    # clear already run separately (matches t2_table2.json uniform_nonoracle exactly): veh/ped
    results = {"clear": {
        "1 (NE)": {"AP_vehicle": 0.0909, "AP_pedestrian": 0.0},
        "2 (NE+SW)": {"AP_vehicle": 0.1622, "AP_pedestrian": 0.0016},
        "4 (all)": {"AP_vehicle": 0.2176, "AP_pedestrian": 0.0227},
    }}
    t0 = time.time()
    for condition in CONDITIONS:
        results[condition] = {}
        for label, sensors in SENSOR_CONFIGS.items():
            ap_res = run_condition_config(condition, sensors)
            results[condition][label] = {
                "AP_vehicle": round(ap_res["vehicle"], 4),
                "AP_pedestrian": round(ap_res["pedestrian"], 4),
            }
            print(f"[{time.time()-t0:6.1f}s] {condition:9s} sensors={label:10s} "
                  f"veh={ap_res['vehicle']:.4f} ped={ap_res['pedestrian']:.4f}", flush=True)

    out = VISTA / "outputs/sensor_ablation.json"
    out.write_text(json.dumps(results, indent=2))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
