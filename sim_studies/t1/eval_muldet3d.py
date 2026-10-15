#!/usr/bin/env python3
"""
Re-evaluate MulDet3D (the paper's pipeline) on a SPECIFIC frame-id split, so it is
scored on the exact same test frames as PointPillars -> apples-to-apples for Theme A.

Reuses carla_detection.py internals (oracle GT dynamic mask, same as paper Table 2).
CPU only (safe to run alongside GPU training).

  python eval_muldet3d.py --condition clear --split test --mode uniform
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

VISTA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VISTA))
from carla_adapter import process_frame as adapter_process_frame
from carla_detection import detect_frame, match_detections, compute_ap, gt_class, det_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", default="uniform", choices=["uniform", "quality_aware"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data_root = VISTA / "data/carla_pcdet" / a.condition
    ids = (data_root / "ImageSets" / f"{a.split}.txt").read_text().split()
    cond_dir = VISTA / "carla_data" / a.condition
    classes = ["vehicle", "pedestrian"]
    tp = {c: [] for c in classes}; fp = {c: [] for c in classes}; n_gt = {c: 0 for c in classes}

    t0 = time.time()
    for i, idx in enumerate(ids):
        fd = cond_dir / f"frame_{idx}"
        pts, cols = adapter_process_frame(fd)
        gt = json.loads((fd / "gt_boxes.json").read_text())
        dets = detect_frame(pts, cols, a.mode)
        for cls in classes:
            gt_cls = [o for o in gt["objects"] if gt_class(o["class"]) == cls]
            det_cls = [d for d in dets if det_class(d["label"]) == cls]
            tp_s, fp_s, ng = match_detections(det_cls, gt_cls, dist_threshold=3.0)
            tp[cls].extend(tp_s); fp[cls].extend(fp_s); n_gt[cls] += ng
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ids)}]  {(i+1)/(time.time()-t0):.1f} fps", flush=True)

    ap_res = {c: compute_ap(tp[c], fp[c], n_gt[c]) for c in classes}
    ap_vals = [v for v in ap_res.values() if not np.isnan(v)]
    result = {
        "detector": "MulDet3D(oracle)", "condition": a.condition, "split": a.split,
        "mode": a.mode,
        "AP_vehicle": round(ap_res["vehicle"], 4),
        "AP_pedestrian": round(ap_res["pedestrian"], 4),
        "mAP": round(float(np.mean(ap_vals)), 4) if ap_vals else float("nan"),
        "n_gt_vehicle": n_gt["vehicle"], "n_gt_pedestrian": n_gt["pedestrian"],
        "n_frames": len(ids),
    }
    print("\n==== RESULT ====")
    print(json.dumps(result, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
