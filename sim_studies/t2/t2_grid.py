#!/usr/bin/env python3
"""
T2 full grid: oracle vs non-oracle (bg-subtract) detection AP for the non-oracle Table 2.
Conditions x {uniform, quality_aware} x 2 classes, on the same test split, with Delta.

Efficient: one process_frame per frame; both masks and both modes reuse it.
Output: outputs/sim_studies/t2/t2_table2.{json,md}
"""
import json, sys, time
from pathlib import Path
import numpy as np

VISTA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VISTA))
sys.path.insert(0, str(VISTA / "sim_studies/t2"))
from carla_adapter import process_frame
from carla_detection import detect_frame, match_detections, compute_ap, gt_class, det_class
from bg_subtract import build_bg, foreground_mask

CONDITIONS = ["clear", "rain_med", "fog", "night"]
MODES = ["uniform", "quality_aware"]
CLASSES = ["vehicle", "pedestrian"]
N_BG = 200


def get_bg(cond):
    bgp = VISTA / "work/bg_models" / f"{cond}_n{N_BG}.npz"
    if bgp.exists():
        z = np.load(bgp); return {k: z[k] for k in z.files}
    m = build_bg(cond, N_BG)
    np.savez_compressed(bgp, **m); return m


def score(cond, mode, mask_kind, models, ids, cond_dir, cache):
    tp = {c: [] for c in CLASSES}; fp = {c: [] for c in CLASSES}; n_gt = {c: 0 for c in CLASSES}
    dmi = None
    for idx in ids:
        pts, cols, gt = cache[idx]
        if dmi is None:
            dmi = cols.index("dynamic_mask")
        if mask_kind == "nonoracle":
            pts = pts.copy()
            pts[:, dmi] = cache_fg[cond][idx]
        dets = detect_frame(pts, cols, mode)
        for cls in CLASSES:
            gt_cls = [o for o in gt["objects"] if gt_class(o["class"]) == cls]
            det_cls = [d for d in dets if det_class(d["label"]) == cls]
            tp_s, fp_s, ng = match_detections(det_cls, gt_cls, 3.0)
            tp[cls].extend(tp_s); fp[cls].extend(fp_s); n_gt[cls] += ng
    ap = {c: compute_ap(tp[c], fp[c], n_gt[c]) for c in CLASSES}
    return {"AP_vehicle": round(ap["vehicle"], 4), "AP_pedestrian": round(ap["pedestrian"], 4),
            "mAP": round(float(np.nanmean(list(ap.values()))), 4)}


cache_fg = {}

def main():
    results = {}
    for cond in CONDITIONS:
        t0 = time.time()
        models = get_bg(cond)
        cond_dir = VISTA / "carla_data" / cond
        # test split = contiguous frames 1600-1999 (same as T1), generated directly
        ids = [f"{i:06d}" for i in range(1600, 2000)
               if (cond_dir / f"frame_{i:06d}").is_dir()]
        # Precompute per-frame: process_frame + gt + non-oracle mask (once)
        cache = {}; cache_fg[cond] = {}
        for idx in ids:
            fd = cond_dir / f"frame_{idx}"
            pts, cols = process_frame(fd)
            gt = json.loads((fd / "gt_boxes.json").read_text())
            cache[idx] = (pts, cols, gt)
            cache_fg[cond][idx] = foreground_mask(fd, models).astype(np.float32)
        results[cond] = {}
        for mode in MODES:
            for kind in ("oracle", "nonoracle"):
                r = score(cond, mode, kind, models, ids, cond_dir, cache)
                results[cond][f"{mode}_{kind}"] = r
        # deltas
        for mode in MODES:
            o = results[cond][f"{mode}_oracle"]; n = results[cond][f"{mode}_nonoracle"]
            results[cond][f"{mode}_delta_mAP"] = round(n["mAP"] - o["mAP"], 4)
        print(f"[{cond}] done in {time.time()-t0:.0f}s  "
              f"uni O/N mAP {results[cond]['uniform_oracle']['mAP']}/{results[cond]['uniform_nonoracle']['mAP']}",
              flush=True)
        del cache; cache_fg[cond].clear()

    out = VISTA / "outputs/sim_studies/t2"
    (out / "t2_table2.json").write_text(json.dumps(results, indent=2))

    # markdown
    lines = ["# T2 non-oracle Table 2 (test split, 3m+11pt AP)\n",
             "| cond | mode | mask | Veh | Ped | mAP | ΔmAP |", "|---|---|---|---|---|---|---|"]
    for cond in CONDITIONS:
        for mode in MODES:
            for kind in ("oracle", "nonoracle"):
                r = results[cond][f"{mode}_{kind}"]
                d = results[cond][f"{mode}_delta_mAP"] if kind == "nonoracle" else ""
                lines.append(f"| {cond} | {mode} | {kind} | {r['AP_vehicle']} | {r['AP_pedestrian']} | {r['mAP']} | {d} |")
    (out / "t2_table2.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("\nwrote", out / "t2_table2.json")


if __name__ == "__main__":
    main()
