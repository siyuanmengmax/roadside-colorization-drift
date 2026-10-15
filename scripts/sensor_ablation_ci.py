#!/usr/bin/env python3
"""
Sensor-count ablation with block-bootstrap CIs (same protocol as sensor_ablation.py:
label-free background-subtraction foreground, uniform mode, 3 m BEV match, 11-pt AP,
test frames 1600-1999 per condition).

Adds, per condition x config: vehicle/pedestrian AP with 95% block-bootstrap CI
(50-frame blocks, 1000 resamples), TP/FP/FN and recall, and PAIRED CIs for the
vehicle-AP differences 2-vs-1 and 4-vs-2 sensors.

  python sensor_ablation_ci.py  ->  outputs/sensor_ablation_prf.json
"""
import json, os, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
VISTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISTA)); sys.path.insert(0, str(VISTA / "sim_studies/t2"))
from carla_adapter import process_frame
from carla_detection import detect_frame, match_detections, compute_ap, gt_class, det_class
from bg_subtract import foreground_mask
sys.path.insert(0, str(VISTA / "scripts"))
import prf_utils as PRF

CONDITIONS = ["clear", "rain_med", "fog", "night"]
SENSOR_CONFIGS = {"1": ["ne"], "2": ["ne", "sw"], "4": None}
CLASSES = ["vehicle", "pedestrian"]
N_BG, DIST_THRESH, BLOCK, N_BOOT = 200, 3.0, 50, 1000
MODELS = {}  # condition -> background models, loaded lazily in each worker


def process(job):
    condition, fd = job
    if condition not in MODELS:
        z = np.load(VISTA / "work/bg_models" / f"{condition}_n{N_BG}.npz")
        MODELS[condition] = {k: z[k] for k in z.files}
    fd = Path(fd)
    pts, cols = process_frame(fd)
    gt = json.loads((fd / "gt_boxes.json").read_text())
    out = {"4_oracle": {}}  # reference: all sensors, foreground from ground-truth boxes
    dets = detect_frame(pts, cols, "uniform", sensors=None)
    for cls in CLASSES:
        out["4_oracle"][cls] = match_detections([d for d in dets if det_class(d["label"]) == cls],
                                                [o for o in gt["objects"] if gt_class(o["class"]) == cls],
                                                dist_threshold=DIST_THRESH)
    pts[:, cols.index("dynamic_mask")] = foreground_mask(fd, MODELS[condition], base_threshold=1.0).astype(np.float32)
    for label, sensors in SENSOR_CONFIGS.items():
        dets = detect_frame(pts, cols, "uniform", sensors=sensors)
        out[label] = {}
        for cls in CLASSES:
            gt_cls = [o for o in gt["objects"] if gt_class(o["class"]) == cls]
            det_cls = [d for d in dets if det_class(d["label"]) == cls]
            out[label][cls] = match_detections(det_cls, gt_cls, dist_threshold=DIST_THRESH)
    return out


def pooled(recs, idx, cfg, cls):
    tp, fp, n = [], [], 0
    for i in idx:
        t, f, g = recs[i][cfg][cls]
        tp.extend(t); fp.extend(f); n += g
    return compute_ap(tp, fp, n), len(tp), len(fp), n


def main():
    t0 = time.time()
    results = {}
    with mp.get_context("spawn").Pool(16) as pool:
        for cond in CONDITIONS:
            cdir = VISTA / "carla_data" / cond
            frames = [str(cdir / f"frame_{i:06d}") for i in range(1600, 2000) if (cdir / f"frame_{i:06d}").is_dir()]
            recs = pool.map(process, [(cond, f) for f in frames], chunksize=4)
            n = len(recs); nblk = int(np.ceil(n / BLOCK))
            blocks = [list(range(i * BLOCK, min((i + 1) * BLOCK, n))) for i in range(nblk)]
            rng = np.random.RandomState(0)
            draws = [[i for b in rng.randint(0, nblk, nblk) for i in blocks[b]] for _ in range(N_BOOT)]
            res, boot, cnt = {"n_frames": n}, {}, {}
            for cfg in [*SENSOR_CONFIGS, "4_oracle"]:
                res[cfg] = {}
                for cls in CLASSES:
                    apv, ntp, nfp, ngt = pooled(recs, range(n), cfg, cls)
                    boot[cfg, cls] = np.array([pooled(recs, dr, cfg, cls)[0] for dr in draws])
                    res[cfg][cls] = {"AP": round(apv, 4),
                                     "CI95": [round(float(np.nanpercentile(boot[cfg, cls], q)), 4) for q in (2.5, 97.5)],
                                     "TP": ntp, "FP": nfp, "FN": ngt - ntp, "recall": round(ntp / max(ngt, 1), 4)}
                    cnt[cfg, cls] = PRF.counts([r[cfg][cls] for r in recs])
                    res[cfg][cls].update(PRF.summarize(cnt[cfg, cls], draws))
            for hi, lo in (("2", "1"), ("4", "2"), ("4", "1"), ("4_oracle", "4")):
                diff = boot[hi, "vehicle"] - boot[lo, "vehicle"]
                ci = [round(float(np.nanpercentile(diff, q)), 4) for q in (2.5, 97.5)]
                res[f"veh_{hi}_minus_{lo}"] = {
                    "delta_AP": round(res[hi]["vehicle"]["AP"] - res[lo]["vehicle"]["AP"], 4),
                    "paired_CI95": ci, "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
                res[f"veh_{hi}_minus_{lo}"].update(PRF.paired(cnt[hi, "vehicle"], cnt[lo, "vehicle"], draws))
                res[f"ped_{hi}_minus_{lo}"] = PRF.paired(cnt[hi, "pedestrian"], cnt[lo, "pedestrian"], draws)
            np.savez(VISTA / f"outputs/sensor_ablation_counts_{cond}.npz",
                     **{f"{c}_{k}": cnt[c, k] for c in [*SENSOR_CONFIGS, "4_oracle"] for k in CLASSES})
            results[cond] = res
            print(f"[{time.time()-t0:5.0f}s] {cond}: " + " | ".join(
                f"{c}: {res[c]['vehicle']['AP']:.3f} {res[c]['vehicle']['CI95']} r={res[c]['vehicle']['recall']:.3f} F1={res[c]['vehicle']['F1']:.3f}"
                for c in SENSOR_CONFIGS), flush=True)
            print("    ", {k: v for k, v in res.items() if k.startswith("veh_")}, flush=True)
    p = VISTA / "outputs/sensor_ablation_prf.json"
    p.write_text(json.dumps(results, indent=2)); print("wrote", p)


if __name__ == "__main__":
    main()
