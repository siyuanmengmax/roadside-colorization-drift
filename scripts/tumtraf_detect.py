#!/usr/bin/env python3
"""
TUMTraf-I detection pipeline (journal C1/C2/C3). Ports carla_detection.detect_frame to
real data: ground-referenced frame (via registration ground matrix), oracle foreground
(points-in-cuboid), clustering uniform vs quality-aware, BEV center-distance 11-pt AP.

Reliability options: 'geo' (paper's calculate_reliability, range-dominated ~uniform) or
'signal' (reflectivity-based, non-uniform on real). Lets us test the C2 quality-aware benefit.
"""
import sys, json, time
from pathlib import Path
import numpy as np

VISTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISTA)); sys.path.insert(0, str(VISTA / "scripts"))
from carla_adapter import COLUMNS   # column layout expected by detect_frame
from carla_detection import detect_frame, match_detections, compute_ap


def det_size_class(d):
    """Label-free 2-class assignment by detected footprint (CARLA classifier does not
    transfer to real vehicle scale). Vehicles are large; pedestrians small."""
    return "vehicle" if max(d["l"], d["w"]) > 2.0 else "pedestrian"
from src.preprocessing.find_background import calculate_reliability
from tumtraf_adapter import list_frames, read_pcd_ascii, label_for, load_boxes, ROOT
from signal_reliability import signal_reliability

SEQS = ["a9_dataset_r02_s01", "a9_dataset_r02_s02", "a9_dataset_r02_s03", "a9_dataset_r02_s04"]
LIDAR = "s110_lidar_ouster_south"


def gt_class(t):
    return "pedestrian" if "PED" in t.upper() else "vehicle"


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def points_in_cuboid(xyz, box, margin=0.15):
    R = quat_to_R(box["quat"])
    local = (xyz - np.array(box["center"])) @ R          # world->box
    hl, hw, hh = np.array(box["lwh"]) / 2 + margin
    return (np.abs(local[:, 0]) <= hl) & (np.abs(local[:, 1]) <= hw) & (np.abs(local[:, 2]) <= hh)


def ground_matrix(seq):
    return np.load(ROOT / seq / "registration" / "source_ground_matrix_label.npy")


def build_frame(pcd, seq, reliability):
    """Return (pts[N,11], cols, gt_list) in ground-referenced frame with oracle mask."""
    d = read_pcd_ascii(pcd)
    xyz = d[:, :3]; refl = d[:, 5]
    rng_m = np.linalg.norm(xyz, axis=1).astype(np.float32)
    boxes = load_boxes(label_for(pcd, seq))
    # oracle dynamic mask (points inside any GT cuboid) — computed in raw LiDAR frame
    dyn = np.zeros(len(xyz), bool)
    for b in boxes:
        dyn |= points_in_cuboid(xyz, b)
    # reliability
    if reliability == "geo":
        rel = calculate_reliability(rng_m, refl.astype(np.float32))
    else:
        rel = signal_reliability(refl)
    # transform to ground frame (z up, ground~0)
    G = ground_matrix(seq)
    xyz_g = (G @ np.hstack([xyz, np.ones((len(xyz), 1))]).T).T[:, :3]

    pts = np.zeros((len(xyz), len(COLUMNS)), np.float32)
    pts[:, 0:3] = xyz_g
    pts[:, COLUMNS.index("range")] = rng_m
    pts[:, COLUMNS.index("reflectivity")] = refl
    pts[:, COLUMNS.index("intensity")] = refl
    pts[:, COLUMNS.index("reliability")] = rel
    pts[:, COLUMNS.index("dynamic_mask")] = dyn.astype(np.float32)
    pts[:, COLUMNS.index("lidar_source")] = 0.0
    # GT centers in ground frame for matching
    gt = []
    for b in boxes:
        c = G @ np.array([*b["center"], 1.0])
        gt.append({"bbox": {"center": [float(c[0]), float(c[1]), float(c[2])]}, "cls": gt_class(b["type"])})
    return pts, COLUMNS, gt


def run(reliability, mode, frames):
    classes = ["vehicle", "pedestrian"]
    tp = {c: [] for c in classes}; fp = {c: [] for c in classes}; ngt = {c: 0 for c in classes}
    for pcd, seq in frames:
        pts, cols, gt = build_frame(pcd, seq, reliability)
        dets = detect_frame(pts, cols, mode)
        for c in classes:
            gt_c = [{"bbox": g["bbox"]} for g in gt if g["cls"] == c]
            det_c = [d for d in dets if det_size_class(d) == c]
            tp_s, fp_s, n = match_detections(det_c, gt_c, 3.0)
            tp[c].extend(tp_s); fp[c].extend(fp_s); ngt[c] += n
    ap = {c: compute_ap(tp[c], fp[c], ngt[c]) for c in classes}
    return {"AP_vehicle": round(ap["vehicle"], 4), "AP_pedestrian": round(ap["pedestrian"], 4),
            "n_gt_vehicle": ngt["vehicle"], "n_gt_pedestrian": ngt["pedestrian"]}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    frames = []
    for s in SEQS:
        for p in list_frames(s)[::a.stride]:
            if label_for(p, s).exists():
                frames.append((p, s))
    if a.limit:
        frames = frames[:a.limit]
    print(f"{len(frames)} frames")

    res = {}
    for reliability in ["geo", "signal"]:
        res[reliability] = {}
        for mode in ["uniform", "quality_aware"]:
            t0 = time.time()
            r = run(reliability, mode, frames)
            res[reliability][mode] = r
            print(f"[{reliability}/{mode}] {r}  ({time.time()-t0:.0f}s)", flush=True)
        u = res[reliability]["uniform"]; q = res[reliability]["quality_aware"]
        res[reliability]["delta_qa_minus_uni"] = {
            "veh": round(q["AP_vehicle"] - u["AP_vehicle"], 4),
            "ped": round(q["AP_pedestrian"] - u["AP_pedestrian"], 4)}
        print(f"  => {reliability} delta (qa-uni): {res[reliability]['delta_qa_minus_uni']}")
    out = VISTA / "outputs/c2_quality_aware.json"
    out.write_text(json.dumps({"n_frames": len(frames), "results": res}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
