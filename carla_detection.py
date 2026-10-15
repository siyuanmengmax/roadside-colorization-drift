#!/usr/bin/env python3
"""
CARLA Detection Pipeline
Experiments 1 & 2: Quality-aware vs Uniform, Multi-LiDAR ablation.

Three detection modes:
  uniform       — baseline, equal point weighting (clustering_stage_1_no_rel)
  quality_aware — reliability-weighted density   (clustering_stage_1)
  color_aware   — quality_aware + color-consistency DBSCAN gate
                  (clustering_stage_1_color; needs camera images in frame dir)

mAP is computed by matching detected bboxes to CARLA GT boxes via BEV IoU.

Usage:
    # Single frame sanity check
    python carla_detection.py --frame-dir carla_data/clear/frame_000004 --mode uniform

    # Full condition
    python carla_detection.py --condition carla_data/clear --mode quality_aware

    # Color-aware mode
    python carla_detection.py --condition carla_data/clear --mode color_aware

    # Ablation: single LiDAR only
    python carla_detection.py --condition carla_data/clear --mode uniform --sensors ne
"""

import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# mock hdbscan so clustering.py can be imported on Python 3.7
import types
sys.modules.setdefault("hdbscan", types.ModuleType("hdbscan"))

from carla_adapter import (process_frame as adapter_process_frame,
                           process_frame_color, COLUMNS, COLUMNS_COLOR, CORNERS)
from src.detection.clustering import (clustering_stage_1, clustering_stage_1_no_rel,
                                      clustering_stage_1_color)
from src.detection.classifying import classifying_stage_1, classifying_stage_2
from src.detection.compute_bbox import compute_bbox


# ── BEV IoU ────────────────────────────────────────────────────────────────────

def _rect_corners(cx, cy, w, l, yaw):
    """Return (4,2) corners of a rotated rectangle in BEV."""
    hw, hl = w / 2, l / 2
    corners = np.array([[-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl]], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return corners @ R.T + np.array([cx, cy])


def _clip_polygon(subject, clip):
    """Sutherland-Hodgman polygon clipping (2-D, convex clip polygon)."""
    def inside(p, a, b):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0

    def intersect(p1, p2, p3, p4):
        d1 = p2 - p1
        d2 = p4 - p3
        cross = d1[0]*d2[1] - d1[1]*d2[0]
        if abs(cross) < 1e-10:
            return p1
        t = ((p3[0]-p1[0])*d2[1] - (p3[1]-p1[1])*d2[0]) / cross
        return p1 + t * d1

    out = list(subject)
    n = len(clip)
    for i in range(n):
        if not out:
            break
        a, b = clip[i], clip[(i+1) % n]
        inp = out
        out = []
        for j in range(len(inp)):
            cur, prev = inp[j], inp[j-1]
            if inside(cur, a, b):
                if not inside(prev, a, b):
                    out.append(intersect(prev, cur, a, b))
                out.append(cur)
            elif inside(prev, a, b):
                out.append(intersect(prev, cur, a, b))
    return out


def _polygon_area(pts):
    if len(pts) < 3:
        return 0.0
    pts = np.array(pts)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def bev_iou(box_a, box_b) -> float:
    """BEV IoU between two boxes. Each box: (cx, cy, w, l, yaw)."""
    ca = _rect_corners(*box_a)
    cb = _rect_corners(*box_b)
    inter_pts = _clip_polygon(list(ca), list(cb))
    inter_area = _polygon_area(inter_pts)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter_area
    return inter_area / (union + 1e-8)


# ── GT class mapping ───────────────────────────────────────────────────────────

def gt_class(carla_class: str) -> str:
    """Map CARLA blueprint class to evaluation category."""
    c = carla_class.lower()
    if "walker" in c or "pedestrian" in c:
        return "pedestrian"
    return "vehicle"   # all vehicle.* blueprints


_LABEL_NAMES = ["person", "small vehicle", "large vehicle", "other"]

def det_class(label) -> str:
    """Map detection label (int index or string) to evaluation category."""
    if isinstance(label, (int, np.integer)):
        label = _LABEL_NAMES[label] if 0 <= label < len(_LABEL_NAMES) else "other"
    label = str(label).lower()
    if "person" in label:
        return "pedestrian"
    if "vehicle" in label:
        return "vehicle"
    return "other"


# ── Per-frame detection ────────────────────────────────────────────────────────

def detect_frame(pts: np.ndarray, cols: List[str], mode: str,
                 sensors: Optional[List[str]] = None,
                 params: dict = None) -> List[dict]:
    """Run detection on one fused point cloud, return list of bbox dicts.

    Args:
        pts:     (N, 11) array from carla_adapter.process_frame()
        cols:    column name list (== COLUMNS)
        mode:    'uniform' | 'quality_aware'
        sensors: restrict to subset of ['ne','nw','se','sw'] (ablation)
        params:  clustering hyperparameters
    """
    if params is None:
        params = {"rho_min": 5, "alpha": 0.0, "eps_min": 0.3, "eps_max": 2.0}

    # Sensor ablation filter
    if sensors:
        src_idx = [CORNERS.index(s) for s in sensors]
        mask = np.isin(pts[:, cols.index("lidar_source")], src_idx)
        pts = pts[mask]

    # Dynamic points only (GT mask replaces background subtraction)
    dyn = pts[:, cols.index("dynamic_mask")] > 0
    dyn_pts = pts[dyn]

    if len(dyn_pts) < params["rho_min"]:
        return []

    # Height filter (remove ground + above-vehicle returns)
    z = dyn_pts[:, cols.index("z")]
    ht_mask = (z >= 0.1) & (z <= 3.5)
    dyn_pts = dyn_pts[ht_mask]

    if len(dyn_pts) < params["rho_min"]:
        return []

    # Clustering
    if mode == "quality_aware":
        clusters, clusters_coords, _ = clustering_stage_1(
            dyn_pts, cols,
            rho_min=params["rho_min"], alpha=params["alpha"],
            eps_min=params["eps_min"], eps_max=params["eps_max"])
    elif mode == "color_aware":
        r_idx   = cols.index("R")
        rgb_arr = dyn_pts[:, r_idx:r_idx + 3]          # R G B columns
        rrel    = dyn_pts[:, cols.index("rgb_rel")]
        clusters, clusters_coords, _ = clustering_stage_1_color(
            dyn_pts, cols, rgb_arr, rrel,
            rho_min=params["rho_min"], alpha=params["alpha"],
            eps_min=params["eps_min"], eps_max=params["eps_max"],
            delta=params.get("delta", 30.0),
            rgb_gate=params.get("rgb_gate", 0.25))
    else:
        clusters, clusters_coords, _ = clustering_stage_1_no_rel(
            dyn_pts, cols,
            rho_min=params["rho_min"], alpha=params["alpha"],
            eps_min=params["eps_min"], eps_max=params["eps_max"])

    if not clusters:
        return []

    # Classification + bbox
    coarse = classifying_stage_1(clusters_coords)
    labels, probs = classifying_stage_2(clusters_coords, coarse)

    rel_col = cols.index("reliability")
    bboxes = []
    for i, cluster in enumerate(clusters):
        bb = compute_bbox(clusters_coords[i])   # [x,y,z, w,l,h, yaw]
        bboxes.append({
            "cx": bb[0], "cy": bb[1], "cz": bb[2],
            "w": bb[3],  "l": bb[4], "h": bb[5], "yaw": bb[6],
            "label": labels[i],
            "confidence": probs[i],
            "n_pts": len(cluster),
            "mean_reliability": float(dyn_pts[cluster, rel_col].mean()),
        })
    return bboxes


# ── mAP evaluation ────────────────────────────────────────────────────────────

def match_detections(dets: List[dict], gt_objs: List[dict],
                     dist_threshold: float = 3.0) -> Tuple[List, List, int]:
    """Match detections to GT objects using BEV center distance.

    Standard metric for roadside LiDAR: a detection is TP if its 2D center
    is within dist_threshold metres of any unmatched GT box center.
    This handles partial-visibility bbox underestimation gracefully.

    Returns:
        tp_scores: confidence scores of true-positive detections
        fp_scores: confidence scores of false-positive detections
        n_gt:      number of GT objects (recall denominator)
    """
    tp_scores, fp_scores = [], []
    matched_gt = set()

    dets_sorted = sorted(dets, key=lambda d: d["confidence"], reverse=True)

    for det in dets_sorted:
        best_dist, best_j = np.inf, -1
        for j, obj in enumerate(gt_objs):
            if j in matched_gt:
                continue
            cx, cy, _ = obj["bbox"]["center"]
            dist = np.sqrt((det["cx"] - cx) ** 2 + (det["cy"] - cy) ** 2)
            if dist < best_dist:
                best_dist, best_j = dist, j

        if best_dist <= dist_threshold and best_j >= 0:
            tp_scores.append(det["confidence"])
            matched_gt.add(best_j)
        else:
            fp_scores.append(det["confidence"])

    return tp_scores, fp_scores, len(gt_objs)


def compute_ap(tp_scores, fp_scores, n_gt) -> float:
    """Compute Average Precision from TP/FP scores."""
    if n_gt == 0:
        return float("nan")
    all_scores = [(s, True) for s in tp_scores] + [(s, False) for s in fp_scores]
    all_scores.sort(key=lambda x: -x[0])

    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []
    for _, is_tp in all_scores:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_gt)

    # 11-point interpolation
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        p_at_r = [p for p, r in zip(precisions, recalls) if r >= thr]
        ap += (max(p_at_r) if p_at_r else 0.0) / 11
    return ap


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_condition(condition_dir: Path, mode: str,
                  sensors: Optional[List[str]] = None,
                  dist_threshold: float = 3.0,
                  params: dict = None,
                  max_frames: int = None) -> dict:
    """Run detection + evaluation on all frames in a condition directory."""
    frame_dirs = sorted(condition_dir.glob("frame_*"))
    if max_frames:
        frame_dirs = frame_dirs[:max_frames]

    sensor_tag = "+".join(sensors) if sensors else "all"
    print(f"\n{'='*60}")
    print(f"Condition : {condition_dir.name}")
    print(f"Mode      : {mode}")
    print(f"Sensors   : {sensor_tag}")
    print(f"Frames    : {len(frame_dirs)}")
    print(f"{'='*60}")

    # Per-class accumulators
    classes = ["vehicle", "pedestrian"]
    tp = {c: [] for c in classes}
    fp = {c: [] for c in classes}
    n_gt = {c: 0 for c in classes}

    per_frame = []
    t0 = time.time()

    for i, fd in enumerate(frame_dirs):
        try:
            if mode == "color_aware":
                pts, cols = process_frame_color(fd)
            else:
                pts, cols = adapter_process_frame(fd)
            gt = json.loads((fd / "gt_boxes.json").read_text())
        except Exception as e:
            print(f"  SKIP {fd.name}: {e}")
            continue

        dets = detect_frame(pts, cols, mode, sensors=sensors, params=params)

        # Separate GT and detections by class
        for cls in classes:
            gt_cls  = [o for o in gt["objects"] if gt_class(o["class"]) == cls]
            det_cls = [d for d in dets        if det_class(d["label"]) == cls]
            tp_s, fp_s, ng = match_detections(det_cls, gt_cls, dist_threshold)
            tp[cls].extend(tp_s)
            fp[cls].extend(fp_s)
            n_gt[cls] += ng

        per_frame.append({
            "frame": fd.name,
            "n_det": len(dets),
            "n_gt":  len(gt["objects"]),
        })

        if (i+1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            fps = (i+1) / elapsed
            print(f"  [{i+1:>4}/{len(frame_dirs)}]  {fps:.1f} fps")

    # Compute AP per class
    ap = {}
    for cls in classes:
        ap[cls] = compute_ap(tp[cls], fp[cls], n_gt[cls])

    ap_vals = [v for v in ap.values() if not np.isnan(v)]
    mAP = float(np.mean(ap_vals)) if ap_vals else float("nan")

    results = {
        "condition":    condition_dir.name,
        "mode":         mode,
        "sensors":      sensor_tag,
        "dist_threshold_m": dist_threshold,
        "n_frames":     len(per_frame),
        "mAP":          round(mAP, 4),
        "AP_vehicle":   round(ap.get("vehicle", float("nan")), 4),
        "AP_pedestrian":round(ap.get("pedestrian", float("nan")), 4),
        "n_gt_vehicle":     n_gt["vehicle"],
        "n_gt_pedestrian":  n_gt["pedestrian"],
    }

    print(f"\n  Results:")
    print(f"    mAP            : {mAP:.4f}")
    print(f"    AP vehicle     : {ap.get('vehicle', float('nan')):.4f}  (GT: {n_gt['vehicle']})")
    print(f"    AP pedestrian  : {ap.get('pedestrian', float('nan')):.4f}  (GT: {n_gt['pedestrian']})")
    return results, per_frame


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CARLA detection pipeline")
    p.add_argument("--mode", choices=["uniform", "quality_aware", "color_aware"], required=True)
    p.add_argument("--frame-dir",  type=Path, help="Single frame (sanity check)")
    p.add_argument("--condition",  type=Path, help="Condition directory (batch)")
    p.add_argument("--sensors", nargs="+", choices=CORNERS,
                   help="Restrict to subset of sensors (ablation). Default: all 4")
    p.add_argument("--dist-threshold", type=float, default=3.0,
                   help="BEV center distance threshold for TP match (metres, default 3.0)")
    p.add_argument("--max-frames", type=int, help="Limit frames per condition (testing)")
    p.add_argument("--out", type=Path, default=Path("results"),
                   help="Output directory for result CSV/JSON")
    p.add_argument("--rho-min",  type=float, default=5.0)
    p.add_argument("--alpha",    type=float, default=0.0)
    p.add_argument("--eps-min",  type=float, default=0.3)
    p.add_argument("--eps-max",  type=float, default=2.0)
    p.add_argument("--delta",    type=float, default=30.0,
                   help="color_aware: max mean-L1 color dist for cluster expansion (default 30)")
    p.add_argument("--rgb-gate", type=float, default=0.25,
                   help="color_aware: min rgb_rel to activate color gating (default 0.25)")
    return p.parse_args()


def main():
    args = parse_args()
    params = {
        "rho_min": args.rho_min, "alpha": args.alpha,
        "eps_min": args.eps_min, "eps_max": args.eps_max,
        "delta":   args.delta,   "rgb_gate": args.rgb_gate,
    }

    if args.frame_dir:
        # Single-frame sanity check
        if args.mode == "color_aware":
            pts, cols = process_frame_color(args.frame_dir)
        else:
            pts, cols = adapter_process_frame(args.frame_dir)
        gt = json.loads((args.frame_dir / "gt_boxes.json").read_text())
        t0 = time.time()
        dets = detect_frame(pts, cols, args.mode, sensors=args.sensors, params=params)
        elapsed = time.time() - t0
        print(f"\nFrame : {args.frame_dir}")
        print(f"Mode  : {args.mode}")
        print(f"Time  : {elapsed:.2f}s")
        print(f"GT objects  : {len(gt['objects'])}")
        print(f"Detections  : {len(dets)}")
        for d in dets:
            print(f"  [{d['label']:>14}]  conf={d['confidence']:.2f}  "
                  f"pts={d['n_pts']:>4}  rel={d['mean_reliability']:.3f}  "
                  f"pos=({d['cx']:.1f}, {d['cy']:.1f}, {d['cz']:.1f})")
        return

    if args.condition:
        results, per_frame = run_condition(
            args.condition, args.mode,
            sensors=args.sensors,
            dist_threshold=args.dist_threshold,
            params=params,
            max_frames=args.max_frames,
        )
        # Save outputs
        args.out.mkdir(parents=True, exist_ok=True)
        sensor_tag = "+".join(args.sensors) if args.sensors else "all"
        stem = f"{args.condition.name}_{args.mode}_{sensor_tag}"
        json_path = args.out / f"{stem}.json"
        csv_path  = args.out / f"{stem}_frames.csv"
        json_path.write_text(json.dumps(results, indent=2))
        pd.DataFrame(per_frame).to_csv(csv_path, index=False)
        print(f"\nSaved: {json_path}")
        print(f"Saved: {csv_path}")
        return

    print("ERROR: specify --frame-dir or --condition")


if __name__ == "__main__":
    main()
