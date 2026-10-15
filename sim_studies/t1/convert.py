#!/usr/bin/env python3
"""
T1 converter: CARLA condition -> OpenPCDet custom dataset (per condition).

Output layout (root = data/carla_pcdet/<cond>/):
  points/<idx>.npy    (N,4) float32  [x, y, z, intensity/255]  world frame
  labels/<idx>.txt    per line: "x y z dx dy dz heading name"   (TRAINING targets)
                      filtered to in-range boxes with >= MIN_PTS points
  ImageSets/{train,val,test}.txt
  frame_stats.json    per-frame per-object {class, range_from_center, n_pts} (ALL objects)

Notes:
- Points come from carla_adapter.process_frame (world frame, matches GT). Raw lidar_*.npy
  are sensor-local and must NOT be used.
- OpenPCDet box = [cx, cy, cz+H/2, L, W, H, deg2rad(yaw)].  center[z] in GT is the BASE.
- Eval does NOT use these labels; eval reads gt_boxes.json via carla_detection.py.
  Label filtering here only affects training quality.
"""
import argparse, json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np

VISTA = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(VISTA))
from carla_adapter import process_frame, _pts_in_obb, CORNERS  # noqa

RANGE_XY = 76.8            # keep boxes whose center is within +/- this in x and y
MIN_PTS  = 5              # min points inside OBB to be a TRAINING target (== paper rho_min)
SPLIT = dict(train=(0, 1400), val=(1400, 1600), test=(1600, 2000))


def cls_name(carla_class: str) -> str:
    c = carla_class.lower()
    return "Pedestrian" if ("walker" in c or "pedestrian" in c) else "Vehicle"


def convert_frame(args):
    frame_dir, out_root = args
    idx = frame_dir.name.split("_")[1]           # '000000'
    try:
        pts, cols = process_frame(frame_dir)
    except Exception as e:
        return (idx, None, f"process_frame failed: {e}")
    xi = [cols.index(c) for c in ("x", "y", "z", "intensity")]
    P = pts[:, xi].astype(np.float32)
    P[:, 3] /= 255.0                              # intensity 0-255 -> 0-1
    np.save(out_root / "points" / f"{idx}.npy", P)

    gt = json.loads((frame_dir / "gt_boxes.json").read_text())
    lines, stats = [], []
    for o in gt.get("objects", []):
        cx, cy, cz = o["bbox"]["center"]
        L, W, H    = o["bbox"]["dimensions"]
        yaw        = o["bbox"]["rotation"][1]      # deg
        n_pts      = int(_pts_in_obb(pts[:, :3], o).sum())
        rng        = float(o.get("distance_from_center", np.hypot(cx, cy)))
        name       = cls_name(o["class"])
        stats.append({"class": name, "range": rng, "n_pts": n_pts})
        in_range = abs(cx) <= RANGE_XY and abs(cy) <= RANGE_XY
        if in_range and n_pts >= MIN_PTS:
            heading = np.deg2rad(yaw)
            lines.append(f"{cx:.4f} {cy:.4f} {cz + H/2:.4f} "
                         f"{L:.4f} {W:.4f} {H:.4f} {heading:.6f} {name}")
    (out_root / "labels" / f"{idx}.txt").write_text("\n".join(lines))
    return (idx, stats, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, help="clear | fog | ...")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    cond_dir = VISTA / "carla_data" / a.condition
    out_root = VISTA / "data/carla_pcdet" / a.condition
    for sub in ("points", "labels", "ImageSets"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    frame_dirs = sorted(cond_dir.glob("frame_*"))
    print(f"{a.condition}: {len(frame_dirs)} frames -> {out_root}")

    all_stats = {}
    with Pool(a.workers) as pool:
        for i, (idx, stats, err) in enumerate(
                pool.imap_unordered(convert_frame,
                                    [(fd, out_root) for fd in frame_dirs], chunksize=8)):
            if err:
                print(f"  SKIP {idx}: {err}")
            else:
                all_stats[idx] = stats
            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(frame_dirs)}]")

    (out_root / "frame_stats.json").write_text(json.dumps(all_stats))

    ids = sorted(all_stats.keys())
    for split, (lo, hi) in SPLIT.items():
        sel = [i for i in ids if lo <= int(i) < hi]
        (out_root / "ImageSets" / f"{split}.txt").write_text("\n".join(sel))
        print(f"  {split}: {len(sel)} frames")
    print("done.")


if __name__ == "__main__":
    main()
