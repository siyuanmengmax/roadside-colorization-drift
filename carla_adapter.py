#!/usr/bin/env python3
"""
CARLA → Pipeline Adapter

Converts a CARLA frame directory into the 11-column format expected by
src/preprocessing/find_background.py and src/detection/clustering.py.

Output columns (world frame, all 4 LiDARs fused):
  idx  name           notes
   0   x              world frame (metres)
   1   y
   2   z
   3   range          sqrt(x²+y²+z²) in sensor-local frame
   4   reflectivity   CARLA intensity × 255  (0–255 scale)
   5   near_ir        0  (CARLA has no near-IR channel)
   6   intensity      same as reflectivity
   7   flags          0
   8   reliability    calculate_reliability(range, reflectivity)
   9   dynamic_mask   1 = inside a GT bounding box, 0 = static
  10   lidar_source   0=ne  1=nw  2=se  3=sw

Usage:
    # Process one frame, get numpy array
    from carla_adapter import process_frame
    pts, cols = process_frame(Path("carla_data/clear/frame_000004"))

    # Batch: save all frames in a condition as .bin files
    python carla_adapter.py --condition carla_data/clear --out pipeline_data/clear
"""

import argparse
import json
import math
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Pull reliability model directly from existing pipeline
sys.path.insert(0, str(Path(__file__).parent))
from src.preprocessing.find_background import calculate_reliability

# ── constants ──────────────────────────────────────────────────────────────────

CORNERS     = ["ne", "nw", "se", "sw"]
CORNER_IDX  = {c: i for i, c in enumerate(CORNERS)}

COLUMNS = [
    "x", "y", "z",
    "range", "reflectivity", "near_ir", "intensity", "flags",
    "reliability", "dynamic_mask", "lidar_source",
]


# ── CARLA coordinate math ──────────────────────────────────────────────────────

def carla_transform_matrix(pose: dict) -> np.ndarray:
    """4×4 UE4 transform matrix from a sensor pose dict."""
    cy = np.cos(np.radians(pose["yaw"]))
    sy = np.sin(np.radians(pose["yaw"]))
    cp = np.cos(np.radians(pose["pitch"]))
    sp = np.sin(np.radians(pose["pitch"]))
    cr = np.cos(np.radians(pose["roll"]))
    sr = np.sin(np.radians(pose["roll"]))
    M = np.eye(4)
    M[0,0]=cp*cy; M[0,1]=cy*sp*sr-sy*cr; M[0,2]=-cy*sp*cr-sy*sr
    M[1,0]=sy*cp; M[1,1]=sy*sp*sr+cy*cr; M[1,2]=-sy*sp*cr+cy*sr
    M[2,0]=sp;    M[2,1]=-cp*sr;          M[2,2]=cp*cr
    M[0,3]=pose["x"]; M[1,3]=pose["y"]; M[2,3]=pose["z"]
    return M


def to_world(pts_xyz: np.ndarray, pose: dict) -> np.ndarray:
    """Transform (N,3) sensor-local points to world frame."""
    M = carla_transform_matrix(pose)
    h = np.hstack([pts_xyz, np.ones((len(pts_xyz), 1))])
    return (M @ h.T).T[:, :3]


# ── GT dynamic mask (OBB test) ─────────────────────────────────────────────────

def _pts_in_obb(pts_world: np.ndarray, obj: dict) -> np.ndarray:
    """Return boolean mask: True for points inside this oriented bounding box.

    Rotates points into the box's local frame (yaw only — pitch/roll negligible
    for ground vehicles), then does an axis-aligned half-extent test.
    A small margin (0.3 m) is added to catch points on object surfaces.
    """
    cx, cy, cz = obj["bbox"]["center"]
    length, width, height = obj["bbox"]["dimensions"]
    yaw_deg = obj["bbox"]["rotation"][1]          # CARLA rotation[1] = yaw

    margin = 0.3
    half_l = length / 2 + margin
    half_w = width  / 2 + margin
    half_h = height / 2 + margin

    # Translate to box-centred frame
    dx = pts_world[:, 0] - cx
    dy = pts_world[:, 1] - cy
    dz = pts_world[:, 2] - cz

    # Rotate by -yaw around z-axis
    th = np.radians(-yaw_deg)
    cos_th, sin_th = np.cos(th), np.sin(th)
    lx =  cos_th * dx + sin_th * dy
    ly = -sin_th * dx + cos_th * dy

    return (
        (np.abs(lx) <= half_l) &
        (np.abs(ly) <= half_w) &
        (np.abs(dz) <= half_h)
    )


def compute_dynamic_mask(pts_world: np.ndarray, gt: dict) -> np.ndarray:
    """Return (N,) float32 mask: 1.0 inside any GT box, 0.0 otherwise."""
    mask = np.zeros(len(pts_world), dtype=bool)
    for obj in gt.get("objects", []):
        mask |= _pts_in_obb(pts_world, obj)
    return mask.astype(np.float32)


# ── Per-sensor conversion ──────────────────────────────────────────────────────

def convert_sensor(pts_local: np.ndarray, pose: dict,
                   gt: dict, source_idx: int) -> np.ndarray:
    """Convert one sensor's raw (N,4) CARLA cloud to the 11-column format.

    Args:
        pts_local:  (N,4) [x,y,z,intensity] in sensor-local UE4 frame.
        pose:       sensor pose dict from metadata.json.
        gt:         gt_boxes.json dict for this frame.
        source_idx: 0=ne, 1=nw, 2=se, 3=sw.

    Returns:
        (N, 11) float32 array with COLUMNS layout.
    """
    N = len(pts_local)

    # Sensor-local xyz + intensity
    xyz_local  = pts_local[:, :3]
    intensity  = pts_local[:, 3]           # normalised 0–1 from CARLA

    # range in sensor-local frame (physically meaningful for reliability model)
    rng = np.sqrt((xyz_local ** 2).sum(axis=1)).astype(np.float32)

    # Reflectivity: scale intensity to 0–255 to match real OS-1 convention
    reflectivity = np.clip(intensity * 255.0, 0, 255).astype(np.float32)

    # Reliability (uses existing pipeline model — zero modification)
    reliability = calculate_reliability(rng, reflectivity)

    # World-frame xyz
    xyz_world = to_world(xyz_local, pose).astype(np.float32)

    # Dynamic mask from GT OBBs (replaces build_background for simulation)
    dyn_mask = compute_dynamic_mask(xyz_world, gt)

    out = np.zeros((N, len(COLUMNS)), dtype=np.float32)
    out[:, 0] = xyz_world[:, 0]        # x  (world)
    out[:, 1] = xyz_world[:, 1]        # y  (world)
    out[:, 2] = xyz_world[:, 2]        # z  (world)
    out[:, 3] = rng                    # range
    out[:, 4] = reflectivity           # reflectivity
    out[:, 5] = 0.0                    # near_ir  (no equivalent in CARLA)
    out[:, 6] = reflectivity           # intensity (same as reflectivity)
    out[:, 7] = 0.0                    # flags
    out[:, 8] = reliability            # reliability
    out[:, 9] = dyn_mask               # dynamic_mask
    out[:, 10] = float(source_idx)     # lidar_source

    return out


# ── Frame-level processing ─────────────────────────────────────────────────────

def process_frame(frame_dir: Path) -> Tuple[np.ndarray, List[str]]:
    """Load and convert one CARLA frame → fused 11-column point cloud.

    Returns:
        pts:     (N_total, 11) float32, all 4 sensors fused in world frame.
        columns: list of column names (== COLUMNS).
    """
    meta = json.loads((frame_dir / "metadata.json").read_text())
    gt   = json.loads((frame_dir / "gt_boxes.json").read_text())
    poses = meta["sensor_poses"]

    parts = []
    for corner in CORNERS:
        p = frame_dir / f"lidar_{corner}.npy"
        if not p.exists():
            continue
        pts_local = np.load(p)                          # (N, 4)
        pose      = poses[corner]
        src_idx   = CORNER_IDX[corner]
        parts.append(convert_sensor(pts_local, pose, gt, src_idx))

    if not parts:
        raise FileNotFoundError(f"No lidar_*.npy files found in {frame_dir}")

    fused = np.vstack(parts)
    return fused, COLUMNS


# ── Color-augmented frame processing ─────────────────────────────────────────

# Extended column list: base 11 + R, G, B, rgb_rel
COLUMNS_COLOR = COLUMNS + ["R", "G", "B", "rgb_rel"]

def process_frame_color(frame_dir: Path):
    """Like process_frame, but also colorizes with born_fused_360 and appends
    R, G, B, rgb_rel columns. Used by the color_aware detection mode.

    Returns:
        pts:     (N_total, 15) float32  — COLUMNS_COLOR layout.
        columns: COLUMNS_COLOR name list.
    """
    import sys as _sys
    import types as _types
    _sys.modules.setdefault("hdbscan", _types.ModuleType("hdbscan"))
    from born_fused import born_fused_360, spectral_reliability, CAM_360_OFFSETS
    import cv2

    meta  = json.loads((frame_dir / "metadata.json").read_text())
    gt    = json.loads((frame_dir / "gt_boxes.json").read_text())
    poses = meta["sensor_poses"]

    parts = []
    for corner in CORNERS:
        p = frame_dir / f"lidar_{corner}.npy"
        if not p.exists():
            continue
        pts_local = np.load(p)                      # (N, 4)
        pose      = poses[corner]
        src_idx   = CORNER_IDX[corner]

        # 11-column base
        base = convert_sensor(pts_local, pose, gt, src_idx)   # (N, 11) world frame

        # Colorize in sensor-local frame (born_fused_360 expects sensor-local pts)
        images = {off: cv2.imread(str(frame_dir / f"camera_{corner}_{off}.png"))
                  for off in CAM_360_OFFSETS}
        colored = born_fused_360(pts_local, images)    # (N, 8): x,y,z,int,R,G,B,has_color
        rgb_colors = colored[:, 4:7].astype(np.float32)  # (N, 3)  R G B 0-255

        rel2 = spectral_reliability(colored)           # (N, 2): geo_rel, rgb_rel
        rgb_r = rel2[:, 1].astype(np.float32)          # (N,)

        # Zero out RGB for points outside camera FOV (has_color == 0)
        no_color = colored[:, 7] < 0.5
        rgb_colors[no_color] = 0.0
        rgb_r[no_color] = 0.0

        extra = np.column_stack([rgb_colors, rgb_r])   # (N, 4)
        parts.append(np.hstack([base, extra]))          # (N, 15)

    if not parts:
        raise FileNotFoundError(f"No lidar_*.npy files found in {frame_dir}")

    fused = np.vstack(parts).astype(np.float32)
    return fused, COLUMNS_COLOR


# ── Batch export ───────────────────────────────────────────────────────────────

def export_condition(condition_dir: Path, out_dir: Path,
                     max_frames: int = None) -> None:
    """Convert all frames in a condition directory to .bin files.

    Output: out_dir/frame_{N:06d}.bin  (float32 raw, shape N×11)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dirs = sorted(condition_dir.glob("frame_*"))
    if max_frames:
        frame_dirs = frame_dirs[:max_frames]

    print(f"{condition_dir.name}: {len(frame_dirs)} frames → {out_dir}")
    for fd in frame_dirs:
        try:
            pts, _ = process_frame(fd)
            out_path = out_dir / f"{fd.name}.bin"
            pts.astype(np.float32).tofile(out_path)
        except Exception as e:
            print(f"  SKIP {fd.name}: {e}")

    print(f"  Done. {len(frame_dirs)} frames exported.")


# ── Quick sanity check ─────────────────────────────────────────────────────────

def sanity_check(frame_dir: Path) -> None:
    """Print per-column stats and dynamic-point counts for one frame."""
    pts, cols = process_frame(frame_dir)
    print(f"\nFrame: {frame_dir}")
    print(f"Total points (4 sensors fused): {len(pts):,}")
    print()
    print(f"{'Column':<16} {'min':>10} {'max':>10} {'mean':>10}")
    print("-" * 50)
    for i, col in enumerate(cols):
        v = pts[:, i]
        print(f"{col:<16} {v.min():>10.3f} {v.max():>10.3f} {v.mean():>10.3f}")

    dyn = pts[:, cols.index("dynamic_mask")] > 0
    n_gt = len(json.loads((frame_dir / "gt_boxes.json").read_text())["objects"])
    print(f"\nGT objects in frame : {n_gt}")
    print(f"Dynamic points      : {dyn.sum():,}  ({100*dyn.mean():.1f}% of cloud)")

    for src, name in enumerate(CORNERS):
        mask = (pts[:, cols.index("lidar_source")] == src) & dyn
        total = (pts[:, cols.index("lidar_source")] == src).sum()
        print(f"  lidar_{name}: {total:>7,} pts  dynamic: {mask.sum():>6,}")

    rel = pts[:, cols.index("reliability")]
    print(f"\nReliability  mean={rel.mean():.3f}  min={rel.min():.3f}  max={rel.max():.3f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CARLA → pipeline format adapter")
    p.add_argument("--frame-dir", type=Path,
                   help="Single frame directory (sanity check mode)")
    p.add_argument("--condition", type=Path,
                   help="Condition directory (e.g. carla_data/clear)")
    p.add_argument("--out", type=Path, default=Path("pipeline_data"),
                   help="Output root for .bin files (batch mode)")
    p.add_argument("--max-frames", type=int,
                   help="Limit number of frames (for testing)")
    args = p.parse_args()

    if args.frame_dir:
        sanity_check(args.frame_dir)
    elif args.condition:
        out = args.out / args.condition.name
        export_condition(args.condition, out, args.max_frames)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
