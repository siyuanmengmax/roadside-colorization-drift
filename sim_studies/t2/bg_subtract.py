#!/usr/bin/env python3
"""
T2: label-free foreground extraction for CARLA via range-image background subtraction.

Reuses the paper's own model (find_background.update_background_model /
detect_foreground_points / calculate_reliability) but re-keys it from array-index to a
stable RAY BIN (channel x azimuth) recovered from each point's sensor-local direction,
so it works on CARLA's variable-length clouds. NO ground-truth boxes are used anywhere.

CARLA LiDAR: 128 channels, fov [-22.5, +22.5] deg, 10 Hz, 1,310,720 pts/s -> 1024 az bins.
Foreground mask replaces the oracle `dynamic_mask` column for the non-oracle Table 2.
"""
import sys
from pathlib import Path
import numpy as np

VISTA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VISTA))
from carla_adapter import process_frame, CORNERS
from src.preprocessing.find_background import (
    update_background_model, detect_foreground_points, calculate_reliability)

CHANNELS = 128
AZ_BINS = 1024
UPPER_FOV = 22.5
LOWER_FOV = -22.5
N_BINS = CHANNELS * AZ_BINS
MIN_RANGE, MAX_RANGE = 0.5, 200.0


def ray_bin(xyz):
    """(N,3) sensor-local -> (bin_idx int32, range float32)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rng = np.sqrt(x * x + y * y + z * z).astype(np.float32)
    rxy = np.sqrt(x * x + y * y)
    el = np.degrees(np.arctan2(z, rxy))
    az = np.degrees(np.arctan2(y, x)) % 360.0
    ch = np.round((el - LOWER_FOV) / (UPPER_FOV - LOWER_FOV) * (CHANNELS - 1))
    ch = np.clip(ch, 0, CHANNELS - 1).astype(np.int64)
    ab = np.clip((az / 360.0 * AZ_BINS).astype(np.int64), 0, AZ_BINS - 1)
    return (ch * AZ_BINS + ab).astype(np.int64), rng


def _load_local(frame_dir, corner):
    p = frame_dir / f"lidar_{corner}.npy"
    return np.load(p) if p.exists() else None


def build_bg(condition, n_bg=200, learning_rate=0.01):
    """Per-sensor background model over the first n_bg frames of a condition."""
    cond_dir = VISTA / "carla_data" / condition
    frames = sorted(cond_dir.glob("frame_*"))[:n_bg]
    models = {}
    for corner in CORNERS:
        mean = np.zeros(N_BINS, np.float32)
        std = np.ones(N_BINS, np.float32)
        weight = np.zeros(N_BINS, np.float32)
        mrefl = np.zeros(N_BINS, np.float32)
        counts = np.zeros(N_BINS, np.int32)
        for fd in frames:
            pl = _load_local(fd, corner)
            if pl is None:
                continue
            bins, rng = ray_bin(pl[:, :3])
            refl = np.clip(pl[:, 3] * 255.0, 0, 255).astype(np.float32)
            m = (rng > MIN_RANGE) & (rng < MAX_RANGE)
            rel = calculate_reliability(rng[m], refl[m])
            mean, std, weight, mrefl, counts = update_background_model(
                mean, std, weight, mrefl, counts,
                bins[m], rng[m], rel, refl[m], learning_rate, N_BINS)
        models[corner] = np.column_stack([mean, std, mrefl, counts.astype(np.float32)])
    return models


def foreground_mask(frame_dir, models, base_threshold=1.0, min_count=10,
                    base_threshold_refl=0.0, range_factor=1.0):
    """Return foreground bool mask aligned with process_frame's fused point order."""
    parts = []
    for corner in CORNERS:
        pl = _load_local(frame_dir, corner)
        if pl is None:
            continue
        bins, rng = ray_bin(pl[:, :3])
        refl = np.clip(pl[:, 3] * 255.0, 0, 255).astype(np.float32)
        m = (rng > MIN_RANGE) & (rng < MAX_RANGE)
        rel = calculate_reliability(rng[m], refl[m])
        bg = models[corner]
        fg_bin = np.zeros(N_BINS, np.bool_)
        fg_bin = detect_foreground_points(
            bins[m], rng[m], rel, refl[m],
            bg[:, 0], bg[:, 1], bg[:, 2], bg[:, 3],
            min_count, base_threshold, base_threshold_refl, fg_bin, range_factor)
        fg = np.zeros(len(pl), np.bool_)
        fg[np.where(m)[0]] = fg_bin[bins[m]]
        parts.append(fg)
    return np.concatenate(parts)


if __name__ == "__main__":
    import argparse, json, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--n_bg", type=int, default=200)
    ap.add_argument("--check_frames", type=int, default=20,
                    help="compare non-oracle mask vs oracle dynamic_mask on this many test frames")
    a = ap.parse_args()

    t0 = time.time()
    models = build_bg(a.condition, a.n_bg)
    bgp = VISTA / "work/bg_models" / f"{a.condition}_n{a.n_bg}.npz"
    np.savez_compressed(bgp, **{k: v for k, v in models.items()})
    valid = {k: int((v[:, 3] > 0).sum()) for k, v in models.items()}
    print(f"[{a.condition}] bg built in {time.time()-t0:.1f}s, valid bins/sensor: {valid} -> {bgp}")

    # Validate against oracle mask on test frames (recall/precision of dynamic points)
    cond_dir = VISTA / "carla_data" / a.condition
    test_ids = (VISTA / "data/carla_pcdet" / a.condition / "ImageSets/test.txt").read_text().split()
    tp = fp = fn = n_fg = n_pts = 0
    for idx in test_ids[:a.check_frames]:
        fd = cond_dir / f"frame_{idx}"
        pts, cols = process_frame(fd)
        oracle = pts[:, cols.index("dynamic_mask")] > 0
        fg = foreground_mask(fd, models)
        tp += int((fg & oracle).sum()); fp += int((fg & ~oracle).sum())
        fn += int((~fg & oracle).sum()); n_fg += int(fg.sum()); n_pts += len(fg)
    rec = tp / (tp + fn + 1e-9); prec = tp / (tp + fp + 1e-9)
    print(json.dumps({
        "condition": a.condition, "n_bg": a.n_bg, "check_frames": a.check_frames,
        "fg_recall_vs_oracle": round(rec, 3), "fg_precision_vs_oracle": round(prec, 3),
        "mean_fg_frac": round(n_fg / n_pts, 3), "oracle_note": "recall=fraction of GT-dynamic pts captured"
    }, indent=2))
