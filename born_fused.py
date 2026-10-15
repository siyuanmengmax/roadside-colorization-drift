#!/usr/bin/env python3
"""
Born-Fused vs Late-Fused Colored Point Cloud Generator
Born-fused vs. late-fused colorization of CARLA frames

Born-fused:  co-located LiDAR + camera, zero calibration error.
             Simulates Ouster REV8's hardware-aligned native color.

Late-fused:  same geometry but with injected calibration noise.
             Simulates long-term deployment miscalibration.

Output arrays (saved as .npy):
  colored_{mode}_{corner}.npy  →  (N, 8) float32
      cols: [x, y, z, intensity, R, G, B, has_color]
      x/y/z/intensity: sensor-local frame (UE4 convention, same as lidar input)
      R/G/B: 0-255 sampled from co-located camera; 0 if outside FOV
      has_color: 1.0 inside FOV, 0.0 outside

  reliability_{corner}.npy  →  (N, 2) float32  [for Experiment 5]
      col 0: geometric reliability = normalized LiDAR intensity
      col 1: RGB reliability = per-point brightness (proxy for illumination quality)

Calibration noise (late-fused only):
  Applied ONCE per frame (simulates fixed drift, not per-point noise).
  Translation: Gaussian σ = noise_cm / 100 metres, 3-axis.
  Rotation:    Gaussian σ = noise_deg°, 3-axis (small-angle perturbation).

Usage:
    # Single frame, born-fused
    python born_fused.py --frame-dir carla_data/clear/frame_000000 --mode born

    # Single frame, late-fused with typical calibration drift
    python born_fused.py --frame-dir carla_data/clear/frame_000000 \\
                         --mode late --noise-cm 2 --noise-deg 0.2

    # Batch: all conditions, three late-fused noise levels
    python born_fused.py --dataset carla_data --mode born
    python born_fused.py --dataset carla_data --mode late --noise-cm 2  --noise-deg 0.2
    python born_fused.py --dataset carla_data --mode late --noise-cm 5  --noise-deg 1.0

    # Compare born vs late on one frame (saves diff visualization)
    python born_fused.py --frame-dir carla_data/clear/frame_000000 --mode compare
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("WARNING: cv2 not found — visualization outputs disabled.")

CORNERS = ["ne", "nw", "se", "sw"]

# ── CARLA coordinate math ──────────────────────────────────────────────────────

# CARLA UE4 sensor frame → standard camera frame.
# From CARLA's lidar_to_camera.py: (x, y, z) → (y, -z, x)
T_UE4_TO_CAM = np.array([
    [0,  1,  0, 0],
    [0,  0, -1, 0],
    [1,  0,  0, 0],
    [0,  0,  0, 1],
], dtype=np.float64)

# Yaw offsets used in 360° camera mode (must match collect_data._CAM_360_OFFSETS)
CAM_360_OFFSETS = [0, 90, 180, 270]


def R_yaw_4x4(yaw_deg: float) -> np.ndarray:
    """4x4 yaw-rotation matrix in CARLA's UE4 left-handed frame.

    Rotates points so that the direction yaw_deg away from sensor +x becomes
    the new +x — i.e., it brings a yaw-offset camera's FOV into alignment.
    Verified: R_yaw(90) maps (0,r,0) → (r,0,0) ✓
    """
    th = np.radians(yaw_deg)
    cy, sy = np.cos(th), np.sin(th)
    return np.array([
        [ cy,  sy,  0,  0],
        [-sy,  cy,  0,  0],
        [  0,   0,  1,  0],
        [  0,   0,  0,  1],
    ], dtype=np.float64)


def _T_for_offset(yaw_offset: float) -> np.ndarray:
    """UE4 sensor frame → camera frame for a camera rotated yaw_offset degrees."""
    return T_UE4_TO_CAM @ R_yaw_4x4(yaw_offset)


def camera_intrinsics(w: int = 1280, h: int = 720, fov: float = 90.0) -> np.ndarray:
    fx = w / (2.0 * np.tan(np.radians(fov / 2.0)))
    return np.array([[fx, 0, w / 2.0],
                     [0, fx, h / 2.0],
                     [0,  0,      1.0]])


K_DEFAULT = camera_intrinsics()


def perturb_extrinsic(T: np.ndarray, t_std_m: float, r_std_deg: float,
                       rng: np.random.RandomState) -> np.ndarray:
    """Add calibration noise to a 4×4 extrinsic matrix.

    Noise is sampled once and held fixed — simulates constant drift,
    not frame-to-frame jitter (which would be unrealistically bad).
    """
    T = T.copy()
    # Translation drift
    T[:3, 3] += rng.normal(0, t_std_m, 3)
    # Rotation drift (small-angle: skew-symmetric perturbation)
    dr = rng.normal(0, np.radians(r_std_deg), 3)
    dR = np.array([[ 1,     -dr[2],  dr[1]],
                   [ dr[2],  1,     -dr[0]],
                   [-dr[1],  dr[0],  1    ]])
    T[:3, :3] = dR @ T[:3, :3]
    return T


# ── Core colorizer ─────────────────────────────────────────────────────────────

def _project_and_sample(pts_xyz: np.ndarray, image_bgr: np.ndarray,
                         T_to_cam: np.ndarray,
                         K: np.ndarray = K_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """Project LiDAR points to camera plane and sample RGB colors.

    Args:
        pts_xyz:    (N, 3) LiDAR points in sensor-local UE4 frame.
        image_bgr:  (H, W, 3) BGR image from co-located camera.
        T_to_cam:   4×4 transform from UE4 sensor frame to camera frame.
        K:          3×3 camera intrinsics.

    Returns:
        rgb:       (N, 3) uint8  — sampled RGB; (0,0,0) for points outside FOV.
        has_color: (N,)   float32 — 1.0 if in FOV, 0.0 otherwise.
    """
    H, W = image_bgr.shape[:2]
    N = len(pts_xyz)

    rgb       = np.zeros((N, 3), dtype=np.uint8)
    has_color = np.zeros(N, dtype=np.float32)

    # UE4 → camera frame
    pts_h   = np.hstack([pts_xyz, np.ones((N, 1))])   # (N, 4)
    pts_cam = (T_to_cam @ pts_h.T).T[:, :3]            # (N, 3)

    # Only points in front of camera
    in_front = pts_cam[:, 2] > 0.1
    if not in_front.any():
        return rgb, has_color

    pc = pts_cam[in_front]
    uvw = K @ pc.T                            # (3, M)
    uv  = (uvw[:2] / uvw[2]).T               # (M, 2)
    u   = uv[:, 0].astype(np.int32)
    v   = uv[:, 1].astype(np.int32)

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # Map back to original point indices
    front_idx  = np.where(in_front)[0]
    valid_idx  = front_idx[in_img]
    u_v, v_v   = u[in_img], v[in_img]

    sampled_bgr = image_bgr[v_v, u_v]                         # (M, 3) BGR
    rgb[valid_idx] = sampled_bgr[:, [2, 1, 0]]                # BGR → RGB
    has_color[valid_idx] = 1.0

    return rgb, has_color


# ── Born-fused ─────────────────────────────────────────────────────────────────

def born_fused(pts: np.ndarray, image_bgr: np.ndarray,
               K: np.ndarray = K_DEFAULT) -> np.ndarray:
    """Create born-fused colored point cloud (zero calibration error).

    Args:
        pts:       (N, 4) LiDAR: [x, y, z, intensity] in sensor-local UE4 frame.
        image_bgr: (H, W, 3) BGR from co-located camera (identical pose).

    Returns:
        (N, 8) float32: [x, y, z, intensity, R, G, B, has_color]
    """
    rgb, has_color = _project_and_sample(pts[:, :3], image_bgr, T_UE4_TO_CAM, K)
    return np.hstack([pts, rgb.astype(np.float32), has_color[:, None]])


# ── Late-fused ─────────────────────────────────────────────────────────────────

def late_fused(pts: np.ndarray, image_bgr: np.ndarray,
               noise_cm: float, noise_deg: float,
               rng: np.random.RandomState,
               K: np.ndarray = K_DEFAULT) -> np.ndarray:
    """Create late-fused colored point cloud with calibration noise.

    Noise is sampled ONCE per call (fixed drift model, not per-point jitter).

    Args:
        pts:       (N, 4) LiDAR: [x, y, z, intensity] in sensor-local UE4 frame.
        image_bgr: (H, W, 3) BGR from co-located camera.
        noise_cm:  Translation noise std (cm).
        noise_deg: Rotation noise std (degrees).
        rng:       RandomState for reproducible noise.
    """
    T_noisy = perturb_extrinsic(T_UE4_TO_CAM, noise_cm / 100.0, noise_deg, rng)
    rgb, has_color = _project_and_sample(pts[:, :3], image_bgr, T_noisy, K)
    return np.hstack([pts, rgb.astype(np.float32), has_color[:, None]])


# ── 360° born-fused / late-fused ──────────────────────────────────────────────

def born_fused_360(pts: np.ndarray,
                   images_by_offset: Dict[int, np.ndarray],
                   K: np.ndarray = K_DEFAULT) -> np.ndarray:
    """Full 360° born-fused colored point cloud using 4 cameras at 90° intervals.

    Each LiDAR point is assigned to the camera whose boresight is nearest its
    azimuth angle (sector width = 90°). No blending at boundaries.

    Args:
        pts:              (N, 4) [x, y, z, intensity] in sensor-local UE4 frame.
        images_by_offset: {0: bgr, 90: bgr, 180: bgr, 270: bgr}

    Returns:
        (N, 8) float32: [x, y, z, intensity, R, G, B, has_color]
    """
    N = len(pts)
    rgb       = np.zeros((N, 3), dtype=np.uint8)
    has_color = np.zeros(N, dtype=np.float32)

    # Assign each point to its azimuth sector (0–3)
    az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360.0
    sectors = (((az + 45.0) % 360.0) / 90.0).astype(int)  # 0,1,2,3

    for sec_idx, yaw_off in enumerate(CAM_360_OFFSETS):
        if yaw_off not in images_by_offset:
            continue
        mask = sectors == sec_idx
        if not mask.any():
            continue
        T = _T_for_offset(yaw_off)
        r, hc = _project_and_sample(pts[mask, :3], images_by_offset[yaw_off], T, K)
        rgb[mask]       = r
        has_color[mask] = hc

    return np.hstack([pts, rgb.astype(np.float32), has_color[:, None]])


def late_fused_360(pts: np.ndarray,
                   images_by_offset: Dict[int, np.ndarray],
                   noise_cm: float, noise_deg: float,
                   rng: np.random.RandomState,
                   K: np.ndarray = K_DEFAULT) -> np.ndarray:
    """Full 360° late-fused with per-camera calibration noise.

    Each camera gets an independent noise draw (simulating independent
    per-camera calibration drift in a multi-camera rig).
    """
    N = len(pts)
    rgb       = np.zeros((N, 3), dtype=np.uint8)
    has_color = np.zeros(N, dtype=np.float32)

    az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360.0
    sectors = (((az + 45.0) % 360.0) / 90.0).astype(int)

    for sec_idx, yaw_off in enumerate(CAM_360_OFFSETS):
        if yaw_off not in images_by_offset:
            continue
        mask = sectors == sec_idx
        if not mask.any():
            continue
        T_noisy = perturb_extrinsic(_T_for_offset(yaw_off),
                                     noise_cm / 100.0, noise_deg, rng)
        r, hc = _project_and_sample(pts[mask, :3], images_by_offset[yaw_off],
                                     T_noisy, K)
        rgb[mask]       = r
        has_color[mask] = hc

    return np.hstack([pts, rgb.astype(np.float32), has_color[:, None]])


# ── Spectral reliability (Experiment 5) ───────────────────────────────────────

def spectral_reliability(colored_pts: np.ndarray) -> np.ndarray:
    """Compute per-point geometric and RGB reliability scores.

    Args:
        colored_pts: (N, 8) output of born_fused / late_fused.

    Returns:
        (N, 2) float32: [geo_reliability, rgb_reliability]
        geo_reliability: LiDAR intensity, normalized to [0, 1].
        rgb_reliability: per-point brightness / 255 — proxy for illumination
                         quality (high = clear day, low = night / heavy fog).
    """
    intensity = colored_pts[:, 3]
    i_min, i_max = intensity.min(), intensity.max()
    geo_rel = (intensity - i_min) / (i_max - i_min + 1e-8)

    R, G, B = colored_pts[:, 4], colored_pts[:, 5], colored_pts[:, 6]
    brightness = (R + G + B) / 3.0
    rgb_rel = np.clip(brightness / 255.0, 0.0, 1.0)

    # Points with no camera coverage get 0 RGB reliability
    has_color = colored_pts[:, 7]
    rgb_rel *= has_color

    return np.stack([geo_rel, rgb_rel], axis=1).astype(np.float32)


# ── Frame-level processing ─────────────────────────────────────────────────────

def load_frame_data(frame_dir: Path) -> Tuple[Dict, Dict]:
    """Load all LiDAR clouds and camera images from a frame directory.

    Returns:
        clouds: {corner: (N,4) array}
        images: one of two formats depending on what was collected —
          Single-camera mode:  {corner: bgr_array}
          360° camera mode:    {corner: {yaw_offset: bgr_array}}
          Auto-detected from file names (camera_ne.png vs camera_ne_0.png).
    """
    clouds, images = {}, {}

    for corner in CORNERS:
        p = frame_dir / f"lidar_{corner}.npy"
        if p.exists():
            clouds[corner] = np.load(p)

        # 360° mode: camera_ne_0.png, camera_ne_90.png, ...
        offset_imgs = {}
        for off in CAM_360_OFFSETS:
            p = frame_dir / f"camera_{corner}_{off}.png"
            if p.exists() and HAS_CV2:
                offset_imgs[off] = cv2.imread(str(p))

        if offset_imgs:
            images[corner] = offset_imgs          # dict of {offset: bgr}
        else:
            p = frame_dir / f"camera_{corner}.png"
            if p.exists() and HAS_CV2:
                images[corner] = cv2.imread(str(p))   # single bgr array

    return clouds, images


def colorize_frame(frame_dir: Path, mode: str,
                   noise_cm: float = 0.0, noise_deg: float = 0.0,
                   seed: Optional[int] = None,
                   save_reliability: bool = True) -> Dict[str, np.ndarray]:
    """Process one frame directory, produce colored clouds for all corners.

    Returns dict {corner: (N, 8) array}.
    Files written to frame_dir:
      colored_{tag}_{corner}.npy
      reliability_{corner}.npy  (if save_reliability)
    """
    clouds, images = load_frame_data(frame_dir)

    if not clouds:
        print(f"  WARNING: no LiDAR files in {frame_dir}")
        return {}
    if not images and mode != "born-no-cam":
        print(f"  WARNING: no camera images in {frame_dir} — skipping colorization")
        return {}

    tag = _mode_tag(mode, noise_cm, noise_deg)
    rng = np.random.RandomState(seed)
    results = {}

    for corner in CORNERS:
        if corner not in clouds or corner not in images:
            continue

        pts    = clouds[corner]
        img    = images[corner]
        is_360 = isinstance(img, dict)

        if mode == "born":
            colored = (born_fused_360(pts, img) if is_360
                       else born_fused(pts, img))
        elif mode == "late":
            colored = (late_fused_360(pts, img, noise_cm, noise_deg, rng) if is_360
                       else late_fused(pts, img, noise_cm, noise_deg, rng))
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'born' or 'late'.")

        out_path = frame_dir / f"colored_{tag}_{corner}.npy"
        np.save(out_path, colored)
        results[corner] = colored

        if save_reliability:
            rel = spectral_reliability(colored)
            np.save(frame_dir / f"reliability_{corner}.npy", rel)

    return results


def _mode_tag(mode: str, noise_cm: float, noise_deg: float) -> str:
    if mode == "born":
        return "born"
    return f"late_{noise_cm:g}cm_{noise_deg:g}deg"


# ── Visualization ──────────────────────────────────────────────────────────────

def visualize_projection(frame_dir: Path, corner: str, colored: np.ndarray,
                          tag: str):
    """Overlay projected colored points on camera image. Saved to frame_dir."""
    if not HAS_CV2:
        return
    # Support both single-camera and 360° mode (use 0° image for overlay)
    img_path = frame_dir / f"camera_{corner}.png"
    if not img_path.exists():
        img_path = frame_dir / f"camera_{corner}_0.png"
    if not img_path.exists():
        return

    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    K = K_DEFAULT

    pts_xyz = colored[:, :3]
    has_color = colored[:, 7].astype(bool)
    pts_in_fov = pts_xyz[has_color]

    # Re-project to get pixel coords for visualization
    pts_h   = np.hstack([pts_in_fov, np.ones((len(pts_in_fov), 1))])
    pts_cam = (T_UE4_TO_CAM @ pts_h.T).T[:, :3]
    front   = pts_cam[:, 2] > 0.1
    pc      = pts_cam[front]
    uvw     = K @ pc.T
    uv      = (uvw[:2] / uvw[2]).T
    u = uv[:, 0].astype(np.int32)
    v = uv[:, 1].astype(np.int32)
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    vis = img.copy()
    depth = pc[in_img, 2]
    d_min, d_max = depth.min(), depth.max() + 1e-6
    for i, (ui, vi, di) in enumerate(zip(u[in_img], v[in_img], depth)):
        t = (di - d_min) / (d_max - d_min)
        color = (0, int(255 * (1 - t)), int(255 * t))   # BGR: green(near)→red(far)
        cv2.circle(vis, (ui, vi), 2, color, -1)

    out = frame_dir / f"vis_{tag}_{corner}.png"
    cv2.imwrite(str(out), vis)


def visualize_diff(frame_dir: Path, corner: str,
                   born: np.ndarray, late: np.ndarray, tag: str):
    """Side-by-side diff: born-fused vs late-fused RGB per point.
    Highlights where calibration noise causes color mis-assignment.
    """
    if not HAS_CV2:
        return

    rgb_born = born[:, 4:7]
    rgb_late = late[:, 4:7]
    has_both = born[:, 7].astype(bool) & late[:, 7].astype(bool)

    if not has_both.any():
        return

    # Pixel-level color difference (per point)
    diff = np.abs(rgb_born[has_both].astype(float) - rgb_late[has_both].astype(float))
    mean_diff = diff.mean()
    max_diff  = diff.max()

    print(f"    [{corner}] color diff (born vs late) — mean: {mean_diff:.1f}  max: {max_diff:.1f}  "
          f"(higher = more calibration-induced color error)")

    # Save a diff heatmap on the camera image (360° mode: use 0° offset)
    img_path = frame_dir / f"camera_{corner}.png"
    if not img_path.exists():
        img_path = frame_dir / f"camera_{corner}_0.png"
    if not img_path.exists() or not HAS_CV2:
        return

    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    K = K_DEFAULT

    pts_xyz  = born[has_both, :3]
    pts_h    = np.hstack([pts_xyz, np.ones((len(pts_xyz), 1))])
    pts_cam  = (T_UE4_TO_CAM @ pts_h.T).T[:, :3]
    front    = pts_cam[:, 2] > 0.1
    pc       = pts_cam[front]
    uvw      = K @ pc.T
    uv       = (uvw[:2] / uvw[2]).T
    u = uv[:, 0].astype(np.int32)
    v = uv[:, 1].astype(np.int32)
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    diff_front = diff[front][in_img]
    d_max = diff_front.max() + 1e-6

    vis = img.copy()
    for i, (ui, vi) in enumerate(zip(u[in_img], v[in_img])):
        intensity = float(diff_front[i].mean()) / d_max
        color = (0, int(255 * (1 - intensity)), int(255 * intensity))  # BGR
        cv2.circle(vis, (ui, vi), 2, color, -1)

    out = frame_dir / f"diff_{tag}_{corner}.png"
    cv2.imwrite(str(out), vis)
    print(f"    Diff visualization: {out}")

    return mean_diff


# ── Dataset-level batch processing ────────────────────────────────────────────

def process_dataset(data_root: Path, conditions: list, mode: str,
                    noise_cm: float = 0.0, noise_deg: float = 0.0,
                    max_frames: Optional[int] = None):
    """Batch colorize all frames in a dataset."""
    tag = _mode_tag(mode, noise_cm, noise_deg)
    print(f"\nBatch processing: mode={mode}  tag={tag}")
    print(f"Conditions: {conditions}")

    total_frames = 0
    for cond in conditions:
        cond_dir = data_root / cond
        if not cond_dir.exists():
            print(f"  SKIP {cond}: directory not found")
            continue

        frame_dirs = sorted(cond_dir.glob("frame_*"))
        if max_frames:
            frame_dirs = frame_dirs[:max_frames]

        print(f"\n  {cond}: {len(frame_dirs)} frames")
        for i, frame_dir in enumerate(frame_dirs):
            colorize_frame(frame_dir, mode, noise_cm, noise_deg, seed=i)
            if (i + 1) % 100 == 0 or i == 0:
                print(f"    {i+1}/{len(frame_dirs)}")
            total_frames += 1

    print(f"\nDone. Processed {total_frames} frames.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Born-fused vs late-fused colored point cloud generator (REV8 sim)"
    )
    p.add_argument("--mode", choices=["born", "late", "compare"], required=True,
                   help="born: zero noise | late: with calibration noise | compare: both + diff")
    p.add_argument("--frame-dir", type=Path,
                   help="Single frame directory to process")
    p.add_argument("--dataset", type=Path,
                   help="Root dataset directory (batch mode)")
    p.add_argument("--conditions", nargs="+", default=["clear", "rain_med", "fog", "night"],
                   help="Weather conditions to process (batch mode only)")
    p.add_argument("--max-frames", type=int,
                   help="Max frames per condition (for testing)")
    p.add_argument("--noise-cm",  type=float, default=2.0,
                   help="Calibration noise std, translation (cm). Default: 2 cm")
    p.add_argument("--noise-deg", type=float, default=0.2,
                   help="Calibration noise std, rotation (deg). Default: 0.2°")
    p.add_argument("--corner", default="ne", choices=CORNERS,
                   help="Corner to visualize in single-frame mode (default: ne)")
    p.add_argument("--no-reliability", action="store_true",
                   help="Skip spectral reliability computation (faster)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for noise (default: 42)")
    return p.parse_args()


def run_single_frame(args):
    frame_dir = args.frame_dir
    save_rel  = not args.no_reliability
    corner    = args.corner

    print(f"Frame: {frame_dir}")
    print(f"Mode:  {args.mode}")

    if args.mode == "compare":
        print("\n--- Born-fused ---")
        born_results = colorize_frame(frame_dir, "born",
                                      save_reliability=save_rel)
        print("\n--- Late-fused ---")
        late_results = colorize_frame(frame_dir, "late",
                                      noise_cm=args.noise_cm,
                                      noise_deg=args.noise_deg,
                                      seed=args.seed,
                                      save_reliability=False)
        print("\n--- Comparison ---")
        if corner in born_results and corner in late_results:
            tag = _mode_tag("late", args.noise_cm, args.noise_deg)
            visualize_projection(frame_dir, corner, born_results[corner], "born")
            visualize_projection(frame_dir, corner, late_results[corner], tag)
            visualize_diff(frame_dir, corner, born_results[corner],
                           late_results[corner], tag)
    else:
        results = colorize_frame(frame_dir, args.mode,
                                 noise_cm=args.noise_cm,
                                 noise_deg=args.noise_deg,
                                 seed=args.seed,
                                 save_reliability=save_rel)
        tag = _mode_tag(args.mode, args.noise_cm, args.noise_deg)
        if corner in results:
            visualize_projection(frame_dir, corner, results[corner], tag)

    # Print summary stats
    print("\n--- Summary ---")
    for c in CORNERS:
        tag = _mode_tag(args.mode if args.mode != "compare" else "born",
                        args.noise_cm, args.noise_deg)
        p_born = frame_dir / f"colored_born_{c}.npy"
        if p_born.exists():
            d = np.load(p_born)
            in_fov = d[:, 7].sum()
            print(f"  born  lidar_{c}: {len(d):>6} pts  in FOV: {int(in_fov):>5} "
                  f"({100*in_fov/len(d):.1f}%)")
        p_late = frame_dir / f"colored_{_mode_tag('late', args.noise_cm, args.noise_deg)}_{c}.npy"
        if p_late.exists():
            d = np.load(p_late)
            in_fov = d[:, 7].sum()
            print(f"  late  lidar_{c}: {len(d):>6} pts  in FOV: {int(in_fov):>5} "
                  f"({100*in_fov/len(d):.1f}%)")


def main():
    args = parse_args()

    if args.frame_dir:
        if not args.frame_dir.exists():
            print(f"ERROR: {args.frame_dir} not found")
            return
        run_single_frame(args)

    elif args.dataset:
        if not args.dataset.exists():
            print(f"ERROR: {args.dataset} not found")
            return
        if args.mode == "compare":
            process_dataset(args.dataset, args.conditions, "born",
                            max_frames=args.max_frames)
            process_dataset(args.dataset, args.conditions, "late",
                            noise_cm=args.noise_cm, noise_deg=args.noise_deg,
                            max_frames=args.max_frames)
        else:
            process_dataset(args.dataset, args.conditions, args.mode,
                            noise_cm=args.noise_cm, noise_deg=args.noise_deg,
                            max_frames=args.max_frames)
    else:
        print("ERROR: specify --frame-dir or --dataset")


if __name__ == "__main__":
    main()
