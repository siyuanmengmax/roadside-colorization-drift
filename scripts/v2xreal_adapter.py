#!/usr/bin/env python3
"""
V2X-Real (UCLA smart intersection, infra subset) adapter.

Second real-world site for drift-law cross-validation -- Westwood Plaza x Charles E.
Young Dr South, UCLA campus, Los Angeles. Distinct physical location/country from
TUMTraf-I's Munich A9 site. Loads real roadside 128/64-beam LiDAR + RGB camera frames with
per-frame calibration (OPV2V-style yaml). Reuses tumtraf_adapter's project()/colorize()
(pure geometry, no TUMTraf-specific assumptions) instead of duplicating them.

Layout: data/v2x_real/{split}/{session}/{agent_id}/{frame:06d}.{bin,yaml,_cam1.jpeg,_cam2.jpeg}
.bin: (N,4) float32 x y z intensity (flattened dense scan buffer, e.g. 128x1024).
yaml: cam1/cam2 {intrinsic (3x3), extrinsic (4x4)}, lidar_pose, infra (bool), vehicles{...}.
Agent id convention (confirmed by inspection): negative ids (-1,-2) = infra (roadside
RSUs, infra:true, 2 cams each); positive ids (1,2,...) = vehicle-mounted (infra:false,
4 cams each). extrinsic is already T_cam_from_lidar (transforms LiDAR-frame points into
the camera frame), same convention tumtraf_adapter.project()/colorize() expect -- no
inversion needed (verified by projection sanity check in v2xreal_sanity.py).
"""
import os
import sys
from pathlib import Path
import numpy as np
import yaml as pyyaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tumtraf_adapter import project, colorize  # noqa: F401  (re-exported, pure geometry)

ROOT = Path(os.environ.get("V2XREAL_ROOT", "data/v2x_real"))
INFRA_AGENTS = ["-1", "-2"]
SPLITS = ["train1", "train4", "test", "val"]


def list_infra_frames(splits=SPLITS):
    """All infra (roadside) .bin frame paths across the given splits."""
    out = []
    for split in splits:
        split_dir = ROOT / split
        if not split_dir.is_dir():
            continue
        for session in sorted(split_dir.iterdir()):
            if not session.is_dir():
                continue
            for agent in INFRA_AGENTS:
                agent_dir = session / agent
                if agent_dir.is_dir():
                    out += sorted(agent_dir.glob("*.bin"))
    return out


def read_bin(path):
    """Return (N,4) float32 array: x y z intensity."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def _load_yaml(bin_path):
    return pyyaml.safe_load(bin_path.with_suffix(".yaml").read_text())


def load_calib(bin_path, cam="cam1"):
    d = _load_yaml(bin_path)[cam]
    K = np.array(d["intrinsic"], dtype=np.float64)
    T_cam_from_lidar = np.array(d["extrinsic"], dtype=np.float64)
    return K, T_cam_from_lidar


def match_image(bin_path, cam="cam1"):
    return bin_path.parent / f"{bin_path.stem}_{cam}.jpeg"


def is_infra(bin_path):
    return bool(_load_yaml(bin_path).get("infra", False))
