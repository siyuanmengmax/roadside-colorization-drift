#!/usr/bin/env python3
"""
DAIR-V2X-I (single-infrastructure-side) adapter for the drift-law sweeps.

Each frame has one roadside LiDAR sweep (x y z intensity, binary_compressed PCD), one 1920x1080
camera image, the camera intrinsics with plumb-bob distortion, and the virtual-LiDAR -> camera
extrinsic. The images are distorted (k1 ~ -0.48), so we undistort them once with the intrinsic K
kept as the new camera matrix; a pinhole projection with K is then valid.
"""
import os
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("DAIR_ROOT", "data/dair-v2x-i"))
SIDE = ROOT / "single-infrastructure-side"


def frames_by_site():
    """{intersection_loc: [frame ids sorted]} from data_info.json."""
    info = json.loads((SIDE / "data_info.json").read_text())
    out = defaultdict(list)
    for x in info:
        out[x["intersection_loc"]].append(Path(x["pointcloud_path"]).stem)
    return {k: sorted(v) for k, v in out.items()}


# Analysis groups. Four intersection_loc labels show the same YiZhuang intersection from nearly the
# same viewpoint with different camera-LiDAR calibrations (rigs Y1-Y3; YiZhuang11North is a 33-frame
# dusk sequence); West152 is a second intersection. Frames are split by mean image grey level
# (dair_brightness.py): daytime >= 80, low light < 80.
GROUPS = {"W": ["YiZhuangdatasetWest152"], "Y1": ["YiZhuangdatasetNorth151"], "Y2": ["YiZhuang11South153"],
          "Y3": ["YiZhuang11North151"],
          "Ylow": ["YiZhuang11North", "YiZhuang11North151", "YiZhuang11South153", "YiZhuangdatasetNorth151"]}
DAY_GREY = 80.0


def dair_group(g):
    b = json.loads(((Path(__file__).resolve().parents[1] / "outputs/dair_brightness.json")).read_text())
    s = frames_by_site()
    fr = sorted(f for site in GROUPS[g] for f in s[site])
    return [f for f in fr if (b[f] < DAY_GREY) == (g == "Ylow")]


def read_pcd(fid):
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(ROOT / "single-infrastructure-side-velodyne" / f"{fid}.pcd"))
    return np.asarray(pc.points, np.float64)


def load_calib(fid):
    """-> K (3x3), D (5,), T (4x4 LiDAR -> camera)."""
    ci = json.loads((SIDE / "calib/camera_intrinsic" / f"{fid}.json").read_text())
    ex = json.loads((SIDE / "calib/virtuallidar_to_camera" / f"{fid}.json").read_text())
    K = np.array(ci["cam_K"], np.float64).reshape(3, 3)
    D = np.array(ci["cam_D"], np.float64)
    T = np.eye(4)
    T[:3, :3] = np.array(ex["rotation"], np.float64)
    T[:3, 3] = np.array(ex["translation"], np.float64).ravel()
    return K, D, T


def load_image(fid, K, D):
    import cv2
    img = cv2.imread(str(ROOT / "single-infrastructure-side-image" / f"{fid}.jpg"))
    if img is None:
        return None
    return cv2.undistort(img, K, D, None, K)


if __name__ == "__main__":
    s = frames_by_site()
    print({k: len(v) for k, v in s.items()})
    fid = s[max(s, key=lambda k: len(s[k]))][0]
    xyz = read_pcd(fid); K, D, T = load_calib(fid)
    print(fid, xyz.shape, np.percentile(np.linalg.norm(xyz, axis=1), [5, 50, 95]).round(1))
    pc = (T @ np.c_[xyz, np.ones(len(xyz))].T).T
    uv = (K @ pc[:, :3].T).T; uv = uv[:, :2] / uv[:, 2:3]
    inimg = (pc[:, 2] > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] < 1080)
    print("points in image:", int(inimg.sum()))
