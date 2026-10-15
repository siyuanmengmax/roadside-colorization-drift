#!/usr/bin/env python3
"""
TUMTraf-I (A9 R02, S110 intersection) adapter.

Loads real roadside Ouster LiDAR + Basler camera frames with calibration from the
per-frame OpenLABEL JSON, and provides late-fused colorization (project real camera
onto real LiDAR). Real-world analog of carla_adapter.py.

PCD fields: x y z intensity t reflectivity ring ambient range  (ASCII).
Calib: OpenLABEL streams[cam].intrinsics_pinhole.camera_matrix_3x4 (K),
       coordinate_systems[cam].pose_wrt_parent.matrix4x4 = camera pose in LiDAR frame
       (=> T_cam_from_lidar = inverse; verified by the projection sanity check).
"""
import os
import json
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("TUMTRAF_ROOT", "data/tumtraf_i"))
PCD_COLS = ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "ambient", "range"]


def list_frames(seq, lidar="s110_lidar_ouster_south"):
    d = ROOT / seq / "point_clouds" / lidar
    return sorted(d.glob("*.pcd"))


def _ts(name):
    # '1646667310_053239541_...' -> float seconds
    a, b = name.split("_")[:2]
    return int(a) + int(b) / 1e9


def read_pcd_ascii(path):
    """Return (N, 9) float array with PCD_COLS."""
    with open(path) as f:
        lines = f.readlines()
    # data starts after the 'DATA ascii' header line
    for i, ln in enumerate(lines):
        if ln.startswith("DATA"):
            start = i + 1
            break
    data = np.array([[float(x) for x in ln.split()] for ln in lines[start:] if ln.strip()],
                    dtype=np.float64)
    return data


def match_image(pcd_path, seq, cam="s110_camera_basler_south2_8mm"):
    """Nearest-timestamp camera image for a LiDAR frame."""
    t0 = _ts(pcd_path.name)
    imgs = sorted((ROOT / seq / "images" / cam).glob("*.jpg"))
    return min(imgs, key=lambda p: abs(_ts(p.name) - t0))


def load_calib(label_path, cam="s110_camera_basler_south2_8mm"):
    d = json.loads(Path(label_path).read_text())["openlabel"]
    K = np.array(d["streams"][cam]["stream_properties"]["intrinsics_pinhole"]["camera_matrix_3x4"])
    T_cam_wrt_lidar = np.array(d["coordinate_systems"][cam]["pose_wrt_parent"]["matrix4x4"]).reshape(4, 4)
    return K, T_cam_wrt_lidar


def load_boxes(label_path):
    """Return list of dicts {type, center(3), quat(xyzw), lwh(3), n_pts}."""
    d = json.loads(Path(label_path).read_text())["openlabel"]
    frames = d["frames"]; fk = list(frames.keys())[0]
    out = []
    for o in frames[fk].get("objects", {}).values():
        od = o["object_data"]; c = od["cuboid"]["val"]
        n = 0
        for nn in od["cuboid"].get("attributes", {}).get("num", []):
            if nn["name"] == "num_points":
                n = nn["val"]
        out.append({"type": od["type"], "center": c[:3], "quat": c[3:7], "lwh": c[7:10], "n_pts": n})
    return out


def label_for(pcd_path, seq, lidar="s110_lidar_ouster_south"):
    return ROOT / seq / "labels_point_clouds" / lidar / (pcd_path.stem + ".json")


def project(points_xyz, K, T_cam_from_lidar, W, H):
    """Project (N,3) LiDAR points -> pixel (u,v). Returns u,v,z_cam,valid mask."""
    N = len(points_xyz)
    ph = np.hstack([points_xyz, np.ones((N, 1))])
    pc = (T_cam_from_lidar @ ph.T).T[:, :3]           # camera frame
    z = pc[:, 2]
    uvw = (K[:3, :3] @ pc.T).T                          # (N,3)
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]
    valid = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z, valid


def colorize(points_xyz, image_bgr, K, T_cam_from_lidar):
    """Late-fused colorization: sample image RGB at each point's projection.
    Returns (N,3) RGB uint8 and has_color bool mask."""
    H, W = image_bgr.shape[:2]
    u, v, z, valid = project(points_xyz, K, T_cam_from_lidar, W, H)
    rgb = np.zeros((len(points_xyz), 3), np.uint8)
    ui = np.clip(u.astype(int), 0, W - 1); vi = np.clip(v.astype(int), 0, H - 1)
    rgb[valid] = image_bgr[vi[valid], ui[valid]][:, ::-1]   # BGR->RGB
    return rgb, valid
