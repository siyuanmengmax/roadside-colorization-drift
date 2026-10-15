#!/usr/bin/env python3
"""Decisive calib check: project 3D GT cuboids onto the camera image; if they wrap
the vehicles, the LiDAR->camera calibration is correct and colorization is valid."""
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tumtraf_adapter import (list_frames, match_image, load_calib, load_boxes,
                             label_for, read_pcd_ascii, project)

SEQ = "a9_dataset_r02_s01"; CAM = "s110_camera_basler_south2_8mm"
OUT = (Path(__file__).resolve().parents[1] / "outputs")


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def corners(center, quat, lwh):
    l, w, h = lwh
    x = np.array([l, l, l, l, -l, -l, -l, -l]) / 2
    y = np.array([w, w, -w, -w, w, w, -w, -w]) / 2
    z = np.array([h, -h, h, -h, h, -h, h, -h]) / 2
    pts = np.vstack([x, y, z]).T
    return (quat_to_R(quat) @ pts.T).T + np.array(center)


EDGES = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]


def main():
    frames = list_frames(SEQ); pcd = frames[len(frames)//2]
    lbl = label_for(pcd, SEQ); img = cv2.imread(str(match_image(pcd, SEQ, CAM)))
    H, W = img.shape[:2]
    K, T = load_calib(lbl, CAM)                 # use direct-T convention (from sanity)
    boxes = load_boxes(lbl)
    drawn = 0
    for b in boxes:
        c8 = corners(b["center"], b["quat"], b["lwh"])
        u, v, z, valid = project(c8, K, T, W, H)
        if z.min() <= 0.1 or valid.sum() < 4:   # box not (mostly) in front of cam
            continue
        col = (0, 255, 0) if b["type"] == "CAR" else (0, 165, 255)
        for a, bb in EDGES:
            cv2.line(img, (int(u[a]), int(v[a])), (int(u[bb]), int(v[bb])), col, 2)
        drawn += 1
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "phase0_boxes_overlay.png"), img)
    print(f"drew {drawn}/{len(boxes)} boxes -> phase0_boxes_overlay.png")


if __name__ == "__main__":
    main()
