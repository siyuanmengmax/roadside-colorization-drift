#!/usr/bin/env python3
"""Mean grey level of every DAIR-V2X-I image (used to split daytime from low-light frames).
  python dair_brightness.py -> outputs/dair_brightness.json  {frame_id: mean grey}"""
import json
import multiprocessing as mp
from pathlib import Path
import cv2
import dair_adapter as A


def grey(fid):
    im = cv2.imread(str(A.ROOT / "single-infrastructure-side-image" / f"{fid}.jpg"), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    return fid, float(im.mean()) if im is not None else None


if __name__ == "__main__":
    fids = [f for v in A.frames_by_site().values() for f in v]
    with mp.get_context("spawn").Pool(12) as p:
        out = dict(p.map(grey, fids, chunksize=50))
    (Path(__file__).resolve().parents[1] / "outputs/dair_brightness.json").write_text(json.dumps(out))
    print(len(out))
