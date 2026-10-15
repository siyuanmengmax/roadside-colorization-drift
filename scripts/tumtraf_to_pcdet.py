#!/usr/bin/env python3
"""
Convert TUMTraf-I -> OpenPCDet custom format (reuses the CarlaDataset class in pointpillars/).
Official split: train = s01+s02, test = s03 (test_adverse = s04, held out).
Ground-referenced frame (registration source_ground_matrix). 2 classes (Vehicle/Pedestrian).

Output: journal/t1_pp/data/tumtraf/{points/<id>.npy, labels/<id>.txt, ImageSets/{train,val,test}.txt}
 points: (N,4) [x,y,z, reflectivity/255]   labels: "x y z dx dy dz heading name" (>=5 pts, in-range)
"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np

VISTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISTA)); sys.path.insert(0, str(VISTA / "scripts"))
from tumtraf_adapter import read_pcd_ascii, label_for, load_boxes, ROOT, list_frames
from tumtraf_detect import quat_to_R, points_in_cuboid, ground_matrix

RANGE_XY = 76.8
MIN_PTS = 5
OUT = VISTA / "journal/t1_pp/data/tumtraf"
SEQS = {"s01": "a9_dataset_r02_s01", "s02": "a9_dataset_r02_s02",
        "s03": "a9_dataset_r02_s03", "s04": "a9_dataset_r02_s04"}


def cls_name(t):
    return "Pedestrian" if "PED" in t.upper() else "Vehicle"


def convert_one(args):
    tag, seq, pcd = args
    idx = f"{tag}_{pcd.stem.split('_')[0]}_{pcd.stem.split('_')[1]}"
    d = read_pcd_ascii(pcd)
    xyz = d[:, :3]; refl = d[:, 5].astype(np.float32)
    G = ground_matrix(seq); Grot = G[:3, :3]
    xyz_g = (G @ np.hstack([xyz, np.ones((len(xyz), 1))]).T).T[:, :3].astype(np.float32)
    P = np.zeros((len(xyz), 4), np.float32)
    P[:, :3] = xyz_g; P[:, 3] = np.clip(refl / 255.0, 0, 1)
    np.save(OUT / "points" / f"{idx}.npy", P)

    lines = []
    for b in load_boxes(label_for(pcd, seq)):
        cg = G @ np.array([*b["center"], 1.0])
        if abs(cg[0]) > RANGE_XY or abs(cg[1]) > RANGE_XY:
            continue
        if int(points_in_cuboid(xyz, b).sum()) < MIN_PTS:
            continue
        l, w, h = b["lwh"]
        fwd = Grot @ quat_to_R(b["quat"]) @ np.array([1.0, 0, 0])   # box +x in ground frame
        heading = float(np.arctan2(fwd[1], fwd[0]))
        lines.append(f"{cg[0]:.4f} {cg[1]:.4f} {cg[2]:.4f} {l:.4f} {w:.4f} {h:.4f} {heading:.6f} {cls_name(b['type'])}")
    (OUT / "labels" / f"{idx}.txt").write_text("\n".join(lines))
    return idx, len(lines)


def main():
    for s in ("points", "labels", "ImageSets"):
        (OUT / s).mkdir(parents=True, exist_ok=True)
    split_frames = {"train": [], "val": [], "test": []}
    tasks = []
    for tag, seq in SEQS.items():
        frames = list_frames(seq)
        split = "train" if tag in ("s01", "s02") else ("test" if tag == "s03" else None)
        if split is None:
            continue
        for p in frames:
            if not label_for(p, seq).exists():
                continue
            tasks.append((tag, seq, p))
    print(f"{len(tasks)} frames to convert")
    ids_by_split = {"train": [], "test": []}
    with Pool(8) as pool:
        for tag_seq_p, (idx, nb) in zip(tasks, pool.imap(convert_one, tasks, chunksize=8)):
            tag = tag_seq_p[0]
            ids_by_split["train" if tag in ("s01", "s02") else "test"].append(idx)
    # write ImageSets: train, val(=first 100 of test for in-training eval), test
    (OUT / "ImageSets/train.txt").write_text("\n".join(sorted(ids_by_split["train"])))
    test_ids = sorted(ids_by_split["test"])
    (OUT / "ImageSets/test.txt").write_text("\n".join(test_ids))
    (OUT / "ImageSets/val.txt").write_text("\n".join(test_ids[:100]))
    print(f"train={len(ids_by_split['train'])} test={len(test_ids)} val=100")


if __name__ == "__main__":
    main()
