#!/usr/bin/env python3
"""
Zero-shot sim-to-real check: score a CARLA-trained PointPillars checkpoint (no
fine-tuning, no training on real data at all) directly on real TUMTraf-I frames,
using the SAME metric as the rest of the paper (carla_detection.match_detections:
BEV center-distance <= 3 m + 11-point AP), so it is directly comparable to the sim
PointPillars/MulDet3D numbers in Table III. This is NOT the same experiment as the
(permanently skipped, see main.tex Limitations/Table III footnote) "train PointPillars
on real data" baseline -- it tests whether a sim-trained detector generalizes zero-shot
to real roadside geometry, which needs no GPU training at all (pure inference, does not
touch the code path that crashed during real-data training).

GT is read from the CarlaDataset label format already on disk
(journal/t1_pp/data/tumtraf/labels/<idx>.txt: "x y z dx dy dz heading name" per line).

Run from OpenPCDet-master/tools with:
  PYTHONPATH=<openpcdet_root> conda run -n myenv python <this> \
      --ckpt <carla_clear_checkpoint.pth> --split test
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

VISTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISTA))
from carla_detection import match_detections, compute_ap, gt_class  # paper eval

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

CFG = "cfgs/carla_models/pointpillar_carla_zeroshot_tumtraf.yaml"
DATA_ROOT = VISTA / "journal/t1_pp/data/tumtraf"
LABEL2CLS = {1: "vehicle", 2: "pedestrian"}   # CLASS_NAMES order: Vehicle, Pedestrian


def load_gt(idx):
    txt = (DATA_ROOT / "labels" / f"{idx}.txt").read_text().strip()
    if not txt:
        return []
    out = []
    for ln in txt.split("\n"):
        v = ln.split()
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        out.append({"class": v[7], "bbox": {"center": [x, y, z]}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--score_thresh", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip_first", type=int, default=0,
                    help="skip the first N frames (match the label-free protocol, whose "
                         "background model is bootstrapped on them)")
    a = ap.parse_args()

    logger = common_utils.create_logger()
    cfg_from_yaml_file(CFG, cfg)
    cfg.DATA_CONFIG.DATA_PATH = str(DATA_ROOT)
    cfg.DATA_CONFIG.DATA_SPLIT.test = "test"  # use the 1032-frame test.txt (s03), not val.txt

    ds, _, _ = build_dataloader(cfg.DATA_CONFIG, cfg.CLASS_NAMES, batch_size=1,
                                dist=False, workers=2, logger=logger,
                                training=False, total_epochs=1)
    ds.set_split(a.split if a.split != "test" else "test")
    logger.info(f"Scoring real TUMTraf-I zero-shot: {len(ds)} frames  ckpt={a.ckpt}")

    model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    model.load_params_from_file(a.ckpt, logger=logger, to_cpu=False)
    model.cuda().eval()

    classes = ["vehicle", "pedestrian"]
    tp = {c: [] for c in classes}
    fp = {c: [] for c in classes}
    n_gt = {c: 0 for c in classes}

    for i in range(a.skip_first, len(ds)):
        idx = ds.sample_id_list[i]
        batch = ds.collate_batch([ds[i]])
        load_data_to_gpu(batch)
        with torch.no_grad():
            pred_dicts, _ = model(batch)
        pb = pred_dicts[0]["pred_boxes"].cpu().numpy()      # (N,7) world frame
        ps = pred_dicts[0]["pred_scores"].cpu().numpy()
        pl = pred_dicts[0]["pred_labels"].cpu().numpy()
        dets = []
        for b, s, l in zip(pb, ps, pl):
            if s < a.score_thresh:
                continue
            dets.append({"cx": float(b[0]), "cy": float(b[1]),
                         "confidence": float(s), "cls": LABEL2CLS.get(int(l), "other")})

        gt_objs = load_gt(idx)
        for cls in classes:
            gt_cls = [o for o in gt_objs if gt_class(o["class"]) == cls]
            det_cls = [d for d in dets if d["cls"] == cls]
            tp_s, fp_s, ng = match_detections(det_cls, gt_cls, dist_threshold=3.0)
            tp[cls].extend(tp_s); fp[cls].extend(fp_s); n_gt[cls] += ng

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(ds)}]", flush=True)

    ap_res = {c: compute_ap(tp[c], fp[c], n_gt[c]) for c in classes}
    ap_vals = [v for v in ap_res.values() if not np.isnan(v)]
    mAP = float(np.mean(ap_vals)) if ap_vals else float("nan")
    result = {
        "detector": "PointPillars (CARLA-trained, zero-shot)", "domain": "TUMTraf-I real",
        "split": a.split, "ckpt": a.ckpt, "score_thresh": a.score_thresh,
        "AP_vehicle": round(ap_res["vehicle"], 4),
        "AP_pedestrian": round(ap_res["pedestrian"], 4),
        "mAP": round(mAP, 4),
        "n_gt_vehicle": n_gt["vehicle"], "n_gt_pedestrian": n_gt["pedestrian"],
        "n_frames": len(ds) - a.skip_first, "skip_first": a.skip_first,
    }
    print("\n==== RESULT ====")
    print(json.dumps(result, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
