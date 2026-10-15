#!/usr/bin/env python3
"""
T4 (Theme C): PR curves, TP/FP/FN counts, recall-at-floor, confidence histograms,
and block-bootstrap 95% CIs on AP. Answers ztiR-W3 (is 0.091 meaningful?) and
uNN2-W5 (signal vs noise). Uses the T2 NON-ORACLE foreground by default.

Bootstrap: resample the 400 test frames in 50-frame contiguous blocks (respects
temporal correlation), 1000 resamples, 95% percentile CI per Table-2 cell.

  python t4_analysis.py --condition clear --modes uniform quality_aware
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

VISTA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VISTA))
sys.path.insert(0, str(VISTA / "sim_studies/t2"))
from carla_adapter import process_frame
from carla_detection import detect_frame, match_detections, compute_ap, gt_class, det_class
from bg_subtract import build_bg, foreground_mask

CLASSES = ["vehicle", "pedestrian"]
BLOCK = 50
N_BOOT = 1000


def pr_from_scores(tp_scores, fp_scores, n_gt):
    all_s = [(s, 1) for s in tp_scores] + [(s, 0) for s in fp_scores]
    all_s.sort(key=lambda x: -x[0])
    tpc = fpc = 0; prec = []; rec = []
    for _, t in all_s:
        tpc += t; fpc += (1 - t)
        prec.append(tpc / (tpc + fpc)); rec.append(tpc / n_gt if n_gt else 0.0)
    return prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--modes", nargs="+", default=["uniform", "quality_aware"])
    ap.add_argument("--mask", default="nonoracle", choices=["oracle", "nonoracle"])
    ap.add_argument("--n_bg", type=int, default=200)
    a = ap.parse_args()

    bgp = VISTA / "work/bg_models" / f"{a.condition}_n{a.n_bg}.npz"
    models = ({k: v for k, v in np.load(bgp).items()} if bgp.exists()
              else build_bg(a.condition, a.n_bg))
    cond_dir = VISTA / "carla_data" / a.condition
    ids = [f"{i:06d}" for i in range(1600, 2000) if (cond_dir / f"frame_{i:06d}").is_dir()]

    # per-frame, per-mode, per-class: (tp_scores, fp_scores, n_gt)
    pf = {m: {c: [] for c in CLASSES} for m in a.modes}
    for idx in ids:
        fd = cond_dir / f"frame_{idx}"
        pts, cols = process_frame(fd)
        if a.mask == "nonoracle":
            pts = pts.copy()
            pts[:, cols.index("dynamic_mask")] = foreground_mask(fd, models).astype(np.float32)
        gt = json.loads((fd / "gt_boxes.json").read_text())
        for m in a.modes:
            dets = detect_frame(pts, cols, m)
            for c in CLASSES:
                gt_c = [o for o in gt["objects"] if gt_class(o["class"]) == c]
                det_c = [d for d in dets if det_class(d["label"]) == c]
                tp_s, fp_s, ng = match_detections(det_c, gt_c, 3.0)
                pf[m][c].append((tp_s, fp_s, ng))

    out = {"condition": a.condition, "mask": a.mask, "n_frames": len(ids),
           "block": BLOCK, "n_boot": N_BOOT, "modes": {}}
    for m in a.modes:
        out["modes"][m] = {}
        for c in CLASSES:
            frames = pf[m][c]
            tp_all = [s for f in frames for s in f[0]]
            fp_all = [s for f in frames for s in f[1]]
            n_gt = sum(f[2] for f in frames)
            TP, FP = len(tp_all), len(fp_all)
            FN = n_gt - TP
            ap_pt = compute_ap(tp_all, fp_all, n_gt)
            prec, rec = pr_from_scores(tp_all, fp_all, n_gt)
            # block bootstrap on AP
            nblk = len(frames) // BLOCK
            blocks = [frames[i*BLOCK:(i+1)*BLOCK] for i in range(nblk)]
            boot = []
            rng = np.random.RandomState(0)
            for _ in range(N_BOOT):
                sel = rng.randint(0, nblk, nblk)
                fr = [f for b in sel for f in blocks[b]]
                tps = [s for f in fr for s in f[0]]
                fps = [s for f in fr for s in f[1]]
                ng = sum(f[2] for f in fr)
                boot.append(compute_ap(tps, fps, ng))
            ci = [round(float(np.percentile(boot, 2.5)), 4),
                  round(float(np.percentile(boot, 97.5)), 4)]
            out["modes"][m][c] = {
                "AP": round(ap_pt, 4), "AP_95CI": ci,
                "TP": TP, "FP": FP, "FN": FN, "n_gt": n_gt,
                "recall_at_floor": round(TP / n_gt, 4) if n_gt else 0.0,
                "max_recall": round(max(rec), 4) if rec else 0.0,
                "mean_conf_tp": round(float(np.mean(tp_all)), 4) if tp_all else None,
                "mean_conf_fp": round(float(np.mean(fp_all)), 4) if fp_all else None,
                "pr_curve": {"precision": [round(p, 4) for p in prec[::max(1, len(prec)//60)]],
                             "recall": [round(r, 4) for r in rec[::max(1, len(rec)//60)]]},
            }
    outp = VISTA / "outputs/sim_studies/t4" / f"t4_{a.condition}_{a.mask}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))

    # overlap check uniform vs quality_aware
    print(f"\n=== T4 {a.condition} ({a.mask}) ===")
    for c in CLASSES:
        line = f"{c:11s}: "
        for m in a.modes:
            r = out["modes"][m][c]
            line += f"{m[:4]} AP={r['AP']:.3f} CI{r['AP_95CI']} TP/FP/FN={r['TP']}/{r['FP']}/{r['FN']} rec={r['recall_at_floor']:.3f}  "
        print(line)
    if len(a.modes) == 2:
        for c in CLASSES:
            a0 = out["modes"][a.modes[0]][c]["AP_95CI"]; a1 = out["modes"][a.modes[1]][c]["AP_95CI"]
            overlap = not (a0[1] < a1[0] or a1[1] < a0[0])
            print(f"  {c}: uniform vs quality_aware CI overlap = {overlap}")
    print("wrote", outp)


if __name__ == "__main__":
    main()
