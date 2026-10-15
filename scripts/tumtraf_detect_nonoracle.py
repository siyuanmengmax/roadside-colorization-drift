#!/usr/bin/env python3
"""
TUMTraf-I detection under the paper's default LABEL-FREE protocol, one sequence at a time.

Replaces tumtraf_detect.py's oracle (points-in-GT-cuboid) foreground with range-image
background subtraction (same model as sim_studies/t2/bg_subtract.py, re-keyed to the
Ouster ring x azimuth ray bin). The background is bootstrapped on the FIRST n_bg frames of
the sequence (no GT), and detection is evaluated on the remaining frames only.

Per sequence it reports, for foreground in {nonoracle, oracle}:
  uniform, geometric quality-aware, signal quality-aware  ->  BEV-dist 11-pt AP,
  block-bootstrap 95% CIs, and PAIRED block-bootstrap CIs of (quality-aware - uniform).

  python tumtraf_detect_nonoracle.py --seq s03
  python tumtraf_detect_nonoracle.py --seq s04
"""
import argparse, json, os, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
VISTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISTA)); sys.path.insert(0, str(VISTA / "scripts"))
from carla_adapter import COLUMNS
from carla_detection import detect_frame, match_detections, compute_ap
from src.preprocessing.find_background import (
    update_background_model, detect_foreground_points, calculate_reliability)
from tumtraf_adapter import list_frames, read_pcd_ascii, label_for, load_boxes
from signal_reliability import signal_reliability
from tumtraf_detect import det_size_class, gt_class, points_in_cuboid, ground_matrix
import prf_utils as PRF

CHANNELS, AZ_BINS = 64, 1024
N_BINS = CHANNELS * AZ_BINS
MIN_RANGE, MAX_RANGE = 0.5, 200.0
CLASSES = ["vehicle", "pedestrian"]
# (name, reliability, mode, rho_min). The quality-aware mode needs sum(r_i) >= rho_min; with r_i < 1 a
# five-point cluster can never pass rho_min = 5, so uniform6 is the size-matched uniform reference,
# and signal_norm rescales the signal reliability to mean 1 over the frame's foreground points.
VARIANTS = [("uniform", "geo", "uniform", 5), ("uniform6", "geo", "uniform", 6),
            ("geo_qa", "geo", "quality_aware", 5), ("signal_qa", "signal", "quality_aware", 5),
            ("signal_norm", "signal_norm", "quality_aware", 5)]
PAIRS = [("geo_qa", "uniform"), ("signal_qa", "uniform"), ("uniform6", "uniform"),
         ("geo_qa", "uniform6"), ("signal_qa", "uniform6"), ("signal_norm", "uniform")]
MASKS = ["nonoracle", "oracle"]
BLOCK, N_BOOT = 50, 1000
BG = None  # background model; workers load it in _init (numba threads do not survive fork)
EMPTY_FG = False  # sensitivity variant: a return in a ray bin with no learned background is foreground
TAU = 20.0  # signal-reliability scale (eq. signalrel)
GT_FILTER = False  # score against the PointPillars label population (tumtraf_to_pcdet.py): >=5 pts, |x|,|y|<=76.8 m
RANGE_XY, MIN_PTS = 76.8, 5


def _init(bg_path, empty_fg, gt_filter, tau=20.0):
    global BG, EMPTY_FG, GT_FILTER, TAU
    BG = np.load(bg_path); EMPTY_FG = empty_fg; GT_FILTER = gt_filter; TAU = tau


def ray_bin(d):
    xyz = d[:, :3]
    rng = np.linalg.norm(xyz, axis=1).astype(np.float32)
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360.0
    ab = np.clip((az / 360.0 * AZ_BINS).astype(np.int64), 0, AZ_BINS - 1)
    ring = np.clip(d[:, 6].astype(np.int64), 0, CHANNELS - 1)
    return ring * AZ_BINS + ab, rng


def build_bg(frames, learning_rate=0.01):
    mean = np.zeros(N_BINS, np.float32); std = np.ones(N_BINS, np.float32)
    weight = np.zeros(N_BINS, np.float32); mrefl = np.zeros(N_BINS, np.float32)
    counts = np.zeros(N_BINS, np.int32)
    for p in frames:
        d = read_pcd_ascii(p)
        bins, rng = ray_bin(d)
        refl = d[:, 5].astype(np.float32)
        m = (rng > MIN_RANGE) & (rng < MAX_RANGE)
        rel = calculate_reliability(rng[m], refl[m])
        mean, std, weight, mrefl, counts = update_background_model(
            mean, std, weight, mrefl, counts, bins[m], rng[m], rel, refl[m], learning_rate, N_BINS)
    return np.column_stack([mean, std, mrefl, counts.astype(np.float32)])


def fg_mask(d, bg, base_threshold=1.0, min_count=10):
    bins, rng = ray_bin(d)
    refl = d[:, 5].astype(np.float32)
    m = (rng > MIN_RANGE) & (rng < MAX_RANGE)
    rel = calculate_reliability(rng[m], refl[m])
    fg_bin = detect_foreground_points(
        bins[m], rng[m], rel, refl[m], bg[:, 0], bg[:, 1], bg[:, 2], bg[:, 3],
        min_count, base_threshold, 0.0, np.zeros(N_BINS, np.bool_), 1.0)
    fg = np.zeros(len(d), np.bool_)
    fg[np.where(m)[0]] = fg_bin[bins[m]]
    if EMPTY_FG:
        fg |= m & (bg[bins, 3] < min_count)
    return fg


def process(job):
    """One frame -> {mask: {variant: {cls: (tp_scores, fp_scores, n_gt)}}} + fg-vs-oracle counts."""
    pcd, seq = job
    d = read_pcd_ascii(pcd)
    xyz = d[:, :3]; refl = d[:, 5].astype(np.float32)
    rng_m = np.linalg.norm(xyz, axis=1).astype(np.float32)
    boxes = load_boxes(label_for(pcd, seq))
    oracle = np.zeros(len(xyz), bool)
    for b in boxes:
        oracle |= points_in_cuboid(xyz, b)
    fg = fg_mask(d, BG)
    G = ground_matrix(seq)
    xyz_g = (G @ np.hstack([xyz, np.ones((len(xyz), 1))]).T).T[:, :3]
    gt = []
    for b in boxes:
        c = G @ np.array([*b["center"], 1.0])
        if GT_FILTER and (abs(c[0]) > RANGE_XY or abs(c[1]) > RANGE_XY
                          or int(points_in_cuboid(xyz, b).sum()) < MIN_PTS):
            continue
        gt.append({"bbox": {"center": [float(c[0]), float(c[1]), float(c[2])]}, "cls": gt_class(b["type"])})
    rel = {"geo": calculate_reliability(rng_m, refl), "signal": signal_reliability(refl, tau=TAU)}

    out = {}
    for mname, mask in (("nonoracle", fg), ("oracle", oracle)):
        out[mname] = {}
        sig = rel["signal"]; mu = float(sig[mask].mean()) if mask.any() else 1.0
        rel["signal_norm"] = sig / max(mu, 1e-6)
        for vname, relname, mode, rho in VARIANTS:
            pts = np.zeros((len(xyz), len(COLUMNS)), np.float32)
            pts[:, 0:3] = xyz_g
            pts[:, COLUMNS.index("range")] = rng_m
            pts[:, COLUMNS.index("reflectivity")] = refl
            pts[:, COLUMNS.index("intensity")] = refl
            pts[:, COLUMNS.index("reliability")] = rel[relname]
            pts[:, COLUMNS.index("dynamic_mask")] = mask.astype(np.float32)
            dets = detect_frame(pts, COLUMNS, mode, params={"rho_min": rho, "alpha": 0.0, "eps_min": 0.3, "eps_max": 2.0})
            out[mname][vname] = {}
            for c in CLASSES:
                gt_c = [{"bbox": g["bbox"]} for g in gt if g["cls"] == c]
                det_c = [x for x in dets if det_size_class(x) == c]
                out[mname][vname][c] = match_detections(det_c, gt_c, 3.0)
    pt = (int((fg & oracle).sum()), int((fg & ~oracle).sum()), int((~fg & oracle).sum()))
    return out, pt


def pooled_ap(recs, idx, mask, variant, cls):
    tp, fp, n = [], [], 0
    for i in idx:
        t, f, g = recs[i][mask][variant][cls]
        tp.extend(t); fp.extend(f); n += g
    return compute_ap(tp, fp, n), len(tp), len(fp), n


def main():
    global BG
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, choices=["s01", "s02", "s03", "s04"])
    ap.add_argument("--n_bg", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bg_all", action="store_true",
                    help="sensitivity: bootstrap the background on ALL frames of the sequence")
    ap.add_argument("--empty_fg", action="store_true",
                    help="sensitivity: returns in ray bins without a learned background are foreground")
    ap.add_argument("--gt_filter", action="store_true",
                    help="score against GT with >=5 points inside +-76.8 m (same population as the PointPillars eval)")
    ap.add_argument("--tau", type=float, default=20.0, help="signal-reliability scale")
    a = ap.parse_args()
    seq = f"a9_dataset_r02_{a.seq}"
    frames = [p for p in list_frames(seq) if label_for(p, seq).exists()]
    bg_frames, ev_frames = frames[:a.n_bg], frames[a.n_bg:]
    if a.limit:
        ev_frames = ev_frames[:a.limit]
    t0 = time.time()
    BG = build_bg(frames if a.bg_all else bg_frames)
    vtag = ("_bgall" if a.bg_all else "") + ("_emptyfg" if a.empty_fg else "") + ("_gtfilt" if a.gt_filter else "") + "_v3" + (f"_tau{a.tau:g}" if a.tau != 20.0 else "")
    print(f"[{a.seq}] bg from {len(bg_frames)} frames ({time.time()-t0:.0f}s), "
          f"valid bins {int((BG[:, 3] >= 10).sum())}/{N_BINS}; eval {len(ev_frames)} frames", flush=True)

    bg_path = VISTA / f"outputs/tumtraf_bg_{a.seq}_n{a.n_bg}{vtag}.npy"
    np.save(bg_path, BG)
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(bg_path, a.empty_fg, a.gt_filter, a.tau)) as pool:
        res = pool.map(process, [(p, seq) for p in ev_frames], chunksize=4)
    recs = [r[0] for r in res]
    tp_p, fp_p, fn_p = (sum(r[1][k] for r in res) for k in range(3))
    print(f"  detection done ({time.time()-t0:.0f}s)", flush=True)

    n = len(recs)
    nblk = int(np.ceil(n / BLOCK))
    blocks = [list(range(i * BLOCK, min((i + 1) * BLOCK, n))) for i in range(nblk)]
    rng = np.random.RandomState(0)
    draws = [[i for b in rng.randint(0, nblk, nblk) for i in blocks[b]] for _ in range(N_BOOT)]

    out = {"seq": a.seq, "n_bg": a.n_bg, "bg_all": a.bg_all, "empty_fg": a.empty_fg, "gt_filter": a.gt_filter, "tau": a.tau, "n_eval_frames": n, "block": BLOCK, "n_boot": N_BOOT,
           "eval_first_frame": ev_frames[0].name, "match": "BEV centre distance 3 m, 11-pt AP",
           "fg_points_vs_oracle": {"recall": round(tp_p / max(tp_p + fn_p, 1), 3),
                                   "precision": round(tp_p / max(tp_p + fp_p, 1), 3)},
           "results": {}}
    full = list(range(n))
    for mask in MASKS:
        out["results"][mask] = {}
        boot, cnt = {}, {}
        for vname, _, _, _ in VARIANTS:
            out["results"][mask][vname] = {}
            for c in CLASSES:
                apv, ntp, nfp, ngt = pooled_ap(recs, full, mask, vname, c)
                boot[vname, c] = np.array([pooled_ap(recs, dr, mask, vname, c)[0] for dr in draws])
                out["results"][mask][vname][c] = {
                    "AP": round(apv, 4),
                    "CI95": [round(float(np.nanpercentile(boot[vname, c], q)), 4) for q in (2.5, 97.5)],
                    "TP": ntp, "FP": nfp, "FN": ngt - ntp, "n_gt": ngt,
                    "recall": round(ntp / max(ngt, 1), 4)}
                cnt[vname, c] = PRF.counts([r[mask][vname][c] for r in recs])
                out["results"][mask][vname][c].update(PRF.summarize(cnt[vname, c], draws))
        for hi, lo in PAIRS:
            key = f"{hi}_minus_{lo}"
            out["results"][mask][key] = {}
            for c in CLASSES:
                diff = boot[hi, c] - boot[lo, c]
                point = out["results"][mask][hi][c]["AP"] - out["results"][mask][lo][c]["AP"]
                ci = [round(float(np.nanpercentile(diff, q)), 4) for q in (2.5, 97.5)]
                out["results"][mask][key][c] = {
                    "delta_AP": round(point, 4), "paired_CI95": ci,
                    "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
                out["results"][mask][key][c].update(PRF.paired(cnt[hi, c], cnt[lo, c], draws))
        np.savez(VISTA / f"outputs/tumtraf_counts_{a.seq}{vtag}_{mask}.npz",
                 **{f"{v}_{c}": cnt[v, c] for v, _, _, _ in VARIANTS for c in CLASSES})
    tag = vtag + (f"_limit{a.limit}" if a.limit else "")
    p = VISTA / f"outputs/tumtraf_nonoracle_{a.seq}{tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)); print("wrote", p, f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
