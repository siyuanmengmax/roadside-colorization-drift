#!/usr/bin/env python3
"""
Re-express the drift sweeps on a pixel-displacement axis.

For every dataset of the drift-law analysis (CARLA Town10HD x 4 weather, CARLA Town05, TUMTraf-I,
V2X-Real) this replays the SAME drift injections as the original sweeps (same seeds, same code path
for the perturbation) and additionally records, per point, how far its image projection moves.

Outputs per dataset and nominal drift level:
  mean_E        mean colour error (identical definition to t3_sweep.py / phase1_drift.py)
  mean_px       mean projection displacement in pixels (frame means averaged over frames)
  mean_mrad     the same divided by the focal length (angular displacement)
  mean_dt_cm    ACTUAL magnitude of the injected translation (CARLA draws N(0, d) per axis,
                the real sweeps use a unit direction times d)
and a pooled per-point curve: mean colour error binned by per-point pixel displacement.

  python pixel_displacement_sweep.py [--sets sim_clear real_tumtraf ...]
      -> outputs/pixel_displacement.json
"""
import argparse, json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

VISTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISTA)); sys.path.insert(0, str(VISTA / "scripts"))
CM = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
EDGES = np.concatenate([[0.0], np.logspace(np.log10(0.05), np.log10(2000.0), 47)])  # px bins
NB = len(EDGES) - 1
PROTOCOL = "gauss"   # CARLA: "gauss" replays t3_sweep (N(0,d) per axis); "unit" = unit direction x d, as on real data
DEG_PER_CM = 0.0     # CARLA unit protocol only: rotation about a random axis, degrees per cm of translation
STATIC_ONLY = False  # CARLA only: drop points inside GT boxes (carla_adapter dynamic_mask), so scene content is identical across weather
MIN_RANGE = 1.0  # m; real sweeps only (0 reproduces the original, unfiltered phase1/phase1c numbers)


def proj(xyz, K, T):
    pc = (T @ np.hstack([xyz, np.ones((len(xyz), 1))]).T).T[:, :3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvw = (K @ pc.T).T
        u = uvw[:, 0] / uvw[:, 2]; v = uvw[:, 1] / uvw[:, 2]
    return u, v, z


def unit(xyz, img, K, T0, Ts):
    """One camera. Returns per-drift (sumE, n, sum_px, bin_sumE, bin_n), and clean depths."""
    H, W = img.shape[:2]
    K = np.asarray(K)[:3, :3]
    u0, v0, z0 = proj(xyz, K, T0)
    ok0 = (z0 > 0.1) & np.isfinite(u0) & np.isfinite(v0)
    ok0 &= (u0.astype(np.int64, copy=False).clip(-1, W) >= 0) if False else ok0
    ui0 = np.where(ok0, u0, -1).astype(np.int64); vi0 = np.where(ok0, v0, -1).astype(np.int64)
    val0 = ok0 & (ui0 >= 0) & (ui0 < W) & (vi0 >= 0) & (vi0 < H) & (u0 >= 0) & (v0 >= 0)
    out = []
    for T in Ts:
        u, v, z = proj(xyz, K, T)
        ok = (z > 0.1) & np.isfinite(u) & np.isfinite(v)
        ui = np.where(ok, u, -1).astype(np.int64); vi = np.where(ok, v, -1).astype(np.int64)
        val = ok & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (u >= 0) & (v >= 0)
        both = val0 & val
        if not both.any():
            out.append((0.0, 0, 0.0, np.zeros(NB), np.zeros(NB))); continue
        c0 = img[vi0[both], ui0[both]].astype(np.float64); c1 = img[vi[both], ui[both]].astype(np.float64)
        err = np.abs(c1 - c0).mean(axis=1)
        px = np.hypot(u[both] - u0[both], v[both] - v0[both])
        b = np.clip(np.searchsorted(EDGES, px, side="right") - 1, 0, NB - 1)
        out.append((float(err.sum()), int(both.sum()), float(px.sum()),
                    np.bincount(b, weights=err, minlength=NB), np.bincount(b, minlength=NB).astype(float)))
    return out, z0[val0]


def pack(per_cam):
    """per_cam: list of (unit_out, depths, dt_cm list, fx). -> frame record (one 'rig')."""
    rec = {"E": [], "px": [], "dt": [], "bE": np.zeros((len(CM), NB)), "bn": np.zeros((len(CM), NB))}
    for k in range(len(CM)):
        sE = sum(c[0][k][0] for c in per_cam); n = sum(c[0][k][1] for c in per_cam)
        spx = sum(c[0][k][2] for c in per_cam)
        rec["E"].append(sE / n if n else np.nan); rec["px"].append(spx / n if n else np.nan)
        rec["dt"].append(float(np.mean([c[2][k] for c in per_cam])))
        for c in per_cam:
            rec["bE"][k] += c[0][k][3]; rec["bn"][k] += c[0][k][4]
    z = np.concatenate([c[1] for c in per_cam]) if per_cam else np.zeros(0)
    rec["z_med"] = float(np.median(z)) if len(z) else np.nan
    rec["fx"] = float(per_cam[0][3])
    return rec


def unit_drift(T0, cm, rng):
    """Translation of exactly `cm` along a random direction (+ optional rotation of DEG_PER_CM*cm)."""
    T = T0.copy()
    d3 = rng.normal(0, 1, 3); d3 /= (np.linalg.norm(d3) + 1e-9)
    ax = rng.normal(0, 1, 3); ax /= (np.linalg.norm(ax) + 1e-9)
    T[:3, 3] += d3 * (cm / 100.0)
    if DEG_PER_CM:
        import cv2
        R, _ = cv2.Rodrigues(ax * np.radians(DEG_PER_CM * cm))
        T[:3, :3] = R @ T[:3, :3]
    return T


def _set(protocol, deg, static_only=False):
    global PROTOCOL, DEG_PER_CM, STATIC_ONLY
    PROTOCOL, DEG_PER_CM, STATIC_ONLY = protocol, deg, static_only


def carla_frame(job):
    """Replays t3_sweep.frame_diffs (pure translation): one record per corner."""
    fd, seed = job
    from born_fused import load_frame_data, CORNERS, CAM_360_OFFSETS, _T_for_offset, perturb_extrinsic, K_DEFAULT
    clouds, images = load_frame_data(Path(fd))
    recs = []
    for corner in CORNERS:
        if corner not in clouds or corner not in images or not isinstance(images[corner], dict):
            continue
        pts = clouds[corner]
        if STATIC_ONLY:
            from carla_adapter import convert_sensor, COLUMNS, CORNER_IDX
            meta = json.loads((Path(fd) / "metadata.json").read_text())
            gt = json.loads((Path(fd) / "gt_boxes.json").read_text())
            dyn = convert_sensor(pts, meta["sensor_poses"][corner], gt, CORNER_IDX[corner])[:, COLUMNS.index("dynamic_mask")] > 0
            pts = pts[~dyn]
        az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360.0
        sectors = (((az + 45.0) % 360.0) / 90.0).astype(int)
        rngs = {cm: np.random.RandomState(seed + CORNERS.index(corner)) for cm in CM}
        per_cam = []
        for si, off in enumerate(CAM_360_OFFSETS):
            m = sectors == si
            if off not in images[corner] or not m.any():
                continue
            T0 = _T_for_offset(off)
            if PROTOCOL == "unit":
                Ts = [unit_drift(T0, cm, rngs[cm]) for cm in CM]
            else:
                Ts = [perturb_extrinsic(T0, cm / 100.0, 0.0, rngs[cm]) for cm in CM]
            dt = [float(np.linalg.norm(T[:3, 3] - T0[:3, 3]) * 100.0) for T in Ts]
            o, z = unit(pts[m, :3], images[corner][off], K_DEFAULT, T0, Ts)
            per_cam.append((o, z, dt, K_DEFAULT[0, 0]))
        if per_cam:
            r = pack(per_cam); r["frame"] = seed; recs.append(r)
    return recs


def real_frame(job):
    kind, path, seed = job
    import cv2
    if kind == "tumtraf":
        from tumtraf_adapter import match_image, load_calib, label_for, read_pcd_ascii
        pcd = Path(path); seq = pcd.parts[pcd.parts.index("tumtraf_i") + 1]
        lbl = label_for(pcd, seq)
        if not lbl.exists():
            return []
        cam = "s110_camera_basler_south2_8mm"
        img = cv2.imread(str(match_image(pcd, seq, cam))); xyz = read_pcd_ascii(pcd)[:, :3]
        K, T0 = load_calib(lbl, cam)
    elif kind == "dair":
        import dair_adapter as DA
        K, Dd, T0 = DA.load_calib(path)
        img = DA.load_image(path, K, Dd)
        if img is None:
            return []
        xyz = DA.read_pcd(path)
    else:
        from v2xreal_adapter import read_bin, load_calib, match_image
        b = Path(path); ip = match_image(b, "cam1")
        if not ip.exists():
            return []
        img = cv2.imread(str(ip))
        if img is None:
            return []
        xyz = read_bin(b)[:, :3]; K, T0 = load_calib(b, "cam1")
    # V2X-Real .bin files pad no-return rays with (0,0,0); those rows all project to the single
    # pixel where the camera sees the LiDAR origin and must not be coloured.
    xyz = xyz[np.linalg.norm(xyz, axis=1) >= MIN_RANGE]
    rng = np.random.RandomState(seed)
    d3 = rng.normal(0, 1, 3); d3 /= (np.linalg.norm(d3) + 1e-9)
    if DEG_PER_CM:  # coupled run: rotation of DEG_PER_CM*cm about one random axis per frame, as in unit_drift
        import cv2
        ax = rng.normal(0, 1, 3); ax /= (np.linalg.norm(ax) + 1e-9)
    Ts = []
    for cm in CM:
        T = T0.copy(); T[:3, 3] = T[:3, 3] + d3 * (cm / 100.0)
        if DEG_PER_CM:
            R, _ = cv2.Rodrigues(ax * np.radians(DEG_PER_CM * cm))
            T[:3, :3] = R @ T[:3, :3]
        Ts.append(T)
    o, z = unit(xyz, img, K, T0, Ts)
    if o[0][1] < 20:
        return []
    r = pack([(o, z, list(CM), np.asarray(K)[0, 0])]); r["frame"] = seed
    return [r]


def jobs_for(name):
    if name.startswith("sim_"):
        d = {"sim_town05_clear": VISTA / "carla_data_town05/clear"}.get(name, VISTA / "carla_data" / name[4:])
        return carla_frame, [(str(f), i) for i, f in enumerate(sorted(d.glob("frame_*"))[::10])]
    if name.startswith("real_tumtraf"):
        # real_tumtraf: all four sequences; _day: s01-s03 (daytime); _s04: s04 (rain at night).
        # Frame indices (and hence drift directions) are those of the pooled sweep in every variant.
        from tumtraf_adapter import list_frames
        keep = {"real_tumtraf": ("s01", "s02", "s03", "s04"), "real_tumtraf_day": ("s01", "s02", "s03"),
                "real_tumtraf_s04": ("s04",)}[name]
        fr = []
        for s in ["a9_dataset_r02_s01", "a9_dataset_r02_s02", "a9_dataset_r02_s03", "a9_dataset_r02_s04"]:
            fr += [(s[-3:], p) for p in list_frames(s)[::6]]
        return real_frame, [("tumtraf", str(p), i) for i, (s, p) in enumerate(fr) if s in keep]
    if name.startswith("real_dair_"):
        import dair_adapter as DA
        fr = DA.dair_group(name[len("real_dair_"):])
        step = max(1, len(fr) // 360)  # about 360 frames per group, as for TUMTraf-I and V2X-Real
        return real_frame, [("dair", f, i) for i, f in enumerate(fr[::step])]
    from v2xreal_adapter import list_infra_frames
    return real_frame, [("v2xreal", str(p), i) for i, p in enumerate(list_infra_frames()[::25])]


ALL = ["sim_clear", "sim_rain_med", "sim_fog", "sim_night", "sim_town05_clear", "real_tumtraf", "real_v2xreal"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sets", nargs="+", default=ALL)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="pixel_displacement.json")
    ap.add_argument("--protocol", default="gauss", choices=["gauss", "unit"])
    ap.add_argument("--deg_per_cm", type=float, default=0.0)
    ap.add_argument("--static_only", action="store_true")
    ap.add_argument("--suffix", default=""); a = ap.parse_args()
    outp = VISTA / "outputs" / a.out
    out = json.loads(outp.read_text()) if outp.exists() else {}
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_set, initargs=(a.protocol, a.deg_per_cm, a.static_only)) as pool:
        for name in a.sets:
            fn, jobs = jobs_for(name)
            recs = [r for rs in pool.map(fn, jobs, chunksize=2) for r in rs]
            E = np.array([r["E"] for r in recs]); px = np.array([r["px"] for r in recs]); dt = np.array([r["dt"] for r in recs])
            bE = sum(r["bE"] for r in recs).sum(axis=0); bn = sum(r["bn"] for r in recs).sum(axis=0)
            fx = float(np.mean([r["fx"] for r in recs]))
            out[name + a.suffix] = {
                "n_records": len(recs), "fx_px": round(fx, 1),
                "median_depth_m": round(float(np.nanmedian([r["z_med"] for r in recs])), 2),
                "nominal_cm": CM,
                "mean_E": np.round(np.nanmean(E, axis=0), 4).tolist(),
                "std_E": np.round(np.nanstd(E, axis=0), 4).tolist(),
                "protocol": ("unit direction x d" if (name.startswith("real") or a.protocol == "unit") else "gaussian N(0,d) per axis"),
                "deg_per_cm": a.deg_per_cm,
                "min_range_m": MIN_RANGE if name.startswith("real") else 0.0,
                "mean_px": np.round(np.nanmean(px, axis=0), 3).tolist(),
                "mean_mrad": np.round(np.nanmean(px, axis=0) / fx * 1000.0, 3).tolist(),
                "mean_dt_cm": np.round(np.nanmean(dt, axis=0), 3).tolist(),
                "bins_px_edges": np.round(EDGES, 4).tolist(),
                "bins_n": bn.tolist(),
                "bins_mean_E": [round(float(e / n), 4) if n > 0 else None for e, n in zip(bE, bn)],
            }
            o = out[name + a.suffix]
            print(f"[{time.time()-t0:5.0f}s] {name}: n={len(recs)} fx={o['fx_px']} zmed={o['median_depth_m']}\n"
                  f"   E   {o['mean_E']}\n   px  {o['mean_px']}\n   dt  {o['mean_dt_cm']}", flush=True)
            outp.write_text(json.dumps(out, indent=1))
            np.savez(outp.with_name(outp.stem + f"_records_{name}{a.suffix}.npz"), E=E, px=px,
                     frame=np.array([r["frame"] for r in recs]))
    print("wrote", outp)


if __name__ == "__main__":
    main()
