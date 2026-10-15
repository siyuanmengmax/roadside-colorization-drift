#!/usr/bin/env python3
"""
Zero-drift colour noise floor: how much does the colour of a STATIC point change between two
consecutive frames when the extrinsic is exact?

For frame t, a LiDAR point is kept as static if frame t+1 has a return within 5 cm of it (the
sensors are fixed, so static surfaces are re-hit; moving objects and surfaces they newly occlude
are not). Static points are projected with the clean extrinsic and coloured from image t and from
image t+1. Noise floor = mean absolute per-channel difference, same units as Eq. (err).
Same frame subsets as the drift sweeps.

  python drift_noise_floor.py  ->  outputs/drift_noise_floor.json
"""
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

VISTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISTA)); sys.path.insert(0, str(VISTA / "scripts"))
from pixel_displacement_sweep import proj, jobs_for, ALL, MIN_RANGE

TOL = 0.05


def cam_noise(xyz, xyz_next, img, img_next, K, T):
    if img is None or img_next is None or len(xyz) == 0 or len(xyz_next) == 0:
        return 0.0, 0, 0
    d, _ = cKDTree(xyz_next).query(xyz, k=1)
    st = xyz[d <= TOL]
    H, W = img.shape[:2]
    u, v, z = proj(st, np.asarray(K)[:3, :3], T)
    ok = (z > 0.1) & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0)
    ui = np.where(ok, u, -1).astype(np.int64); vi = np.where(ok, v, -1).astype(np.int64)
    ok &= (ui < W) & (vi < H)
    if not ok.any():
        return 0.0, 0, len(xyz)
    e = np.abs(img_next[vi[ok], ui[ok]].astype(float) - img[vi[ok], ui[ok]].astype(float)).mean(axis=1)
    return float(e.sum()), int(ok.sum()), len(xyz)


def carla(job):
    fd, _ = job
    from born_fused import load_frame_data, CORNERS, CAM_360_OFFSETS, _T_for_offset, K_DEFAULT
    fd = Path(fd); nx = fd.parent / f"frame_{int(fd.name.split('_')[1]) + 1:06d}"
    if not nx.is_dir():
        return []
    c0, i0 = load_frame_data(fd); c1, i1 = load_frame_data(nx)
    out = []
    for corner in CORNERS:
        if corner not in c0 or corner not in c1 or not isinstance(i0.get(corner), dict):
            continue
        p0, p1 = c0[corner][:, :3], c1[corner][:, :3]
        az = np.degrees(np.arctan2(p0[:, 1], p0[:, 0])) % 360.0
        sec = (((az + 45.0) % 360.0) / 90.0).astype(int)
        s = n = 0
        for si, off in enumerate(CAM_360_OFFSETS):
            if off in i0[corner] and off in i1.get(corner, {}):
                a, b, _ = cam_noise(p0[sec == si], p1, i0[corner][off], i1[corner][off], K_DEFAULT, _T_for_offset(off))
                s += a; n += b
        if n:
            out.append(s / n)
    return out


def real(job):
    kind, path, _, nxt = job
    import cv2
    if nxt is None:
        return []
    if kind == "tumtraf":
        from tumtraf_adapter import match_image, load_calib, label_for, read_pcd_ascii
        p, q = Path(path), Path(nxt); seq = p.parts[p.parts.index("tumtraf_i") + 1]
        lbl = label_for(p, seq)
        if not lbl.exists():
            return []
        cam = "s110_camera_basler_south2_8mm"
        i0 = cv2.imread(str(match_image(p, seq, cam))); i1 = cv2.imread(str(match_image(q, seq, cam)))
        x0, x1 = read_pcd_ascii(p)[:, :3], read_pcd_ascii(q)[:, :3]; K, T = load_calib(lbl, cam)
    else:
        from v2xreal_adapter import read_bin, load_calib, match_image
        p, q = Path(path), Path(nxt)
        if not match_image(p, "cam1").exists() or not match_image(q, "cam1").exists():
            return []
        i0 = cv2.imread(str(match_image(p, "cam1"))); i1 = cv2.imread(str(match_image(q, "cam1")))
        x0, x1 = read_bin(p)[:, :3], read_bin(q)[:, :3]; K, T = load_calib(p, "cam1")
    x0 = x0[np.linalg.norm(x0, axis=1) >= MIN_RANGE]; x1 = x1[np.linalg.norm(x1, axis=1) >= MIN_RANGE]
    s, n, tot = cam_noise(x0, x1, i0, i1, K, T)
    return [s / n] if n >= 20 else []


def real_jobs(name):
    if name.startswith("real_tumtraf"):
        from tumtraf_adapter import list_frames
        keep = {"real_tumtraf": ("s01", "s02", "s03", "s04"), "real_tumtraf_day": ("s01", "s02", "s03"),
                "real_tumtraf_s04": ("s04",)}[name]
        jobs, i = [], 0
        for sq in ["a9_dataset_r02_s01", "a9_dataset_r02_s02", "a9_dataset_r02_s03", "a9_dataset_r02_s04"]:
            fr = list_frames(sq)
            for k in range(0, len(fr), 6):
                if sq[-3:] in keep:
                    jobs.append(("tumtraf", str(fr[k]), i, str(fr[k + 1]) if k + 1 < len(fr) else None))
                i += 1
        return jobs
    from v2xreal_adapter import list_infra_frames
    fr = list_infra_frames()
    return [("v2xreal", str(fr[k]), k // 25, str(fr[k + 1]) if k + 1 < len(fr) and fr[k + 1].parent == fr[k].parent else None)
            for k in range(0, len(fr), 25)]


def main():
    sw = json.loads((VISTA / "outputs/drift_sweep_matched.json").read_text())
    names = sys.argv[1:] or ALL  # optional subset; results are merged into the existing file
    outp = VISTA / "outputs/drift_noise_floor.json"
    out = json.loads(outp.read_text()) if (sys.argv[1:] and outp.exists()) else {}
    with mp.get_context("spawn").Pool(20) as pool:
        for name in names:
            if name.startswith("sim_"):
                _, jobs = jobs_for(name); vals = [v for r in pool.map(carla, jobs, chunksize=2) for v in r]
            else:
                vals = [v for r in pool.map(real, real_jobs(name), chunksize=2) for v in r]
            vals = np.array(vals); N = float(vals.mean())
            d = np.array(sw[name]["nominal_cm"]); e = np.array(sw[name]["mean_E"])
            dn = float(np.interp(N, e, d)) if N < e[-1] else None
            out[name] = {"n": int(len(vals)), "noise_floor": round(N, 3), "median": round(float(np.median(vals)), 3),
                         "p10_p90": [round(float(np.percentile(vals, q)), 3) for q in (10, 90)],
                         "drift_at_noise_floor_cm": round(dn, 2) if dn is not None else None,
                         "E_at_1cm": e[1], "ratio_E1cm_over_floor": round(float(e[1] / N), 2)}
            print(name, out[name], flush=True)
    outp.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
