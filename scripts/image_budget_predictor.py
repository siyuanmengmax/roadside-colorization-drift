#!/usr/bin/env python3
"""
Does an image statistic measured WITHOUT any drift sweep predict the drift budget?

For every frame of the real drift sweeps (same frames as pixel_displacement_sweep.py) we compute,
under the nominal extrinsic only:
  G_lidar  mean |I(p) - I(p + o)| over the 8 one-pixel offsets o, averaged over channels and over
           the pixels p that LiDAR points project to (the colour variation the points actually sample)
  G_image  the same 8-neighbour difference averaged over the whole image (no LiDAR involved)
and join them with that frame's measured drift curve (records npz) to get per-frame E(1 cm) and the
per-frame drift at which E reaches 5 units (inf if not reached by 15 cm).

Reports Spearman correlations at three levels: across groups, across frames pooled, and across
frames within each group (which does not depend on how many sites there are).

  python image_budget_predictor.py -> outputs/image_budget_predictor.json
"""
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

VISTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISTA)); sys.path.insert(0, str(VISTA / "scripts"))
CM = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0])
OUT = VISTA / "outputs"
GROUPS = {  # name -> records file
    "real_tumtraf_day": "drift_sweep_matched_records_real_tumtraf_day.npz",
    "real_tumtraf_s04": "drift_sweep_matched_records_real_tumtraf_s04.npz",
    "real_v2xreal": "drift_sweep_matched_records_real_v2xreal.npz",
    "real_dair_W": "drift_sweep_dair_records_real_dair_W.npz",
    "real_dair_Y1": "drift_sweep_dair_records_real_dair_Y1.npz",
    "real_dair_Y2": "drift_sweep_dair_records_real_dair_Y2.npz",
    "real_dair_Y3": "drift_sweep_dair_records_real_dair_Y3.npz",
    "real_dair_Ylow": "drift_sweep_dair_records_real_dair_Ylow.npz",
}
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def load(job):
    """Same loading path as pixel_displacement_sweep.real_frame -> (img, K, T0, xyz) or None."""
    kind, path, seed = job
    import cv2
    if kind == "tumtraf":
        from tumtraf_adapter import match_image, load_calib, label_for, read_pcd_ascii
        pcd = Path(path); seq = pcd.parts[pcd.parts.index("tumtraf_i") + 1]
        lbl = label_for(pcd, seq)
        if not lbl.exists():
            return None
        cam = "s110_camera_basler_south2_8mm"
        img = cv2.imread(str(match_image(pcd, seq, cam))); xyz = read_pcd_ascii(pcd)[:, :3]
        K, T0 = load_calib(lbl, cam)
    elif kind == "dair":
        import dair_adapter as DA
        K, Dd, T0 = DA.load_calib(path); img = DA.load_image(path, K, Dd); xyz = DA.read_pcd(path)
    else:
        from v2xreal_adapter import read_bin, load_calib, match_image
        b = Path(path); ip = match_image(b, "cam1")
        if not ip.exists():
            return None
        img = cv2.imread(str(ip)); xyz = read_bin(b)[:, :3]; K, T0 = load_calib(b, "cam1")
    if img is None:
        return None
    xyz = xyz[np.linalg.norm(xyz, axis=1) >= 1.0]
    return img, np.asarray(K)[:3, :3], np.asarray(T0), xyz


def stats(job):
    r = load(job)
    if r is None:
        return None
    img, K, T, xyz = r
    H, W = img.shape[:2]; im = img.astype(np.float32)
    pc = (T @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvw = (K @ pc.T).T; u = uvw[:, 0] / uvw[:, 2]; v = uvw[:, 1] / uvw[:, 2]
    ok = (pc[:, 2] > 0.1) & np.isfinite(u) & np.isfinite(v) & (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)
    if ok.sum() < 20:
        return None
    ui, vi = u[ok].astype(np.int64), v[ok].astype(np.int64)
    c0 = im[vi, ui]
    g_l = np.mean([np.abs(im[vi + dy, ui + dx] - c0).mean() for dx, dy in OFFS])
    core = im[1:-1, 1:-1]
    g_i = np.mean([np.abs(im[1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx] - core).mean() for dx, dy in OFFS])
    return job[2], float(g_l), float(g_i)


def budget(e, b=5.0):
    return float(np.interp(b, e, CM)) if e[-1] >= b else np.inf


def main():
    from scipy.stats import spearmanr
    from pixel_displacement_sweep import jobs_for
    res, frames = {}, []
    with mp.get_context("spawn").Pool(16) as pool:
        for g, rec in GROUPS.items():
            _, jobs = jobs_for(g)
            st = {k: (a, b) for k, a, b in (x for x in pool.map(stats, jobs, chunksize=2) if x)}
            z = np.load(OUT / rec); E, fr = z["E"], z["frame"]
            rows = [(st[k][0], st[k][1], E[i, 1], budget(E[i])) for i, k in enumerate(fr) if k in st]
            a = np.array(rows)
            for r in rows:
                frames.append((g, *r))
            rho_w = spearmanr(a[:, 0], a[:, 3]).correlation
            rho_w1 = spearmanr(a[:, 0], a[:, 2]).correlation
            mean_curve = E[np.isin(fr, list(st))].mean(axis=0)
            res[g] = {"n_frames": len(rows), "G_lidar": round(float(a[:, 0].mean()), 3), "G_image": round(float(a[:, 1].mean()), 3),
                      "budget5": round(budget(mean_curve), 2) if np.isfinite(budget(mean_curve)) else None,
                      "E_1cm": round(float(mean_curve[1]), 3),
                      "within_rho_Glidar_budget5": round(float(rho_w), 3),
                      "within_rho_Glidar_E1cm": round(float(rho_w1), 3)}
            print(g, res[g], flush=True)
    gs = list(res)
    b5 = [res[g]["budget5"] if res[g]["budget5"] else 99.0 for g in gs]
    out = {"groups": res, "across_groups": {
        k: round(float(spearmanr([res[g][k] for g in gs], b5).correlation), 3) for k in ("G_lidar", "G_image")}}
    out["across_groups_p"] = {k: round(float(spearmanr([res[g][k] for g in gs], b5).pvalue), 4) for k in ("G_lidar", "G_image")}
    F = np.array([f[1:] for f in frames], float)
    out["pooled_frames"] = {"n": len(F),
                            "rho_Glidar_budget5": round(float(spearmanr(F[:, 0], F[:, 3]).correlation), 3),
                            "rho_Gimage_budget5": round(float(spearmanr(F[:, 1], F[:, 3]).correlation), 3),
                            "rho_Glidar_E1cm": round(float(spearmanr(F[:, 0], F[:, 2]).correlation), 3)}
    # least-squares fit E(1 cm) = k * G_lidar * px_per_cm across frames (one constant for all real data)
    sw = json.loads((OUT / "drift_sweep_matched.json").read_text()); sw.update(json.loads((OUT / "drift_sweep_dair.json").read_text()))
    pxcm = np.array([sw[f[0]]["mean_px"][1] for f in frames])
    x = F[:, 0] * pxcm; k = float((x * F[:, 2]).sum() / (x * x).sum())
    pred = k * x; ss = ((F[:, 2] - pred) ** 2).sum(); tot = ((F[:, 2] - F[:, 2].mean()) ** 2).sum()
    out["linear_E1cm_from_G"] = {"k": round(k, 3), "R2_frames": round(float(1 - ss / tot), 3)}
    gm = {g: (np.mean([p for f, p in zip(frames, pred) if f[0] == g]), res[g]["E_1cm"]) for g in gs}
    out["linear_E1cm_from_G"]["group_pred_vs_meas"] = {g: [round(float(a), 3), b] for g, (a, b) in gm.items()}
    (OUT / "image_budget_predictor.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "groups"}, indent=1))


if __name__ == "__main__":
    main()
