#!/usr/bin/env python3
"""
Signal-quality reliability for real Ouster returns (journal C2/C3).

Motivation: the paper's geometric reliability (calculate_reliability) is range-dominated
and near-uniform on both CARLA (~0.994) and real data (std~0.003) because most points are
close. A SIGNAL-quality reliability uses the per-point return strength (calibrated
reflectivity) + range attenuation, which is near-uniform in CARLA (synthetic high/flat
intensity) but genuinely non-uniform on real Ouster (dark/grazing/far surfaces return weakly).

    r = 1 - exp(-reflectivity/tau)         in [0,1]   (reflectivity-only, default)

Low when the calibrated surface return is weak (dark/absorptive/grazing). This is FLAT in
CARLA (synthetic reflectivity is uniformly high, median ~234 -> r~1.000, std 0.000) but highly
variable on real Ouster (reflectivity median ~7, range 0-255 -> std ~0.24). This explains why the
paper's SIM showed ~0 quality-aware benefit (a synthetic-reflectivity artifact, not a method flaw).
An optional range term is kept for ablation but OFF by default (geometric range reliability is a
separate axis already modelled by calculate_reliability).
"""
import numpy as np


def signal_reliability(reflectivity, range_m=None, tau=20.0, R0=None):
    reflectivity = np.asarray(reflectivity, np.float32)
    r = 1.0 - np.exp(-np.clip(reflectivity, 0, None) / tau)
    if R0 is not None and range_m is not None:
        r = r * np.exp(-np.asarray(range_m, np.float32) / R0)
    return r.astype(np.float32)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.preprocessing.find_background import calculate_reliability
    from tumtraf_adapter import list_frames, read_pcd_ascii
    from carla_adapter import process_frame

    # REAL
    d = read_pcd_ascii(list_frames("a9_dataset_r02_s01")[100])
    xyz = d[:, :3]; refl_r = d[:, 5]; rng_r = np.linalg.norm(xyz, axis=1)
    rel_geo_r = calculate_reliability(rng_r.astype(np.float32), refl_r.astype(np.float32))
    rel_sig_r = signal_reliability(refl_r, rng_r)

    # CARLA
    pts, cols = process_frame((Path(__file__).resolve().parents[1] / "carla_data/clear/frame_000100"))
    refl_c = pts[:, cols.index("reflectivity")]; rng_c = pts[:, cols.index("range")]
    rel_geo_c = calculate_reliability(rng_c.astype(np.float32), refl_c.astype(np.float32))
    rel_sig_c = signal_reliability(refl_c, rng_c)

    def stat(name, v):
        print(f"  {name:26s} mean={v.mean():.3f} std={v.std():.3f} "
              f"p05={np.percentile(v,5):.3f} p95={np.percentile(v,95):.3f}")
    print("REAL (TUMTraf-I):")
    stat("geometric reliability", rel_geo_r); stat("SIGNAL reliability", rel_sig_r)
    print("SIM (CARLA clear):")
    stat("geometric reliability", rel_geo_c); stat("SIGNAL reliability", rel_sig_c)
    print(f"\nContrast (std): SIGNAL real={rel_sig_r.std():.3f} vs sim={rel_sig_c.std():.3f} "
          f"-> ratio {rel_sig_r.std()/(rel_sig_c.std()+1e-9):.1f}x")
