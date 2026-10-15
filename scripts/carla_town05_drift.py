#!/usr/bin/env python3
"""
Second-CARLA-town drift-law cross-validation (simulated counterpart of
phase1c_drift_v2xreal.py). Same pure-translation protocol as the paper's headline sim
curve (outputs/sim_studies/t3/t3_sweep_transonly.json, clear condition: slope=1.1438
RGB/cm, R2=0.923, 15cm=18.15), but on a DIFFERENT intersection in a DIFFERENT CARLA
town (Town05, auto-detected junction, vs. the original Town10HD intersection) —
different geometry, building layout, road width, traffic density.

Reuses frame_diffs()/fit_stats()/CM from t3_sweep.py verbatim (both are pure functions
over an arbitrary frame_dir, no Town10HD-specific assumption) rather than duplicating
them; only the data root changes.
"""
import json, sys
from pathlib import Path
import numpy as np

VISTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISTA / "sim_studies/t3"))
from t3_sweep import frame_diffs, fit_stats, CM  # noqa: E402

STRIDE = 10
DATA_DIR = VISTA / "carla_data_town05" / "clear"
OUT = VISTA / "outputs"


def main():
    frame_diffs.deg_per_cm = 0.0  # pure translation, matches transonly protocol
    frames = sorted(DATA_DIR.glob("frame_*"))[::STRIDE]
    print(f"{len(frames)} frames (Town05 clear, stride {STRIDE})")
    accum = {cm: [] for cm in CM}
    used = 0
    for i, fd in enumerate(frames):
        fdif = frame_diffs(fd, seed=i)
        if fdif is None:
            continue
        used += 1
        for cm in CM:
            if not np.isnan(fdif[cm]):
                accum[cm].append(fdif[cm])
        if used % 50 == 0:
            print(f"  [{used}/{len(frames)}]", flush=True)

    mean_diff = {cm: round(float(np.mean(accum[cm])), 4) for cm in CM}
    std_diff = {cm: round(float(np.std(accum[cm])), 4) for cm in CM}
    fit = fit_stats(CM, [mean_diff[cm] for cm in CM])
    res = {"dataset": "CARLA Town05 (second intersection)", "condition": "clear",
           "n_frames": used, "mean_color_diff": mean_diff, "std_color_diff": std_diff,
           "fit": fit}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "town05_drift.json").write_text(json.dumps(res, indent=2))
    print(f"\n[Town05 clear] slope={fit['slope']} R2={fit['R2']} "
          f"CI={fit['slope_95CI']} dAIC(lin-quad)={fit['dAIC_lin_minus_quad']}")
    print("wrote", OUT / "town05_drift.json")


if __name__ == "__main__":
    main()
