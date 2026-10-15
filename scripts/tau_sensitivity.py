#!/usr/bin/env python3
"""
Sensitivity of the signal-quality reliability r = 1 - exp(-rho/tau) to its scale tau.

Collects the TUMTraf-I detection runs at tau in {5, 10, 20, 40, 80}
(tumtraf_detect_nonoracle.py --gt_filter [--tau T]) and, per tau, the reliability spread on real
(TUMTraf-I) and simulated (CARLA clear) returns.

  python tau_sensitivity.py  ->  outputs/tau_sensitivity.json
"""
import json, sys
from pathlib import Path
import numpy as np

VISTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISTA)); sys.path.insert(0, str(VISTA / "scripts"))
from signal_reliability import signal_reliability

TAUS = [5, 10, 20, 40, 80]
OUT = VISTA / "outputs"
PAIRS = ["signal_qa_minus_uniform6", "signal_norm_minus_uniform"]


def run_file(seq, tau):
    return OUT / f"tumtraf_nonoracle_{seq}_gtfilt_v3{'' if tau == 20 else f'_tau{tau:g}'}.json"


def reliability_spread():
    from tumtraf_adapter import list_frames, read_pcd_ascii
    from carla_adapter import process_frame
    refl_r = np.concatenate([read_pcd_ascii(p)[:, 5] for p in list_frames("a9_dataset_r02_s03")[200::100]])
    pts, cols = process_frame(VISTA / "carla_data/clear/frame_001600")
    refl_c = pts[:, cols.index("reflectivity")]
    return {t: {"real_std": round(float(signal_reliability(refl_r, tau=t).std()), 3),
                "real_mean": round(float(signal_reliability(refl_r, tau=t).mean()), 3),
                "sim_std": round(float(signal_reliability(refl_c, tau=t).std()), 3),
                "sim_mean": round(float(signal_reliability(refl_c, tau=t).mean()), 3)} for t in TAUS}


def main():
    out = {"reliability": reliability_spread(), "detection": {}}
    for seq in ("s03", "s04"):
        out["detection"][seq] = {}
        for t in TAUS:
            p = run_file(seq, t)
            if not p.exists():
                continue
            R = json.loads(p.read_text())["results"]
            row = {}
            for mask in ("nonoracle", "oracle"):
                for v in ("signal_qa", "signal_norm"):
                    for c in ("vehicle", "pedestrian"):
                        d = R[mask][v][c]
                        row[f"{mask}/{v}/{c}"] = {k: d[k] for k in ("P", "R", "F1", "TP", "FP")}
                for pr in PAIRS:
                    for c in ("vehicle", "pedestrian"):
                        d = R[mask][pr][c]
                        if c == "pedestrian" and seq == "s04":
                            continue
                        row[f"{mask}/{pr}/{c}"] = {"dF1": d["delta_F1"], "CI95": d["delta_F1_CI95"],
                                                   "excl0": d["delta_F1_excl0"]}
            out["detection"][seq][str(t)] = row
    (OUT / "tau_sensitivity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["reliability"], indent=1))
    for seq, rows in out["detection"].items():
        for mask in ("nonoracle", "oracle"):
            for pr in PAIRS:
                for c in ("vehicle", "pedestrian"):
                    k = f"{mask}/{pr}/{c}"
                    cells = [f"{t}:{rows[t][k]['dF1']:+.3f}{'*' if rows[t][k]['excl0'] else ' '}{rows[t][k]['CI95']}"
                             for t in rows if k in rows[t]]
                    if cells:
                        print(f"{seq} {k:42s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
