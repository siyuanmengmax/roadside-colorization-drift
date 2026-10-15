#!/usr/bin/env python3
"""T4 PR-curve figure (revision) from t4_<cond>_<mask>.json."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VISTA = Path(__file__).resolve().parents[2]
cond = sys.argv[1] if len(sys.argv) > 1 else "clear"
mask = sys.argv[2] if len(sys.argv) > 2 else "nonoracle"
d = json.loads((VISTA / f"outputs/sim_studies/t4/t4_{cond}_{mask}.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, cls in zip(axes, ["vehicle", "pedestrian"]):
    for m in d["modes"]:
        r = d["modes"][m][cls]
        pr = r["pr_curve"]
        ax.plot(pr["recall"], pr["precision"], "-o", ms=2,
                label=f"{m} (AP={r['AP']:.3f}, CI[{r['AP_95CI'][0]:.2f},{r['AP_95CI'][1]:.2f}])")
    ax.set_title(f"{cls}  (TP/FP/FN={d['modes']['uniform'][cls]['TP']}/"
                 f"{d['modes']['uniform'][cls]['FP']}/{d['modes']['uniform'][cls]['FN']})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.25); ax.legend(fontsize=7)
fig.suptitle(f"PR curves — {cond} ({mask}), 3 m BEV match")
fig.tight_layout()
outp = VISTA / f"outputs/sim_studies/t4/t4_pr_{cond}.pdf"
fig.savefig(outp); fig.savefig(outp.with_suffix(".png"), dpi=110)
print("wrote", outp)
