import json
from pathlib import Path

VISTA = Path(__file__).resolve().parents[2]
out = {}
for c in ["clear", "rain_med", "fog", "night"]:
    p = VISTA / f"outputs/sim_studies/t4/t4_{c}_nonoracle.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text()); out[c] = {}
    for cls in ["vehicle", "pedestrian"]:
        u = d["modes"]["uniform"][cls]; q = d["modes"]["quality_aware"][cls]
        ov = not (u["AP_95CI"][1] < q["AP_95CI"][0] or q["AP_95CI"][1] < u["AP_95CI"][0])
        out[c][cls] = {"uni_AP": u["AP"], "uni_CI": u["AP_95CI"], "qa_AP": q["AP"],
                       "qa_CI": q["AP_95CI"], "TP_FP_FN": f"{u['TP']}/{u['FP']}/{u['FN']}",
                       "recall": u["recall_at_floor"], "CI_overlap": ov}
(VISTA / "outputs/sim_studies/t4/T4_summary.json").write_text(json.dumps(out, indent=2))
for c in out:
    for cls in out[c]:
        r = out[c][cls]
        print(f"{c:9s} {cls:11s} uni {r['uni_AP']:.3f} qa {r['qa_AP']:.3f} "
              f"rec {r['recall']:.3f} CI_overlap={r['CI_overlap']}")
print("wrote T4_summary.json")