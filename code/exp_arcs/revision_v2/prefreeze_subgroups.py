"""Pre-freeze evidence for the new-site study (EMSE plan v3, P1-B): how did the combined
detector compare with Isolation Forest and robust-z on the ORIGINAL DNS / AO injection
sites? Read-only re-aggregation of frozen artifacts; analysis identity D (post-hoc,
computed after Step 2 results were seen). It only sets the direction of the new-site
predictions; it is not itself evidence for them.

Inputs (read-only):
  ec_manifest_corruption_v1.json  combined detector per cell (drop_true|p=0.0, which the
                                  E-C self-check ties to e3_detector_v1 combined)
  revision_v2/step2_baselines_v1.json  IF / robust-z per cell (manifest-aware, alpha=0.106)
  rq2_combined_topk_v1.json       combined tie-aware Detect@1 by variant

Output: revision_v2/prefreeze_subgroups_v1.json
"""
from __future__ import annotations
import hashlib
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from evaluation.stat_test import bootstrap_ci  # noqa: E402

D = REPO / "experiments/exp_arcs"
EC = D / "ec_manifest_corruption_v1.json"
S2 = D / "revision_v2/step2_baselines_v1.json"
TOPK = D / "rq2_combined_topk_v1.json"
OUT = D / "revision_v2/prefreeze_subgroups_v1.json"
VARIANTS = ("F4@dns", "F4@active_observation", "F4-2host-dns")
BASELINES = ("iforest", "robust_z")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def org_ci(vals: list[float]) -> dict:
    m, lo, hi = bootstrap_ci(vals, B=10_000, seed=42)
    return {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "n_orgs": len(vals)}


def main() -> int:
    ec = json.loads(EC.read_text())
    s2 = json.loads(S2.read_text())["baselines"]
    topk = json.loads(TOPK.read_text())
    tv = (topk.get("summary") or topk)["by_fault_variant"]
    comb = {(c["org"], c["cell"]): c["results"]["drop_true|p=0.0"] for c in ec["cells"]}
    out = {"identity": "D (post-hoc; computed after Step 2 results were seen)",
           "purpose": "direction of P1-B predictions only",
           "inputs": {p.name: sha(p) for p in (EC, S2, TOPK)}, "variants": {}}
    for v in VARIANTS:
        rec = {"combined": {"detect1_tie_aware": tv[v]["detect1_tie_aware"]}}
        orgs = sorted(o for o, c in comb if c == v)
        cc = [comb[(o, v)]["cost_over_Cstar"] for o in orgs]
        rec["combined"].update({"success": round(st.mean(comb[(o, v)]["success"] for o in orgs), 4),
                                "cost_over_Cstar": round(st.mean(cc), 4)})
        for b in BASELINES:
            cells = {(x["org"], x["cell"]): x for x in
                     s2[f"{b}|manifest_aware"]["by_alpha"]["primary"]["cells"]}
            bc = [cells[(o, v)]["cost"] / cells[(o, v)]["C_star"] for o in orgs]
            diff = [a - b_ for a, b_ in zip(cc, bc)]
            rec[b] = {"detect1": round(st.mean(cells[(o, v)]["detect1"] for o in orgs), 4),
                      "success": round(st.mean(float(cells[(o, v)]["success"]) for o in orgs), 4),
                      "cost_over_Cstar": round(st.mean(bc), 4),
                      "paired_cost_diff_combined_minus_baseline": org_ci(diff),
                      "n_orgs_combined_cheaper": sum(d < 0 for d in diff),
                      "n_orgs_tie": sum(d == 0 for d in diff)}
        out["variants"][v] = rec
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["variants"], indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
