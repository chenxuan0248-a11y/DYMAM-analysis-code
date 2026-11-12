"""experiments/exp_arcs/e3_cliffs_delta — Cliff's delta effect size for the 3
pre-registered primaries (H3a Success@B0, H3b cost, H4 harm), org-level.

Closes the Phase-8 effect-size gap (the pipeline had wilcoxon/bootstrap/BH but no
Cliff's delta). Reads the frozen E3 cells, aggregates each of the 20 orgs to
org-level (avoiding pseudoreplication), then computes Cliff's delta between the
paired conditions. Writes e3_cliffs_delta_v1.json.

    poetry run python -m experiments.exp_arcs.e3_cliffs_delta
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
from collections import OrderedDict

from evaluation.stat_test import cliffs_delta

D = Path("experiments/exp_arcs")
B0 = 3

def _per_org(cells):
    orgs = OrderedDict()
    for c in cells:
        orgs.setdefault(c["org"], []).append(c)
    out = {}
    for org, cs in orgs.items():
        def sb(m):  # Success@B0 rate over this org's fault types
            return st.mean(1.0 if (x["methods"][m]["success"] and x["methods"][m]["cost"] <= B0) else 0.0 for x in cs)
        def cost(m):
            return st.mean(x["methods"][m]["cost"] for x in cs)
        def harm(m):
            return st.mean(x["methods"][m].get("harm", 0) for x in cs)
        out[org] = {
            "sb_arcs": sb("arcs"), "sb_single": sb("single_point_oracle"),
            "cost_arcs": cost("arcs"), "cost_global": cost("global"),
            "harm_arcs": harm("arcs"), "harm_global": harm("global"),
        }
    return out

def main() -> int:
    cells = json.loads((D/"e3_arcs_v1.json").read_text())["cells"]
    po = _per_org(cells)
    orgs = list(po)
    def col(k): return [po[o][k] for o in orgs]

    primaries = {
        # H3a: Success@B0 — ARCS vs single-point (higher is better → expect delta>0)
        "H3a_success_at_B0_arcs_vs_single": cliffs_delta(col("sb_arcs"), col("sb_single")),
        # H3b: cost — global vs ARCS (global higher → expect delta>0, i.e. ARCS cheaper)
        "H3b_cost_global_vs_arcs": cliffs_delta(col("cost_global"), col("cost_arcs")),
        # H4: harm — global vs ARCS (global higher → expect delta>0, i.e. ARCS safer)
        "H4_harm_global_vs_arcs": cliffs_delta(col("harm_global"), col("harm_arcs")),
    }
    result = {
        "phase": "8 — Cliff's delta effect size (org-level, n=%d)" % len(orgs),
        "unit": "organization", "n_orgs": len(orgs), "B0": B0,
        "orgs": orgs,
        "primaries": primaries,
        "note": ("Cliff's delta complements the wilcoxon p / bootstrap CI already in "
                 "e3_oracle_stats_v1.json. delta=+1 = complete separation (every org's "
                 "value in the first condition exceeds the second) — expected here for the "
                 "deterministic Success@B0 margin and the org-scaling cost/harm gaps."),
    }
    out = D/"e3_cliffs_delta_v1.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Cliff's delta (org-level, n={len(orgs)}):")
    for k, v in primaries.items():
        print(f"  {k:38s} delta={v['delta']:+.3f}  ({v['magnitude']})")
    print(f"\nwrote {out}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
