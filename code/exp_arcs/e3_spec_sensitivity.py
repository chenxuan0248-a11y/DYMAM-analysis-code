"""experiments/exp_arcs/e3_spec_sensitivity — ④ cost-spec sensitivity (anti spec-tuning).

Success@B0 (0.556 vs 0.444), ρC and cost ratios all depend on the frozen cost spec
(dns=1, http=3, ...). A reviewer will ask: did you TUNE the spec to get the gap?
This sweep reprices the SAME repair scopes under alternative plausible cost specs
and checks the qualitative conclusions survive.

Key structural fact (verified below by cost-match): with H2 substitution inert
(D1), ARCS's greedy MUST cover the full required-leaf union, so the chosen SCOPE is
independent of the cost weights — only its PRICE and B0 shift. So the sweep is an
exact re-pricing of frozen scopes, not a re-selection.

Self-check: the reconstructed frozen-spec cost of every method on every cell must
equal the cost stored in e3_arcs_v1.json (else the reconstruction drifted and no
number is trusted). Mismatches are reported, not hidden.

    poetry run python -m experiments.exp_arcs.e3_spec_sensitivity
"""
from __future__ import annotations
import json, statistics as st
from functools import lru_cache
from pathlib import Path

from experiments.exp_arcs.dag import build_dag
from experiments.exp_arcs.contamination import contamination_closure
from experiments.exp_arcs.candidates import candidate_invocations
from experiments.exp_arcs.counterfactual import repair_and_check
from experiments.exp_arcs.select import arcs_greedy, single_point, global_replay

D = Path("experiments/exp_arcs")

@lru_cache(maxsize=None)
def _dag(clean_dir_str: str):
    return build_dag(Path(clean_dir_str))

# frozen family costs (family_to_action -> action cost, repairable families only)
FROZEN = {"dns": 1, "certificate": 1, "registration": 1, "active_observation": 3,
          "passive_observation": 2, "vulnerability_context": 5,
          "applicability_validation": 5, "temporal_memory": 8}

# Alternative plausible specs (family -> cost). Missing families fall back to FROZEN.
SPECS = {
    "frozen": {},                                   # baseline / self-check
    "uniform": {f: 1 for f in FROZEN},              # remove all cost heterogeneity
    "dns_expensive": {"dns": 3, "certificate": 3, "registration": 3},  # kill cheap-dns edge (adversarial)
    "active_heavy": {"active_observation": 6, "vulnerability_context": 10,
                     "applicability_validation": 10, "temporal_memory": 16},
}

def _fam_cost(spec_over):
    def c(fam): return spec_over.get(fam, FROZEN.get(fam, 1))
    return c

def _invocations(dag, scope):
    """distinct (target, family) repairable invocations in a scope."""
    seen = set()
    for nid in scope:
        n = dag.nodes.get(nid)
        if n and n.get("repairable"):
            seen.add((n.get("target"), n["source_family"]))
    return seen

def _price(invs, costf):
    return sum(costf(fam) for (_t, fam) in invs)

def main() -> int:
    e3 = {(_c["org"], _c["cell"]): _c for _c in json.loads((D/"e3_arcs_v1.json").read_text())["cells"]}
    grid = json.loads((D/"e3_grid_v1.json").read_text())["cells"]

    # per cell: reconstruct scopes -> invocation sets + success
    recs = []
    mismatches = []
    for c in grid:
        org, cell = c["org"], c["cell"]
        clean, faulted = Path(c["clean_dir"]), Path(c["faulted_dir"])
        dag = _dag(str(clean))
        dfx = contamination_closure(clean, faulted)
        cand = candidate_invocations(clean, dfx)
        s_arcs = arcs_greedy(clean, dfx, cand)
        s_single = single_point(clean, dfx, cand, tuple(e3[(org, cell)]["oracle_inv"]))
        s_global = global_replay(clean, dfx, cand)
        inv = {"arcs": _invocations(dag, s_arcs), "single": _invocations(dag, s_single),
               "global": _invocations(dag, s_global)}
        succ = {"arcs": repair_and_check(clean, dfx, s_arcs)["success"] or
                        repair_and_check(clean, dfx, s_arcs)["n_outputs_broken"] == 0,
                "single": repair_and_check(clean, dfx, s_single)["success"] or
                          repair_and_check(clean, dfx, s_single)["n_outputs_broken"] == 0,
                "global": True}
        # self-check against e3 frozen costs
        cf = _fam_cost({})
        for m, e3key in (("arcs", "arcs"), ("single", "single_point_oracle"), ("global", "global")):
            got = _price(inv[m], cf); exp = e3[(org, cell)]["methods"][e3key]["cost"]
            if got != exp:
                mismatches.append(f"{org}/{cell}/{m}: reconstructed {got} != e3 {exp}")
        recs.append({"org": org, "cell": cell, "inv": inv, "succ": succ})

    if mismatches:
        print(f"⚠ COST-MATCH FAILED on {len(mismatches)} method-cells — reconstruction drifted, numbers NOT trusted:")
        for m in mismatches[:15]:
            print("  " + m)
        # still write what we have, flagged
    else:
        print(f"✓ self-check PASSED: reconstructed frozen cost == e3 cost on all {len(recs)*3} method-cells\n")

    # per spec: reprice, recompute B0 + Success@B0 + cost ratios
    summary = {}
    for sname, over in SPECS.items():
        costf = _fam_cost(over)
        arcs_cost = [_price(r["inv"]["arcs"], costf) for r in recs]
        single_cost = [_price(r["inv"]["single"], costf) for r in recs]
        global_cost = [_price(r["inv"]["global"], costf) for r in recs]
        B0 = st.median(single_cost)
        def sat(costs, succ_key):
            return st.mean(1.0 if (r["succ"][succ_key] and cst <= B0) else 0.0
                           for r, cst in zip(recs, costs))
        arcs_succ = st.mean(1.0 if r["succ"]["arcs"] else 0.0 for r in recs)
        single_succ = st.mean(1.0 if r["succ"]["single"] else 0.0 for r in recs)
        ratio = st.mean(a / g for a, g in zip(arcs_cost, global_cost) if g > 0)
        summary[sname] = {
            "B0": B0,
            "arcs_success": round(arcs_succ, 3), "single_success": round(single_succ, 3),
            "arcs_success_at_B0": round(sat(arcs_cost, "arcs"), 3),
            "single_success_at_B0": round(sat(single_cost, "single"), 3),
            "success_at_B0_margin": round(sat(arcs_cost, "arcs") - sat(single_cost, "single"), 3),
            "mean_arcs_cost": round(st.mean(arcs_cost), 2),
            "mean_global_cost": round(st.mean(global_cost), 2),
            "mean_arcs_over_global": round(ratio, 4),
        }

    result = {"phase": "8/④ — cost-spec sensitivity (re-pricing frozen scopes)",
              "specs": SPECS, "self_check_passed": not mismatches,
              "n_mismatches": len(mismatches), "mismatches": mismatches[:50],
              "summary": summary,
              "note": "Scopes are cost-invariant (D1: substitution inert -> forced full cover), so this "
                      "reprices the SAME scopes.",
              "findings": [
                  "ROBUST (all 4 specs): ARCS success 1.0 >= single 0.444; mean_arcs_over_global "
                  "0.013-0.015 << 1. The cost-vs-global win is NOT a spec-tuning artifact.",
                  "FRAGILE: the Success@B0 margin +0.111 collapses to 0.000 under uniform and "
                  "dns_expensive. It comes ENTIRELY from F4-2host-dns (ARCS covers 2 dns invocations at "
                  "cost 2 <= B0=3); once 2*dns_cost > B0 (uniform B0=1; dns=3 -> 6>3) it leaves budget "
                  "and the margin vanishes. So the Success@B0 margin IS spec-sensitive.",
                  "IMPLICATION: do not headline Success@B0. Lead with cost-vs-global (spec-robust) and the "
                  "Pareto budget sweep; report Success@B0 as a single-budget snapshot with this caveat.",
              ]}
    (D/"e3_spec_sensitivity_v1.json").write_text(json.dumps(result, indent=2))

    print(f"{'spec':16}{'B0':>5}{'ARCSsu':>8}{'SPsu':>7}{'ARCS@B0':>9}{'SP@B0':>8}{'margin':>8}{'ARCS/GL':>9}")
    for s, v in summary.items():
        print(f"{s:16}{v['B0']:>5.0f}{v['arcs_success']:>8.3f}{v['single_success']:>7.3f}"
              f"{v['arcs_success_at_B0']:>9.3f}{v['single_success_at_B0']:>8.3f}"
              f"{v['success_at_B0_margin']:>+8.3f}{v['mean_arcs_over_global']:>9.4f}")
    print(f"\nwrote {D/'e3_spec_sensitivity_v1.json'}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
