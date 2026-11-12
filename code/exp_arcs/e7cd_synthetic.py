"""experiments/exp_arcs/e7cd_synthetic — E7-B/C/D on a SYNTHETIC constrained grid.

⚠️ SYNTHETIC / ILLUSTRATIVE — NOT the recon corpus. On the real recon corpus every
repair is idempotent, harm≤1, no evidence family maps to a harmful/irreversible
action, and D1 measured substantive redundancy = 0, so ARCS's safety/substitution
constraints are INERT there (ARCS ≡ min-cost cover ≡ ILP, ρC≡1). This benchmark
constructs candidate sets where each constraint BINDS, purely to demonstrate the
selector's machinery does the right thing when it is exercised. It is reported as
an illustration of mechanism, NEVER as a main empirical result. It fills the plan's
E7-B (−substitution) / E7-C (−reversibility) / E7-D (−harm) ablations, which are
un-runnable on the corpus because the constraints never activate.

Each scenario runs `arcs_greedy` under 4 configs and shows the chosen scope /
cost / harm / #irreversible diverge when the constraint is toggled:
  full        : β=1, H_max bound, forbid_irreversible=True, substitution ON
  −substitution (E7-B): substitution OFF (alt={})
  −harm (E7-D): H_max=∞, β=0
  −reversible (E7-C): forbid_irreversible=False

    poetry run python -m experiments.exp_arcs.e7cd_synthetic
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
from experiments.exp_arcs.ilp_oracle import load_config
from experiments.exp_arcs.select import SelectionInfeasible, arcs_greedy


def _cfg(base, *, H_max=None, beta=None, forbid=None):
    c = copy.deepcopy(base)
    if H_max is not None: c["constraints"]["H_max"] = H_max
    if beta is not None: c["objective"]["beta"] = beta
    if forbid is not None: c["constraints"]["forbid_irreversible"] = forbid
    return c


def _run(cand, alt, cfg):
    try:
        scope = arcs_greedy(None, None, cand, alt=alt or {}, cfg=cfg)
    except SelectionInfeasible as e:
        # coverage is a hard constraint: no partial repair is executed; the cell is
        # escalated (paper §3.4). The partial selection is reported for audit only.
        return {"status": "infeasible", "escalate": "global_replay",
                "reasons": sorted(e.reasons), "cost": 0, "harm": 0, "n_irreversible": 0,
                "success": False, "invocations": [],
                "partial_scope_not_executed": sorted(e.partial_scope)}
    # summarise the chosen scope over the candidate invocations
    inv_of = {}
    for k, leaves in cand["invocations"].items():
        for l in leaves: inv_of[l] = k
    chosen = {inv_of[l] for l in scope if l in inv_of}
    cost = sum(cand["inv_cost"][k] for k in chosen)
    harm = sum(cand["inv_harm"][k] for k in chosen)
    nirr = sum(1 for k in chosen if cand["inv_irreversible"].get(k))
    # success = every non-alt-satisfiable output's required leaves covered
    sat = set((alt or {}).keys())
    need = set()
    for o, req in cand["coverage_map"].items():
        if o not in sat: need |= set(req)
    ok = need <= set(scope)
    return {"status": "ok", "cost": cost, "harm": harm, "n_irreversible": nirr,
            "success": ok, "invocations": sorted(str(k) for k in chosen)}


def scenarios():
    # S1 — substitution (E7-B): o1 has a surviving independent-family alt cover;
    # o2 must be repaired via inv B (cost 5). With substitution, o1 is free.
    S1 = {"invocations": {("a", "f1"): ["L1"], ("b", "f2"): ["L2"]},
          "inv_cost": {("a", "f1"): 3, ("b", "f2"): 5},
          "inv_harm": {("a", "f1"): 0, ("b", "f2"): 0},
          "inv_irreversible": {("a", "f1"): False, ("b", "f2"): False},
          "coverage_map": {"o1": ["L1"], "o2": ["L2"]}}
    S1_alt = {"o1": ["surviving_independent_family"]}

    # S2 — harm budget (E7-D): o1 via safe inv (cost 5, harm 0); o2 via cheap-but-
    # harmful inv (cost 1, harm 10). H_max=5 → ARCS refuses o2 (stays within budget,
    # leaves o2 broken); −harm re-collects o2 (harm 10).
    S2 = {"invocations": {("a", "safe"): ["L1"], ("b", "harm"): ["L2"]},
          "inv_cost": {("a", "safe"): 5, ("b", "harm"): 1},
          "inv_harm": {("a", "safe"): 0, ("b", "harm"): 10},
          "inv_irreversible": {("a", "safe"): False, ("b", "harm"): False},
          "coverage_map": {"o1": ["L1"], "o2": ["L2"]}}

    # S3 — irreversibility (E7-C): o2's only cover is irreversible (cost 1). ARCS
    # refuses it (0 irreversible, o2 broken); −reversible re-collects via it.
    S3 = {"invocations": {("a", "rev"): ["L1"], ("b", "irr"): ["L2"]},
          "inv_cost": {("a", "rev"): 4, ("b", "irr"): 1},
          "inv_harm": {("a", "rev"): 0, ("b", "irr"): 0},
          "inv_irreversible": {("a", "rev"): False, ("b", "irr"): True},
          "coverage_map": {"o1": ["L1"], "o2": ["L2"]}}
    return [("S1_substitution", S1, S1_alt), ("S2_harm_budget", S2, None),
            ("S3_irreversibility", S3, None)]


def main() -> int:
    base = load_config()
    base["objective"]["alpha"] = 1.0
    configs = {
        "full":          _cfg(base, H_max=5, beta=1.0, forbid=True),
        "-substitution": _cfg(base, H_max=5, beta=1.0, forbid=True),   # alt off at call
        "-harm":         _cfg(base, H_max=10**9, beta=0.0, forbid=True),
        "-reversible":   _cfg(base, H_max=5, beta=1.0, forbid=False),
    }
    out = {"note": "SYNTHETIC illustration of constraint binding — NOT the recon "
                   "corpus (where constraints are inert; see PHASE6_REFRAME.md). "
                   "Demonstrates E7-B/C/D: toggling a constraint changes ARCS's outcome. "
                   "Coverage is a hard constraint: when no constraint-satisfying cover "
                   "exists the selector reports status=infeasible and the cell escalates "
                   "(no partial repair is executed).",
           "scenarios": {}}
    print(out["note"], "\n")
    for name, cand, alt in scenarios():
        row = {}
        for cfg_name, cfg in configs.items():
            use_alt = alt if cfg_name != "-substitution" else None
            row[cfg_name] = _run(cand, use_alt, cfg)
        out["scenarios"][name] = row
        print(f"== {name} ==")
        print(f"  {'config':16}{'status':>11}{'cost':>6}{'harm':>6}{'irrev':>7}{'success':>9}  invocations")
        for cfg_name in configs:
            r = row[cfg_name]
            print(f"  {cfg_name:16}{r['status']:>11}{r['cost']:>6}{r['harm']:>6}{r['n_irreversible']:>7}"
                  f"{str(r['success']):>9}  {r['invocations']}"
                  + (f"  reasons={r['reasons']}" if r['status'] == 'infeasible' else ""))
        print()
    Path("experiments/exp_arcs/e7cd_synthetic_v2.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    # verdict per scenario: does full differ from the ablation that should matter?
    v = {
        "E7-B substitution changes scope": out["scenarios"]["S1_substitution"]["full"]["cost"]
            != out["scenarios"]["S1_substitution"]["-substitution"]["cost"],
        "E7-D harm cap changes the outcome": (
            out["scenarios"]["S2_harm_budget"]["full"]["status"],
            out["scenarios"]["S2_harm_budget"]["full"]["harm"])
            != (out["scenarios"]["S2_harm_budget"]["-harm"]["status"],
                out["scenarios"]["S2_harm_budget"]["-harm"]["harm"]),
        "E7-C irreversibility ban changes the outcome": (
            out["scenarios"]["S3_irreversibility"]["full"]["status"],
            out["scenarios"]["S3_irreversibility"]["full"]["n_irreversible"])
            != (out["scenarios"]["S3_irreversibility"]["-reversible"]["status"],
                out["scenarios"]["S3_irreversibility"]["-reversible"]["n_irreversible"]),
    }
    print("Machinery-binds check:", json.dumps(v))
    print("wrote experiments/exp_arcs/e7cd_synthetic_v2.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
