"""experiments/exp_arcs/arcs_pilot — Phase 4 first pass (offline).

For each of the 25 local cells (`local_oracle_v1.json`): compute Global / Single-
point / ARCS-greedy / ILP repair, under Oracle and semi-oracle lineage
localization. Success = A1 test (every leaf-repairable output covered), cost/harm
at invocation grain. Outputs the Phase-5/G3 inputs: ρC = C(S_ARCS)/C*, Success@B0,
scope precision/recall. No pipeline runs (record-replay A1 counterfactual).

    poetry run python -m experiments.exp_arcs.arcs_pilot [local_oracle_v1.json] [out.json]
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

from experiments.exp_arcs.candidates import candidate_invocations
from experiments.exp_arcs.contamination import contamination_closure
from experiments.exp_arcs.dag import build_dag
from experiments.exp_arcs.ilp_oracle import load_config
from experiments.exp_arcs.localizer import lineage_blame
from experiments.exp_arcs.select import (
    arcs_greedy, evaluate_scope, global_replay, single_point)
from experiments.exp_arcs.substitution import alt_covers


def _oracle_inv(inject_node: list, cand: dict) -> tuple | None:
    """The single localized invocation a single-point baseline would repair.

    Handles both single-host inject_node ('target', host[, family]) and multi-point
    ('targets', [hosts][, family]). For a multi-point / multi-family fault this one
    invocation necessarily under-covers — that is the point of the baseline. Pick
    the host-set's invocation with the most contaminated leaves (deterministic)."""
    kind = inject_node[0]
    if kind == "targets":
        hosts = set(inject_node[1])
        family = inject_node[2] if len(inject_node) >= 3 else None
    else:
        hosts = {inject_node[1]}
        family = inject_node[2] if len(inject_node) >= 3 else None
    invs = [(k, v) for k, v in cand["invocations"].items()
            if k[0] in hosts and (family is None or k[1] == family)]
    if not invs:
        return None
    return max(invs, key=lambda kv: (len(kv[1]), str(kv[0])))[0]


def run_cell(cell: dict, cfg: dict) -> dict[str, Any]:
    clean, faulted = Path(cell["clean_dir"]), Path(cell["faulted_dir"])
    dag = build_dag(clean)
    dfx = contamination_closure(clean, faulted)
    cand = candidate_invocations(clean, dfx)
    sub = alt_covers(clean, dfx)
    C_star = cell["C_star"]

    oinv = _oracle_inv(cell["inject_node"], cand)
    linv = lineage_blame(clean, faulted)["ranked_tf"][0]

    def ev(scope):
        return evaluate_scope(clean, dfx, scope, dag=dag)

    m_global = ev(global_replay(clean, dfx, cand))
    m_sp_oracle = ev(single_point(clean, dfx, cand, oinv)) if oinv else None
    m_sp_lineage = ev(single_point(clean, dfx, cand, linv))
    m_arcs = ev(arcs_greedy(clean, dfx, cand, alt=sub["alt_covers"], cfg=cfg))

    rho_c = round(m_arcs["cost"] / C_star, 4) if C_star else None
    return {
        "org": cell["org"], "cell": cell["cell"], "inject_node": cell["inject_node"],
        "C_star": C_star, "global_recollect_cost": cell["global_recollect_cost"],
        "n_substitutable_outputs": sub["n_substitutable"],
        "oracle_inv": list(oinv) if oinv else None,
        "lineage_inv": list(linv) if linv else None,
        # host-level match (F10 has no family truth, so the family may differ)
        "loc_match_oracle_vs_lineage": (oinv[0] == linv[0]) if (oinv and linv) else None,
        "methods": {
            "global": m_global,
            "single_point_oracle": m_sp_oracle,
            "single_point_lineage": m_sp_lineage,
            "arcs": m_arcs,
            "ilp": {"cost": C_star, "success": True},   # GATE-verified in Phase 2
        },
        "rho_C_arcs_vs_ilp": rho_c,
    }


def _rate(cells, path, pred):
    vals = [pred(c) for c in cells if pred(c) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    src = Path(argv[0]) if argv else Path("experiments/exp_arcs/local_oracle_v1.json")
    out = Path(argv[1]) if len(argv) > 1 else Path("experiments/exp_arcs/arcs_pilot_v1.json")
    cfg = load_config()
    cells = json.loads(src.read_text())["cells"]

    rows = [run_cell(c, cfg) for c in cells]

    # B0 = median single-point (oracle) cost
    sp_costs = [r["methods"]["single_point_oracle"]["cost"] for r in rows
                if r["methods"]["single_point_oracle"]]
    B0 = statistics.median(sp_costs)

    def succ_at_B0(r, method):
        m = r["methods"][method]
        if m is None:
            return None
        return 1.0 if (m["success"] and m["cost"] <= B0) else 0.0

    def succ(r, method):
        m = r["methods"][method]
        return None if m is None else (1.0 if m["success"] else 0.0)

    methods = ["global", "single_point_oracle", "single_point_lineage", "arcs", "ilp"]
    summary = {
        "n_cells": len(rows), "B0_median_singlepoint_cost": B0,
        "success_rate": {m: _rate(rows, None, lambda r, m=m: succ(r, m)) for m in methods},
        "success_at_B0": {m: _rate(rows, None, lambda r, m=m: succ_at_B0(r, m)) for m in methods},
        "mean_cost": {m: round(statistics.mean(
            [r["methods"][m]["cost"] for r in rows if r["methods"][m]]), 3) for m in methods},
        "mean_rho_C_arcs_vs_ilp": round(statistics.mean(
            [r["rho_C_arcs_vs_ilp"] for r in rows if r["rho_C_arcs_vs_ilp"] is not None]), 4),
        "arcs_beats_singlepoint_cells": sum(
            1 for r in rows
            if r["methods"]["arcs"]["success"] and not r["methods"]["single_point_oracle"]["success"]),
        "total_substitutable_outputs": sum(r["n_substitutable_outputs"] for r in rows),
        "localization_oracle_eq_lineage": all(
            r["loc_match_oracle_vs_lineage"] for r in rows if r["loc_match_oracle_vs_lineage"] is not None),
    }
    result = {"phase": "4.pilot", "spec": "25 local cells x {Global, Single-point, ARCS, ILP}",
              "source": str(src), "summary": summary, "cells": rows}
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(json.dumps(summary, indent=2))
    print("\nARCS > single-point cells (adaptive scope pays off):")
    for r in rows:
        a, sp = r["methods"]["arcs"], r["methods"]["single_point_oracle"]
        if a["success"] and sp and not sp["success"]:
            print(f"  {r['org']:11}/{r['cell']:20} single_pt cost={sp['cost']} succ={sp['success']} "
                  f"-> ARCS cost={a['cost']} succ={a['success']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
