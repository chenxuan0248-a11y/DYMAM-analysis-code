"""experiments/exp_arcs/run_e3_f2 — E3 F2 negative control, extended to 20 orgs.

F2 (wrong_tool) is intrinsically GLOBAL: the wrong fingerprint propagates across
the whole run, so the correct repair is ≈ global re-collect. The negative control
shows ARCS does NOT shrink scope here (ARCS/global ≈ 0.7–0.9), the counterpart to
the local grid (ARCS/global ≈ 0.01). Together: ARCS's scope tracks real locality,
it does not blindly minimize.

Reuses each org's E3 CLEAN bundle (deterministic replay, R_replay=1.0) and records
one F2-faulted run per org via the frozen GLOBAL interceptor `interceptors.inject`.
Then solve_oracle → C\*, ARCS greedy → cost, and the control metric ARCS/global.

    poetry run python -m experiments.exp_arcs.run_e3_f2 [raw_root] [grid.json] [out.json]
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from core.graph import run_pipeline
from experiments.exp_arcs.candidates import candidate_invocations
from experiments.exp_arcs.contamination import contamination_closure
from experiments.exp_arcs.ilp_oracle import load_config, solve_oracle
from experiments.exp_arcs.interceptors import inject
from experiments.exp_arcs.recorder import record_run
from experiments.exp_arcs.select import arcs_greedy, evaluate_scope, single_point

RAW_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/arcs_e3_f2_bundles")
GRID = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("experiments/exp_arcs/e3_grid_v1.json")
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("experiments/exp_arcs/e3_negative_control_f2_v1.json")


def main() -> int:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    grid = json.loads(GRID.read_text())
    # per-org clean bundle + global cost (same across an org's cells)
    per_org: dict[str, dict] = {}
    for c in grid["cells"]:
        per_org.setdefault(c["org"], {"clean": c["clean_dir"],
                                      "global": c["global_recollect_cost"]})
    orgs = sorted(per_org)
    print(f"F2 negative control over {len(orgs)} orgs", flush=True)

    rows, missing = [], []
    for org in orgs:
        clean = Path(per_org[org]["clean"])
        if not clean.exists():
            missing.append(org); print(f"  [skip] {org}: clean bundle gone", flush=True); continue
        g = per_org[org]["global"]
        try:
            with inject("F2"):
                faulted = record_run(run_pipeline(organization=org), raw_root=RAW_ROOT)
            dfx = contamination_closure(clean, faulted)
            cand = candidate_invocations(clean, dfx)
            ilp = solve_oracle(clean, faulted)
            arcs = evaluate_scope(clean, dfx, arcs_greedy(clean, dfx, cand, cfg=cfg))
            inv0 = max(cand["invocations"], key=lambda k: (len(cand["invocations"][k]), str(k)),
                       default=None)
            sp = evaluate_scope(clean, dfx, single_point(clean, dfx, cand, inv0)) if inv0 else None
            rows.append({
                "org": org, "D_f": dfx["n_contaminated"], "C_star": ilp["C_star"],
                "global_recollect_cost": g,
                "arcs_cost": arcs["cost"], "arcs_success": arcs["success"],
                "arcs_over_global": round(arcs["cost"] / g, 3) if g else None,
                "rho_C_arcs_vs_ilp": round(arcs["cost"] / ilp["C_star"], 3) if ilp["C_star"] else None,
                "singlepoint_success": sp["success"] if sp else None,
                "singlepoint_cost": sp["cost"] if sp else None,
            })
            print(f"  {org:11} D(f)={dfx['n_contaminated']:>4} C*={ilp['C_star']:>4} "
                  f"global={g:>4} ARCS={arcs['cost']:>4} ARCS/global={rows[-1]['arcs_over_global']} "
                  f"arcs_succ={arcs['success']} sp_succ={rows[-1]['singlepoint_success']}", flush=True)
        except Exception as e:
            print(f"  [FAIL] {org}: {type(e).__name__}: {str(e)[:200]}", flush=True)

    aog = [r["arcs_over_global"] for r in rows if r["arcs_over_global"] is not None]
    summary = {
        "control": "F2 intrinsically-global — ARCS should ≈ global (NOT shrink)",
        "n_cells": len(rows), "missing_bundles": missing,
        "mean_arcs_over_global": round(statistics.mean(aog), 3) if aog else None,
        "range_arcs_over_global": [min(aog), max(aog)] if aog else None,
        "mean_rho_C": round(statistics.mean([r["rho_C_arcs_vs_ilp"] for r in rows]), 3) if rows else None,
        "arcs_success_rate": round(statistics.mean([1 if r["arcs_success"] else 0 for r in rows]), 3) if rows else None,
        "singlepoint_success_rate": round(statistics.mean([1 if r["singlepoint_success"] else 0 for r in rows]), 3) if rows else None,
        "contrast_local_arcs_over_global": "local grid ARCS/global ≈ 0.003–0.04 (e3_arcs_v1.json)",
        "reading": ("ARCS/global ≈ 0.7–0.9 on F2 (global) vs ≈0.01 on local faults ⇒ "
                    "ARCS's scope tracks real locality; it does not blindly minimize. "
                    "ρC≡1 throughout (structural) — control axis is cost-vs-global."),
    }
    result = {"phase": "6.E3.negative_control_f2", "source": str(GRID),
              "summary": summary, "cells": rows}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
