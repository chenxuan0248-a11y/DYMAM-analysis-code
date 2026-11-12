"""experiments/exp_arcs/negative_control_f2 — F2 intrinsically-global NEGATIVE CONTROL.

Runs ARCS on the 5-org F2 cells (from `ilp_oracle_pilot_v1.json`, global-injected)
and shows ARCS ≈ global re-collect — i.e. when a fault is intrinsically global
(F2's adaptive re-probe amplification), ARCS correctly does NOT shrink the repair
scope. This is the counterpart to the local grid ({local → ARCS ≪ global}); the
pair rebuts "ARCS just always minimizes": its scope tracks real locality.

Distinguishing axis = ARCS_cost / global (≈0.8 for F2 vs ≈0.01 for local faults).
NOTE ρC = ARCS/C* ≡ 1 here too (no substitution → forced cover → ARCS = optimum),
so the control is about cost-vs-global, NOT ρC.

Reads bundle dirs recorded in `ilp_oracle_pilot_v1.json`. If those (prior-session)
bundles are gone, regenerate them with the frozen global interceptor:
`ilp_oracle_pilot.py` (F2 ∈ its FAULTS).

    poetry run python -m experiments.exp_arcs.negative_control_f2
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from experiments.exp_arcs.candidates import candidate_invocations
from experiments.exp_arcs.contamination import contamination_closure
from experiments.exp_arcs.ilp_oracle import load_config
from experiments.exp_arcs.select import arcs_greedy, evaluate_scope, single_point


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    src = Path(argv[0]) if argv else Path("experiments/exp_arcs/ilp_oracle_pilot_v1.json")
    out = Path(argv[1]) if len(argv) > 1 else Path("experiments/exp_arcs/arcs_negative_control_f2_v1.json")
    cfg = load_config()
    cells = [c for c in json.loads(src.read_text())["cells"] if c["fault"] == "F2"]

    rows = []
    missing = []
    for c in cells:
        clean, faulted = Path(c["clean_dir"]), Path(c["faulted_dir"])
        if not (clean.exists() and faulted.exists()):
            missing.append(c["org"]); continue
        dfx = contamination_closure(clean, faulted)
        cand = candidate_invocations(clean, dfx)
        arcs = evaluate_scope(clean, dfx, arcs_greedy(clean, dfx, cand, cfg=cfg))
        # single-point on a global fault: the single largest invocation (must fail)
        inv0 = max(cand["invocations"], key=lambda k: (len(cand["invocations"][k]), str(k)),
                   default=None)
        sp = evaluate_scope(clean, dfx, single_point(clean, dfx, cand, inv0)) if inv0 else None
        g = c["global_recollect_cost"]
        rows.append({
            "org": c["org"], "D_f": c["D_f_contaminated"], "C_star": c["C_star"],
            "global_recollect_cost": g,
            "arcs_cost": arcs["cost"], "arcs_success": arcs["success"],
            "arcs_over_global": round(arcs["cost"] / g, 3) if g else None,
            "rho_C_arcs_vs_ilp": round(arcs["cost"] / c["C_star"], 3) if c["C_star"] else None,
            "singlepoint_success": sp["success"] if sp else None,
            "singlepoint_cost": sp["cost"] if sp else None,
        })

    summary = {
        "control": "F2 intrinsically-global — ARCS should ≈ global (NOT shrink)",
        "n_cells": len(rows), "missing_bundles": missing,
        "mean_arcs_over_global": round(statistics.mean([r["arcs_over_global"] for r in rows]), 3) if rows else None,
        "range_arcs_over_global": [min(r["arcs_over_global"] for r in rows),
                                   max(r["arcs_over_global"] for r in rows)] if rows else None,
        "mean_rho_C": round(statistics.mean([r["rho_C_arcs_vs_ilp"] for r in rows]), 3) if rows else None,
        "arcs_success_rate": round(statistics.mean([1 if r["arcs_success"] else 0 for r in rows]), 3) if rows else None,
        "singlepoint_success_rate": round(statistics.mean([1 if r["singlepoint_success"] else 0 for r in rows]), 3) if rows else None,
        "contrast_local_arcs_over_global": "0.003–0.017 (see arcs_pilot_all_v1.json)",
        "reading": ("ARCS/global ≈ 0.7–0.9 on F2 vs ≈0.01 on local faults ⇒ ARCS's "
                    "scope tracks real locality; it does not blindly minimize. ρC≡1 "
                    "throughout (structural), so the control axis is cost-vs-global."),
    }
    result = {"phase": "5.negative_control_f2", "source": str(src),
              "summary": summary, "cells": rows}
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2))
    if missing:
        print(f"\nWARNING missing bundles for: {missing} — regenerate via ilp_oracle_pilot.py")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
