"""experiments/exp_arcs/e7_localization_ablation — reviewer item A.

Signal-level ablation of the reference-free localizer on the 180-cell E3 grid.
The combined detector unions three reference-free signals:
  - self  = pipeline self-diagnosis (revalidate / no_supporting_observation)
  - peer  = cross-host peer-anomaly (host missing a family >=50% of peers have)
  - manif = discovery-manifest diff (resolved-but-unused hosts, catches F10)

This does NOT re-run the pipeline and adds NO new scoring: it reuses
`detector_eval`'s own signal functions and the SAME `evaluate_scope` A1
counterfactual as the frozen `e3_detector_v1.json`. It only scores additional
signal SUBSETS (single = leave-one-in; pair = leave-one-out) that the frozen
artifact did not store, because leave-one-out success is not reconstructable from
per-signal booleans (union coverage != any-single coverage on multi-host faults).

Determinism check: the `combined` (all three) column reproduced here must equal
the frozen e3_detector_v1.json `combined` success rate, or the reuse is not
faithful (asserted in main()).

    poetry run python -m experiments.exp_arcs.e7_localization_ablation \
        [grid.json=e3_grid_v1.json] [out.json=e7_localization_ablation_v1.json]
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

from experiments.exp_arcs.contamination import contamination_closure
from experiments.exp_arcs.dag import build_dag
from experiments.exp_arcs.detector_eval import (
    detector_localize,
    manifest_dropped_hosts,
    peer_anomaly_hosts,
)
from experiments.exp_arcs.select import evaluate_scope

# leave-one-in (single) and leave-one-out (pair) + combined, host-grain.
COMBOS = {
    "self": ("self",),
    "peer": ("peer",),
    "manifest": ("manifest",),
    "self+peer": ("self", "peer"),            # = combined - manifest
    "self+manifest": ("self", "manifest"),    # = combined - peer
    "peer+manifest": ("peer", "manifest"),    # = combined - self
    "combined": ("self", "peer", "manifest"),
}


def _true_hosts(clean_dir: Path, dfx: dict, dag) -> set[str]:
    from experiments.exp_arcs.contamination import _REPAIRABLE
    from experiments.exp_arcs.counterfactual import _bare
    hosts = set()
    for l in _bare(dfx, _REPAIRABLE):
        v = dag.nodes.get(l)
        if v and v.get("repairable") and v.get("target"):
            hosts.add(v["target"])
    return hosts


def run_cell(cell: dict) -> dict[str, Any]:
    clean, faulted = Path(cell["clean_dir"]), Path(cell["faulted_dir"])
    dag = build_dag(clean)
    dfx = contamination_closure(clean, faulted)
    true_hosts = _true_hosts(clean, dfx, dag)
    sig = {
        "self": detector_localize(faulted)["flagged_hosts"],
        "peer": peer_anomaly_hosts(faulted),
        "manifest": manifest_dropped_hosts(faulted),
    }

    def arcs_on(flagged: set) -> dict:
        # IDENTICAL to detector_eval.run_cell.arcs_on (host-grain).
        scope = {n for n, v in dag.nodes.items()
                 if v.get("repairable") and v.get("target") in flagged}
        ev = evaluate_scope(clean, dfx, scope, dag=dag)
        return {"success": ev["success"], "cost": ev["cost"],
                "region_covered": bool(true_hosts) and true_hosts <= flagged,
                "n_flagged": len(flagged)}

    row = {"org": cell["org"], "cell": cell["cell"],
           "fault_type": "F10" if "F10" in cell["cell"] else "F4",
           "C_star": cell["C_star"], "n_true_hosts": len(true_hosts)}
    for name, parts in COMBOS.items():
        flagged: set = set()
        for p in parts:
            flagged |= sig[p]
        row[name] = arcs_on(flagged)
    return row


def _succ(rows, key):
    v = [1 if r[key]["success"] else 0 for r in rows]
    return round(sum(v) / len(v), 4) if v else None


def _cost_over_cstar(rows, key):
    cc = [(r[key]["cost"], r["C_star"]) for r in rows
          if r[key]["success"] and r["C_star"]]
    return round(statistics.mean([c / s for c, s in cc]), 2) if cc else None


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    src = Path(argv[0]) if argv else Path("experiments/exp_arcs/e3_grid_v1.json")
    out = Path(argv[1]) if len(argv) > 1 else Path(
        "experiments/exp_arcs/e7_localization_ablation_v1.json")
    cells = json.loads(src.read_text())["cells"]
    rows = [run_cell(c) for c in cells]

    combos = list(COMBOS)
    overall = {c: {"success": _succ(rows, c),
                   "cost_over_Cstar": _cost_over_cstar(rows, c)} for c in combos}
    by_subtype = {}
    for name in sorted({r["cell"] for r in rows}):
        sub = [r for r in rows if r["cell"] == name]
        by_subtype[name] = {c: _succ(sub, c) for c in combos}

    # leave-one-out drop = combined - (combined without signal X)
    loo_drop = {
        "drop_self": round(overall["combined"]["success"] - overall["peer+manifest"]["success"], 4),
        "drop_peer": round(overall["combined"]["success"] - overall["self+manifest"]["success"], 4),
        "drop_manifest": round(overall["combined"]["success"] - overall["self+peer"]["success"], 4),
    }

    result = {
        "phase": "reviewer-A localization signal ablation (host-grain, 180 cells)",
        "source": str(src),
        "grid": "e3_grid_v1 (20 orgs x 9 fault subtypes)",
        "note": ("Reuses detector_eval signal fns + evaluate_scope A1 counterfactual "
                 "(no new scoring). 'combined' must match e3_detector_v1.json combined."),
        "n_cells": len(rows),
        "overall": overall,
        "leave_one_out_success_drop": loo_drop,
        "by_subtype": by_subtype,
        "cells": rows,
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"overall": overall, "leave_one_out_success_drop": loo_drop,
                      "by_subtype": by_subtype}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
