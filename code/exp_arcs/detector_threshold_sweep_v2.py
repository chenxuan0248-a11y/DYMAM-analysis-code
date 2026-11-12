"""detector_threshold_sweep_v2 — extends detector_threshold_sweep (Reviewer item B)
to the high-precision operating points recommended in the Discussion (t in
[0.85, 0.97]) and to the downstream scope-quality metrics.

detector_threshold_sweep_v1.json only swept t in [0.3, 0.8] and reported
combined_success / mean_cost_over_Cstar. detector_pr_roc_table.tex separately
reports classifier precision/recall/clean-FA at t=0.9/0.95, and
detector_threshold_cv_v1.json separately reports LOOO held-out detection-only
metrics at t*=0.97 — none of the three includes end-to-end repair success,
cost, or scope precision/recall at the SAME high-t operating point, so the
paper's practitioner recommendation (t in [0.9, 0.97]) and its RQ2 headline
(success at t=0.5) were being read off different artifacts. This script closes
that gap: same scoring path as v1 (score_cell -> evaluate_scope), extended
threshold grid, and scope_precision/scope_recall/mean_harm added per threshold.

    poetry run python -m experiments.exp_arcs.detector_threshold_sweep_v2
"""
from __future__ import annotations
import json
import statistics as st
import sys
from pathlib import Path

from experiments.exp_arcs.contamination import contamination_closure
from experiments.exp_arcs.dag import build_dag
from experiments.exp_arcs.detector_eval import (
    detector_localize,
    manifest_dropped_hosts,
    peer_anomaly_hosts,
)
from experiments.exp_arcs.select import evaluate_scope

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97]
OUT = Path("experiments/exp_arcs/detector_threshold_sweep_v2.json")


def score_cell(cell, thr):
    clean, faulted = Path(cell["clean_dir"]), Path(cell["faulted_dir"])
    dag = build_dag(clean)
    dfx = contamination_closure(clean, faulted)
    flagged = (detector_localize(faulted)["flagged_hosts"]
               | peer_anomaly_hosts(faulted, min_base_rate=thr)
               | manifest_dropped_hosts(faulted))
    scope = {n for n, v in dag.nodes.items()
             if v.get("repairable") and v.get("target") in flagged}
    ev = evaluate_scope(clean, dfx, scope, dag=dag)
    return {
        "success": ev["success"], "cost": ev["cost"], "harm": ev["harm"],
        "C_star": cell["C_star"], "n_flagged": len(flagged),
        "scope_precision": ev["scope_precision"], "scope_recall": ev["scope_recall"],
    }


def main() -> int:
    grid = json.loads(Path("experiments/exp_arcs/e3_grid_v1.json").read_text())["cells"]
    sweep = {}
    for thr in THRESHOLDS:
        res = [score_cell(c, thr) for c in grid]
        n = len(res)
        succ = sum(1 for r in res if r["success"]) / n
        ccs = [r["cost"] / r["C_star"] for r in res if r["success"] and r["C_star"]]
        sweep[str(thr)] = {
            "combined_success": round(succ, 4),
            "mean_cost_over_Cstar": round(st.mean(ccs), 2) if ccs else None,
            "mean_hosts_flagged": round(st.mean(r["n_flagged"] for r in res), 2),
            "mean_harm": round(st.mean(r["harm"] for r in res), 4),
            "mean_scope_precision": round(st.mean(r["scope_precision"] for r in res), 4),
            "mean_scope_recall": round(st.mean(r["scope_recall"] for r in res), 4),
        }
    frozen = json.loads(Path("experiments/exp_arcs/e3_detector_v1.json").read_text()
                        )["summary"]["detector_only_success"]["combined"]
    ok = abs(sweep["0.5"]["combined_success"] - frozen) < 1e-9
    result = {
        "phase": "detector_threshold_sweep_v2 — extends reviewer-B sweep to "
                 "t in [0.85, 0.97] with downstream scope P/R and harm, so the "
                 "recommended high-t operating point and the RQ2 headline share "
                 "one artifact",
        "thresholds": THRESHOLDS,
        "faithfulness_check_at_0.5": {"sweep": sweep["0.5"]["combined_success"],
                                      "frozen": frozen, "match": ok},
        "sweep": sweep,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"{'thr':>5} {'success':>8} {'cost/C*':>8} {'harm':>7} {'scopeP':>7} {'scopeR':>7} {'flagged':>8}")
    for thr in THRESHOLDS:
        r = sweep[str(thr)]
        print(f"{thr:>5} {r['combined_success']:>8} {r['mean_cost_over_Cstar']:>8} "
              f"{r['mean_harm']:>7} {r['mean_scope_precision']:>7} "
              f"{r['mean_scope_recall']:>7} {r['mean_hosts_flagged']:>8}")
    print(f"\nfaithfulness @0.5 vs frozen 0.806: match={ok}")
    print(f"wrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
