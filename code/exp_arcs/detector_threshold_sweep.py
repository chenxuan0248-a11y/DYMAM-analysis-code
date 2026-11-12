"""Reviewer item B (part 1) — peer-anomaly threshold sensitivity.

The combined detector's peer-anomaly signal flags a host missing a family that
>= PEER_MIN_BASE_RATE of peers have. The frozen runs use 0.5. This sweeps that
threshold on the 180-cell E3 grid and reports end-to-end combined ARCS success +
cost/C* (host-grain), so "0.5" is shown to be a robust operating point, not a
tuned constant. Reuses detector_eval's signal fns + evaluate_scope (no new
scoring); at 0.5 it must reproduce the frozen combined 0.806 (asserted).

    poetry run python -m experiments.exp_arcs.detector_threshold_sweep
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

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
OUT = Path("experiments/exp_arcs/detector_threshold_sweep_v1.json")


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
    return ev["success"], ev["cost"], cell["C_star"], len(flagged)


def main() -> int:
    grid = json.loads(Path("experiments/exp_arcs/e3_grid_v1.json").read_text())["cells"]
    sweep = {}
    for thr in THRESHOLDS:
        res = [score_cell(c, thr) for c in grid]
        n = len(res)
        succ = sum(1 for s, *_ in res if s) / n
        ccs = [c / cs for s, c, cs, _ in res if s and cs]
        flags = st.mean(nf for *_, nf in res)
        sweep[str(thr)] = {
            "combined_success": round(succ, 4),
            "mean_cost_over_Cstar": round(st.mean(ccs), 2) if ccs else None,
            "mean_hosts_flagged": round(flags, 2),
        }
    # faithfulness: 0.5 must equal frozen combined 0.806
    frozen = json.loads(Path("experiments/exp_arcs/e3_detector_v1.json").read_text()
                        )["summary"]["detector_only_success"]["combined"]
    ok = abs(sweep["0.5"]["combined_success"] - frozen) < 1e-9
    result = {
        "phase": "reviewer-B peer-anomaly threshold sensitivity (180 cells, host-grain)",
        "thresholds": THRESHOLDS,
        "faithfulness_check_at_0.5": {"sweep": sweep["0.5"]["combined_success"],
                                      "frozen": frozen, "match": ok},
        "sweep": sweep,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"{'thr':>5} {'success':>8} {'cost/C*':>8} {'hosts_flagged':>14}")
    for thr in THRESHOLDS:
        r = sweep[str(thr)]
        print(f"{thr:>5} {r['combined_success']:>8} {r['mean_cost_over_Cstar']:>8} "
              f"{r['mean_hosts_flagged']:>14}")
    print(f"\nfaithfulness @0.5 vs frozen 0.806: match={ok}")
    print(f"wrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
