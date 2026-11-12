"""experiments/exp_arcs/counterfactual — Step 2.1 valid repair space V(f).

Decisions A1 + B1 (see ``phase2_design_choices.md``):

* B1 — DAG-level counterfactual: repairing a leaf = substituting its clean value.
  Under deterministic record-replay a re-collected leaf yields exactly its clean
  cached value, so contamination-reachability on the clean DAG is the exact
  success test — no pipeline re-run.
* A1 — success in record space: repairing scope S restores a contaminated output
  (report_item) iff every contaminated leaf that is an ancestor of that output in
  the clean DAG is in S.

So V(f) is characterized by the coverage map ``{output -> required leaves}`` (the
ILP's ``D(f) ⊆ Cov(S)`` constraint), not enumerated. The minimal valid scope is
the union of required leaves — it can be strictly smaller than all contaminated
leaves (leaves feeding no surviving report need no repair), which is the ILP's
real freedom. Outputs with NO contaminated leaf ancestor are not leaf-repairable
(gate-logic / orphan-claim contamination) and reported separately — this is why
gate_logic faults (F8) are out of Phase 2 scope.

    poetry run python -m experiments.exp_arcs.counterfactual \
        --clean <clean_run_dir> --faulted <faulted_run_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable

from experiments.exp_arcs.contamination import _REPAIRABLE, contamination_closure
from experiments.exp_arcs.dag import Dag, build_dag, scope_cost

_OUTPUT_TYPE = "report_item"   # the "result" whose correctness defines success


def _bare(dfx: dict[str, Any], types: tuple[str, ...]) -> set[str]:
    return {i.split(":", 1)[1] for i in dfx["contaminated_ids"]
            if i.split(":", 1)[0] in types}


def _ancestors(dag: Dag, targets: Iterable[str]) -> set[str]:
    """All nodes upstream of ``targets`` (reverse reachability)."""
    radj: dict[str, list[str]] = {}
    for s, d in dag.edges:
        radj.setdefault(d, []).append(s)
    seen: set[str] = set()
    queue = deque(targets)
    while queue:
        n = queue.popleft()
        for p in radj.get(n, []):
            if p not in seen:
                seen.add(p)
                queue.append(p)
    return seen


def coverage_map(clean_dir: Path, dfx: dict[str, Any]) -> dict[str, Any]:
    """Per contaminated output: its contaminated *repairable* leaf ancestors.

    An output is unrepairable-by-leaf if it has NO contaminated leaf ancestor
    (gate-logic / orphan claim) OR any contaminated ancestor is a non-repairable
    (full_replay/derived) leaf — repairing the rest cannot restore it.
    """
    dag = build_dag(clean_dir)
    cont_by_type = {l for l in _bare(dfx, _REPAIRABLE) if l in dag.nodes}
    repairable = {l for l in cont_by_type if dag.nodes[l].get("repairable")}
    nonrepairable = cont_by_type - repairable
    cont_outputs = {o for o in _bare(dfx, (_OUTPUT_TYPE,)) if o in dag.nodes}

    cov: dict[str, list[str]] = {}
    unrepairable: list[str] = []
    for o in cont_outputs:
        anc = _ancestors(dag, {o})
        anc_rep = anc & repairable
        anc_nonrep = anc & nonrepairable
        if anc_nonrep or not anc_rep:
            unrepairable.append(o)
        else:
            cov[o] = sorted(anc_rep)
    return {
        "cont_leaves": sorted(repairable),
        "cont_leaves_nonrepairable": sorted(nonrepairable),
        "coverage_map": cov,
        "unrepairable_outputs": sorted(unrepairable),
    }


def repair_and_check(clean_dir: Path, dfx: dict[str, Any],
                     scope: Iterable[str]) -> dict[str, Any]:
    """A1 success test for a candidate scope S (set of leaf ids)."""
    cm = coverage_map(clean_dir, dfx)
    S = set(scope)
    broken = [o for o, req in cm["coverage_map"].items() if not set(req) <= S]
    unrep = cm["unrepairable_outputs"]
    return {
        "success": not broken and not unrep,
        "n_outputs_restored": len(cm["coverage_map"]) - len(broken),
        "n_outputs_broken": len(broken),
        "n_outputs_unrepairable_by_leaf": len(unrep),
    }


def vf(clean_dir: str | Path, faulted_dir: str | Path) -> dict[str, Any]:
    """V(f) characterization + costs (invocation grain) for one bundle pair."""
    clean_dir, faulted_dir = Path(clean_dir), Path(faulted_dir)
    dag = build_dag(clean_dir)
    dfx = contamination_closure(clean_dir, faulted_dir)
    cm = coverage_map(clean_dir, dfx)
    minimal = sorted({l for req in cm["coverage_map"].values() for l in req})

    # baselines at (target, family) invocation grain
    minimal_cost = scope_cost(dag, minimal)
    all_repairable = [n for n, v in dag.nodes.items() if v.get("repairable")]
    global_cost = scope_cost(dag, all_repairable)   # re-collect the whole run
    return {
        "n_contaminated": dfx["n_contaminated"],
        "n_contaminated_leaves": len(cm["cont_leaves"]),
        "n_contaminated_leaves_nonrepairable": len(cm["cont_leaves_nonrepairable"]),
        "n_contaminated_outputs": len(cm["coverage_map"]) + len(cm["unrepairable_outputs"]),
        "n_outputs_leaf_repairable": len(cm["coverage_map"]),
        "n_outputs_unrepairable_by_leaf": len(cm["unrepairable_outputs"]),
        "minimal_repair_scope_size": len(minimal),
        "minimal_cost": minimal_cost,          # C(S*_min): cost/harm/invocations
        "global_replay_cost": global_cost,     # re-collect all repairable leaves
        "minimal_repair_scope": minimal,
        "coverage_map": cm["coverage_map"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--faulted", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    result = vf(args.clean, args.faulted)
    print(f"contaminated leaves        : {result['n_contaminated_leaves']}")
    print(f"contaminated outputs       : {result['n_contaminated_outputs']} "
          f"(leaf-repairable {result['n_outputs_leaf_repairable']}, "
          f"unrepairable-by-leaf {result['n_outputs_unrepairable_by_leaf']})")
    print(f"minimal repair scope       : {result['leaves_needed_vs_contaminated']} "
          f"(leaves needed / contaminated leaves)")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
