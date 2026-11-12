"""experiments/exp_arcs/candidates — Step 4.1 candidate repair-scope generation.

From a localized fault + the clean DAG, enumerate the candidate repair units on
I(f). The action grain is the (target, source_family) INVOCATION (same grain as
`scope_cost` and the ILP oracle): one invocation re-collects all of a host's
evidence for that family. A candidate = the set of contaminated repairable leaves
belonging to one invocation; selecting it re-collects (and, under record-replay,
restores to the clean value) all of them at one invocation's cost.

Candidates are restricted to the contaminated repairable leaves (the only leaves a
repair could change), grouped by invocation. This is I(f) at the repair grain.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.exp_arcs.contamination import _REPAIRABLE, contamination_closure
from experiments.exp_arcs.counterfactual import _bare, coverage_map
from experiments.exp_arcs.dag import build_dag


def candidate_invocations(clean_dir: str | Path,
                          dfx: dict[str, Any] | None = None,
                          faulted_dir: str | Path | None = None) -> dict[str, Any]:
    """Return candidate invocations and the leaf-repairable coverage map.

    ``invocations``: {(target, family) -> sorted[leaf ids]} over the contaminated
    repairable leaves present in the clean DAG.
    ``inv_cost/inv_harm/inv_irreversible``: per-invocation cost/harm/irreversible.
    ``coverage_map``: {output -> required contaminated repairable leaves} (from
    counterfactual; the success target).
    """
    if dfx is None:
        assert faulted_dir is not None, "need dfx or faulted_dir"
        dfx = contamination_closure(Path(clean_dir), Path(faulted_dir))
    dag = build_dag(Path(clean_dir))
    cm = coverage_map(Path(clean_dir), dfx)

    cont_leaves = [l for l in _bare(dfx, _REPAIRABLE)
                   if l in dag.nodes and dag.nodes[l].get("repairable")]
    inv_leaves: dict[tuple, list[str]] = defaultdict(list)
    inv_cost: dict[tuple, float] = {}
    inv_harm: dict[tuple, float] = {}
    inv_irr: dict[tuple, bool] = {}
    for l in cont_leaves:
        v = dag.nodes[l]
        key = (v.get("target"), v["source_family"])
        inv_leaves[key].append(l)
        inv_cost[key] = v["cost"]
        inv_harm[key] = v["harm"]
        inv_irr[key] = v["reversibility_rank"] == 3
    return {
        "invocations": {k: sorted(v) for k, v in inv_leaves.items()},
        "inv_cost": inv_cost, "inv_harm": inv_harm, "inv_irreversible": inv_irr,
        "coverage_map": cm["coverage_map"],
        "unrepairable_outputs": cm["unrepairable_outputs"],
        "all_repairable_leaves": [n for n, v in dag.nodes.items() if v.get("repairable")],
    }
