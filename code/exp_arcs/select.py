"""experiments/exp_arcs/select — Step 4.3/4.4 ARCS scope selection + baselines.

All methods return a repair scope as a set of leaf ids, then are scored uniformly:
cost/harm/invocations via `dag.scope_cost` (invocation grain), success via
`counterfactual.repair_and_check` (A1 test: every leaf-repairable output covered),
and scope precision/recall against the contaminated repairable leaves D(f).

Methods:
  - global_replay : re-collect every repairable leaf (the upper-cost anchor).
  - single_point  : repair only the one localized invocation.
  - arcs_greedy   : cost/harm-greedy cover of the leaf-repairable outputs, under
                    H_max / budget / no-irreversible, with optional substitution
                    (alt_covers) letting an output be satisfied by a surviving
                    independent family instead of a repair.

Coverage is a HARD constraint (paper §3.3): if the constraints make a full cover
impossible, arcs_greedy raises SelectionInfeasible (default) so the caller escalates
(§3.4) instead of executing a partial scope. Budget-sweep evaluations that measure
"what is achievable within budget B" (pareto, Success@B0) opt into the legacy
partial result with on_infeasible="partial".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from experiments.exp_arcs.counterfactual import repair_and_check
from experiments.exp_arcs.dag import build_dag, scope_cost
from experiments.exp_arcs.ilp_oracle import load_config


class SelectionInfeasible(Exception):
    """No scope satisfies coverage under the active constraints.

    ``reasons``: subset of {"irreversible", "harm_budget", "cost_budget", "no_cover"} —
    why the uncovered leaves could not be reached. ``partial_scope``: leaves the greedy
    had selected when it stopped (never to be executed as a repair; kept for audit).
    ``uncovered``: target leaves left uncovered.
    """

    def __init__(self, reasons: set[str], partial_scope: set[str], uncovered: set[str]):
        self.reasons = set(reasons)
        self.partial_scope = set(partial_scope)
        self.uncovered = set(uncovered)
        super().__init__(f"infeasible ({', '.join(sorted(self.reasons))}); "
                         f"{len(self.uncovered)} target leaves uncovered")


def evaluate_scope(clean_dir: str | Path, dfx: dict[str, Any],
                   scope: Iterable[str], dag=None) -> dict[str, Any]:
    dag = dag or build_dag(Path(clean_dir))
    S = set(scope)
    sc = scope_cost(dag, S)
    chk = repair_and_check(Path(clean_dir), dfx, S)
    D_rep = {i.split(":", 1)[1] for i in dfx["repairable_ids"]}
    inter = S & D_rep
    return {
        "n_leaves": len(S),
        "cost": sc["cost"], "harm": sc["harm"],
        "n_invocations": sc["n_invocations"],
        "n_irreversible": sc["n_irreversible"],
        "success": chk["success"] or chk["n_outputs_broken"] == 0,
        "n_outputs_broken": chk["n_outputs_broken"],
        "n_outputs_restored": chk["n_outputs_restored"],
        "scope_precision": round(len(inter) / len(S), 4) if S else 1.0,
        "scope_recall": round(len(inter) / len(D_rep), 4) if D_rep else 1.0,
    }


def global_replay(clean_dir, dfx, cand) -> set[str]:
    return set(cand["all_repairable_leaves"])


def fixed_topk(region_invocations: dict[tuple, list[str]],
               ranking: list[tuple], k: int) -> set[str]:
    """Non-adaptive E7-A baseline: repair the k highest-ranked invocations in the
    localized region, for a k FIXED across all cells — regardless of how many
    invocations the fault actually needs.

    ``region_invocations``: {(target, family) -> leaf ids} over ALL repairable
    invocations of the localized host(s) (the region pool — includes non-culprit
    invocations, so a large fixed k over-repairs narrow faults). ``ranking``: a
    precomputed order over those keys; the runner ranks by contaminated-leaf mass
    first (a STEELMAN — even a well-ranked fixed-k is compared, so ARCS's win is
    conservative). Returns the union of the top-k invocations' leaves, capped at
    the pool size (cannot repair what the region does not contain).

    This isolates ARCS's *adaptive* scope (arcs_greedy picks as many invocations
    as needed, per cell) from the trivial "just repair k more nodes" strategy: the
    rebuttal to "more repairs = higher success" is that no single fixed k matches ARCS on BOTH
    success and cost across the grid."""
    chosen: set[str] = set()
    for key in ranking[:k]:
        chosen |= set(region_invocations.get(key, []))
    return chosen


def single_point(clean_dir, dfx, cand, localized_inv: tuple) -> set[str]:
    return set(cand["invocations"].get(localized_inv, []))


def arcs_greedy(clean_dir, dfx, cand, alt: dict | None = None,
                budget: float | None = None, cfg: dict | None = None,
                on_infeasible: str = "raise") -> set[str]:
    if on_infeasible not in ("raise", "partial"):
        raise ValueError(f"on_infeasible must be 'raise' or 'partial', not {on_infeasible!r}")
    cfg = cfg or load_config()
    alpha = float(cfg["objective"]["alpha"]); beta = float(cfg["objective"]["beta"])
    H_max = float(cfg["constraints"]["H_max"])
    forbid_irr = bool(cfg["constraints"]["forbid_irreversible"])

    invs = cand["invocations"]
    inv_cost, inv_harm, inv_irr = cand["inv_cost"], cand["inv_harm"], cand["inv_irreversible"]
    cov = {o: set(req) for o, req in cand["coverage_map"].items()}

    # substitution: an output already coverable by a surviving independent family
    # (alt cover with no contaminated leaf) needs no repair — drop it up front.
    alt = (alt or {})
    satisfiable_by_alt = set(alt.keys())

    # success = every surviving leaf-repairable output's required leaves selected,
    # i.e. cover the UNION of their required leaves. Weighted set cover over
    # invocations by required-leaf coverage (partial progress credited — an output
    # whose leaves span >1 invocation must not stall the greedy).
    target_leaves: set[str] = set()
    for o, req in cov.items():
        if o not in satisfiable_by_alt:
            target_leaves |= req

    chosen: set[tuple] = set()
    chosen_leaves: set[str] = set()
    harm_used = 0.0
    cost_used = 0.0
    remaining = set(target_leaves)
    while remaining:
        best = None; best_gain = 0.0
        for k, leaves in invs.items():
            if k in chosen:
                continue
            if forbid_irr and inv_irr.get(k):
                continue
            if harm_used + inv_harm.get(k, 0) > H_max:
                continue
            if budget is not None and cost_used + inv_cost.get(k, 0) > budget:
                continue
            newly = len(remaining & set(leaves))
            if newly <= 0:
                continue
            price = alpha * inv_cost.get(k, 0) + beta * inv_harm.get(k, 0)
            gain = newly / price if price > 0 else newly
            if gain > best_gain:
                best_gain = gain; best = k
        if best is None:
            break  # budget/constraints exhausted before full coverage
        chosen.add(best)
        chosen_leaves |= set(invs[best])
        harm_used += inv_harm.get(best, 0)
        cost_used += inv_cost.get(best, 0)
        remaining -= set(invs[best])
    if remaining and on_infeasible == "raise":
        raise SelectionInfeasible(
            _blocking_reasons(remaining, invs, chosen, inv_cost, inv_harm, inv_irr,
                              forbid_irr, H_max, harm_used, budget, cost_used),
            chosen_leaves, remaining)
    return chosen_leaves


def _blocking_reasons(remaining, invs, chosen, inv_cost, inv_harm, inv_irr,
                      forbid_irr, H_max, harm_used, budget, cost_used) -> set[str]:
    """Why each still-uncovered leaf was unreachable when the greedy stopped."""
    reasons: set[str] = set()
    for leaf in remaining:
        options = [k for k, leaves in invs.items() if k not in chosen and leaf in leaves]
        if not options:
            reasons.add("no_cover")
        for k in options:
            if forbid_irr and inv_irr.get(k):
                reasons.add("irreversible")
            elif harm_used + inv_harm.get(k, 0) > H_max:
                reasons.add("harm_budget")
            elif budget is not None and cost_used + inv_cost.get(k, 0) > budget:
                reasons.add("cost_budget")
    return reasons
