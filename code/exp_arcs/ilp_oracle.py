"""experiments/exp_arcs/ilp_oracle — Step 2.2 constrained ILP repair oracle.

Given a clean/faulted bundle pair, solve for the minimum (cost + harm) repair
scope S* that restores every leaf-repairable contaminated output, under the
frozen ``ilp_config.yaml`` constraints (H_max cap, no irreversible invocation;
E_min = 1.0 ⇒ full success).

Decision unit = a ``(target, source_family)`` tool INVOCATION (``dag.scope_cost``
grain): one invocation re-collects all of a host's evidence for that family, so
an output is restored iff every invocation feeding its required leaves is chosen.
With no substitution (H2 Opportunistic) every required invocation is forced, so
S* is the minimal cover; the GATE cross-checks C* against ``counterfactual.py``.
Substitution (Phase 4.2) plugs in as ALTERNATIVE covering invocations per output
(``alt_covers`` below), which is where the ILP gains real freedom.

    poetry run python -m experiments.exp_arcs.ilp_oracle \
        --clean <clean_dir> --faulted <faulted_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pulp
import yaml

from experiments.exp_arcs.counterfactual import coverage_map
from experiments.exp_arcs.contamination import contamination_closure
from experiments.exp_arcs.dag import build_dag, scope_cost

_CONFIG = Path(__file__).with_name("ilp_config.yaml")


def load_config(path: Path = _CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _invocation(dag_nodes: dict[str, Any], leaf: str) -> tuple[Any, str]:
    v = dag_nodes[leaf]
    return (v.get("target"), v["source_family"])


def solve_oracle(clean_dir: str | Path, faulted_dir: str | Path,
                 cfg: dict[str, Any] | None = None,
                 alt_covers: dict[str, list[list[str]]] | None = None) -> dict[str, Any]:
    """Solve S*, C* for one fault. ``alt_covers`` (Phase 4.2) maps output ->
    list of alternative covering leaf-sets; default = the one actual cover."""
    cfg = cfg or load_config()
    alpha = float(cfg["objective"]["alpha"])
    beta = float(cfg["objective"]["beta"])
    H_max = float(cfg["constraints"]["H_max"])
    forbid_irr = bool(cfg["constraints"]["forbid_irreversible"])

    dag = build_dag(Path(clean_dir))
    dfx = contamination_closure(Path(clean_dir), Path(faulted_dir))
    cm = coverage_map(Path(clean_dir), dfx)

    # required covering invocations per leaf-repairable output (one cover each,
    # unless alt_covers supplies alternatives).
    inv_cost: dict[tuple[Any, str], float] = {}
    inv_harm: dict[tuple[Any, str], float] = {}
    inv_irr: dict[tuple[Any, str], bool] = {}
    output_covers: dict[str, list[set[tuple[Any, str]]]] = {}

    def leaves_to_invs(leaves: list[str]) -> set[tuple[Any, str]]:
        invs = set()
        for l in leaves:
            if l not in dag.nodes:
                continue
            i = _invocation(dag.nodes, l)
            invs.add(i)
            inv_cost[i] = dag.nodes[l]["cost"]
            inv_harm[i] = dag.nodes[l]["harm"]
            inv_irr[i] = dag.nodes[l]["reversibility_rank"] == 3
        return invs

    for o, req_leaves in cm["coverage_map"].items():
        covers = (alt_covers or {}).get(o) or [req_leaves]
        output_covers[o] = [leaves_to_invs(c) for c in covers]

    candidates = sorted({i for covers in output_covers.values()
                         for cov in covers for i in cov}, key=str)

    prob = pulp.LpProblem("arcs_repair", pulp.LpMinimize)
    y = {i: pulp.LpVariable(f"y_{k}", cat="Binary") for k, i in enumerate(candidates)}

    prob += pulp.lpSum((alpha * inv_cost[i] + beta * inv_harm[i]) * y[i]
                       for i in candidates)

    # coverage: each output needs at least one of its covers fully selected.
    # single-cover case (pilot) → every invocation in that cover forced to 1.
    for o, covers in output_covers.items():
        if len(covers) == 1:
            for i in covers[0]:
                prob += y[i] >= 1, f"cover_{o}_{i}"
        else:  # multi-cover (substitution): pick ≥1 whole cover via cover-select vars
            zs = []
            for ci, cov in enumerate(covers):
                z = pulp.LpVariable(f"z_{o}_{ci}", cat="Binary")
                zs.append(z)
                for i in cov:
                    prob += y[i] >= z, f"covsel_{o}_{ci}_{i}"
            prob += pulp.lpSum(zs) >= 1, f"anycover_{o}"

    if forbid_irr:
        for i in candidates:
            if inv_irr[i]:
                prob += y[i] == 0, f"noirr_{i}"

    prob += pulp.lpSum(inv_harm[i] * y[i] for i in candidates) <= H_max, "H_max"

    status = pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=False))]
    selected = [i for i in candidates if y[i].value() and y[i].value() > 0.5]
    C_star = sum(inv_cost[i] for i in selected)
    H_star = sum(inv_harm[i] for i in selected)
    return {
        "status": status,
        "n_candidate_invocations": len(candidates),
        "n_selected_invocations": len(selected),
        "C_star": C_star,
        "H_star": H_star,
        "objective": pulp.value(prob.objective) if status == "Optimal" else None,
        "n_outputs_covered": len(cm["coverage_map"]),
        "n_outputs_unrepairable_by_leaf": len(cm["unrepairable_outputs"]),
        "_selected": selected,
        "_coverage_map": cm["coverage_map"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--faulted", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    r = solve_oracle(args.clean, args.faulted)
    print(f"status={r['status']}  S* invocations={r['n_selected_invocations']}  "
          f"C*={r['C_star']}  H*={r['H_star']}  objective={r['objective']}")
    print(f"outputs covered={r['n_outputs_covered']} "
          f"unrepairable-by-leaf={r['n_outputs_unrepairable_by_leaf']}")

    if args.out:
        out = {k: v for k, v in r.items() if not k.startswith("_")}
        Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
