"""experiments/exp_arcs/substitution — Step 4.2 alternative-evidence substitution.

H2: when a contaminated claim is *also* supported by an INDEPENDENT source family
whose evidence is uncontaminated, the claim can be restored by substituting that
family instead of re-collecting the contaminated one. This yields ALTERNATIVE
covers for an output, which is the ILP's only real freedom (`solve_oracle`'s
`alt_covers`) and ARCS's only route to beat single-point on cost.

D1 (FROZEN, `D1_FROZEN.md`) verdict = **H2 Opportunistic**: on the substantive
claim types ARCS repairs (service_observation / technology_identification /
vulnerability_applicability) the independent-second-family redundancy is 0.000.
So this hook is expected to find essentially NO substitutable outputs on this
corpus — it is implemented in full (real independent-family logic, not a stub) so
that the −substitution ablation (E7-B) is meaningful and the mechanism is present;
it simply, and correctly, returns almost nothing here.

Independent families (frozen, D1): {dns, certificate, active_observation,
registration}. Non-independent / derived families cannot substitute.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.exp_arcs.contamination import _REPAIRABLE, contamination_closure
from experiments.exp_arcs.counterfactual import _ancestors, _bare, _OUTPUT_TYPE, coverage_map
from experiments.exp_arcs.dag import build_dag

INDEPENDENT_FAMILIES = frozenset(
    {"dns", "certificate", "active_observation", "registration"})

# D1 (FROZEN): substitution counts ONLY for the substantive claim types ARCS
# repairs. Redundancy in asset_identity / organization_association (asset
# existence / org attribution) does NOT make a service/tech/CVE finding
# substitutable — counting it would overstate H2 (it is D1's context-only pool).
SUBSTANTIVE_CLAIM_TYPES = frozenset(
    {"service_observation", "technology_identification", "vulnerability_applicability"})


def _output_claim_types(clean_dir: Path) -> dict[str, str]:
    types: dict[str, str] = {}
    path = clean_dir / "report_item.jsonl"
    if path.exists():
        for ln in path.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                types[r["report_item_id"]] = r.get("claim_type")
    return types


def alt_covers(clean_dir: str | Path,
               dfx: dict[str, Any] | None = None,
               faulted_dir: str | Path | None = None) -> dict[str, Any]:
    """Per leaf-repairable output, alternative covers by independent family.

    Returns ``{"alt_covers": {output -> [cover_leafset, ...]}, "n_substitutable":
    int, ...}``. An output is substitutable iff (a) its claim type is substantive
    (D1), (b) its required repairable-leaf ancestors span ≥2 independent families,
    and (c) at least one such family is uncontaminated (survives to cover it alone).
    """
    if dfx is None:
        assert faulted_dir is not None
        dfx = contamination_closure(Path(clean_dir), Path(faulted_dir))
    dag = build_dag(Path(clean_dir))
    cm = coverage_map(Path(clean_dir), dfx)
    cont_leaves = {l for l in _bare(dfx, _REPAIRABLE) if l in dag.nodes}
    claim_types = _output_claim_types(Path(clean_dir))

    alt: dict[str, list[list[str]]] = {}
    n_sub = 0
    n_substantive = 0
    for o, req in cm["coverage_map"].items():
        if claim_types.get(o) not in SUBSTANTIVE_CLAIM_TYPES:
            continue                      # D1: non-substantive redundancy excluded
        n_substantive += 1
        # all repairable-leaf ancestors of o in the clean DAG, by family
        anc = _ancestors(dag, {o})
        fam_leaves: dict[str, list[str]] = defaultdict(list)
        for l in anc:
            v = dag.nodes.get(l)
            if v and v.get("repairable") and v["source_family"] in INDEPENDENT_FAMILIES:
                fam_leaves[v["source_family"]].append(l)
        indep = set(fam_leaves)
        if len(indep) < 2:
            continue
        # a family can cover o on its own iff it is uncontaminated (survives)
        clean_families = [f for f, ls in fam_leaves.items()
                          if not (set(ls) & cont_leaves)]
        if clean_families:
            n_sub += 1
            alt[o] = [sorted(fam_leaves[f]) for f in sorted(clean_families)]
    return {
        "alt_covers": alt,
        "n_substitutable": n_sub,
        "n_substantive_outputs": n_substantive,
        "n_leaf_repairable_outputs": len(cm["coverage_map"]),
        "independent_families": sorted(INDEPENDENT_FAMILIES),
    }
