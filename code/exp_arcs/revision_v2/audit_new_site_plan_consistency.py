"""Audit saved new-site plans against saved counterfactual evaluations.

This read-only check uses the retained site labels, plans, and evaluation rows.
It does not regenerate candidates, scores, plans, or counterfactual outcomes.
The output contains counts only; no site or organization identifiers are printed.

Run from the repository root:
  python3 -I experiments/exp_arcs/revision_v2/audit_new_site_plan_consistency.py
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARCS = HERE.parent
SITES = HERE / "p1b_site_list_v1.json"
PLANS = ARCS / "_audit/new_site_common_planner_20261020/plans.json"
EVALUATION = ARCS / "_audit/new_site_common_planner_20261020/evaluation.json"
METHODS = ("combined", "iforest", "robust_z")
POLICIES = ("selective", "fixed_four_family")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    sites = {row["site_id"]: row for row in read(SITES)["sites"]}
    plans = read(PLANS)
    evaluation = read(EVALUATION)
    assert evaluation["plans_sha256"] == hashlib.sha256(PLANS.read_bytes()).hexdigest()
    assert plans["n_sites"] == 187 and plans["n_plans"] == 561

    plan_rows = {(row["method"], row["site_id"]): row["plan"]
                 for row in plans["rows"]}
    evaluation_rows = {(row["method"], row["site_id"]): row
                       for row in evaluation["rows"]}
    assert len(plan_rows) == len(evaluation_rows) == 561
    assert plan_rows.keys() == evaluation_rows.keys()
    assert {method for method, _ in plan_rows} == set(METHODS)
    assert {site_id for _, site_id in plan_rows} <= sites.keys()

    counts: Counter[str] = Counter()
    by_method: dict[str, Counter[str]] = {method: Counter() for method in METHODS}
    for (method, site_id), row in evaluation_rows.items():
        if not row["has_consequence"]:
            continue
        site = sites[site_id]
        plan = plan_rows[(method, site_id)]
        truth_host, missing_family = site["host"], site["family"]
        alerted = truth_host in plan["alert_hosts"]
        outcomes = {}
        for policy in POLICIES:
            actions = plan["policies"][policy]["actions"]
            action_pairs = {(action["host"], action["family"]) for action in actions}
            assert len(action_pairs) == len(actions)
            covers_missing = (truth_host, missing_family) in action_pairs
            recovered = bool(row["policies"][policy]["offline_counterfactual_success"])
            outcomes[policy] = recovered
            counts[f"{policy}_success"] += recovered
            counts[f"{policy}_truth_host_alert_mismatch"] += alerted != recovered
            counts[f"{policy}_missing_family_action_mismatch"] += covers_missing != recovered
        counts["consequential_method_site_rows"] += 1
        counts["policy_outcome_disagreements"] += outcomes[POLICIES[0]] != outcomes[POLICIES[1]]
        by_method[method]["consequential_rows"] += 1
        by_method[method]["recovered_under_each_policy"] += outcomes[POLICIES[0]]

    assert counts == Counter({
        "consequential_method_site_rows": 558,
        "selective_success": 420,
        "fixed_four_family_success": 420,
    })
    assert {method: dict(values) for method, values in by_method.items()} == {
        "combined": {"consequential_rows": 186, "recovered_under_each_policy": 142},
        "iforest": {"consequential_rows": 186, "recovered_under_each_policy": 144},
        "robust_z": {"consequential_rows": 186, "recovered_under_each_policy": 134},
    }
    print(json.dumps({"counts": dict(counts),
                      "by_method": {method: dict(values) for method, values in by_method.items()}},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
