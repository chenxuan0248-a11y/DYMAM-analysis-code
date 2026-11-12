"""Build a *local, unapproved* cell-level candidate from retained E3 results.

No raw responses, host names, organization names, case IDs, or score vectors are
written. The output still needs a rights and re-identification review before sharing.
"""
from __future__ import annotations

import json
import secrets
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release_candidate_emse_derived_v1" / "derived_cells.json"


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def main() -> None:
    coverage = read("revision_v2/candidate_coverage_v1.json")
    access = read("revision_v2/information_access_v1.json")
    plans = read("revision_v2/repair_plan_no_clean_eval_v1.json")
    common = read("revision_v2/common_planner_eval_20261020.json")
    base = {(r["org"], r["cell"]): r for r in access["cells"]["records"]}
    assert len(base) == 180
    missing = defaultdict(lambda: [0, 0])
    for item in coverage["uncovered_pairs"]:
        key = item["org"], item["cell"]
        assert key in base and item["covered_E"] is False
        missing[key][0] += 1
        missing[key][1] += not item["covered_ER"]
    by_plan = {(r["org"], r["cell"]): r for r in plans["cells"]}
    by_common = defaultdict(dict)
    for r in common["cells"]:
        key = r["org"], r["cell"]
        assert r["method"] not in by_common[key]
        by_common[key][r["method"]] = r
    assert set(base) == set(by_plan) == set(by_common)

    # The balanced design has nine cells per organization, so the mean of all
    # 180 rows equals the mean of 20 organization means. Do not export groups.
    org_counts = defaultdict(int)
    for org, _ in base:
        org_counts[org] += 1
    assert len(org_counts) == 20 and set(org_counts.values()) == {9}
    keys = list(base)
    secrets.SystemRandom().shuffle(keys)
    rows = []
    for key in keys:
        truth = base[key]["n_truth"]
        e_missing, er_missing = missing[key]
        assert 0 <= er_missing <= e_missing <= truth
        a = {mode: access["cells"][mode] for mode in
             ("records", "manifest", "paired_clean_oracle")}
        access_row = {}
        for mode, values in a.items():
            hit = next(r for r in values if (r["org"], r["cell"]) == key)
            assert hit["n_truth"] == truth
            access_row[mode] = hit["truth_covered"]
        plan_row = by_plan[key]
        method_rows = by_common[key]
        assert set(method_rows) == {"combined", "iforest", "robust_z", "contract_b"}
        rows.append({
            "truth_pairs": truth,
            "visible_evidence": truth - e_missing,
            "visible_evidence_manifest": truth - er_missing,
            "access_covered": access_row,
            "no_clean_plan": {
                policy: {
                    "recovered": bool(plan_row["policies"][policy]["offline_counterfactual_success"]),
                    "spec_cost": plan_row["policies"][policy]["tool_action_spec_cost"],
                } for policy in ("no_action", "selective", "fixed_four_family", "global_four_family")
            },
            "same_planner": {
                method: {
                    "detect_at_1": method_rows[method]["detect1"],
                    "truth_host_alert_recall": method_rows[method]["truth_host_recall"],
                    "alerted_hosts": method_rows[method]["n_alerted"],
                    "policies": {
                        policy: {
                            "recovered": bool(method_rows[method]["policies"][policy]["offline_counterfactual_success"]),
                            "spec_cost": method_rows[method]["policies"][policy]["tool_action_spec_cost"],
                        } for policy in ("selective", "fixed_four_family")
                    },
                } for method in ("combined", "iforest", "robust_z", "contract_b")
            },
        })
    assert sum(r["truth_pairs"] for r in rows) == 260
    assert sum(r["visible_evidence"] for r in rows) == 129
    assert sum(r["visible_evidence_manifest"] for r in rows) == 243
    assert sum(r["no_clean_plan"]["selective"]["recovered"] for r in rows) == 119
    assert sum(r["no_clean_plan"]["fixed_four_family"]["recovered"] for r in rows) == 145
    assert sum(r["same_planner"]["iforest"]["policies"]["fixed_four_family"]["recovered"] for r in rows) == 168
    doc = {
        "schema_version": "0.1",
        "status": "local_candidate_not_cleared_for_distribution",
        "derivation": "retained per-cell result JSON; no original response or scoring replay",
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} local candidate rows to {OUT}")


if __name__ == "__main__":
    main()
