"""Read-only, standard-library audit of four numbers used in the abstract.

Candidate visibility is recalculated directly from frozen faulted bundles and
the injection grid, without importing the study's candidate-generation code.
Plan costs are resummed from saved actions. An independent record-ID difference
identifies affected tool families and checks whether the saved plans cover them.
This tests action coverage separately from the saved counterfactual evaluator;
it does not independently regenerate the downstream repaired outputs.

Run from any directory:
  python3 -I experiments/exp_arcs/revision_v2/audit_abstract_chain.py

The output contains aggregate counts only. Input bundles and plans may contain
host identifiers and are not cleared for distribution.
"""
from __future__ import annotations

import hashlib
import json
import argparse
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ARCS = REPO / "experiments/exp_arcs"
REV = ARCS / "revision_v2"
POLICIES = ("selective", "fixed_four_family")
TOOL_FAMILIES = {"dns", "certificate", "registration", "active_observation"}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_pairs(run_dir: Path) -> set[tuple[str, str]]:
    pairs = set()
    with (run_dir / "evidence_event.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                pairs.add((item["target"], item.get("source_family", "other")))
    return pairs


def evidence_index(run_dir: Path) -> dict[str, dict]:
    records = {}
    with (run_dir / "evidence_event.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                records[item["evidence_event_id"]] = item
    return records


def normalized(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "run_id"}


def truth_hosts(inject_node: list) -> set[str]:
    assert inject_node[0] in ("target", "targets")
    return set(inject_node[1]) if inject_node[0] == "targets" else {inject_node[1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-output", type=Path,
                        help="write a local, restricted per-cell review record")
    args = parser.parse_args()
    grid_path = ARCS / "e3_grid_v1.json"
    grid = read(grid_path)["cells"]
    plans_path = REV / "repair_plan_no_clean_plans_v1.json"
    eval_path = REV / "repair_plan_no_clean_eval_v1.json"
    plans = read(plans_path)
    evaluations = read(eval_path)
    assert len(grid) == plans["n_cells"] == evaluations["n_cells"] == 180
    assert evaluations["plans_sha256"] == hashlib.sha256(plans_path.read_bytes()).hexdigest()
    plan_rows = {(row["org"], row["cell"]): row["plan"] for row in plans["rows"]}
    eval_rows = {(row["org"], row["cell"]): row for row in evaluations["cells"]}
    assert len(plan_rows) == len(eval_rows) == 180

    counts: Counter[str] = Counter()
    costs = {policy: 0 for policy in POLICIES}
    audit_rows: list[dict] = []
    for grid_index, cell in enumerate(grid):
        key = (cell["org"], cell["cell"])
        plan, evaluated = plan_rows[key], eval_rows[key]
        faulted = REPO / cell["faulted_dir"]
        clean = REPO / cell["clean_dir"]
        truth = truth_hosts(cell["inject_node"])
        surviving = evidence_pairs(faulted)
        manifest_path = faulted / "discovery_manifest.json"
        manifest = set(read(manifest_path).get("resolved_hosts", [])) if manifest_path.exists() else set()
        evidence_hosts = {host for host, _ in surviving}
        audit_row = {
            "grid_index": grid_index,
            "fault_variant": cell["cell"],
            "truth_host_pairs": len(truth),
            "visible_from_evidence": len(truth & evidence_hosts),
            "visible_from_evidence_or_manifest": len(truth & (evidence_hosts | manifest)),
            "clean_evidence_sha256": sha256(clean / "evidence_event.jsonl"),
            "faulted_evidence_sha256": sha256(faulted / "evidence_event.jsonl"),
            "faulted_manifest_sha256": sha256(manifest_path) if manifest_path.exists() else None,
        }
        counts["truth_pairs"] += len(truth)
        counts["visible_from_evidence"] += len(truth & evidence_hosts)
        counts["visible_from_evidence_or_manifest"] += len(truth & (evidence_hosts | manifest))

        clean_records, faulted_records = evidence_index(clean), evidence_index(faulted)
        needed_actions = {
            (record["target"], record["source_family"])
            for record_id, record in clean_records.items()
            if record["source_family"] in TOOL_FAMILIES
            and (record_id not in faulted_records
                 or normalized(record) != normalized(faulted_records[record_id]))
        }
        counts["needed_action_count_matches_evaluator"] += (
            len(needed_actions) == evaluated["n_needed_actions_evaluator_only"])
        audit_row["necessary_action_count"] = len(needed_actions)
        audit_row["saved_evaluator_necessary_action_count"] = evaluated[
            "n_needed_actions_evaluator_only"]
        audit_row["policies"] = {}
        for policy in POLICIES:
            actions = plan["policies"][policy]["actions"]
            action_pairs = {(action["host"], action["family"]) for action in actions}
            assert len(actions) == len(action_pairs)
            cost = sum(action["spec_cost"] for action in actions)
            assert cost == plan["policies"][policy]["tool_action_spec_cost"]
            assert cost == evaluated["policies"][policy]["tool_action_spec_cost"]
            costs[policy] += cost
            saved_success = bool(evaluated["policies"][policy]["offline_counterfactual_success"])
            coverage_success = needed_actions <= action_pairs
            counts[f"{policy}_saved_evaluator_success"] += saved_success
            counts[f"{policy}_independent_action_coverage"] += coverage_success
            counts[f"{policy}_coverage_evaluator_mismatch"] += coverage_success != saved_success
            audit_row["policies"][policy] = {
                "planned_action_count": len(actions),
                "planned_spec_cost": cost,
                "necessary_actions_covered": coverage_success,
                "saved_counterfactual_success": saved_success,
            }
        audit_rows.append(audit_row)

    assert counts["truth_pairs"] == 260
    assert (counts["visible_from_evidence"], counts["visible_from_evidence_or_manifest"]) == (129, 243)
    assert (counts["selective_saved_evaluator_success"],
            counts["fixed_four_family_saved_evaluator_success"]) == (119, 145)
    assert counts["needed_action_count_matches_evaluator"] == 180
    assert counts["selective_coverage_evaluator_mismatch"] == 0
    assert counts["fixed_four_family_coverage_evaluator_mismatch"] == 0
    assert {policy: round(total / 180, 2) for policy, total in costs.items()} == {
        "selective": 36.92, "fixed_four_family": 69.43}
    summary = {"counts": dict(counts),
               "mean_planned_spec_cost_from_actions": {
                   policy: round(total / 180, 2) for policy, total in costs.items()},
               "scope": "visibility and needed actions from frozen bundle records; "
                        "costs from saved actions; action coverage independently "
                        "agrees with saved counterfactual outcomes"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.audit_output:
        report = {
            "status": "local restricted review record; not cleared for release",
            "input_sha256": {
                "grid": sha256(grid_path),
                "saved_plans": sha256(plans_path),
                "saved_counterfactual_evaluation": sha256(eval_path),
            },
            "summary": summary,
            "rows": audit_rows,
            "limits": [
                "Input cache acquisition is not reconstructed.",
                "Detector scores and action plans are not regenerated.",
                "The downstream counterfactual evaluator is not independently rerun.",
                "Per-cell rows may be linkable despite omitted host and organization names.",
            ],
        }
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
        print(f"Wrote restricted per-cell review record: {args.audit_output}")


if __name__ == "__main__":
    main()
