"""Generate no-clean-DAG plans, then evaluate them in a separate invocation.

Usage:
  .venv/bin/python -m experiments.exp_arcs.revision_v2.repair_plan_no_clean_run plan OUT.json
  .venv/bin/python -m experiments.exp_arcs.revision_v2.repair_plan_no_clean_run evaluate PLANS.json OUT.json
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
D = REPO / "experiments/exp_arcs"
GRID = D / "e3_grid_v1.json"
SPEC = D / "cost_harm_spec.yaml"
PROTOCOL = HERE / "REPAIR_PLAN_NO_CLEAN_PROTOCOL_20261013.md"
WORKER = HERE / "repair_plan_no_clean_worker.py"
FAMILIES = ("dns", "certificate", "registration", "active_observation")
FORBIDDEN_INPUT_KEYS = {"clean_dir", "inject_node", "truth", "C_star", "fault_id", "cell", "org"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def frozen_inputs() -> dict[str, str]:
    return {"grid_sha256": digest(GRID), "spec_sha256": digest(SPEC),
            "protocol_sha256": digest(PROTOCOL), "worker_sha256": digest(WORKER)}


def contract() -> dict:
    import yaml
    spec = yaml.safe_load(SPEC.read_text())
    mapping = spec["family_to_action"]
    return {"families": list(FAMILIES), "threshold": 0.5,
            "action_by_family": {f: mapping[f] for f in FAMILIES},
            "cost_by_family": {f: spec["actions"][mapping[f]]["cost"] for f in FAMILIES}}


def payload(cell: dict, fixed_contract: dict) -> dict:
    # Cell metadata is used only to locate the faulted bundle. None is sent to the planner.
    b = REPO / cell["faulted_dir"]
    manifest_path = b / "discovery_manifest.json"
    manifest = json.loads(manifest_path.read_text()).get("resolved_hosts", []) if manifest_path.exists() else []
    evidence = [{"target": e["target"], "source_family": e.get("source_family", "other")}
                for e in lines(b / "evidence_event.jsonl")]
    claims = [{"claim_id": c["claim_id"], "subject": c.get("subject")}
              for c in lines(b / "claim.jsonl")]
    decisions = [{"claim_id": d.get("claim_id"), "decision": d.get("decision"),
                  "reason_codes": d.get("reason_codes") or []}
                 for d in lines(b / "decision.jsonl")]
    case_id = hashlib.sha256(cell["faulted_dir"].encode()).hexdigest()[:24]
    result = {"case_id": case_id, "contract": fixed_contract, "evidence": evidence,
              "claims": claims, "decisions": decisions, "manifest_hosts": manifest}
    assert not (set(result) & FORBIDDEN_INPUT_KEYS)
    assert not any(k in canonical(result) for k in ('"clean_dir"', '"inject_node"', '"C_star"'))
    return result


def plan(out: Path) -> None:
    cells = json.loads(GRID.read_text())["cells"]
    assert len(cells) == 180 and len({c["org"] for c in cells}) == 20
    fixed_contract = contract()
    rows = []
    for c in cells:
        p = payload(c, fixed_contract)
        data = canonical(p)
        # The child sees one sanitized JSON value, not the grid or the clean bundle.
        cp = subprocess.run([sys.executable, str(WORKER)], input=data, text=True,
                            capture_output=True, check=True, cwd="/private/tmp")
        policy = json.loads(cp.stdout)
        assert policy["case_id"] == p["case_id"]
        assert policy["filesystem_open_attempts"] == 0
        assert set(policy["policies"]) == {"no_action", "selective", "fixed_four_family", "global_four_family"}
        candidates = set(policy["candidate_hosts"])
        for item in policy["policies"].values():
            assert len(item["actions"]) == item["n_invocations"]
            assert all(a["host"] in candidates and a["family"] in FAMILIES for a in item["actions"])
        rows.append({"org": c["org"], "cell": c["cell"], "case_id": p["case_id"],
                     "planner_input_sha256": hashlib.sha256(data.encode()).hexdigest(),
                     "plan": policy})
    result = {"phase": "plan_only_no_clean_dag", "study_status": "retrospective_existing_corpus",
              "input_hashes": frozen_inputs(), "n_cells": len(rows), "n_orgs": 20,
              "planner_io": "sanitized_faulted_records_plus_same_run_manifest_and_static_contract_only",
              "cost_kind": "tool_action_spec_units_excludes_downstream_execution",
              "rows": rows}
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(canonical({"plans_written": len(rows), "output_sha256": digest(out)}))


def evaluate(plans_path: Path, out: Path) -> None:
    # Only this function imports the clean-dependent evaluator.
    from experiments.exp_arcs.contamination import contamination_closure
    from experiments.exp_arcs.dag import build_dag
    from experiments.exp_arcs.detector_pr_roc import _host_scores
    from experiments.exp_arcs.select import evaluate_scope

    plan_doc = json.loads(plans_path.read_text())
    assert plan_doc["phase"] == "plan_only_no_clean_dag"
    assert plan_doc["input_hashes"] == frozen_inputs(), "Frozen inputs changed after planning"
    cells = json.loads(GRID.read_text())["cells"]
    lookup = {(r["org"], r["cell"]): r for r in plan_doc["rows"]}
    assert len(cells) == len(lookup) == 180
    output = []
    for c in cells:
        r = lookup[(c["org"], c["cell"])]
        assert r["case_id"] == hashlib.sha256(c["faulted_dir"].encode()).hexdigest()[:24]
        assert r["planner_input_sha256"] == hashlib.sha256(canonical(payload(c, contract())).encode()).hexdigest()
        assert r["plan"]["filesystem_open_attempts"] == 0
        clean, faulted = REPO / c["clean_dir"], REPO / c["faulted_dir"]
        assert r["plan"]["host_scores"] == _host_scores(faulted)
        dag = build_dag(clean)
        dfx = contamination_closure(clean, faulted)
        needed_leaves = {x.split(":", 1)[1] for x in dfx["repairable_ids"]}
        needed_actions = {(v["target"], v["source_family"])
                          for n in needed_leaves if (v := dag.nodes.get(n)) and v.get("repairable")}
        policy_rows = {}
        for name, item in r["plan"]["policies"].items():
            planned = {(a["host"], a["family"]) for a in item["actions"]}
            assert len(planned) == item["n_invocations"]
            scope = {n for n, v in dag.nodes.items() if v.get("repairable")
                     and (v["target"], v["source_family"]) in planned}
            scored = evaluate_scope(clean, dfx, scope, dag=dag)
            policy_rows[name] = {
                "plan_emitted": bool(planned), "n_planned_actions": len(planned),
                "tool_action_spec_cost": item["tool_action_spec_cost"],
                "mapped_clean_leaf_count": len(scope),
                "needed_action_coverage": len(planned & needed_actions) / len(needed_actions) if needed_actions else 1.0,
                "n_unnecessary_actions": len(planned - needed_actions),
                "n_missing_needed_actions": len(needed_actions - planned),
                "offline_counterfactual_success": bool(scored["success"]),
                "n_outputs_broken": scored["n_outputs_broken"],
                "cost_over_cstar": item["tool_action_spec_cost"] / c["C_star"] if c["C_star"] else None,
            }
        output.append({"org": c["org"], "cell": c["cell"], "case_id": r["case_id"],
                       "n_flagged_hosts": len(r["plan"]["flagged_hosts"]),
                       "n_needed_actions_evaluator_only": len(needed_actions),
                       "policies": policy_rows})

    summaries = {}
    policies = tuple(output[0]["policies"])
    for name in policies:
        by_org = defaultdict(list)
        for row in output:
            by_org[row["org"]].append(row["policies"][name])
        summaries[name] = {
            "plan_emitted_cells": sum(x["policies"][name]["plan_emitted"] for x in output),
            "offline_success_cells": sum(x["policies"][name]["offline_counterfactual_success"] for x in output),
            "offline_success_org_mean": round(st.mean(st.mean(p["offline_counterfactual_success"] for p in v)
                                                       for v in by_org.values()), 4),
            "tool_action_spec_cost_org_mean": round(st.mean(st.mean(p["tool_action_spec_cost"] for p in v)
                                                         for v in by_org.values()), 4),
            "cost_over_cstar_org_mean": round(st.mean(st.mean(p["cost_over_cstar"] for p in v)
                                                 for v in by_org.values()), 4),
            "unnecessary_actions_org_mean": round(st.mean(st.mean(p["n_unnecessary_actions"] for p in v)
                                                      for v in by_org.values()), 4),
        }
    paired = {}
    for reference in ("fixed_four_family", "global_four_family"):
        paired["selective_minus_" + reference] = {}
        for field in ("offline_counterfactual_success", "tool_action_spec_cost", "n_unnecessary_actions"):
            by_org = defaultdict(list)
            for row in output:
                by_org[row["org"]].append(row["policies"]["selective"][field]
                                          - row["policies"][reference][field])
            differences = [st.mean(by_org[o]) for o in sorted(by_org)]
            rng = random.Random(42)
            draws = sorted(st.mean(rng.choice(differences) for _ in differences) for _ in range(10_000))
            paired["selective_minus_" + reference][field] = {
                "org_mean_difference": round(st.mean(differences), 4),
                "bootstrap_ci95": [round(draws[249], 4), round(draws[9749], 4)],
                "n_orgs": len(differences)}
    result = {"phase": "post_plan_oracle_evaluation", "plans_sha256": digest(plans_path),
              "input_hashes": frozen_inputs(), "n_cells": len(output), "n_orgs": 20,
              "canonical_scores_matched": len(output), "planner_zero_file_open_cells": len(output),
              "evaluator_script_sha256": digest(Path(__file__)),
              "cost_kind": "tool_action_spec_units_excludes_downstream_execution",
              "success_kind": "offline_counterfactual_clean_dag_evaluator_only",
              "summary": summaries, "paired_differences": paired, "cells": output}
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "plan":
        plan(Path(sys.argv[2]))
    elif len(sys.argv) == 4 and sys.argv[1] == "evaluate":
        evaluate(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit("usage: repair_plan_no_clean_run.py plan OUT.json | evaluate PLANS.json OUT.json")
