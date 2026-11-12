"""Plan and score four localizers through one isolated faulted-input planner.

Run `plan` and `evaluate` in separate invocations. Neither step calls external tools.
This is a retrospective same-corpus analysis; see COMMON_PLANNER_PROTOCOL_20261020.md.
"""
from __future__ import annotations

import hashlib
import json
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from experiments.exp_arcs.contamination import contamination_closure
from experiments.exp_arcs.dag import build_dag
from experiments.exp_arcs.detector_eval import detector_localize, manifest_dropped_hosts, peer_anomaly_hosts
from experiments.exp_arcs.localizer_baselines import _truth
from experiments.exp_arcs.rq2_combined_topk import tie_aware_hit
from experiments.exp_arcs.select import evaluate_scope
from experiments.exp_arcs.revision_v2 import step2_baselines as s2
from experiments.exp_arcs.revision_v2.repair_plan_no_clean_run import contract, payload

REPO = Path(__file__).resolve().parents[3]
D = REPO / "experiments/exp_arcs"
HERE = Path(__file__).resolve().parent
GRID = D / "e3_grid_v1.json"
STEP2 = HERE / "step2_baselines_v1.json"
SPEC = D / "cost_harm_spec.yaml"
PROTOCOL = HERE / "COMMON_PLANNER_PROTOCOL_20261020.md"
WORKER = HERE / "common_planner_worker.py"
METHODS = ("combined", "iforest", "robust_z", "contract_b")
POLICIES = ("selective", "fixed_four_family")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def frozen_hashes() -> dict[str, str]:
    return {k: digest(v) for k, v in {
        "grid": GRID, "step2": STEP2, "spec": SPEC, "protocol": PROTOCOL,
        "worker": WORKER, "runner": Path(__file__), "step2_code": Path(s2.__file__),
    }.items()}


def host_view(bundle: Path) -> tuple[set[str], dict[str, set[str]]]:
    fams: dict[str, set[str]] = defaultdict(set)
    for line in (bundle / "evidence_event.jsonl").read_text().splitlines():
        if line.strip():
            event = json.loads(line)
            fams[event["target"]].add(event.get("source_family", "other"))
    manifest = json.loads((bundle / "discovery_manifest.json").read_text())
    return set(fams) | set(manifest.get("resolved_hosts", [])), fams


def alert_and_score(bundle: Path, method: str, org: str, thresholds: dict) -> tuple[set[str], dict[str, float]]:
    candidates, fams = host_view(bundle)
    if method == "combined":
        from experiments.exp_arcs.detector_pr_roc import _host_scores
        scores = _host_scores(bundle)
        alerted = (detector_localize(bundle)["flagged_hosts"] |
                   peer_anomaly_hosts(bundle, min_base_rate=0.5) |
                   manifest_dropped_hosts(bundle))
    elif method in ("iforest", "robust_z"):
        scores = dict(s2.scores(str(bundle), method, True))
        tau = thresholds[f"{method}|manifest_aware"][org]
        alerted = {h for h, score in scores.items() if score >= tau}
    elif method == "contract_b":
        alerted = {h for h in candidates if not ("dns" in fams.get(h, ()) and
                                                 "active_observation" in fams.get(h, ()))}
        scores = {h: float(h in alerted) for h in candidates}
    else:
        raise ValueError(method)
    assert set(scores) >= candidates, (method, org, candidates - set(scores))
    return set(alerted) & candidates, {h: float(scores[h]) for h in candidates}


def plan(out: Path) -> None:
    grid = json.loads(GRID.read_text())["cells"]
    assert len(grid) == 180 and len({c["org"] for c in grid}) == 20
    saved_step2 = json.loads(STEP2.read_text())
    thresholds = {m: saved_step2["baselines"][m]["thresholds"]["primary"]
                  for m in ("iforest|manifest_aware", "robust_z|manifest_aware")}
    rows = []
    static = contract()
    for cell in grid:
        bundle = REPO / cell["faulted_dir"]
        base = payload(cell, static)
        candidates = {e["target"] for e in base["evidence"]} | set(base["manifest_hosts"])
        for method in METHODS:
            alerted, scores = alert_and_score(bundle, method, cell["org"], thresholds)
            assert alerted <= candidates and set(scores) == candidates
            task = {**base, "alert_hosts": sorted(alerted)}
            assert not ({"clean_dir", "inject_node", "truth", "C_star", "fault_id", "cell", "org"} & set(task))
            serialized = canonical(task)
            cp = subprocess.run([sys.executable, str(WORKER)], input=serialized,
                                text=True, capture_output=True, check=True, cwd="/private/tmp")
            result = json.loads(cp.stdout)
            assert result["filesystem_open_attempts"] == 0
            assert result["case_id"] == task["case_id"]
            assert set(result["policies"]) == set(POLICIES)
            assert set(result["alert_hosts"]) == alerted
            assert set(result["candidate_hosts"]) == candidates
            rows.append({"org": cell["org"], "cell": cell["cell"], "method": method,
                         "case_id": task["case_id"], "payload_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                         "scores": scores, "plan": result})
    clean_of = {}
    for cell in grid:
        clean_of.setdefault(cell["org"], cell["clean_dir"])
    clean_alerts = {}
    for method in METHODS:
        counts = []
        for org, rel in sorted(clean_of.items()):
            bundle = REPO / rel
            alerted, scores = alert_and_score(bundle, method, org, thresholds)
            counts.append({"org": org, "n_alerted": len(alerted), "n_candidates": len(scores)})
        clean_alerts[method] = counts
    result = {"phase": "common_planner_plan_only", "status": "posthoc_existing_corpus",
              "n_cells": 180, "n_orgs": 20, "hashes": frozen_hashes(),
              "candidate_permission": "faulted_evidence_union_same_run_manifest",
              "methods": list(METHODS), "policies": list(POLICIES),
              "clean_alerts": clean_alerts, "rows": rows}
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(canonical({"plans": len(rows), "sha256": digest(out)}))


def evaluate(plans: Path, out: Path) -> None:
    saved = json.loads(plans.read_text())
    assert saved["phase"] == "common_planner_plan_only" and saved["hashes"] == frozen_hashes()
    grid = json.loads(GRID.read_text())["cells"]
    assert len(saved["rows"]) == 180 * len(METHODS)
    lookup = {(r["org"], r["cell"], r["method"]): r for r in saved["rows"]}
    assert len(lookup) == len(saved["rows"])
    outcomes = []
    static = contract()
    saved_step2 = json.loads(STEP2.read_text())
    thresholds = {m: saved_step2["baselines"][m]["thresholds"]["primary"]
                  for m in ("iforest|manifest_aware", "robust_z|manifest_aware")}
    for cell in grid:
        clean = REPO / cell["clean_dir"]
        faulted = REPO / cell["faulted_dir"]
        dag = build_dag(clean)
        closure = contamination_closure(clean, faulted)
        repairable = {x.split(":", 1)[1] for x in closure["repairable_ids"]}
        needed = {(v["target"], v["source_family"])
                  for node in repairable if (v := dag.nodes.get(node)) and v.get("repairable")}
        truth, _ = _truth(cell["inject_node"])
        base = payload(cell, static)
        for method in METHODS:
            row = lookup[(cell["org"], cell["cell"], method)]
            alerted, scores = alert_and_score(faulted, method, cell["org"], thresholds)
            task = {**base, "alert_hosts": sorted(alerted)}
            assert row["payload_sha256"] == hashlib.sha256(canonical(task).encode()).hexdigest()
            assert row["scores"] == scores and set(row["plan"]["alert_hosts"]) == alerted
            item = {"org": cell["org"], "cell": cell["cell"], "method": method,
                    "n_candidates": len(scores), "n_alerted": len(alerted),
                    "truth_host_recall": len(truth & alerted) / len(truth),
                    "detect1": tie_aware_hit(scores, truth, 1), "policies": {}}
            for policy in POLICIES:
                planned = {(a["host"], a["family"]) for a in row["plan"]["policies"][policy]["actions"]}
                assert all(h in alerted and f in static["families"] for h, f in planned)
                assert len(planned) == len(row["plan"]["policies"][policy]["actions"])
                scope = {node for node, value in dag.nodes.items() if value.get("repairable")
                         and (value["target"], value["source_family"]) in planned}
                scored = evaluate_scope(clean, closure, scope, dag=dag)
                plan_cost = row["plan"]["policies"][policy]["tool_action_spec_cost"]
                item["policies"][policy] = {
                    "offline_counterfactual_success": bool(scored["success"]),
                    "tool_action_spec_cost": plan_cost, "n_actions": len(planned),
                    "n_families": len({f for _, f in planned}),
                    "n_unnecessary_actions": len(planned - needed),
                    "action_precision": len(planned & needed) / len(planned) if planned else 0.0,
                }
            outcomes.append(item)
    summary = {}
    for method in METHODS:
        by_method = [r for r in outcomes if r["method"] == method]
        clean = saved["clean_alerts"][method]
        entry = {"clean_alert_macro_rate": round(st.mean(r["n_alerted"] / r["n_candidates"] for r in clean), 4),
                 "detect1_org_mean": round(st.mean(st.mean(r["detect1"] for r in by_method if r["org"] == org)
                                                  for org in sorted({r["org"] for r in by_method})), 4),
                 "truth_host_alert_recall_org_mean": round(st.mean(r["truth_host_recall"] for r in by_method), 4),
                 "policies": {}}
        for policy in POLICIES:
            rows = [r["policies"][policy] for r in by_method]
            entry["policies"][policy] = {
                "recovery_cells": sum(r["offline_counterfactual_success"] for r in rows),
                "recovery_org_mean": round(st.mean(r["offline_counterfactual_success"] for r in rows), 4),
                "planned_cost_org_mean": round(st.mean(r["tool_action_spec_cost"] for r in rows), 4),
                "alerted_hosts_org_mean": round(st.mean(r["n_alerted"] for r in by_method), 4),
                "actions_org_mean": round(st.mean(r["n_actions"] for r in rows), 4),
                "families_org_mean": round(st.mean(r["n_families"] for r in rows), 4),
                "unnecessary_actions_org_mean": round(st.mean(r["n_unnecessary_actions"] for r in rows), 4),
                "action_precision_org_mean": round(st.mean(r["action_precision"] for r in rows), 4),
                "recovery_by_variant": {variant: sum(r["policies"][policy]["offline_counterfactual_success"]
                                                     for r in by_method if r["cell"] == variant)
                                        for variant in sorted({r["cell"] for r in by_method})},
            }
        summary[method] = entry
    result = {"phase": "post_plan_oracle_scoring", "status": "posthoc_existing_corpus",
              "plans_sha256": digest(plans), "input_hashes": frozen_hashes(),
              "n_cells": 180, "n_orgs": 20,
              "cost_kind": "planned_tool_action_spec_units_excludes_execution_and_refusion",
              "success_kind": "offline_counterfactual_clean_dag_evaluation_only",
              "summary": summary, "cells": outcomes}
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "plan":
        plan(Path(sys.argv[2]))
    elif len(sys.argv) == 4 and sys.argv[1] == "evaluate":
        evaluate(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit("usage: common_planner_compare.py plan OUT | evaluate PLANS OUT")
