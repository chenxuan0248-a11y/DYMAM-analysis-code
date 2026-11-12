"""Exploratory common-planner check on the existing 187 new injection positions.

Run `plan OUT.json` and `evaluate PLANS.json OUT.json` separately. The planner subprocess
sees only faulted-run records. Neither invocation makes external requests.
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
from experiments.exp_arcs.counterfactual import vf
from experiments.exp_arcs.dag import build_dag
from experiments.exp_arcs.localizer_baselines import _truth
from experiments.exp_arcs.rq2_combined_topk import tie_aware_hit
from experiments.exp_arcs.select import evaluate_scope
from experiments.exp_arcs.revision_v2 import common_planner_compare as cc

HERE = Path(__file__).resolve().parent
RUN = cc.D / "_audit/p1b_formal_run_20261009T082000Z.json"
SITES = HERE / "p1b_site_list_v1.json"
PROTOCOL = HERE / "NEW_SITE_COMMON_PLANNER_PROTOCOL_20261020.md"
METHODS = ("combined", "iforest", "robust_z")
STRATA = ("dns_only", "dns_multi", "ao")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hashes() -> dict[str, str]:
    paths = {"formal_run": RUN, "sites": SITES, "protocol": PROTOCOL,
             "runner": Path(__file__), "common_runner": Path(cc.__file__),
             "worker": cc.WORKER, "spec": cc.SPEC, "step2": cc.STEP2}
    return {key: digest(path) for key, path in paths.items()}


def inputs() -> tuple[list[dict], dict[str, dict], dict[str, str]]:
    run = json.loads(RUN.read_text())
    sites = {site["site_id"]: site for site in json.loads(SITES.read_text())["sites"]}
    cells = run["cells"]
    assert len(cells) == 187 and all(c["status"] == "valid" for c in cells)
    assert len({c["site_id"] for c in cells}) == len(cells)
    clean = {}
    for item in json.loads(cc.GRID.read_text())["cells"]:
        clean.setdefault(item["org"], str(cc.REPO / item["clean_dir"]))
    assert {sites[c["site_id"]]["org"] for c in cells} <= set(clean)
    return cells, sites, clean


def thresholds() -> dict:
    saved = json.loads(cc.STEP2.read_text())
    return {method: saved["baselines"][method]["thresholds"]["primary"]
            for method in ("iforest|manifest_aware", "robust_z|manifest_aware")}


def plan(out: Path) -> None:
    cells, sites, _ = inputs()
    tau, fixed = thresholds(), cc.contract()
    rows = []
    for cell in cells:
        site = sites[cell["site_id"]]
        org, faulted = site["org"], Path(cell["bundle_dir"])
        assert faulted.is_dir()
        bare = {"faulted_dir": str(faulted)}
        base = cc.payload(bare, fixed)
        candidates = {e["target"] for e in base["evidence"]} | set(base["manifest_hosts"])
        for method in METHODS:
            alerted, scores = cc.alert_and_score(faulted, method, org, tau)
            assert alerted <= candidates and set(scores) == candidates
            task = {**base, "alert_hosts": sorted(alerted)}
            assert not ({"clean_dir", "inject_node", "truth", "C_star", "org", "host", "stratum"} & set(task))
            serialized = canonical(task)
            child = subprocess.run([sys.executable, str(cc.WORKER)], input=serialized,
                                   text=True, capture_output=True, check=True, cwd="/private/tmp")
            result = json.loads(child.stdout)
            assert result["filesystem_open_attempts"] == 0
            assert set(result["policies"]) == set(cc.POLICIES)
            rows.append({"site_id": cell["site_id"], "method": method,
                         "payload_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                         "scores": scores, "plan": result})
    doc = {"phase": "new_site_plan_only", "status": "retrospective_same_snapshot",
           "n_sites": len(cells), "n_plans": len(rows), "hashes": hashes(), "rows": rows}
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(canonical({"n_sites": len(cells), "n_plans": len(rows), "sha256": digest(out)}))


def evaluate(plans: Path, out: Path) -> None:
    saved = json.loads(plans.read_text())
    assert saved["phase"] == "new_site_plan_only" and saved["hashes"] == hashes()
    cells, sites, clean_of = inputs()
    lookup = {(row["site_id"], row["method"]): row for row in saved["rows"]}
    assert len(lookup) == len(cells) * len(METHODS)
    tau, fixed = thresholds(), cc.contract()
    rows = []
    for cell in cells:
        site = sites[cell["site_id"]]
        org, faulted = site["org"], Path(cell["bundle_dir"])
        clean = Path(clean_of[org])
        dag = build_dag(clean)
        closure = contamination_closure(clean, faulted)
        consequential = vf(str(clean), str(faulted))["n_outputs_leaf_repairable"] > 0
        repairable = {node.split(":", 1)[1] for node in closure["repairable_ids"]}
        needed = {(value["target"], value["source_family"])
                  for node in repairable if (value := dag.nodes.get(node)) and value.get("repairable")}
        truth = {site["host"]}
        for method in METHODS:
            saved_row = lookup[(cell["site_id"], method)]
            alerted, scores = cc.alert_and_score(faulted, method, org, tau)
            base = cc.payload({"faulted_dir": str(faulted)}, fixed)
            task = {**base, "alert_hosts": sorted(alerted)}
            assert saved_row["payload_sha256"] == hashlib.sha256(canonical(task).encode()).hexdigest()
            assert saved_row["scores"] == scores
            item = {"site_id": cell["site_id"], "org": org, "stratum": site["stratum"],
                    "method": method, "has_consequence": consequential,
                    "n_candidates": len(scores), "n_alerted": len(alerted),
                    "detect1": tie_aware_hit(scores, truth, 1),
                    "truth_host_alert_recall": float(bool(truth & alerted)), "policies": {}}
            for policy in cc.POLICIES:
                actions = saved_row["plan"]["policies"][policy]["actions"]
                planned = {(a["host"], a["family"]) for a in actions}
                assert len(planned) == len(actions)
                assert all(host in alerted and fam in fixed["families"] for host, fam in planned)
                scope = {node for node, value in dag.nodes.items() if value.get("repairable")
                         and (value["target"], value["source_family"]) in planned}
                scored = evaluate_scope(clean, closure, scope, dag=dag)
                item["policies"][policy] = {
                    "offline_counterfactual_success": bool(scored["success"]),
                    "tool_action_spec_cost": saved_row["plan"]["policies"][policy]["tool_action_spec_cost"],
                    "n_actions": len(planned), "n_unnecessary_actions": len(planned - needed)}
            rows.append(item)
    summary = {}
    for stratum in STRATA:
        summary[stratum] = {}
        for method in METHODS:
            subset = [r for r in rows if r["stratum"] == stratum and r["method"] == method]
            used = [r for r in subset if r["has_consequence"]]
            orgs = sorted({r["org"] for r in used})
            assert used and orgs

            def org_mean(getter) -> float:
                return round(st.mean(st.mean(getter(r) for r in used if r["org"] == org)
                                     for org in orgs), 4)

            entry = {"n_sites": len(subset), "n_consequential": len(used), "n_orgs": len(orgs),
                     "detect1": org_mean(lambda r: r["detect1"]),
                     "truth_host_alert_recall": org_mean(lambda r: r["truth_host_alert_recall"]),
                     "alerted_hosts": org_mean(lambda r: r["n_alerted"]), "policies": {}}
            for policy in cc.POLICIES:
                entry["policies"][policy] = {
                    "recovery_cells": sum(r["policies"][policy]["offline_counterfactual_success"] for r in used),
                    "recovery_org_mean": org_mean(lambda r: r["policies"][policy]["offline_counterfactual_success"]),
                    "planned_cost_org_mean": org_mean(lambda r: r["policies"][policy]["tool_action_spec_cost"])}
            summary[stratum][method] = entry
    result = {"phase": "new_site_post_plan_scoring", "status": "retrospective_same_snapshot",
              "plans_sha256": digest(plans), "hashes": hashes(), "summary": summary, "rows": rows}
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "plan":
        plan(Path(sys.argv[2]))
    elif len(sys.argv) == 4 and sys.argv[1] == "evaluate":
        evaluate(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit("usage: new_site_common_planner.py plan OUT | evaluate PLANS OUT")
