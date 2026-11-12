"""File-isolated action planner for the retrospective common-detector comparison."""
from __future__ import annotations

import builtins
import io
import json
import os
import sys
from collections import defaultdict

FAMILIES = ("dns", "certificate", "registration", "active_observation")


def plan(data: dict) -> dict:
    contract = data["contract"]
    assert tuple(contract["families"]) == FAMILIES
    assert set(contract["cost_by_family"]) == set(FAMILIES)
    assert set(contract["action_by_family"]) == set(FAMILIES)
    fams: dict[str, set[str]] = defaultdict(set)
    fam_hosts: dict[str, set[str]] = defaultdict(set)
    for item in data["evidence"]:
        host, family = item["target"], item["source_family"]
        fams[host].add(family)
        fam_hosts[family].add(host)
    evidence_hosts = set(fams)
    candidate_hosts = evidence_hosts | set(data["manifest_hosts"])
    alerted = set(data["alert_hosts"])
    assert alerted <= candidate_hosts
    subjects = {c["claim_id"]: c.get("subject") for c in data["claims"]}
    self_flag = set()
    for decision in data["decisions"]:
        hit = decision.get("decision") == "revalidate" or (
            "no_supporting_observation" in (decision.get("reason_codes") or ()))
        if hit and subjects.get(decision.get("claim_id")):
            self_flag.add(subjects[decision["claim_id"]])
    rates = {f: len(hosts) / len(evidence_hosts) for f, hosts in fam_hosts.items()} if evidence_hosts else {}

    def emit(families_by_host: dict[str, list[str]], reason: dict[str, str]) -> dict:
        actions = []
        for host in sorted(alerted):
            for family in families_by_host[host]:
                actions.append({
                    "host": host, "family": family,
                    "action": contract["action_by_family"][family],
                    "spec_cost": contract["cost_by_family"][family],
                    "reason": reason[host],
                })
        return {"actions": actions, "n_hosts": len(alerted),
                "n_invocations": len(actions),
                "tool_action_spec_cost": sum(a["spec_cost"] for a in actions)}

    selective, reasons = {}, {}
    for host in sorted(alerted):
        missing_common = [f for f in FAMILIES if f not in fams.get(host, ()) and rates.get(f, 0) >= 0.5]
        if host not in evidence_hosts:
            selective[host], reasons[host] = list(FAMILIES), "restart_host_boundary"
        elif host in self_flag:
            selective[host], reasons[host] = list(FAMILIES), "self_diagnosis_conservative"
        elif missing_common:
            selective[host], reasons[host] = missing_common, "peer_missing_family"
        else:
            selective[host], reasons[host] = list(FAMILIES), "no_family_attribution_fallback"
    fixed = {host: list(FAMILIES) for host in alerted}
    return {"case_id": data["case_id"], "alert_hosts": sorted(alerted),
            "candidate_hosts": sorted(candidate_hosts),
            "policies": {"selective": emit(selective, reasons),
                         "fixed_four_family": emit(fixed, {h: "fixed_rule" for h in alerted})}}


def main() -> None:
    old_open, old_io_open, old_os_open = builtins.open, io.open, os.open
    attempts = 0

    def deny(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise AssertionError("planner attempted filesystem access")

    builtins.open = deny
    io.open = deny
    os.open = deny
    try:
        result = plan(json.load(sys.stdin))
    finally:
        builtins.open, io.open, os.open = old_open, old_io_open, old_os_open
    result["filesystem_open_attempts"] = attempts
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
