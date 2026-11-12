"""Standard-library-only faulted-input repair planner; JSON stdin -> JSON stdout.

This process intentionally receives neither E3 grid rows nor paired-clean paths.
It does not read files, import project modules, or execute any repair action.
"""
from __future__ import annotations

import json
import sys
import builtins
import io
import os
from collections import defaultdict

FAMILIES = ("dns", "certificate", "registration", "active_observation")
THRESHOLD = 0.5


def plan(data: dict) -> dict:
    assert tuple(data["contract"]["families"]) == FAMILIES
    assert data["contract"]["threshold"] == THRESHOLD
    cost = data["contract"]["cost_by_family"]
    action = data["contract"]["action_by_family"]
    assert set(cost) == set(action) == set(FAMILIES)

    host_fams: dict[str, set[str]] = defaultdict(set)
    fam_hosts: dict[str, set[str]] = defaultdict(set)
    for e in data["evidence"]:
        h, f = e["target"], e["source_family"]
        host_fams[h].add(f)
        fam_hosts[f].add(h)

    subjects = {c["claim_id"]: c.get("subject") for c in data["claims"]}
    self_flag: set[str] = set()
    for d in data["decisions"]:
        hit = d.get("decision") == "revalidate" or "no_supporting_observation" in (d.get("reason_codes") or ())
        if hit and subjects.get(d.get("claim_id")):
            self_flag.add(subjects[d["claim_id"]])
    manifest = set(data["manifest_hosts"])
    evidence_hosts = set(host_fams)
    candidate_hosts = evidence_hosts | manifest | {h for h in subjects.values() if h}
    n = len(evidence_hosts)
    base_rate = {f: len(hs) / n for f, hs in fam_hosts.items()} if n else {}
    scores = {h: max((rate for f, rate in base_rate.items() if f not in host_fams[h]), default=0.0)
              for h in evidence_hosts}
    for h in self_flag | (manifest - evidence_hosts):
        scores[h] = 1.0
    flagged = sorted(h for h, s in scores.items() if s >= THRESHOLD)

    def actions(hosts: list[str], families_by_host: dict[str, list[str]], reasons: dict[str, str]) -> dict:
        out = []
        for h in sorted(hosts):
            for f in families_by_host[h]:
                out.append({"host": h, "family": f, "action": action[f],
                            "downstream_scope": ("discovery_boundary_to_projection"
                                                 if h not in evidence_hosts else "family_evidence_to_projection"),
                            "reason": reasons[h], "spec_cost": cost[f]})
        return {"actions": out, "n_invocations": len(out),
                "tool_action_spec_cost": sum(x["spec_cost"] for x in out),
                "n_hosts": len(hosts)}

    selective_fams: dict[str, list[str]] = {}
    reasons = {}
    for h in flagged:
        missing_common = [f for f in FAMILIES if f not in host_fams.get(h, set())
                          and base_rate.get(f, 0.0) >= THRESHOLD]
        if h not in evidence_hosts:
            selective_fams[h] = list(FAMILIES)
            reasons[h] = "restart_host_boundary"
        elif h in self_flag:
            selective_fams[h] = list(FAMILIES)
            reasons[h] = "self_diagnosis_conservative"
        elif missing_common:
            selective_fams[h] = missing_common
            reasons[h] = "peer_missing_family"
        else:
            selective_fams[h] = list(FAMILIES)
            reasons[h] = "no_family_attribution_fallback"

    fixed_fams = {h: list(FAMILIES) for h in flagged}
    global_fams = {h: list(FAMILIES) for h in candidate_hosts}
    return {"case_id": data["case_id"], "candidate_hosts": sorted(candidate_hosts),
            "flagged_hosts": flagged, "host_scores": {h: scores[h] for h in sorted(scores)},
            "self_diagnosis_hosts": sorted(self_flag),
            "manifest_only_hosts": sorted(manifest - evidence_hosts),
            "policies": {
                "no_action": actions([], {}, {}),
                "selective": actions(flagged, selective_fams, reasons),
                "fixed_four_family": actions(flagged, fixed_fams,
                                             {h: "fixed_rule" for h in flagged}),
                "global_four_family": actions(sorted(candidate_hosts), global_fams,
                                              {h: "global_recollect" for h in candidate_hosts}),
            }}


def main() -> None:
    # All required information arrives over stdin. Fail closed on any later file open,
    # including pathlib's io.open path and low-level os.open.
    original_builtin_open, original_io_open, original_os_open = builtins.open, io.open, os.open
    attempts = 0

    def deny_file(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise AssertionError("planner attempted filesystem access")

    builtins.open = deny_file
    io.open = deny_file
    os.open = deny_file
    try:
        data = json.load(sys.stdin)
        result = plan(data)
    finally:
        builtins.open, io.open, os.open = original_builtin_open, original_io_open, original_os_open
    result["filesystem_open_attempts"] = attempts
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
