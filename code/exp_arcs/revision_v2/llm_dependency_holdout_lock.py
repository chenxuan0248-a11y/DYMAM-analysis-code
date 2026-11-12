"""Validate and hash an authorized E-A holdout target registry before collection."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REGISTRY = HERE / "llm_dependency_holdout_targets_v1.json"
SELECTED = HERE / "llm_dependency_holdout_candidates_v1.json"
PROTOCOL = HERE / "llm_dependency_holdout_protocol.md"
GRID = REPO / "experiments/exp_arcs/e3_grid_v1.json"
OUT = HERE / "llm_dependency_holdout_lock_v1.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verified_local_file(path_value: object, expected_hash: object, label: str) -> bytes:
    if (not isinstance(path_value, str) or not path_value
            or not isinstance(expected_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)):
        raise ValueError(f"{label}: path and real SHA-256 are required")
    path = (REPO / path_value).resolve()
    if not path.is_relative_to(REPO) or not path.is_file():
        raise ValueError(f"{label}: saved file must exist inside the repository")
    raw = path.read_bytes()
    if digest(raw) != expected_hash:
        raise ValueError(f"{label}: saved file hash mismatch")
    return raw


def main() -> int:
    if not REGISTRY.exists():
        print("BLOCKED: authorized target registry has not been supplied")
        return 2
    raw = REGISTRY.read_bytes()
    doc = json.loads(raw)
    targets = doc.get("targets")
    selected = json.loads(SELECTED.read_text())
    selected_roots = {
        row["organization"]: row["base_domains"]
        for row in selected["targets"]
    }
    if not isinstance(targets, list) or len(targets) != 5:
        raise ValueError("holdout registry must contain exactly five target organizations")
    if doc.get("status") != "authorized_for_collection":
        raise ValueError("target registry is not authorized for collection")
    old = json.loads(GRID.read_text())
    old_orgs = set(old["orgs_target"])
    old_domains = set()
    for org in old_orgs:
        gt = REPO / f"data/gt/{org}_gt.json"
        old_domains.update(d.lower().rstrip(".") for d in json.loads(gt.read_text())["base_domains"])
    seen_orgs: set[str] = set()
    seen_domains: set[str] = set()
    for row in targets:
        org = row.get("organization")
        domains = row.get("base_domains")
        source = row.get("scope_source")
        window = row.get("collection_window_utc")
        if not isinstance(org, str) or not org or org.lower() != org or org in old_orgs or org in seen_orgs:
            raise ValueError(f"invalid/reused organization: {org!r}")
        if not isinstance(domains, list) or not (1 <= len(domains) <= 4):
            raise ValueError(f"{org}: expected 1-4 base domains")
        if not isinstance(source, str) or not source.strip() or not isinstance(window, str) or not window.strip():
            raise ValueError(f"{org}: scope_source and collection_window_utc are required")
        if domains != selected_roots.get(org):
            raise ValueError(f"{org}: must match the five preselected root domains")
        if source != next(x["scope_source"] for x in selected["targets"]
                          if x["organization"] == org):
            raise ValueError(f"{org}: scope source does not match selection record")
        verified_local_file(row.get("scope_snapshot_path"),
                            row.get("scope_snapshot_sha256"), f"{org} scope snapshot")
        hosts = row.get("approved_active_hosts")
        exclusions = row.get("excluded_hosts")
        if not isinstance(hosts, list) or not hosts or not isinstance(exclusions, list):
            raise ValueError(f"{org}: exact approved_active_hosts and excluded_hosts are required")
        for host in hosts + exclusions:
            if (not isinstance(host, str) or host.lower() != host or "*" in host
                    or not any(host == d or host.endswith("." + d) for d in domains)):
                raise ValueError(f"{org}: invalid or out-of-root exact host {host!r}")
        if set(hosts) & set(exclusions):
            raise ValueError(f"{org}: a host cannot be both approved and excluded")
        operations = row.get("permitted_operations")
        if (not isinstance(operations, list) or not operations
                or not set(operations) <= {"dns_lookup", "http_get"}):
            raise ValueError(f"{org}: permitted operations must be explicitly limited")
        rate = row.get("max_requests_per_second")
        if not isinstance(rate, (int, float)) or not 0 < rate <= 1:
            raise ValueError(f"{org}: conservative request rate must be in (0, 1]")
        identity = row.get("researcher_identity_reference")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(f"{org}: program identity or written approval reference required")
        strict_raw = verified_local_file(row.get("strict_policy_path"),
                                         row.get("strict_policy_sha256"),
                                         f"{org} strict policy")
        strict = json.loads(strict_raw)
        if strict.get("schema_version") != 1:
            raise ValueError(f"{org}: unsupported strict policy")
        allowed_by_tool = strict.get("allowed_targets_by_tool")
        if not isinstance(allowed_by_tool, dict) or not allowed_by_tool:
            raise ValueError(f"{org}: strict policy must authorize exact targets by tool")
        tool_operations = {"dns_resolve": "dns_lookup", "httpx": "http_get"}
        for tool, tool_hosts in allowed_by_tool.items():
            if tool not in tool_operations or tool_operations[tool] not in operations:
                raise ValueError(f"{org}: unapproved live tool {tool}")
            if not isinstance(tool_hosts, list) or not tool_hosts or not set(tool_hosts) <= set(hosts):
                raise ValueError(f"{org}: strict tool targets exceed approved active hosts")
        if not set(exclusions) <= set(strict.get("denied_hosts", [])):
            raise ValueError(f"{org}: registry exclusions missing from strict policy")
        strict_rate = strict.get("max_requests_per_second")
        if not isinstance(strict_rate, (int, float)) or not 0 < strict_rate <= rate:
            raise ValueError(f"{org}: strict policy exceeds approved rate")
        seen_orgs.add(org)
        for domain in domains:
            if not isinstance(domain, str) or not domain or "*" in domain or domain.lower() != domain:
                raise ValueError(f"{org}: invalid base domain {domain!r}")
            d = domain.rstrip(".")
            if d in old_domains or d in seen_domains:
                raise ValueError(f"{org}: reused base domain {d}")
            seen_domains.add(d)
    lock = {
        "status": "target_registry_locked_before_collection",
        "registry_path": str(REGISTRY),
        "registry_sha256": digest(raw),
        "protocol_sha256": digest(PROTOCOL.read_bytes()),
        "candidate_selection_sha256": digest(SELECTED.read_bytes()),
        "e3_grid_sha256": digest(GRID.read_bytes()),
        "n_new_organizations": len(targets),
        "organization_ids": sorted(seen_orgs),
    }
    OUT.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
