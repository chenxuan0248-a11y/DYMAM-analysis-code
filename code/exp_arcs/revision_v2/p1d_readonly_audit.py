"""P1-D: read-only replay-degradation and clean-alert feasibility audit.

Reads only saved JSON/JSONL files. Never imports or calls a tool, LLM, database,
or pipeline runner. See P1D_READONLY_PROTOCOL.md for scope and grading.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import ipaddress
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from experiments.exp_arcs.detector_eval import (
    detector_localize,
    manifest_dropped_hosts,
    peer_anomaly_hosts,
)

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LOCAL = {"wappalyzer_match", "geolocation", "cve_rag", "kev_epss"}
MISS_RE = re.compile(r"^(\w+): no cached entry for (\{.*\}) — ")
# These are the source-family labels produced by the audited tool sources and
# core.state._source_family. No family is inferred from the tool name alone.
MISS_FAMILY = {
    "httpx": "active_observation",
    "favicon_hash": "other",
    "greynoise": "other",
    "nmap": "active_observation",
    "asn_lookup": "other",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ident(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def read_jsonl(path: Path) -> list[dict]:
    # Literal newline is required: JSON strings may contain U+0085/U+2028.
    return [json.loads(s) for s in path.read_text().split("\n") if s.strip()]


def source_runs() -> list[dict]:
    smoke = json.loads((HERE / "smoke_restore_result_v1.json").read_text())
    ea = json.loads((HERE / "e_a_offline_recheck_v1.json").read_text())
    out = []
    for row in smoke["runs"]:
        if row["pass"] != "A" or row["cell"] != "clean":
            continue
        assert row["exit"] == 0 and row["restore_ok"] and row["vs_frozen"]
        rd = Path(row["dir"])
        bundles = list((rd / "bundles").glob("run_*/run_manifest.json"))
        assert len(bundles) == 1
        out.append({"source": "smoke_A", "org": row["org"], "dir": rd,
                    "bundle": bundles[0].parent, "stage_trace": False})
    for org in ea["organizations"]:
        row = ea["runs"][f"full_cached|{org}|clean"]
        assert row["status"] == "ok" and row["neo4j_projection_succeeded"]
        assert row["result"]["matches_frozen_bundle"]
        rd = Path(row["out_dir"])
        out.append({"source": "ea_full_cached", "org": org, "dir": rd,
                    "bundle": Path(row["result"]["bundle_dir"]), "stage_trace": True})
    assert len(out) == 7 and len({r["org"] for r in out}) == 7
    return sorted(out, key=lambda r: r["org"])


def normalize_host(value: str) -> str:
    v = value.strip().lower()
    if not v:
        return v
    parts = urlsplit(v if "://" in v else "//" + v)
    return (parts.hostname or v.split("/")[0]).rstrip(".")


def clean_flags(bundle: Path) -> tuple[set[str], dict[str, set[str]], set[str]]:
    val = detector_localize(bundle)["flagged_hosts"]
    peer = peer_anomaly_hosts(bundle)
    manifest = manifest_dropped_hosts(bundle)
    all_flags = val | peer | manifest
    components = {"validator": val, "peer": peer, "manifest": manifest}
    hosts = {e["target"] for e in read_jsonl(bundle / "evidence_event.jsonl") if e.get("target")}
    return all_flags, components, hosts


def source_families(evidence: list[dict]) -> dict[str, set[str]]:
    by_host: dict[str, set[str]] = defaultdict(set)
    for e in evidence:
        if e.get("target") and e.get("source_family"):
            by_host[e["target"]].add(e["source_family"])
    return by_host


def ip_index(evidence: list[dict]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = defaultdict(set)
    for e in evidence:
        claim = str((e.get("payload") or {}).get("claim") or "")
        if ":" not in claim:
            continue
        maybe_ip = claim.split(":", 1)[1]
        try:
            ip = str(ipaddress.ip_address(maybe_ip))
        except ValueError:
            continue
        if e.get("target"):
            found[ip].add(e["target"])
    return found


def classify_error(tool: str, error: str, detail: str) -> str:
    d = detail.lower()
    if tool == "wappalyzer_match" and "fingerprint_rules.parquet" in d:
        return "fingerprint_rules_file_missing"
    if tool == "geolocation" and ".mmdb" in d:
        return "geolite_database_file_missing"
    if tool == "cve_rag" and "cve_corpus" in d and "not found: collection" in d:
        return "qdrant_cve_collection_missing"
    if tool == "kev_epss" and "dymam.duckdb" in d and "database does not exist" in d:
        return "duckdb_database_file_missing"
    return f"other_{error}"


def audit_run(run: dict) -> tuple[dict, list[dict], list[dict]]:
    rd, bundle = run["dir"], run["bundle"]
    assert bundle.is_dir() and rd.is_dir()
    calls = read_jsonl(rd / "tool_calls.jsonl")
    assert len({c["seq"] for c in calls}) == len(calls)
    misses = [m for m in read_jsonl(rd / "miss_log.jsonl") if m.get("kind") == "tool_miss"]
    local_status: dict[str, Counter] = defaultdict(Counter)
    for c in calls:
        if c["tool"] in LOCAL:
            local_status[c["tool"]][c["status"]] += 1

    stage_errors: dict[str, Counter] = defaultdict(Counter)
    stage_reasons: dict[str, Counter] = defaultdict(Counter)
    if run["stage_trace"]:
        for stage in read_jsonl(rd / "stage_trace.jsonl"):
            for e in stage.get("stage_errors") or []:
                tool = e.get("tool")
                if tool in LOCAL:
                    stage_errors[tool][e.get("error", "unknown")] += 1
                    stage_reasons[tool][classify_error(
                        tool, e.get("error", "unknown"), str(e.get("detail") or ""))] += 1
        for tool in LOCAL:
            assert sum(stage_errors[tool].values()) == local_status[tool]["error"], (run["org"], tool)

    flagged, components, hosts = clean_flags(bundle)
    evidence = read_jsonl(bundle / "evidence_event.jsonl")
    families = source_families(evidence)
    ips = ip_index(evidence)
    resolved = set(json.loads((bundle / "discovery_manifest.json").read_text()).get("resolved_hosts") or [])
    known_hosts = hosts | resolved
    host_lookup: dict[str, set[str]] = defaultdict(set)
    for host in known_hosts:
        host_lookup[normalize_host(host)].add(host)

    host_misses: dict[str, Counter] = defaultdict(Counter)
    unmapped: list[dict] = []
    mapping_counts = Counter()
    logged_miss_keys = Counter((c["tool"], c["cache_key"]) for c in calls if c["status"] == "miss")
    parsed_miss_keys = Counter()
    for m in misses:
        seq = m["seq"]
        parsed = MISS_RE.match(m["detail"])
        assert parsed and parsed.group(1) == m["tool"]
        kwargs = ast.literal_eval(parsed.group(2))
        assert isinstance(kwargs, dict) and len(kwargs) == 1
        expected_key = f"tool:{m['tool']}:v1:" + hashlib.sha1(
            json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest()
        parsed_miss_keys[(m["tool"], expected_key)] += 1
        (arg, value), = kwargs.items()
        if arg in {"url", "target", "host"}:
            possible = host_lookup.get(normalize_host(str(value)), set())
            basis = "exact_url_host"
        elif arg == "ip":
            possible = ips.get(str(value), set())
            basis = "observed_ip_claim"
        else:
            possible = set()
            basis = "unsupported_argument"
        if len(possible) == 1:
            host = next(iter(possible))
            host_misses[host][m["tool"]] += 1
            mapping_counts[f"unique_{basis}"] += 1
        else:
            reason = "ambiguous" if possible else "unmapped"
            mapping_counts[reason] += 1
            unmapped.append({"org": run["org"], "source": run["source"],
                             "seq": seq, "tool": m["tool"], "input_id": ident(str(value)),
                             "reason": reason, "mapping_basis": basis,
                             "possible_host_count": len(possible)})
    assert parsed_miss_keys == logged_miss_keys, run["org"]

    mapped_hosts = set(host_misses)
    index: list[dict] = []
    for host in sorted(flagged | mapped_hosts):
        miss_tools = host_misses.get(host, Counter())
        index.append({
            "org": run["org"], "source": run["source"], "host_id": ident(host),
            "alert": host in flagged,
            "validator": host in components["validator"],
            "peer": host in components["peer"],
            "manifest": host in components["manifest"],
            "miss_call_count": sum(miss_tools.values()),
            "miss_tools": dict(sorted(miss_tools.items())),
            "remaining_families": sorted(families.get(host, set())),
            "miss_families_with_surviving_alternative": sorted({
                MISS_FAMILY[t] for t in miss_tools
                if t in MISS_FAMILY and MISS_FAMILY[t] in families.get(host, set())
            }),
            "grade": ("coincident_unconfirmed" if miss_tools and host in flagged else
                      "mapped_no_alert" if miss_tools else "alert_without_mapped_miss"),
        })

    formal_report = json.loads((bundle / "rendered_report.json").read_text())
    summary_present = None
    if run["stage_trace"]:
        summary_present = bool(json.loads((rd / "cell_result.json").read_text())["report_summary_present"])
    else:
        summary_present = False if "report LLM unavailable" in (rd / "console.txt").read_text() else None
    run_out = {
        "org": run["org"], "source": run["source"], "completed": True,
        "frozen_bundle_match": True, "tool_calls": len(calls), "tool_miss_calls": len(misses),
        "input_sha256": {
            "tool_calls.jsonl": digest(rd / "tool_calls.jsonl"),
            "miss_log.jsonl": digest(rd / "miss_log.jsonl"),
            "evidence_event.jsonl": digest(bundle / "evidence_event.jsonl"),
            "claim.jsonl": digest(bundle / "claim.jsonl"),
            "decision.jsonl": digest(bundle / "decision.jsonl"),
            "discovery_manifest.json": digest(bundle / "discovery_manifest.json"),
            "rendered_report.json": digest(bundle / "rendered_report.json"),
            **({"stage_trace.jsonl": digest(rd / "stage_trace.jsonl"),
                "cell_result.json": digest(rd / "cell_result.json")}
               if run["stage_trace"] else
               {"console.txt": digest(rd / "console.txt")}),
        },
        "local_tool_status": {t: dict(sorted(c.items())) for t, c in sorted(local_status.items())},
        "local_stage_error_types": {t: dict(sorted(c.items())) for t, c in sorted(stage_errors.items())},
        "local_stage_error_reasons": {t: dict(sorted(c.items())) for t, c in sorted(stage_reasons.items())},
        "evidence_hosts": len(hosts), "clean_alert_hosts": len(flagged),
        "clean_alert_components": {k: len(v) for k, v in components.items()},
        "mapped_miss_hosts": len(mapped_hosts),
        "alerted_mapped_miss_hosts": len(flagged & mapped_hosts),
        "mapped_miss_calls": sum(sum(v.values()) for v in host_misses.values()),
        "miss_mapping_calls": dict(sorted(mapping_counts.items())),
        "report_summary_present": summary_present,
        "formal_report_sections": len(formal_report.get("sections") or []),
    }
    return run_out, index, unmapped


def original_clean_alert_total() -> dict:
    grid = json.loads((EXP / "e3_grid_v1.json").read_text())["cells"]
    by_org = {}
    for cell in grid:
        by_org.setdefault(cell["org"], Path(cell["clean_dir"]))
    assert len(by_org) == 20
    counts = [clean_flags(bundle) for bundle in by_org.values()]
    n_alert = sum(len(flags) for flags, _, _ in counts)
    n_hosts = sum(len(hosts) for _, _, hosts in counts)
    assert (n_alert, n_hosts) == (210, 1892), (n_alert, n_hosts)
    return {"organizations": 20, "alert_hosts": n_alert, "evidence_hosts": n_hosts}


def main() -> None:
    out_runs, index, unmapped = [], [], []
    for run in source_runs():
        result, rows, unknown = audit_run(run)
        out_runs.append(result)
        index.extend(rows)
        unmapped.extend(unknown)
    grades = Counter(r["grade"] for r in index)
    assert grades["coincident_unconfirmed"] == sum(r["alerted_mapped_miss_hosts"] for r in out_runs)
    assert (grades["coincident_unconfirmed"] + grades["mapped_no_alert"]
            == sum(r["mapped_miss_hosts"] for r in out_runs))
    summary = {
        "distinct_organizations": len(out_runs),
        "clean_runs": len(out_runs),
        "clean_alert_hosts_in_scope": sum(r["clean_alert_hosts"] for r in out_runs),
        "mapped_miss_hosts": sum(r["mapped_miss_hosts"] for r in out_runs),
        "alerted_mapped_miss_hosts": sum(r["alerted_mapped_miss_hosts"] for r in out_runs),
        "replay_miss_calls": sum(r["tool_miss_calls"] for r in out_runs),
        "mapped_replay_miss_calls": sum(r["mapped_miss_calls"] for r in out_runs),
        "unmapped_or_ambiguous_miss_calls": len(unmapped),
        "host_grades": dict(sorted(grades.items())),
        "confirmed_natural_faults": 0,
        "confirmed_natural_faults_note": "No collection-time truth or independent adjudication; zero means none confirmed, not none exist.",
    }
    result = {
        "protocol": "P1D_READONLY_PROTOCOL.md",
        "classification": "revision-stage descriptive; not independent or natural-fault validation",
        "inputs_sha256": {
            "smoke_restore_result_v1.json": digest(HERE / "smoke_restore_result_v1.json"),
            "e_a_offline_recheck_v1.json": digest(HERE / "e_a_offline_recheck_v1.json"),
            "e3_grid_v1.json": digest(EXP / "e3_grid_v1.json"),
            "provenance_local_tools_v2.json": digest(HERE / "provenance_local_tools_v2.json"),
        },
        "original_clean_alert_crosscheck": original_clean_alert_total(),
        "summary": summary, "runs": out_runs,
        "host_evidence_index": index,
        "unmapped_miss_index": unmapped,
    }
    (HERE / "p1d_readonly_v1.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (HERE / "p1d_evidence_index_v1.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["org", "source", "host_id", "grade", "alert",
                                                "validator", "peer", "manifest", "miss_call_count",
                                                "miss_tools", "remaining_families",
                                                "miss_families_with_surviving_alternative"])
        writer.writeheader()
        for row in index:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
