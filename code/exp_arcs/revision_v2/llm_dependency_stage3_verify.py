"""Verify Stage-3 raw runs, correcting only the graph-success summary predicate.

The raw runner counted any graph_writer stage error as Neo4j failure; an empty
Qdrant organization corpus produces independent 404 fallback errors. This
verifier leaves raw results intact and checks graph write and Redis commit from
the recorded stage statistics and pipeline log.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from experiments.exp_arcs.revision_v2 import llm_dependency_stage3 as s3

RAW = s3.OUT
OUT = s3.HERE / "llm_dependency_stage3_verified_v1.json"
REPORT = s3.HERE / "LLM_DEPENDENCY_STAGE3_VERIFIED.md"


def main() -> int:
    raw_bytes = RAW.read_bytes()
    doc = json.loads(raw_bytes)
    verified = {}
    for key, row in doc["runs"].items():
        out_dir = Path(row["out_dir"])
        trace = s3._trace(out_dir / "stage_trace.jsonl")
        graph = next((x for x in trace if x["stage"] == "graph_writer"), {})
        neo4j_errors = [x for x in graph.get("stage_errors", [])
                        if x.get("tool") == "neo4j"]
        graph_stats = graph.get("graph_stats", {})
        log = (out_dir / "pipeline.log").read_text() if (out_dir / "pipeline.log").exists() else ""
        committed = "snapshot persisted" in log
        successful = bool(graph_stats.get("nodes", 0) > 0 and not neo4j_errors and committed)
        result = row.get("result", {})
        verified[key] = {
            "status": row["status"],
            "neo4j_projection_succeeded": successful,
            "neo4j_nodes": graph_stats.get("nodes", 0),
            "neo4j_relationships": graph_stats.get("relationships", 0),
            "redis_snapshot_committed": committed,
            "neo4j_errors": neo4j_errors,
            "qdrant_corpus_fallback_errors": sum(
                x.get("tool") == "organization_corpus" for x in graph.get("stage_errors", [])),
            "tool_live_calls": result.get("tool_status", {}).get("live", 0),
            "llm_calls": result.get("llm_calls"),
            "llm_cached": result.get("llm_cached"),
            "report_summary": result.get("report_summary"),
            "matches_frozen_bundle": result.get("matches_frozen_bundle"),
        }
    comparison = doc["comparisons"]
    report_modes = ("report_disabled", "report_template", "report_perturbed")
    report_formal_identical = all(
        block["status"] == "ok" and not block.get("formal_differences")
        for key, block in comparison.items() if key.rsplit("|", 1)[-1] in report_modes)
    all_graph = all(r["status"] == "ok" and r["neo4j_projection_succeeded"]
                    for r in verified.values())
    no_live = all(r["tool_live_calls"] == 0 for r in verified.values())
    full_cache = all(r["llm_calls"] == r["llm_cached"] for key, r in verified.items()
                     if key.startswith("full_cached|"))
    status = ("PASS_HEALTHY_GRAPH_REPORT_INVARIANT" if
              all_graph and no_live and full_cache and report_formal_identical
              else "INCOMPLETE_OR_DIFFERENCE")
    output = {
        "experiment": "E-A Stage 3 verified",
        "status": status,
        "raw_result": str(RAW),
        "raw_result_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "correction": "Qdrant organization_corpus 404 is independent of Neo4j projection success; raw runner incorrectly treated every graph_writer error as Neo4j failure",
        "n_runs": len(verified),
        "all_neo4j_projection_succeeded": all_graph,
        "no_live_tool_calls": no_live,
        "full_mode_all_llm_cached": full_cache,
        "report_modes_formal_identical": report_formal_identical,
        "service_images": {
            image: subprocess.run(
                ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
            for image in ("neo4j:5-community", "qdrant/qdrant:latest")
        },
        "redis_version": subprocess.run(
            ["redis-server", "--version"], capture_output=True,
            text=True, check=False,
        ).stdout.strip(),
        "runs": verified,
        "comparisons": comparison,
    }
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    lines = ["# E-A Stage 3 — verified healthy-graph result", "",
             f"Status: **{status}**; {len(verified)} runs.", "",
             "The raw runner's `graph=False` flag was a summary-classification error:",
             "it counted the empty Qdrant organization-corpus fallback as a Neo4j failure.",
             "This verification checks actual Neo4j write counts, absence of Neo4j errors,",
             "and a recorded Redis snapshot commit for every run. Raw cell outputs and",
             "stage traces were not changed.", "",
             f"Neo4j projection succeeded: {sum(r['neo4j_projection_succeeded'] for r in verified.values())}/{len(verified)}.",
             f"Zero live tool calls: {no_live}; all full-mode LLM calls cached: {full_cache}.",
             f"Reporting modes formally identical: {report_formal_identical}.", "",
             "| Organization/cell | First difference when all LLM calls disabled | Formal output difference |",
             "|---|---|---|"]
    for org in s3.ORGS:
        for cell in s3.CELLS:
            block = comparison[f"{org}|{cell}|global_llm_disabled"]
            first = (block.get("stage_trace") or {}).get("first_semantic_difference")
            if first:
                first_name = first["stage"] + ": " + ", ".join(first["differences"])
            else:
                first_name = "none observed"
            formal = ", ".join(block.get("formal_differences") or []) or "none"
            lines.append(f"| {org}/{cell} | {first_name} | {formal} |")
    lines.extend(["", "Detailed per-stage field diffs are in the verified JSON.",
                  "This is still a two-organization development-set result, not holdout validation.", ""])
    REPORT.write_text("\n".join(lines))
    print(status)
    return 0 if status == "PASS_HEALTHY_GRAPH_REPORT_INVARIANT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
