"""Finalize existing E-A offline runs without repeating any pipeline cell.

The first 30-run attempt completed, but the borrowed Stage-3 JSONL reader
treated Unicode U+0085 inside a valid string as a line break. This postprocessor
reads only literal newline delimiters, preserves the original cell files, and
records the recovery explicitly.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from experiments.exp_arcs.revision_v2 import e_a_offline_recheck as run


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m ...e_a_offline_recheck_finalize RUN_ROOT")
    root = Path(sys.argv[1]).resolve()
    if not root.is_dir() or not root.is_relative_to(run.D / "_audit"):
        raise SystemExit("run root must be an existing exp_arcs/_audit directory")
    lock = json.loads(run.LOCK.read_text())
    for path, field in ((run.PROTOCOL, "protocol_sha256"),
                        (run.prior.GRID, "grid_sha256"),
                        (run.RDB, "redis_rdb_sha256")):
        if run.sha(path) != lock[field]:
            raise SystemExit(f"frozen input changed: {path}")
    orgs, cells, modes = (lock["selected_organizations_in_hash_rank_order"],
                          lock["cells"], lock["modes"])
    grid = json.loads(run.prior.GRID.read_text())
    by_key = {(r["org"], r["cell"]): r for r in grid["cells"]
              if r["org"] in orgs and r["cell"] in cells[1:]}
    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    raw_hashes: dict[str, dict[str, str]] = {}
    for mode in modes:
        for org in orgs:
            for cell in cells:
                key = (mode, org, cell)
                out = root / mode / org / cell.replace("@", "_")
                result_file = out / "cell_result.json"
                row: dict[str, Any] = {"out_dir": str(out),
                                       "status": "ok" if result_file.is_file() else "incomplete"}
                if result_file.is_file():
                    row["result"] = json.loads(result_file.read_text())
                error_file = out / "error.txt"
                if error_file.is_file():
                    row["error"] = error_file.read_text()[-4000:]
                trace = run.trace_rows(out / "stage_trace.jsonl")
                graph = next((x for x in trace if x["stage"] == "graph_writer"), {})
                neo_errors = [e for e in graph.get("stage_errors", [])
                              if e.get("tool") == "neo4j"]
                row["neo4j_projection_succeeded"] = bool(
                    graph.get("graph_stats", {}).get("nodes", 0) > 0 and not neo_errors)
                if row.get("result", {}).get("tool_status", {}).get("live", 0):
                    row["status"] = "invalid_live_tool_call"
                results[key] = row
                raw_hashes["|".join(key)] = {
                    name: run.sha(out / name) for name in
                    ("cell_result.json", "stage_trace.jsonl", "tool_calls.jsonl",
                     "miss_log.jsonl", "pipeline.log") if (out / name).is_file()
                }
    run.evaluate_faults(results, by_key)
    pairs: dict[str, dict[str, Any]] = {}
    for org in orgs:
        for cell in cells:
            full = results[("full_cached", org, cell)]
            off = results[("global_llm_disabled", org, cell)]
            pair: dict[str, Any] = {
                "status": "ok" if full["status"] == off["status"] == "ok" else "incomplete",
                "full_status": full["status"], "off_status": off["status"],
                "tool_miss_counts": {
                    "full": full.get("result", {}).get("tool_status", {}).get("miss", 0),
                    "off": off.get("result", {}).get("tool_status", {}).get("miss", 0),
                },
            }
            if pair["status"] == "ok":
                a = run.prior.comparison_fields(full["result"], cell)
                b = run.prior.comparison_fields(off["result"], cell)
                if cell != "clean":
                    a["offline_evaluation"] = full["result"].get("offline_evaluation")
                    b["offline_evaluation"] = off["result"].get("offline_evaluation")
                pair["formal_differences"] = sorted(k for k in a if a[k] != b[k])
                pair["stage_trace"] = run.stage3._stage_comparison(
                    Path(full["out_dir"]), Path(off["out_dir"]))
                pair["full_evaluation"] = full["result"].get("offline_evaluation")
                pair["off_evaluation"] = off["result"].get("offline_evaluation")
            pairs[f"{org}|{cell}"] = pair
    rows = list(results.values())
    success = sum(r["status"] == "ok" for r in rows)
    complete = sum(p["status"] == "ok" for p in pairs.values())
    live = sum(r.get("result", {}).get("tool_status", {}).get("live", 0) for r in rows)
    eval_ok = all(r.get("result", {}).get("offline_evaluation") is not None
                  for (mode, org, cell), r in results.items() if cell != "clean")
    graph_ok = all(r.get("neo4j_projection_succeeded") for r in rows)
    status = ("COMPLETE_EXISTING_CORPUS_RECHECK" if success == 30 and complete == 15
              and live == 0 and eval_ok and graph_ok else "INCOMPLETE_RECHECK")
    full_rows = [results[("full_cached", org, cell)] for org in orgs for cell in cells]
    fault_pairs = [p for key, p in pairs.items() if "|clean" not in key and p["status"] == "ok"]
    metric_names = ("candidate_coverage", "detect_at_1", "detect_at_3",
                    "repair_success", "repair_cost")
    metric_change_counts = {
        name: sum(p["full_evaluation"] is not None and
                  p["off_evaluation"] is not None and
                  p["full_evaluation"][name] != p["off_evaluation"][name]
                  for p in fault_pairs)
        for name in metric_names
    }
    discovery_confidence_first = sum(
        (first := p.get("stage_trace", {}).get("first_semantic_difference")) is not None
        and first["stage"] == "discovery"
        and first["differences"].get("assets", {}).get("field", "").endswith(".confidence")
        for p in pairs.values()
    )
    doc = {
        "experiment": "E-A retrospective offline recheck",
        "status": status,
        "executed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_root": str(root),
        "postprocess_recovery": "30 cells completed; original summary reader split embedded Unicode line separators; no cell rerun",
        "selection_lock_sha256": run.sha(run.LOCK),
        "finalizer_sha256": run.sha(Path(__file__)),
        "organizations": orgs, "cells": cells, "modes": modes,
        "successful_runs": success, "complete_pairs": complete,
        "live_tool_calls": live, "evaluation_complete": eval_ok,
        "graph_projection_complete": graph_ok,
        "full_cached_bundle_matches_frozen": sum(r.get("result", {}).get("matches_frozen_bundle") is True
                                                 for r in full_rows),
        "full_cached_llm_calls": sum(r.get("result", {}).get("llm_calls", 0) for r in full_rows),
        "full_cached_llm_cache_hits": sum(r.get("result", {}).get("llm_cached", 0) for r in full_rows),
        "full_cached_report_summary_present": sum(r.get("result", {}).get("report_summary_present") is True
                                                  for r in full_rows),
        "full_cached_tool_misses": sum(r.get("result", {}).get("tool_status", {}).get("miss", 0)
                                       for r in full_rows),
        "first_difference_discovery_technology_confidence_pairs": discovery_confidence_first,
        "tool_miss_count_equal_pairs": sum(p["tool_miss_counts"]["full"] == p["tool_miss_counts"]["off"]
                                           for p in pairs.values()),
        "fault_metric_change_counts": metric_change_counts,
        "runs": {"|".join(k): v for k, v in results.items()},
        "pairs": pairs,
        "raw_artifact_hashes": raw_hashes,
    }
    run.OUT.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    run.write_report(doc)
    print(json.dumps({k: doc[k] for k in (
        "status", "successful_runs", "complete_pairs", "live_tool_calls",
        "evaluation_complete", "graph_projection_complete",
        "full_cached_bundle_matches_frozen", "full_cached_llm_calls",
        "full_cached_llm_cache_hits", "full_cached_report_summary_present",
        "full_cached_tool_misses")}, sort_keys=True))
    return 0 if status == "COMPLETE_EXISTING_CORPUS_RECHECK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
