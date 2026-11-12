"""E-A Stage 3: paired replay with fresh Redis/Neo4j state per cell.

An already running isolated Qdrant must be reachable at QDRANT_URL. The script
manages only Redis port 6391 and container arcs-ea-neo4j-stage3. All model
requests remain cache guarded by llm_dependency_cell. Never points at project
Neo4j/Redis data volumes.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.exp_arcs.revision_v2 import llm_dependency_run as stage2

REPO = stage2.REPO
D = stage2.D
HERE = stage2.HERE
SOURCE_RDB = D / "_audit/p1b_pilot_forced_20261009T065255Z/services/redis/dump.rdb"
REDIS_PORT = 6391
NEO4J_NAME = "arcs-ea-neo4j-stage3"
NEO4J_PORT = 7688
MODES = stage2.MODES
ORGS = stage2.ORGS
CELLS = stage2.CELLS
OUT = HERE / "llm_dependency_stage3_v1.json"
RESULT = HERE / "LLM_DEPENDENCY_STAGE3_RESULT.md"


def _cmd(args: list[str], *, timeout: int = 90, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=REPO, text=True, capture_output=True,
                          timeout=timeout, check=check)


def _stop_isolated() -> None:
    _cmd(["redis-cli", "-p", str(REDIS_PORT), "shutdown", "nosave"],
         timeout=15, check=False)
    _cmd(["docker", "stop", NEO4J_NAME], timeout=45, check=False)


def _start_isolated(redis_dir: Path) -> None:
    _stop_isolated()
    redis_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SOURCE_RDB, redis_dir / "dump.rdb")
    _cmd(["redis-server", "--port", str(REDIS_PORT), "--dir", str(redis_dir),
          "--dbfilename", "dump.rdb", "--appendonly", "no", "--daemonize", "yes"])
    _cmd(["docker", "run", "--rm", "-d", "--name", NEO4J_NAME,
          "-p", f"{NEO4J_PORT}:7687",
          "-e", "NEO4J_AUTH=neo4j/dymam2026",
          "-e", "NEO4J_server_memory_heap_max__size=512m",
          "-e", "NEO4J_server_memory_pagecache_size=512m",
          "neo4j:5-community"], timeout=60)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        probe = _cmd(["docker", "exec", NEO4J_NAME, "cypher-shell",
                      "-u", "neo4j", "-p", "dymam2026", "RETURN 1 AS ok"],
                     timeout=15, check=False)
        if probe.returncode == 0 and "1" in probe.stdout:
            return
        time.sleep(2)
    raise RuntimeError("isolated Neo4j did not become ready in 90 seconds")


def _trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def _tool_events(out_dir: Path, stage: dict[str, Any]) -> list[dict[str, Any]]:
    path = out_dir / "tool_calls.jsonl"
    if not path.exists():
        return []
    lo, hi = stage["tool_log_seq_start"], stage["tool_log_seq_end"]
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    return [{k: v for k, v in row.items() if k != "seq"}
            for row in rows if lo <= row["seq"] <= hi]


def _first_diff(left: Any, right: Any, path: str = "") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"field": path, "full": left, "other": right}
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            here = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                return {"field": here, "full": left.get(key), "other": right.get(key)}
            found = _first_diff(left[key], right[key], here)
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"field": f"{path}.length", "full": len(left), "other": len(right)}
        for i, (a, b) in enumerate(zip(left, right)):
            found = _first_diff(a, b, f"{path}[{i}]")
            if found:
                return found
        return None
    if left != right:
        return {"field": path, "full": left, "other": right}
    return None


def _stage_comparison(full_dir: Path, other_dir: Path) -> dict[str, Any]:
    full, other = _trace(full_dir / "stage_trace.jsonl"), _trace(other_dir / "stage_trace.jsonl")
    if len(full) != len(other):
        return {"stage_sequence_equal": False, "full_stages": [x["stage"] for x in full],
                "other_stages": [x["stage"] for x in other]}
    rows = []
    for a, b in zip(full, other):
        if a["stage"] != b["stage"]:
            return {"stage_sequence_equal": False, "full_stages": [x["stage"] for x in full],
                    "other_stages": [x["stage"] for x in other]}
        fields = {}
        for key in ("assets", "new_evidence", "ledgers", "report_summary"):
            found = _first_diff(a.get(key), b.get(key), key)
            if found:
                fields[key] = found
        tool_diff = _first_diff(_tool_events(full_dir, a), _tool_events(other_dir, b), "tool_events")
        if tool_diff:
            fields["tool_events"] = tool_diff
        rows.append({"stage": a["stage"], "index": a["index"], "differences": fields,
                     "full_llm_calls": a["llm_calls_in_stage"],
                     "other_llm_calls": b["llm_calls_in_stage"]})
    first = next((r for r in rows if r["differences"]), None)
    return {"stage_sequence_equal": True, "first_semantic_difference": first,
            "stages": rows}


def _write_result(doc: dict[str, Any]) -> None:
    lines = ["# E-A Stage 3 — stage divergence and healthy Neo4j path", "",
             f"Executed: {doc['executed_at']}", "",
             f"Status: **{doc['status']}**", "",
             "Each cell started with a copy of the same frozen Redis RDB and a new, empty Neo4j container.",
             "Tool responses and non-report LLM completions used the strict replay cache guard.", "",
             "| Pair | Report disable | Template | Perturbed | Global LLM off |",
             "|---|---|---|---|---|"]
    for org in ORGS:
        for cell in CELLS:
            vals = []
            for mode in MODES[1:]:
                block = doc["comparisons"].get(f"{org}|{cell}|{mode}", {})
                first = (block.get("stage_trace") or {}).get("first_semantic_difference")
                if block.get("status") != "ok":
                    vals.append("blocked")
                elif first is None:
                    vals.append("no semantic change")
                else:
                    keys = ", ".join(first["differences"])
                    vals.append(f"{first['stage']} ({keys})")
            lines.append(f"| {org}/{cell} | " + " | ".join(vals) + " |")
    lines.extend(["", f"Run root: `{doc['run_root']}`", "",
                  "Raw comparison: `experiments/exp_arcs/revision_v2/llm_dependency_stage3_v1.json`", ""])
    RESULT.write_text("\n".join(lines))


def main() -> int:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = D / "_audit" / f"llm_dependency_stage3_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    grid = json.loads(stage2.GRID.read_text())["cells"]
    by_key = {(c["org"], c["cell"]): c for c in grid if c["org"] in ORGS and c["cell"] in CELLS}
    clean_by_org = {org: next(c["clean_dir"] for c in grid if c["org"] == org) for org in ORGS}
    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    env = dict(os.environ, PYTHONPATH=str(REPO), REDIS_URL=f"redis://127.0.0.1:{REDIS_PORT}",
               NEO4J_URI=f"bolt://127.0.0.1:{NEO4J_PORT}", DYMAM_API_URL="")
    env.setdefault("QDRANT_URL", "http://127.0.0.1:6335")

    try:
        for mode in MODES:
            for org in ORGS:
                for cell in CELLS:
                    key = (mode, org, cell)
                    out_dir = root / mode / org / cell.replace("@", "_")
                    out_dir.mkdir(parents=True, exist_ok=False)
                    source = by_key.get((org, cell))
                    inode = None if cell == "clean" else source["inject_node"]
                    frozen = clean_by_org[org] if cell == "clean" else source["faulted_dir"]
                    print(f"RUN {mode} {org} {cell}", flush=True)
                    row: dict[str, Any] = {"status": "blocked_or_failed", "out_dir": str(out_dir)}
                    results[key] = row
                    try:
                        _start_isolated(out_dir / "redis")
                        cmd = [sys.executable, "-m",
                               "experiments.exp_arcs.revision_v2.llm_dependency_stage_trace_cell",
                               str(out_dir), org, cell, json.dumps(inode),
                               str(stage2.resolve(frozen)), mode]
                        with (out_dir / "pipeline.log").open("w") as log:
                            proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log,
                                                  stderr=subprocess.STDOUT, timeout=420)
                        row["exit_code"] = proc.returncode
                        row["status"] = "ok" if proc.returncode == 0 else "blocked_or_failed"
                        result_file = out_dir / "cell_result.json"
                        if result_file.exists():
                            row["result"] = json.loads(result_file.read_text())
                        trace = _trace(out_dir / "stage_trace.jsonl")
                        graph = next((x for x in trace if x["stage"] == "graph_writer"), {})
                        # An empty isolated Qdrant collection may emit a 404 during
                        # optional vector lookup; that is not a Neo4j write failure.
                        neo4j_errors = [e for e in graph.get("stage_errors", [])
                                        if e.get("tool") == "neo4j"]
                        row["neo4j_projection_succeeded"] = bool(
                            graph.get("graph_stats", {}).get("nodes", 0) > 0
                            and not neo4j_errors)
                    except Exception as exc:
                        row["error"] = repr(exc)
                    finally:
                        _stop_isolated()
                    print(f"DONE {mode} {org} {cell}: {row['status']} graph={row.get('neo4j_projection_succeeded')}", flush=True)
    finally:
        _stop_isolated()

    stage2.add_repair_metrics({}, results, by_key)
    comparisons = {}
    for org in ORGS:
        for cell in CELLS:
            full_row = results[("full_cached", org, cell)]
            full_dir = Path(full_row["out_dir"])
            for mode in MODES[1:]:
                other = results[(mode, org, cell)]
                status = "ok" if full_row["status"] == other["status"] == "ok" else "blocked"
                block = {"status": status}
                if status == "ok":
                    block["formal_differences"] = [k for k, v in stage2.comparison_fields(
                        full_row["result"], cell).items()
                        if v != stage2.comparison_fields(other["result"], cell)[k]]
                    block["stage_trace"] = _stage_comparison(full_dir, Path(other["out_dir"]))
                comparisons[f"{org}|{cell}|{mode}"] = block
    rows = list(results.values())
    status = ("PASS_HEALTHY_GRAPH" if all(r["status"] == "ok" and
                                          r.get("neo4j_projection_succeeded") for r in rows)
              else "INCOMPLETE_OR_GRAPH_FAILURE")
    doc = {"experiment": "E-A Stage 3", "executed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "status": status, "run_root": str(root),
           "input_hashes": {"rdb_sha256": hashlib.sha256(SOURCE_RDB.read_bytes()).hexdigest(),
                            "grid_sha256": stage2.sha_file(stage2.GRID),
                            "trace_cell_sha256": stage2.sha_file(HERE / "llm_dependency_stage_trace_cell.py"),
                            "runner_sha256": stage2.sha_file(Path(__file__))},
           "reset_contract": "fresh frozen Redis RDB and new empty Neo4j container before every cell",
           "runs": {"|".join(k): v for k, v in results.items()}, "comparisons": comparisons}
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    _write_result(doc)
    print(f"STATUS {status}", flush=True)
    return 0 if status == "PASS_HEALTHY_GRAPH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
