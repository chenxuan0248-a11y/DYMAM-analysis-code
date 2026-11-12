"""Retrospective five-organization E-A replay, with no live target tools.

Run only after the selection lock and frozen input hashes pass preflight.
Creates ephemeral local Redis, Neo4j, and Qdrant services; never contacts a
target organization or fills a missing target/LLM cache entry.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.exp_arcs.detector_eval import (
    detector_localize, manifest_dropped_hosts, peer_anomaly_hosts,
)
from experiments.exp_arcs.detector_pr_roc import _host_scores
from experiments.exp_arcs.rq2_combined_topk import ranked_hosts, tie_aware_hit
from experiments.exp_arcs.revision_v2.e_a_trace_jsonl import trace_rows
from experiments.exp_arcs.revision_v2 import llm_dependency_run as prior
from experiments.exp_arcs.revision_v2 import llm_dependency_stage3 as stage3

REPO, D, HERE = prior.REPO, prior.D, prior.HERE
LOCK = HERE / "e_a_offline_recheck_lock_v1.json"
PROTOCOL = HERE / "E_A_OFFLINE_RECHECK_PROTOCOL.md"
RDB = stage3.SOURCE_RDB
REDIS_PORT, NEO4J_PORT, QDRANT_PORT = 6392, 7689, 6336
NEO4J_NAME = "arcs-ea-neo4j-offline-recheck"
QDRANT_NAME = "arcs-ea-qdrant-offline-recheck"
OUT = HERE / "e_a_offline_recheck_v1.json"
REPORT = HERE / "E_A_OFFLINE_RECHECK_RESULT.md"


stage3._trace = trace_rows


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd(args: list[str], *, timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=REPO, text=True, capture_output=True,
                          timeout=timeout, check=check)


def port_free(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def preflight() -> tuple[dict, list[dict], list[str]]:
    lock = json.loads(LOCK.read_text())
    checks = {
        PROTOCOL: "protocol_sha256",
        prior.GRID: "grid_sha256",
        RDB: "redis_rdb_sha256",
        HERE / "llm_dependency_stage_trace_cell.py": "stage_trace_wrapper_sha256",
        HERE / "llm_dependency_cell.py": "cell_implementation_sha256",
        HERE / "llm_dependency_run.py": "comparison_implementation_sha256",
    }
    for path, field in checks.items():
        if not path.is_file() or sha(path) != lock[field]:
            raise RuntimeError(f"frozen input missing or changed: {path}")
    grid = json.loads(prior.GRID.read_text())
    seed = lock["selection_seed"]
    eligible = sorted(
        set(grid["orgs_target"]) - set(lock["excluded_prior_ea_organizations"]),
        key=lambda org: (hashlib.sha256(f"{seed}:{org}".encode()).hexdigest(), org),
    )
    orgs = eligible[:5]
    if orgs != lock["selected_organizations_in_hash_rank_order"]:
        raise RuntimeError("selection lock no longer matches E3 grid")
    cells = lock["cells"]
    rows = [row for row in grid["cells"]
            if row["org"] in orgs and row["cell"] in cells[1:]]
    if len(rows) != 10 or len({(r["org"], r["cell"]) for r in rows}) != 10:
        raise RuntimeError("fixed 5 x 2 fault-cell grid is incomplete")
    missing = sorted({p for r in rows for p in (r["clean_dir"], r["faulted_dir"])
                      if not prior.resolve(p).is_dir()})
    if missing:
        raise RuntimeError(f"frozen bundle directories missing: {missing}")
    for port in (REDIS_PORT, NEO4J_PORT, QDRANT_PORT):
        if not port_free(port):
            raise RuntimeError(f"isolated service port {port} is already in use")
    for program in ("redis-server", "docker"):
        if shutil.which(program) is None:
            raise RuntimeError(f"required local service program unavailable: {program}")
    for name in (NEO4J_NAME, QDRANT_NAME):
        probe = cmd(["docker", "container", "inspect", name], check=False, timeout=20)
        if probe.returncode == 0:
            raise RuntimeError(f"isolated container name already exists: {name}")
    return lock, rows, orgs


def start_container(name: str, args: list[str]) -> str:
    result = cmd(["docker", "run", "--pull", "never", "--rm", "-d",
                  "--name", name, *args], timeout=90)
    return result.stdout.strip()


def stop_container(container_id: str | None) -> None:
    if container_id:
        cmd(["docker", "stop", container_id], check=False, timeout=60)


def wait_neo4j() -> None:
    deadline = time.monotonic() + 100
    while time.monotonic() < deadline:
        probe = cmd(["docker", "exec", NEO4J_NAME, "cypher-shell", "-u", "neo4j",
                     "-p", "dymam2026", "RETURN 1 AS ok"], check=False, timeout=15)
        if probe.returncode == 0 and "1" in probe.stdout:
            return
        time.sleep(2)
    raise RuntimeError("isolated Neo4j did not become ready")


def start_redis(redis_dir: Path) -> subprocess.Popen:
    redis_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(RDB, redis_dir / "dump.rdb")
    log = (redis_dir / "redis.log").open("w")
    proc = subprocess.Popen(
        ["redis-server", "--port", str(REDIS_PORT), "--dir", str(redis_dir),
         "--dbfilename", "dump.rdb", "--appendonly", "no", "--save", ""],
        cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
    )
    log.close()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"isolated Redis exited {proc.returncode}")
        ping = cmd(["redis-cli", "-p", str(REDIS_PORT), "ping"], check=False, timeout=5)
        if ping.returncode == 0 and ping.stdout.strip() == "PONG":
            return proc
        time.sleep(0.25)
    proc.terminate()
    proc.wait(timeout=10)
    raise RuntimeError("isolated Redis did not become ready")


def stop_redis(proc: subprocess.Popen | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def evaluate_faults(results: dict[tuple[str, str, str], dict], grid: dict) -> None:
    for (mode, org, cell), row in results.items():
        if cell == "clean" or row["status"] != "ok":
            continue
        clean = results.get((mode, org, "clean"), {})
        if clean.get("status") != "ok":
            row["evaluation_status"] = "clean_pair_missing"
            continue
        try:
            source = grid[(org, cell)]
            clean_bundle = Path(clean["result"]["bundle_dir"])
            fault_bundle = Path(row["result"]["bundle_dir"])
            scores = _host_scores(fault_bundle)
            ranked = ranked_hosts(scores)
            true = source["inject_node"][1]
            truth = set(true) if isinstance(true, list) else {true}
            flagged = (detector_localize(fault_bundle)["flagged_hosts"]
                       | peer_anomaly_hosts(fault_bundle, min_base_rate=0.5)
                       | manifest_dropped_hosts(fault_bundle))
            repair_cell = dict(source, clean_dir=str(clean_bundle), faulted_dir=str(fault_bundle))
            repair = prior.s2.repair(repair_cell, frozenset(flagged))
            row["result"]["offline_evaluation"] = {
                "candidate_hosts": ranked,
                "true_hosts": sorted(truth),
                "candidate_coverage": bool(truth & set(ranked)),
                "detect_at_1": int(bool(truth & set(ranked[:1]))),
                "detect_at_3": int(bool(truth & set(ranked[:3]))),
                "detect_at_1_tie_aware": tie_aware_hit(scores, truth, 1),
                "detect_at_3_tie_aware": tie_aware_hit(scores, truth, 3),
                "flagged_hosts": sorted(flagged),
                "selected_invocations": prior.selected_invocations(clean_bundle, flagged),
                "repair_success": bool(repair["success"]),
                "repair_cost": float(repair["cost"]),
                "repair_cost_over_cstar": float(repair["cost"]) / float(source["C_star"]),
            }
        except Exception as exc:
            row["evaluation_status"] = "failed"
            row["evaluation_error"] = repr(exc)


def write_report(doc: dict[str, Any]) -> None:
    lines = ["# E-A retrospective offline recheck", "",
             f"Status: **{doc['status']}**; executed {doc['executed_at']}.", "",
             "This is an existing-corpus replay, not a new holdout. No target-facing tool call was permitted.", "",
             f"Successful runs: {doc['successful_runs']}/30; complete pairs: {doc['complete_pairs']}/15; "
             f"live tool calls observed: {doc['live_tool_calls']}.", "",
             f"Full cached bundles matching frozen originals: {doc.get('full_cached_bundle_matches_frozen', 'not computed')}/15; "
             f"cached LLM calls: {doc.get('full_cached_llm_cache_hits', 'not computed')}/"
             f"{doc.get('full_cached_llm_calls', 'not computed')}; "
             f"non-empty report summaries: {doc.get('full_cached_report_summary_present', 'not computed')}/15.",
             f"Full cached tool replay misses: {doc.get('full_cached_tool_misses', 'not computed')}; "
             "these were logged and handled by the existing pipeline fallback, not filled online.", "",
             "| Organization / cell | Pair | First semantic difference | Formal fields changed |",
             "|---|---|---|---|"]
    for key, pair in doc["pairs"].items():
        if pair["status"] != "ok":
            lines.append(f"| {key.replace('|', ' / ')} | incomplete | — | — |")
            continue
        first = pair["stage_trace"].get("first_semantic_difference")
        stage = "none" if first is None else first["stage"] + ": " + ", ".join(first["differences"])
        fields = ", ".join(pair["formal_differences"]) or "none"
        lines.append(f"| {key.replace('|', ' / ')} | complete | {stage} | {fields} |")
    lines += ["", "## Fault-cell metrics", "",
              "Each value is `full cached / global LLM off`; costs are the frozen specification units, not measured currency or latency.",
              "", "| Organization / cell | Candidate coverage | Detect@1 | Detect@3 | Repair success | Repair cost | Tool misses |",
              "|---|---|---|---|---|---|---|"]
    for org in doc["organizations"]:
        for cell in doc["cells"][1:]:
            pair = doc["pairs"][f"{org}|{cell}"]
            if pair["status"] != "ok" or not pair.get("full_evaluation") or not pair.get("off_evaluation"):
                lines.append(f"| {org} / {cell} | incomplete | — | — | — | — | — |")
                continue
            a, b = pair["full_evaluation"], pair["off_evaluation"]
            def both(name: str) -> str:
                return f"{a[name]} / {b[name]}"
            miss = pair["tool_miss_counts"]
            lines.append(f"| {org} / {cell} | {both('candidate_coverage')} | "
                         f"{both('detect_at_1')} | {both('detect_at_3')} | "
                         f"{both('repair_success')} | {both('repair_cost')} | "
                         f"{miss['full']} / {miss['off']} |")
    lines += ["", "Failed or missing runs remain in the 30-run denominator. See JSON and per-cell logs.",
              "No claim of independent cross-organization or new-time validity follows from these results.",
              "The original postprocessor misread embedded Unicode line separators in valid JSONL strings. "
              "The saved cell files were reparsed using literal newline delimiters; no cell was rerun.",
              "The full cached runs did not capture generated natural-language report summaries.", ""]
    REPORT.write_text("\n".join(lines))


def main() -> int:
    lock, grid_rows, orgs = preflight()
    if "--preflight" in sys.argv[1:]:
        print(json.dumps({"status": "PREFLIGHT_PASS_NO_OUTCOMES_READ",
                          "organizations": orgs, "fault_cells": len(grid_rows),
                          "intended_runs": lock["intended_runs"]}, sort_keys=True))
        return 0
    cells, modes = lock["cells"], lock["modes"]
    by_key = {(r["org"], r["cell"]): r for r in grid_rows}
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = D / "_audit" / f"e_a_offline_recheck_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(REPO), REPLAY_MODE="true",
               LLM_CACHE_ENABLED="true",
               REDIS_URL=f"redis://127.0.0.1:{REDIS_PORT}",
               NEO4J_URI=f"bolt://127.0.0.1:{NEO4J_PORT}",
               QDRANT_URL=f"http://127.0.0.1:{QDRANT_PORT}", DYMAM_API_URL="")
    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    qdrant_id: str | None = None
    try:
        qdrant_id = start_container(QDRANT_NAME, ["-p", f"127.0.0.1:{QDRANT_PORT}:6333",
                                                    "qdrant/qdrant:latest"])
        for mode in modes:
            for org in orgs:
                for cell in cells:
                    key = (mode, org, cell)
                    out = root / mode / org / cell.replace("@", "_")
                    out.mkdir(parents=True, exist_ok=False)
                    source = None if cell == "clean" else by_key[(org, cell)]
                    inode = None if source is None else source["inject_node"]
                    clean_source = by_key[(org, "F4@dns")]["clean_dir"]
                    frozen = clean_source if source is None else source["faulted_dir"]
                    row: dict[str, Any] = {"status": "blocked_or_failed", "out_dir": str(out)}
                    results[key] = row
                    redis_proc: subprocess.Popen | None = None
                    neo4j_id: str | None = None
                    print(f"RUN {mode} {org} {cell}", flush=True)
                    try:
                        redis_proc = start_redis(out / "redis")
                        neo4j_id = start_container(NEO4J_NAME, [
                            "-p", f"127.0.0.1:{NEO4J_PORT}:7687",
                            "-e", "NEO4J_AUTH=neo4j/dymam2026",
                            "-e", "NEO4J_server_memory_heap_max__size=512m",
                            "-e", "NEO4J_server_memory_pagecache_size=512m",
                            "neo4j:5-community",
                        ])
                        wait_neo4j()
                        args = [sys.executable, "-m",
                                "experiments.exp_arcs.revision_v2.e_a_offline_recheck_cell",
                                str(out), org, cell, json.dumps(inode),
                                str(prior.resolve(frozen)), mode]
                        with (out / "pipeline.log").open("w") as log:
                            proc = subprocess.run(args, cwd=REPO, env=env, stdout=log,
                                                  stderr=subprocess.STDOUT, timeout=420)
                        row["exit_code"] = proc.returncode
                        row["status"] = "ok" if proc.returncode == 0 else "blocked_or_failed"
                        result_file = out / "cell_result.json"
                        if result_file.is_file():
                            row["result"] = json.loads(result_file.read_text())
                        trace = trace_rows(out / "stage_trace.jsonl")
                        graph = next((x for x in trace if x["stage"] == "graph_writer"), {})
                        neo_errors = [e for e in graph.get("stage_errors", [])
                                      if e.get("tool") == "neo4j"]
                        row["neo4j_projection_succeeded"] = bool(
                            graph.get("graph_stats", {}).get("nodes", 0) > 0 and not neo_errors)
                        if row.get("result", {}).get("tool_status", {}).get("live", 0):
                            row["status"] = "invalid_live_tool_call"
                    except Exception as exc:
                        row["error"] = repr(exc)
                    finally:
                        stop_container(neo4j_id)
                        stop_redis(redis_proc)
                    print(f"DONE {mode} {org} {cell}: {row['status']}", flush=True)
    finally:
        stop_container(qdrant_id)

    evaluate_faults(results, by_key)
    pairs: dict[str, dict] = {}
    for org in orgs:
        for cell in cells:
            full = results[("full_cached", org, cell)]
            off = results[("global_llm_disabled", org, cell)]
            key = f"{org}|{cell}"
            pair: dict[str, Any] = {"status": "ok" if full["status"] == off["status"] == "ok"
                                     else "incomplete", "full_status": full["status"],
                                     "off_status": off["status"]}
            if pair["status"] == "ok":
                a = prior.comparison_fields(full["result"], cell)
                b = prior.comparison_fields(off["result"], cell)
                if cell != "clean":
                    a["offline_evaluation"] = full["result"].get("offline_evaluation")
                    b["offline_evaluation"] = off["result"].get("offline_evaluation")
                pair["formal_differences"] = sorted(k for k in a if a[k] != b[k])
                pair["stage_trace"] = stage3._stage_comparison(
                    Path(full["out_dir"]), Path(off["out_dir"]))
                pair["full_evaluation"] = full["result"].get("offline_evaluation")
                pair["off_evaluation"] = off["result"].get("offline_evaluation")
            pairs[key] = pair
    run_rows = list(results.values())
    live = sum(r.get("result", {}).get("tool_status", {}).get("live", 0) for r in run_rows)
    success = sum(r["status"] == "ok" for r in run_rows)
    complete = sum(p["status"] == "ok" for p in pairs.values())
    evaluation_ok = all(r.get("result", {}).get("offline_evaluation") is not None
                        for (mode, org, cell), r in results.items() if cell != "clean")
    status = ("COMPLETE_EXISTING_CORPUS_RECHECK" if success == 30 and complete == 15 and
              live == 0 and evaluation_ok and
              all(r.get("neo4j_projection_succeeded") for r in run_rows)
              else "INCOMPLETE_RECHECK")
    doc = {"experiment": "E-A retrospective offline recheck", "status": status,
           "executed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "run_root": str(root), "selection_lock_sha256": sha(LOCK),
           "offline_cell_sha256": sha(HERE / "e_a_offline_recheck_cell.py"),
           "runner_sha256": sha(Path(__file__)),
           "organizations": orgs, "cells": cells, "modes": modes,
           "successful_runs": success, "complete_pairs": complete,
           "live_tool_calls": live, "evaluation_complete": evaluation_ok,
           "runs": {"|".join(k): v for k, v in results.items()},
           "pairs": pairs}
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    write_report(doc)
    print(f"STATUS {status}", flush=True)
    return 0 if status == "COMPLETE_EXISTING_CORPUS_RECHECK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
