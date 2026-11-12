"""Orchestrate E-A Stage 2 over mozilla/reddit × clean/F4/F10 × LLM modes."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.exp_arcs.dag import build_dag
from experiments.exp_arcs.detector_eval import (
    detector_localize,
    manifest_dropped_hosts,
    peer_anomaly_hosts,
)
from experiments.exp_arcs.detector_pr_roc import _host_scores
from experiments.exp_arcs.revision_v2 import step2_baselines as s2

REPO = Path(__file__).resolve().parents[3]
D = REPO / "experiments/exp_arcs"
HERE = D / "revision_v2"
GRID = D / "e3_grid_v1.json"
OUT = HERE / "llm_dependency_stage2_v1.json"
RESULT = HERE / "LLM_DEPENDENCY_STAGE2_RESULT.md"
PROTOCOL = HERE / "llm_dependency_protocol.md"
ORGS = ("mozilla", "reddit")
CELLS = ("clean", "F4@dns", "F10-local")
MODES = (
    "full_cached",
    "report_disabled",
    "report_template",
    "report_perturbed",
    "global_llm_disabled",
)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else REPO / p


def selected_invocations(clean: Path, flagged: set[str]) -> list[list[str]]:
    dag = build_dag(clean)
    inv = {
        (str(v.get("target")), str(v.get("source_family")))
        for v in dag.nodes.values()
        if v.get("repairable") and v.get("target") in flagged
    }
    return [list(pair) for pair in sorted(inv)]


def add_repair_metrics(run: dict[str, Any], generated: dict[tuple[str, str, str], dict],
                       grid_by_key: dict[tuple[str, str], dict]) -> None:
    for mode in MODES:
        for org in ORGS:
            clean_row = generated.get((mode, org, "clean"))
            if not clean_row or clean_row.get("status") != "ok":
                continue
            clean = Path(clean_row["result"]["bundle_dir"])
            for cell_name in CELLS[1:]:
                fault_row = generated.get((mode, org, cell_name))
                if not fault_row or fault_row.get("status") != "ok":
                    continue
                faulted = Path(fault_row["result"]["bundle_dir"])
                cell = dict(grid_by_key[(org, cell_name)])
                cell["clean_dir"] = str(clean)
                cell["faulted_dir"] = str(faulted)
                scores = _host_scores(faulted)
                flagged = (
                    detector_localize(faulted)["flagged_hosts"]
                    | peer_anomaly_hosts(faulted, min_base_rate=0.5)
                    | manifest_dropped_hosts(faulted)
                )
                repair = s2.repair(cell, frozenset(flagged))
                fault_row["result"]["stage2_evaluation"] = {
                    "candidate_hosts": sorted(scores),
                    "score_sha256": hashlib.sha256(json.dumps(
                        scores, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    "flagged_hosts": sorted(flagged),
                    "selected_invocations": selected_invocations(clean, flagged),
                    "success": bool(repair["success"]),
                    "cost": float(repair["cost"]),
                    "cost_over_Cstar": float(repair["cost"]) / float(cell["C_star"]),
                }


def comparison_fields(result: dict[str, Any], cell: str) -> dict[str, Any]:
    fields = {
        "bundle_fingerprint": result.get("bundle_fingerprint"),
        "evidence_semantic_sha256": result.get("evidence_semantic_sha256"),
        "asset_hosts": result.get("asset_hosts"),
        "combined_candidate_hosts": result.get("combined_candidate_hosts"),
        "combined_scores_sha256": result.get("combined_scores_sha256"),
        "combined_flagged_hosts": result.get("combined_flagged_hosts"),
        "formal_slices": result.get("formal_slices"),
    }
    if cell != "clean":
        fields["stage2_evaluation"] = result.get("stage2_evaluation")
    return fields


def compare(generated: dict[tuple[str, str, str], dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for org in ORGS:
        for cell in CELLS:
            key = f"{org}|{cell}"
            base_row = generated.get(("full_cached", org, cell))
            # If full_cached is blocked by a report-summary cache miss, the
            # frozen bundle remains the actual full-existing replay reference.
            baseline = None
            baseline_kind = None
            if base_row and base_row.get("status") == "ok":
                baseline = comparison_fields(base_row["result"], cell)
                baseline_kind = "full_cached_rerun"
            variants = {}
            for mode in MODES:
                row = generated.get((mode, org, cell), {})
                if row.get("status") != "ok":
                    variants[mode] = {"status": row.get("status"), "exit_code": row.get("exit_code")}
                    continue
                current = comparison_fields(row["result"], cell)
                variants[mode] = {
                    "status": "ok",
                    "equals_full_cached": current == baseline if baseline is not None else None,
                    "differing_fields": sorted(
                        k for k in current if baseline is not None and current.get(k) != baseline.get(k)
                    ),
                    "report_summary": row["result"].get("report_summary"),
                    "matches_frozen_bundle": row["result"].get("matches_frozen_bundle"),
                }
            out[key] = {"baseline": baseline_kind, "variants": variants}
    return out


def write_markdown(doc: dict[str, Any]) -> None:
    lines = [
        "# E-A LLM dependency audit — Stage 2 result",
        "",
        f"Executed: {doc['executed_at']}  ",
        f"Overall status: **{doc['status']}**",
        "",
        "## Execution matrix",
        "",
        "| Mode | Successful cells | Blocked/failed cells | Live tool calls |",
        "|---|---:|---:|---:|",
    ]
    for mode in MODES:
        rows = [r for k, r in doc["runs"].items() if k.startswith(mode + "|")]
        ok = sum(r["status"] == "ok" for r in rows)
        live = sum(r.get("result", {}).get("tool_status", {}).get("live", 0) for r in rows)
        lines.append(f"| {mode} | {ok} | {len(rows) - ok} | {live} |")
    lines.extend([
        "",
        "## Paired formal-output comparison",
        "",
        "| Organization/cell | report-disabled | fixed-template | perturbed-text | global LLM disabled |",
        "|---|---|---|---|---|",
    ])
    for key, block in doc["comparisons"].items():
        vals = []
        for mode in ("report_disabled", "report_template", "report_perturbed", "global_llm_disabled"):
            row = block["variants"][mode]
            if row.get("status") != "ok":
                vals.append(row.get("status", "missing"))
            elif row.get("equals_full_cached") is True:
                vals.append("identical")
            elif row.get("equals_full_cached") is False:
                vals.append("diff: " + ", ".join(row.get("differing_fields") or []))
            else:
                vals.append("no full baseline")
        lines.append(f"| {key} | " + " | ".join(vals) + " |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        doc["interpretation"],
        "",
        "This is a two-organization development-set diagnostic. It cannot establish cross-organization invariance by itself.",
        "",
        f"Raw result: `{OUT.relative_to(REPO)}`",
    ])
    RESULT.write_text("\n".join(lines) + "\n")


def main() -> int:
    grid = json.loads(GRID.read_text())["cells"]
    grid_by_key = {(c["org"], c["cell"]): c for c in grid if c["org"] in ORGS and c["cell"] in CELLS}
    clean_by_org = {}
    for c in grid:
        if c["org"] in ORGS:
            clean_by_org.setdefault(c["org"], c["clean_dir"])

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = D / "_audit" / f"llm_dependency_stage2_{stamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    generated: dict[tuple[str, str, str], dict] = {}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)

    for mode in MODES:
        for org in ORGS:
            for cell in CELLS:
                if cell == "clean":
                    inode = None
                    frozen = resolve(clean_by_org[org])
                else:
                    source = grid_by_key[(org, cell)]
                    inode = source["inject_node"]
                    frozen = resolve(source["faulted_dir"])
                out_dir = run_root / mode / org / cell.replace("@", "_")
                cmd = [
                    sys.executable, "-m", "experiments.exp_arcs.revision_v2.llm_dependency_cell",
                    str(out_dir), org, cell, json.dumps(inode), str(frozen), mode,
                ]
                print(f"RUN {mode} {org} {cell}", flush=True)
                proc = subprocess.run(cmd, cwd=REPO, env=env, timeout=420)
                row: dict[str, Any] = {"status": "ok" if proc.returncode == 0 else "blocked_or_failed",
                                       "exit_code": proc.returncode, "out_dir": str(out_dir)}
                result_file = out_dir / "cell_result.json"
                if result_file.exists():
                    row["result"] = json.loads(result_file.read_text())
                error_file = out_dir / "error.txt"
                if error_file.exists():
                    row["error"] = error_file.read_text()[-4000:]
                generated[(mode, org, cell)] = row

    add_repair_metrics({}, generated, grid_by_key)
    runs = {"|".join(k): v for k, v in generated.items()}
    comparisons = compare(generated)
    live_calls = sum(
        row.get("result", {}).get("tool_status", {}).get("live", 0)
        for row in runs.values()
    )
    report_modes_ok = all(
        row["status"] == "ok"
        for key, row in runs.items()
        if key.split("|", 1)[0] in {"report_disabled", "report_template", "report_perturbed"}
    )
    full_ok = all(row["status"] == "ok" for key, row in runs.items() if key.startswith("full_cached|"))
    report_identical = full_ok and all(
        block["variants"][mode].get("equals_full_cached") is True
        for block in comparisons.values()
        for mode in ("report_disabled", "report_template", "report_perturbed")
    )
    global_ok = all(row["status"] == "ok" for key, row in runs.items() if key.startswith("global_llm_disabled|"))
    global_identical = full_ok and global_ok and all(
        block["variants"]["global_llm_disabled"].get("equals_full_cached") is True
        for block in comparisons.values()
    )
    if live_calls:
        status = "INVALID_LIVE_TOOL_CALL"
    elif report_identical:
        status = "PASS_REPORT_TEXT_INVARIANT"
    elif not full_ok and report_modes_ok:
        status = "PARTIAL_FULL_CACHE_BLOCKED"
    else:
        status = "DIFFERENCE_OR_BLOCKED"
    interpretation = (
        "Reporting-mode formal outputs and evaluated decisions were identical to full cached replay in all six cells. "
        if report_identical else
        "The reporting-mode comparison was incomplete or produced a formal difference; inspect per-cell differing fields. "
    )
    if global_identical:
        interpretation += "Global LLM disablement was also identical in all six development cells."
    elif global_ok:
        interpretation += "Global LLM disablement completed but changed at least one formal field or evaluation result."
    else:
        interpretation += "Global LLM disablement did not complete in every cell."

    doc = {
        "experiment": "E-A LLM dependency audit",
        "stage": 2,
        "executed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "run_root": str(run_root),
        "input_hashes": {
            "protocol_sha256": sha_file(PROTOCOL),
            "cell_script_sha256": sha_file(HERE / "llm_dependency_cell.py"),
            "runner_script_sha256": sha_file(Path(__file__)),
            "grid_sha256": sha_file(GRID),
        },
        "organizations": list(ORGS),
        "cells": list(CELLS),
        "modes": list(MODES),
        "live_tool_calls": live_calls,
        "full_cached_complete": full_ok,
        "report_modes_complete": report_modes_ok,
        "report_modes_identical": report_identical,
        "global_llm_disabled_complete": global_ok,
        "global_llm_disabled_identical": global_identical,
        "interpretation": interpretation,
        "runs": runs,
        "comparisons": comparisons,
    }
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    write_markdown(doc)
    print(f"STATUS {status}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0 if status in {"PASS_REPORT_TEXT_INVARIANT", "PARTIAL_FULL_CACHE_BLOCKED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
