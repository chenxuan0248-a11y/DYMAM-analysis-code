"""Run an E-A cell with read-only per-agent state checkpoints.

Uses the original Stage-2 cell runner and its strict cache/tool guards. This
wrapper changes only observation: it records each agent's returned state after
the node executes and never changes the node's returned value.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from experiments.exp_arcs.revision_v2 import llm_dependency_cell as cell
from agents._base import BaseAgent


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _asset_view(assets: Any) -> dict[str, Any]:
    view: dict[str, Any] = {}
    for host, item in sorted((assets or {}).items()):
        asset = _plain(item)
        # Wall-clock timestamps and tracing provenance do not encode the
        # detector's input. All substantive fields remain available for diff.
        for key in ("first_seen", "last_seen", "source_agents"):
            asset.pop(key, None)
        view[str(host)] = asset
    return view


def _ledger_view(value: Any) -> list[dict[str, Any]]:
    out = []
    for item in value or []:
        row = _plain(item)
        if isinstance(row, dict):
            for key in ("run_id", "observed_at", "created_at", "updated_at"):
                row.pop(key, None)
        out.append(row)
    return out


TRACE = Path(sys.argv[1]) / "stage_trace.jsonl"
_original_run = BaseAgent.run
_counter = [0]


def _traced_run(self: BaseAgent, state: dict[str, Any]) -> dict[str, Any]:
    start_seq = cell._SEQ[0]
    out = _original_run(self, state)
    _counter[0] += 1
    assets = _asset_view(out.get("assets", state.get("assets", {})))
    ledgers = {}
    for key in ("evidence_ledger", "claim_ledger", "decision_ledger", "report_items"):
        if key in out or key in state:
            records = _ledger_view(out.get(key, state.get(key)))
            ledgers[key] = {"count": len(records), "sha256": _hash(records)}
    technology = {
        host: asset.get("technologies", []) for host, asset in assets.items()
    }
    new_evidence = _ledger_view(out.get("evidence"))
    row = {
        "index": _counter[0], "stage": self.name,
        "tool_log_seq_start": start_seq + 1,
        "tool_log_seq_end": cell._SEQ[0],
        "n_assets": len(assets),
        "asset_hosts_sha256": _hash(sorted(assets)),
        "asset_semantic_sha256": _hash(assets),
        "technology_sha256": _hash(technology),
        "assets": assets,
        "new_evidence_count": len(new_evidence),
        "new_evidence_sha256": _hash(new_evidence),
        "new_evidence": new_evidence,
        "ledgers": ledgers,
        "graph_stats": _plain(out.get("graph_stats", {})),
        "report_summary": _plain((out.get("report") or {}).get("summary")),
        "llm_calls_in_stage": len(out.get("llm_calls") or []),
        "stage_errors": _plain(out.get("errors") or []),
    }
    with TRACE.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    return out


BaseAgent.run = _traced_run

if __name__ == "__main__":
    raise SystemExit(cell.main())
