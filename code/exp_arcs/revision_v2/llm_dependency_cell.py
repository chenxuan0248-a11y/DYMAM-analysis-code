"""Run one E-A Stage-2 replay cell in a fresh process.

No live LLM call is allowed.  Non-report LLM cache misses terminate with exit 10.
Reporting-summary behavior is selected by ``mode``.

argv: OUT ORG CELL INJECT_NODE_JSON_OR_NULL FROZEN_BUNDLE MODE
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

OUT = Path(sys.argv[1])
ORG = sys.argv[2]
CELL = sys.argv[3]
INODE = json.loads(sys.argv[4])
FROZEN = Path(sys.argv[5])
MODE = sys.argv[6]
VALID_MODES = {
    "full_cached", "report_disabled", "report_template",
    "report_perturbed", "global_llm_disabled",
}
if MODE not in VALID_MODES:
    raise SystemExit(f"invalid mode: {MODE}")

OUT.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()
_TOOL_LOG = (OUT / "tool_calls.jsonl").open("w")
_MISS_LOG = (OUT / "miss_log.jsonl").open("w")
_SEQ = [0]


def _emit(fh, rec: dict) -> None:
    with _LOCK:
        _SEQ[0] += 1
        fh.write(json.dumps({"seq": _SEQ[0], **rec}, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _kw_hash(kwargs: dict) -> str:
    return hashlib.sha256(json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest()[:16]


# Strict LLM guard.  The template/perturbation paths are local deterministic
# replacements, not model calls.
import core.llm_router as llm_router  # noqa: E402
from core.exceptions import LLMError  # noqa: E402
from core.prompts import get_prompt  # noqa: E402

_orig_complete = llm_router.LLMRouter._complete
_SUMMARY_SYSTEM = get_prompt("reasoning.report_summary", "v1")["system"]


def _is_report_summary(messages) -> bool:
    return bool(messages) and str(messages[0].get("content", "")).startswith(_SUMMARY_SYSTEM)


def _guarded_complete(self, model, messages, temperature, max_tokens, role,
                      json_mode=False, prompt_version="v1"):
    key = self._cache_key(model, messages, temperature, max_tokens, json_mode)
    summary = role == "reasoning" and _is_report_summary(messages)
    if summary and MODE == "report_disabled":
        _emit(_MISS_LOG, {"kind": "report_disabled", "key": key, "model": model, "role": role})
        raise LLMError("E-A reporting LLM disabled; use coded stats-only fallback")
    if summary and MODE in {"report_template", "report_perturbed"}:
        text = ("FIXED_TEMPLATE_REPORT_SUMMARY" if MODE == "report_template"
                else "PERTURBED_REPORT_TEXT_SENTINEL_9f6c4b1a")
        self.calls.append({
            "model": "E-A-local-replacement", "role": role, "provider": "none",
            "prompt_version": prompt_version, "prompt_tokens": 0,
            "completion_tokens": 0, "cost_usd": 0.0, "latency_s": 0.0,
            "cached": False, "ea_mode": MODE,
        })
        _emit(_MISS_LOG, {"kind": MODE, "key": key, "model": model, "role": role})
        return json.dumps({"summary": text})
    if self._cache_get(key) is None:
        _emit(_MISS_LOG, {"kind": "llm_cache_miss_blocked", "key": key,
                          "model": model, "role": role, "summary": summary})
        os._exit(10)
    return _orig_complete(self, model, messages, temperature, max_tokens, role,
                          json_mode, prompt_version)


llm_router.LLMRouter._complete = _guarded_complete


# Tool observer.  Existing agent fallback behavior is preserved; all misses and
# errors are recorded.  This audit does not turn a replay miss into a live call.
import tools._base as tbase  # noqa: E402

_orig_call = tbase.ReplayableTool.__call__


def _observed_call(self, *, mode=None, **kwargs):
    rec = {
        "tool": self.name,
        "kwargs_hash": _kw_hash(kwargs),
        "local": bool(self.local),
        "cache_key": None if self.local else self._cache_key(**kwargs),
    }
    try:
        res = _orig_call(self, mode=mode, **kwargs)
    except tbase.ReplayMiss as exc:
        _emit(_TOOL_LOG, {**rec, "status": "miss"})
        _emit(_MISS_LOG, {"kind": "tool_miss", "tool": self.name, "detail": str(exc)[:300]})
        raise
    except Exception as exc:
        _emit(_TOOL_LOG, {**rec, "status": "error", "error": type(exc).__name__})
        raise
    _emit(_TOOL_LOG, {
        **rec,
        "status": "local" if self.local else ("hit" if res.cache_hit else "live"),
        "mode": res.mode,
    })
    return res


tbase.ReplayableTool.__call__ = _observed_call

from core.graph import run_pipeline  # noqa: E402
from core.state import RunConfig  # noqa: E402
from experiments.exp_arcs.contamination import _RECORD_FILES, _index  # noqa: E402
from experiments.exp_arcs.detector_eval import (  # noqa: E402
    detector_localize, manifest_dropped_hosts, peer_anomaly_hosts,
)
from experiments.exp_arcs.detector_pr_roc import _host_scores  # noqa: E402
from experiments.exp_arcs.interceptors_local import inject_local  # noqa: E402
from experiments.exp_arcs.local_oracle import bundle_fingerprint  # noqa: E402
from experiments.exp_arcs.recorder import record_run  # noqa: E402


def _inject_kwargs(cell: str, inode: list) -> dict:
    fault = "F10" if cell.startswith("F10") else "F4"
    kind, who, *fam = inode
    out = {"fault_id": fault, ("target" if kind == "target" else "targets"): who}
    if fam:
        out["family"] = fam[0]
    return out


def _normalized_slice_hashes(bundle: Path) -> dict[str, dict]:
    out = {}
    for rtype, (filename, id_field) in sorted(_RECORD_FILES.items()):
        idx = _index(bundle, filename, id_field)
        payload = json.dumps(idx, sort_keys=True, separators=(",", ":")).encode()
        out[rtype] = {"rows": len(idx), "sha256": hashlib.sha256(payload).hexdigest()}
    return out


def _as_dict(item):
    return item.model_dump() if hasattr(item, "model_dump") else dict(item)


def main() -> int:
    t0 = time.perf_counter()
    try:
        cfg = RunConfig(organization=ORG, llm_enabled=(MODE != "global_llm_disabled"))
        if INODE is None:
            state = run_pipeline(organization=ORG, config=cfg)
        else:
            with inject_local(**_inject_kwargs(CELL, INODE)) as info:
                want = sorted(INODE[1]) if INODE[0] == "targets" else [INODE[1]]
                assert sorted(info["targets"]) == want, (info["targets"], INODE)
                assert info["family"] == (INODE[2] if len(INODE) > 2 else None), (info, INODE)
                state = run_pipeline(organization=ORG, config=cfg)
        pipeline_s = time.perf_counter() - t0
        bundle = record_run(state, raw_root=OUT / "bundles")
    except Exception:
        (OUT / "error.txt").write_text(traceback.format_exc())
        _TOOL_LOG.close()
        _MISS_LOG.close()
        return 20

    _TOOL_LOG.close()
    _MISS_LOG.close()
    calls = [json.loads(line) for line in (OUT / "tool_calls.jsonl").read_text().splitlines() if line.strip()]
    misses = [json.loads(line) for line in (OUT / "miss_log.jsonl").read_text().splitlines() if line.strip()]
    llm_calls = list(state.get("llm_calls") or [])
    report = dict(state.get("report") or {})
    evidence = [_as_dict(row) for row in (state.get("evidence_ledger") or [])]
    evidence_signature = [{
        "asset": row.get("asset"), "claim": row.get("claim"),
        "source": row.get("source"), "source_family": row.get("source_family"),
        "confidence": row.get("confidence"),
    } for row in evidence]
    evidence_sha = hashlib.sha256(json.dumps(
        evidence_signature, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

    scores = _host_scores(bundle)
    flagged = (
        detector_localize(bundle)["flagged_hosts"]
        | peer_anomaly_hosts(bundle, min_base_rate=0.5)
        | manifest_dropped_hosts(bundle)
    )
    summary = {
        "org": ORG,
        "cell": CELL,
        "mode": MODE,
        "inject_node": INODE,
        "pipeline_seconds": round(pipeline_s, 3),
        "bundle_dir": str(bundle),
        "bundle_fingerprint": bundle_fingerprint(bundle),
        "frozen_bundle": str(FROZEN),
        "frozen_bundle_fingerprint": bundle_fingerprint(FROZEN),
        "matches_frozen_bundle": bundle_fingerprint(bundle) == bundle_fingerprint(FROZEN),
        "formal_slices": _normalized_slice_hashes(bundle),
        "asset_hosts": sorted((state.get("assets") or {}).keys()),
        "n_assets": len(state.get("assets") or {}),
        "n_evidence_ledger": len(state.get("evidence_ledger") or []),
        "n_claim_ledger": len(state.get("claim_ledger") or []),
        "n_decision_ledger": len(state.get("decision_ledger") or []),
        "n_report_items": len(state.get("report_items") or []),
        "evidence_semantic_sha256": evidence_sha,
        "report_summary": report.get("summary"),
        "report_summary_present": bool(report.get("summary")),
        "report_stats": {k: report.get(k) for k in (
            "n_assets", "n_resolved", "n_with_services", "n_unique_cves",
            "n_kev", "avg_confidence", "policy_blocked",
        )},
        "combined_candidate_hosts": sorted(scores),
        "combined_scores_sha256": hashlib.sha256(json.dumps(
            scores, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "combined_flagged_hosts": sorted(flagged),
        "llm_calls": len(llm_calls),
        "llm_cached": sum(1 for row in llm_calls if row.get("cached")),
        "llm_call_roles": dict(sorted(Counter(str(row.get("role")) for row in llm_calls).items())),
        "tool_status": dict(sorted(Counter(str(row.get("status")) for row in calls).items())),
        "tool_calls": len(calls),
        "miss_kinds": dict(sorted(Counter(str(row.get("kind")) for row in misses).items())),
    }
    (OUT / "cell_result.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
