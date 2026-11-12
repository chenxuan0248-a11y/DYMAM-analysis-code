"""P1-B new-site study — run ONE cell in a fresh process (STEP4 protocol v1.1).

Identical to smoke_cell.py except for one documented change to the LLM strict guard:
a cache miss on the report agent's summary call (prompt "reasoning.report_summary") is
answered by raising core.exceptions.LLMError, so ReportAgent takes its own coded fallback
("stats-only report"). That call's output is written only to state["report"]["summary"],
which the recorder never stores (rendered_report.json is a fixed shell), so no bundle file
can change. Every other LLM cache miss still terminates the cell with exit code 10.
With P1B_FORCE_SUMMARY_FALLBACK=1 the fallback is forced even on a cache hit; this is used
only to verify that the fallback leaves bundles byte-identical.

argv: <out_dir> <org> <cell> <inject_node_json|null> <frozen_bundle_dir|->
Exit codes: 0 ok, 10 LLM cache miss (strict, any other role), 20 pipeline/record error.
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

OUT = Path(sys.argv[1]); ORG = sys.argv[2]; CELL = sys.argv[3]
INODE = json.loads(sys.argv[4]); FROZEN = None if sys.argv[5] == "-" else Path(sys.argv[5])
OUT.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()
_TOOL_LOG = (OUT / "tool_calls.jsonl").open("w")
_MISS_LOG = (OUT / "miss_log.jsonl").open("w")
_SEQ = [0]


def _emit(fh, rec: dict) -> None:
    with _LOCK:
        _SEQ[0] += 1
        fh.write(json.dumps({"seq": _SEQ[0], **rec}, sort_keys=True, default=str) + "\n")
        fh.flush(); os.fsync(fh.fileno())


def _kw_hash(kwargs: dict) -> str:
    return hashlib.sha256(json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ── LLM strict guard: a miss must never reach a live model ───────────────────
import core.llm_router as llm_router  # noqa: E402

_orig_complete = llm_router.LLMRouter._complete


from core.exceptions import LLMError  # noqa: E402
from core.prompts import get_prompt  # noqa: E402

_SUMMARY_SYSTEM = get_prompt("reasoning.report_summary", "v1")["system"]
_FORCE = os.environ.get("P1B_FORCE_SUMMARY_FALLBACK") == "1"


def _is_report_summary(messages) -> bool:
    return bool(messages) and str(messages[0].get("content", "")).startswith(_SUMMARY_SYSTEM)


def _guarded_complete(self, model, messages, temperature, max_tokens, role,
                      json_mode=False, prompt_version="v1"):
    key = self._cache_key(model, messages, temperature, max_tokens, json_mode)
    summary = role == "reasoning" and _is_report_summary(messages)
    if summary and (_FORCE or self._cache_get(key) is None):
        _emit(_MISS_LOG, {"kind": "llm_summary_fallback", "key": key, "model": model, "role": role,
                          "forced": _FORCE})
        raise LLMError("report summary: cache miss answered by the coded stats-only fallback")
    if self._cache_get(key) is None:
        _emit(_MISS_LOG, {"kind": "llm_miss", "key": key, "model": model, "role": role})
        os._exit(10)
    return _orig_complete(self, model, messages, temperature, max_tokens, role,
                          json_mode, prompt_version)


llm_router.LLMRouter._complete = _guarded_complete

# ── tool observer: one event per call attempt (hit / miss / local / error) ──
import tools._base as tbase  # noqa: E402

_orig_call = tbase.ReplayableTool.__call__


def _observed_call(self, *, mode=None, **kwargs):
    rec = {"tool": self.name, "kwargs_hash": _kw_hash(kwargs), "local": bool(self.local),
           "cache_key": None if self.local else self._cache_key(**kwargs)}
    try:
        res = _orig_call(self, mode=mode, **kwargs)
    except tbase.ReplayMiss as e:
        _emit(_TOOL_LOG, {**rec, "status": "miss"})
        _emit(_MISS_LOG, {"kind": "tool_miss", "tool": self.name, "detail": str(e)[:300]})
        raise
    except Exception as e:
        _emit(_TOOL_LOG, {**rec, "status": "error", "error": type(e).__name__})
        raise
    _emit(_TOOL_LOG, {**rec, "status": "local" if self.local else ("hit" if res.cache_hit else "live"),
                      "mode": res.mode})
    return res


tbase.ReplayableTool.__call__ = _observed_call

from core.graph import run_pipeline  # noqa: E402
from experiments.exp_arcs.interceptors_local import inject_local  # noqa: E402
from experiments.exp_arcs.local_oracle import bundle_fingerprint  # noqa: E402
from experiments.exp_arcs.recorder import record_run  # noqa: E402


def _inject_kwargs(cell: str, inode: list) -> dict:
    fault = "F10" if cell.startswith("F10") else "F4"
    kind, who, *fam = inode
    kw = {"fault_id": fault, ("target" if kind == "target" else "targets"): who}
    if fam:
        kw["family"] = fam[0]
    return kw


def main() -> int:
    t0 = time.perf_counter()
    try:
        if INODE is None:
            state = run_pipeline(organization=ORG)
        else:
            with inject_local(**_inject_kwargs(CELL, INODE)) as info:
                # compare semantics (host set + family), not the node's recorded shape:
                # run_e3 records F4-host as ['targets', [h]], inject_local as ('target', h)
                want = sorted(INODE[1]) if INODE[0] == "targets" else [INODE[1]]
                assert sorted(info["targets"]) == want, (info["targets"], INODE)
                assert info["family"] == (INODE[2] if len(INODE) > 2 else None), (info, INODE)
                state = run_pipeline(organization=ORG)
        t_pipe = time.perf_counter() - t0
        bundle = record_run(state, raw_root=OUT / "bundles")
    except Exception:
        (OUT / "error.txt").write_text(traceback.format_exc())
        return 20
    wall = time.perf_counter() - t0

    llm = list(state.get("llm_calls", []))
    with (OUT / "llm_calls.jsonl").open("w") as fh:
        for i, c in enumerate(llm):
            fh.write(json.dumps({"seq": i + 1, **c}, sort_keys=True, default=str) + "\n")
    _TOOL_LOG.close(); _MISS_LOG.close()

    calls = [json.loads(l) for l in (OUT / "tool_calls.jsonl").read_text().splitlines() if l.strip()]
    by_tool: dict = {}
    for c in calls:
        d = by_tool.setdefault(c["tool"], Counter())
        d[c["status"]] += 1
    summary = {
        "org": ORG, "cell": CELL, "inject_node": INODE,
        "pipeline_tool_call_attempts": len(calls),
        "by_tool_status": {t: dict(v) for t, v in sorted(by_tool.items())},
        "distinct_cache_keys": len({c["cache_key"] for c in calls if c["cache_key"]}),
        "llm_calls": len(llm),
        "llm_cached": sum(1 for c in llm if c.get("cached")),
        "prompt_tokens_equiv": sum(int(c.get("prompt_tokens") or 0) for c in llm),
        "completion_tokens_equiv": sum(int(c.get("completion_tokens") or 0) for c in llm),
        "replay_wall_s_pipeline": round(t_pipe, 3), "replay_wall_s_total": round(wall, 3),
    }
    (OUT / "usage_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    ref = {"bundle_dir": str(bundle), "bundle_fingerprint": bundle_fingerprint(bundle),
           "frozen_bundle_dir": str(FROZEN) if FROZEN else None,
           "frozen_bundle_fingerprint": bundle_fingerprint(FROZEN) if FROZEN else None}
    (OUT / "bundle_ref.json").write_text(json.dumps(ref, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
