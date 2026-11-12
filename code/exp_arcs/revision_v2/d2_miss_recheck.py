"""Revision v2 · step-1 audit A — exact-input recheck of the smoke's tool cache misses (D2).

Protocol: experiments/exp_arcs/revision_v2/STEP1_AUDIT_PROTOCOL.md §A. Read-only.

    .venv/bin/python experiments/exp_arcs/revision_v2/d2_miss_recheck.py \
        [smoke_result.json] [out.json]
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
HERE = REPO / "experiments/exp_arcs/revision_v2"
P0_CHECK = REPO / "experiments/exp_arcs/_audit/phase0_20261005T074543Z/replay/check_misses.py"
MISS_RE = re.compile(r"(\w+): no cached entry for (\{.*\}) — ")
TOOL_MODULES = ["tools.httpx_replay", "tools.favicon_hash", "tools.greynoise_lookup",
                "tools.nmap_replay", "tools.asn_lookup"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tool_instances() -> dict:
    from tools._base import ReplayableTool
    out = {}
    for m in TOOL_MODULES:
        for v in vars(importlib.import_module(m)).values():
            if isinstance(v, ReplayableTool):
                out[v.name] = v
    return out


def norm(v) -> str:
    s = str(v).strip().lower()
    s = re.sub(r"^https?://", "", s)
    return s.rstrip("/")


def evidence_inputs(bundle: Path) -> dict[str, set[str]]:
    """tool_name -> set of normalized inputs, from payload.claim = '<kind>:<input>'."""
    idx: dict[str, set[str]] = defaultdict(set)
    for line in (bundle / "evidence_event.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        claim = str((e.get("payload") or {}).get("claim", ""))
        if ":" in claim:
            idx[e.get("tool_name")].add(norm(claim.split(":", 1)[1]))
    return idx


def main() -> int:
    res_p = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "smoke_restore_result_v1.json"
    out_p = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "d2_miss_recheck_v1.json"
    smoke = json.loads(res_p.read_text())
    spec = importlib.util.spec_from_file_location("check_misses", P0_CHECK)
    cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
    tools = tool_instances()

    records, key_check = [], {"runs": 0, "runs_key_multiset_equal": 0, "mismatch_runs": []}
    idx_cache: dict[str, dict] = {}
    old_idx_cache: dict[str, dict] = {}
    for run in smoke["runs"]:
        rd = Path(run["dir"])
        frozen = json.loads((rd / "bundle_ref.json").read_text())["frozen_bundle_dir"]
        if frozen not in idx_cache:
            idx_cache[frozen] = evidence_inputs(Path(frozen))
            old_idx_cache[frozen] = cm.evidence_index(Path(frozen))
        idx, old_idx = idx_cache[frozen], old_idx_cache[frozen]

        misses = [json.loads(l) for l in (rd / "miss_log.jsonl").read_text().splitlines()
                  if l.strip() and '"tool_miss"' in l]
        logged_keys = Counter(c["cache_key"] for c in map(json.loads, (rd / "tool_calls.jsonl").read_text().splitlines())
                              if c.get("status") == "miss")
        recomputed: Counter = Counter()
        per_key: dict[tuple, dict] = {}
        for m in misses:
            mm = MISS_RE.match(m["detail"])
            tool, args = mm.group(1), ast.literal_eval(mm.group(2))
            key = tools[tool]._cache_key(**args)
            recomputed[key] += 1
            (arg_name, arg_val), = args.items()
            inp = norm(arg_val)
            if (tool, key) in per_key:
                per_key[(tool, key)]["count"] += 1
                continue
            new_hit = inp in idx.get(tool, set())
            old = cm.explain({"kind": "tool_miss", "detail": m["detail"]}, old_idx)
            per_key[(tool, key)] = {
                "pass": run["pass"], "org": run["org"], "cell": run["cell"],
                "tool": tool, "arg": arg_name, "input": inp, "cache_key": key, "count": 1,
                "frozen_bundle": Path(frozen).name,
                "frozen_tool_evidence_inputs": len(idx.get(tool, set())),
                "old_explained": old["explained"], "old_reason": old["reason"],
                "new_explained": not new_hit,
                "new_reason": (f"frozen bundle has {tool} evidence for input {inp}" if new_hit else
                               f"frozen bundle has no {tool} evidence for input {inp} "
                               f"({len(idx.get(tool, set()))} other {tool} inputs present)"),
            }
        key_check["runs"] += 1
        if recomputed == logged_keys:
            key_check["runs_key_multiset_equal"] += 1
        else:
            key_check["mismatch_runs"].append(run["key"])
        records += per_key.values()

    by_tool: dict = {}
    for r in records:
        d = by_tool.setdefault(r["tool"], Counter())
        d["miss_calls"] += r["count"]; d["unique_run_keys"] += 1
        d["old_unexplained_calls"] += r["count"] * (not r["old_explained"])
        d["new_unexplained_calls"] += r["count"] * (not r["new_explained"])
    tot = Counter()
    for d in by_tool.values():
        tot.update(d)
    flipped = [r for r in records if r["old_explained"] != r["new_explained"]]
    out = {
        "protocol": "experiments/exp_arcs/revision_v2/STEP1_AUDIT_PROTOCOL.md §A",
        "inputs": {"smoke_result": str(res_p.relative_to(REPO)), "smoke_result_sha256": sha256_file(res_p),
                   "protocol_sha256": sha256_file(HERE / "STEP1_AUDIT_PROTOCOL.md"),
                   "script_sha256": sha256_file(Path(__file__)), "phase0_checker_sha256": sha256_file(P0_CHECK),
                   "work_root": smoke["work_root"]},
        "cache_key_crosscheck": key_check,
        "summary": {"total": dict(tot), "by_tool": {t: dict(v) for t, v in sorted(by_tool.items())},
                    "n_records_old_vs_new_differ": len(flipped),
                    "flipped_distinct_inputs": sorted({(r["org"], r["tool"], r["input"]) for r in flipped})},
        "records": sorted(records, key=lambda r: (r["pass"], r["org"], r["cell"], r["tool"], r["input"])),
    }
    out_p.write_text(json.dumps(out, indent=1, sort_keys=True, default=list))
    print(json.dumps({"total": dict(tot), "key_crosscheck": {k: v for k, v in key_check.items() if k != "mismatch_runs"},
                      "flipped_inputs": out["summary"]["flipped_distinct_inputs"]}, default=list))
    print(f"wrote {out_p}  sha256={sha256_file(out_p)}")
    return 0 if tot["new_unexplained_calls"] == 0 and not key_check["mismatch_runs"] else 1


if __name__ == "__main__":
    sys.exit(main())
