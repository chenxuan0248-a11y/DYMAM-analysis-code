"""P1-B new-site study — run controller (STEP4_NEW_SITE_PROTOCOL_v1.1.md section 4).

Reuses smoke_restore.py's isolated services, restore, state fingerprint, preflight and
main-data guard, and smoke_cell.py to run each cell in a fresh process.

Per organization: restore -> clean preflight, compared with the org's frozen clean bundle
(bundle fingerprint, rendered report without run_id, five ledgers via replay_equal). An org
whose preflight does not match is excluded as a technical failure.
Per site: restore -> F4 at (host, family) -> technical-validity checks:
  restore fingerprint, exit code (10 = LLM strict miss, 20 = pipeline/record error),
  tool-miss set subset of the org's clean-preflight miss set, injection effective
  (no (host, family) evidence left). Collateral change of other (host, family) evidence
  counts is recorded, not treated as failure. No resampling; no live fallback.

    .venv/bin/python experiments/exp_arcs/revision_v2/p1b_run.py <work_root> <result.json> --role pilot|formal

Requires Docker running with NO dymam-* container up (smoke_restore.preflight enforces it).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_restore as sr  # noqa: E402  (sets the isolated-service environment on import)

REPO = sr.REPO
PROTOCOL = HERE / "STEP4_NEW_SITE_PROTOCOL_v1.1.md"
SITES = HERE / "p1b_site_list_v1.json"
GRID = REPO / "experiments/exp_arcs/e3_grid_v1.json"
CELL_PY = HERE / "p1b_cell.py"     # smoke_cell.py plus the report-summary fallback (protocol v1.1)


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def miss_set(rd: Path) -> set:
    """Tool misses only; LLM misses and report-summary fallbacks are counted separately."""
    return {(m["kind"], m.get("tool"), m.get("detail")) for m in jl(rd / "miss_log.jsonl") if m["kind"] == "tool_miss"}


def ev_counts(bundle: Path) -> Counter:
    return Counter((e["target"], e.get("source_family")) for e in jl(bundle / "evidence_event.jsonl"))


def run_one(rd: Path, org: str, cell: str, inode, frozen: str | None, svc: Path, redis_base, base_fp,
            gate_base, env) -> dict:
    t0 = time.time()
    sr.restore_redis(svc); sr.restore_neo4j(svc); sr.restore_qdrant(svc)
    before = sr.state_fp(redis_base)
    rec = {"org": org, "cell": cell, "inject_node": inode, "dir": str(rd),
           "restore_ok": gate_base(before) == gate_base(base_fp), "restore_s": round(time.time() - t0, 1)}
    rd.mkdir(parents=True)
    (rd / "state_before.json").write_text(json.dumps(before, indent=1, sort_keys=True))
    if not rec["restore_ok"]:
        rec["exit"] = "restore_fail"
        return rec
    with (rd / "console.txt").open("w") as con:
        p = subprocess.run([str(sr.PY), str(CELL_PY), str(rd), org, cell, json.dumps(inode), frozen or "-"],
                           cwd=REPO, env=env, stdout=con, stderr=subprocess.STDOUT, timeout=1800)
    rec["exit"] = p.returncode
    after = sr.state_fp(redis_base)
    (rd / "state_after.json").write_text(json.dumps(after, indent=1, sort_keys=True))
    rec["llm_miss"] = sum(1 for m in jl(rd / "miss_log.jsonl") if m["kind"] == "llm_miss")
    rec["summary_fallback"] = sum(1 for m in jl(rd / "miss_log.jsonl") if m["kind"] == "llm_summary_fallback")
    if (rd / "bundle_ref.json").exists():
        rec["bundle_ref"] = json.loads((rd / "bundle_ref.json").read_text())
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("work_root"); ap.add_argument("result")
    ap.add_argument("--role", choices=("pilot", "formal"), required=True)
    ap.add_argument("--orgs", default="", help="optional subset of the role's orgs")
    ap.add_argument("--force-summary-fallback", action="store_true",
                    help="equivalence check only: force the report-summary fallback even on cache hits")
    args = ap.parse_args()
    work = Path(args.work_root).expanduser().resolve()
    if work.exists():
        raise SystemExit(f"STOP: {work} exists")
    work.mkdir(parents=True)
    log = (work / "controller.log").open("a")
    say = lambda m: (print(m, flush=True), log.write(m + "\n"), log.flush())

    site_doc = json.loads(SITES.read_text())
    sites = [s for s in site_doc["sites"] if s["role"] == args.role]
    if args.orgs:
        keep = set(args.orgs.split(","))
        sites = [s for s in sites if s["org"] in keep]
    orgs = sorted({s["org"] for s in sites})
    clean_of = {}
    for c in json.loads(GRID.read_text())["cells"]:
        clean_of.setdefault(c["org"], c["clean_dir"])

    meta = sr.preflight()
    meta.update({"step4_protocol_sha256": sr.sha256_file(PROTOCOL), "site_list_sha256": sr.sha256_file(SITES),
                 "p1b_run_sha256": sr.sha256_file(Path(__file__)), "role": args.role})
    say(f"preflight: neo4j_base_agg_match={meta['neo4j_base_agg_match']} head={meta['git']['head'][:8]} "
        f"step4={meta['step4_protocol_sha256'][:12]} sites={meta['site_list_sha256'][:12]} role={args.role}")
    if not meta["neo4j_base_agg_match"]:
        raise SystemExit("STOP: Neo4j baseline aggregate hash mismatch")
    guard_before = sr.main_guard()
    svc = work / "services"
    sr.restore_redis(svc)
    redis_base = sr.redis_lines(sr.ISO_ENV["REDIS_URL"])
    sr.restore_neo4j(svc); sr.restore_qdrant(svc)
    base_fp = sr.state_fp(redis_base)
    gate_base = lambda s: (s["redis_content_digest"], {k: v for k, v in s["neo4j"].items() if k != "node_prop_digest"},
                           s["qdrant"], s["neo4j"]["node_prop_digest"])
    env = {**os.environ, **sr.ISO_ENV}
    if args.force_summary_fallback:
        env["P1B_FORCE_SUMMARY_FALLBACK"] = "1"
    meta.update({"cell_py_sha256": sr.sha256_file(CELL_PY), "force_summary_fallback": args.force_summary_fallback})
    from experiments.exp_arcs.replay_equal import replay_equal

    preflights, cells = {}, []
    for org in orgs:
        frozen = REPO / clean_of[org]
        rd = work / org / "clean"
        pre = run_one(rd, org, "clean", None, str(frozen), svc, redis_base, base_fp, gate_base, env)
        ok = pre["restore_ok"] and pre["exit"] == 0 and pre.get("llm_miss", 0) == 0 and "bundle_ref" in pre
        if ok:
            new = Path(pre["bundle_ref"]["bundle_dir"])
            eq, diffs = replay_equal(frozen, new)
            pre.update({"replay_equal": eq, "replay_equal_diffs": diffs[:5] if diffs else [],
                        "fingerprint_equal": pre["bundle_ref"]["bundle_fingerprint"] == pre["bundle_ref"]["frozen_bundle_fingerprint"],
                        "report_equal": sr._report_wo_runid(frozen) == sr._report_wo_runid(new)})
            ok = eq and pre["fingerprint_equal"] and pre["report_equal"]
        pre["preflight_pass"] = bool(ok)
        preflights[org] = pre
        say(f"[preflight] {org}: pass={pre['preflight_pass']} exit={pre.get('exit')} {pre.get('wall_s')}s")
        base_miss = miss_set(rd)
        clean_ev = ev_counts(frozen)
        for s in [x for x in sites if x["org"] == org]:
            cell = f"F4@{s['family']}"
            rec = {"site_id": s["site_id"], "stratum": s["stratum"], "host": s["host"], "family": s["family"]}
            if not pre["preflight_pass"]:
                rec.update({"status": "org_excluded_preflight"}); cells.append(rec); continue
            rd = work / org / s["stratum"] / s["host"]
            r = run_one(rd, org, cell, s["inject_node"], None, svc, redis_base, base_fp, gate_base, env)
            rec.update({k: r.get(k) for k in ("exit", "restore_ok", "llm_miss", "summary_fallback", "wall_s", "dir")})
            if not r["restore_ok"]:
                status = "restore_fail"
            elif r["exit"] == 10:
                status = "llm_miss"
            elif r["exit"] != 0 or "bundle_ref" not in r:
                status = "pipeline_error"
            else:
                extra = miss_set(rd) - base_miss
                bdir = Path(r["bundle_ref"]["bundle_dir"])
                fev = ev_counts(bdir)
                left = fev.get((s["host"], s["family"]), 0)
                coll = {f"{h}|{f}": [clean_ev.get((h, f), 0), fev.get((h, f), 0)]
                        for (h, f) in set(clean_ev) | set(fev)
                        if (h, f) != (s["host"], s["family"]) and clean_ev.get((h, f), 0) != fev.get((h, f), 0)}
                rec.update({"bundle_dir": str(bdir), "bundle_fingerprint": r["bundle_ref"]["bundle_fingerprint"],
                            "extra_tool_misses": sorted(map(list, extra))[:20], "n_extra_tool_misses": len(extra),
                            "injected_family_evidence_left": left, "clean_injected_family_evidence": clean_ev.get((s["host"], s["family"]), 0),
                            "collateral_changes": coll, "n_collateral_changes": len(coll)})
                status = ("tool_miss_beyond_baseline" if extra else
                          "injection_not_effective" if left else "valid")
            rec["status"] = status
            cells.append(rec)
            say(f"[{s['stratum']}] {s['site_id']}: {status} collateral={rec.get('n_collateral_changes')} {rec.get('wall_s')}s")

    sr.sh("docker", "rm", "-f", sr.C_NEO, sr.C_QD, check=False)
    sr.sh("redis-cli", "-p", str(sr.PORTS["redis"]), "shutdown", "nosave", check=False)
    guard_after = sr.main_guard()
    res = {"protocol": str(PROTOCOL.relative_to(REPO)), "meta": meta, "work_root": str(work),
           "main_guard_before": guard_before, "main_guard_after": guard_after,
           "isolation_ok": guard_before == guard_after,
           "preflights": preflights, "cells": cells,
           "status_counts": dict(Counter(c["status"] for c in cells)),
           "orgs_preflight_pass": sorted(o for o, p in preflights.items() if p["preflight_pass"])}
    Path(args.result).write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    say(f"DONE isolation_ok={res['isolation_ok']} status={res['status_counts']} "
        f"preflight_pass={len(res['orgs_preflight_pass'])}/{len(orgs)}")
    return 0 if res["isolation_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
