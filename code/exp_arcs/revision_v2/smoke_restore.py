"""Revision v2 · three-state-restore smoke — controller.

Protocol: experiments/exp_arcs/revision_v2/SMOKE_RESTORE_PROTOCOL.md (hash recorded in the result).
Before EVERY cell: restore isolated Redis (6380, Phase 0 dump), Neo4j (17687, Phase 0 baseline
clone) and Qdrant (16333, empty), verify the state fingerprint, then run the cell in a fresh
subprocess (smoke_cell.py). Pass A runs all cells, then pass B. Finally compares A vs B.

    .venv/bin/python experiments/exp_arcs/revision_v2/smoke_restore.py <work_root> <result.json> [--orgs mozilla,reddit]

Requires Docker running with NO dymam-* container up (they mount the main data/ dirs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
P0 = REPO / "experiments/exp_arcs/_audit/phase0_20261005T074543Z"
DUMP = P0 / "redis/dump_20261005T074543Z.rdb"
DUMP_SHA = "13f757b775111d8ed6020d1f4197dfbf7d07d1cc0c40a0496e814cd70813b1a2"
NEO4J_BASE = Path.home() / "arcs-audit/neo4j_verify_20261005T074543Z"
NEO4J_BASE_AGG = (P0 / "services/neo4j_baseline.agg.sha256").read_text().split()[0]
NEO4J_IMAGE = (P0 / "services/neo4j_image_id.txt").read_text().strip()
QDRANT_IMAGE = (P0 / "services/qdrant_image_id.txt").read_text().strip()
GRID = REPO / "experiments/exp_arcs/e3_grid_v1.json"
PROTOCOL = REPO / "experiments/exp_arcs/revision_v2/SMOKE_RESTORE_PROTOCOL.md"
CELL_PY = REPO / "experiments/exp_arcs/revision_v2/smoke_cell.py"
PY = REPO / ".venv/bin/python"
PORTS = {"redis": 6380, "neo4j": 17687, "qdrant": 16333}
C_NEO, C_QD = "arcs-rv2-neo4j", "arcs-rv2-qdrant"
COUNTER_PREFIXES = ("cache:stats:", "tool:stats:")

ISO_ENV = {
    "REDIS_URL": f"redis://127.0.0.1:{PORTS['redis']}",
    "NEO4J_URI": f"bolt://127.0.0.1:{PORTS['neo4j']}",
    "QDRANT_URL": f"http://127.0.0.1:{PORTS['qdrant']}",
    "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
    "ALL_PROXY": "http://127.0.0.1:9", "NO_PROXY": "localhost,127.0.0.1,::1",
    "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9",
    "no_proxy": "localhost,127.0.0.1,::1",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPO),
}
for k, v in ISO_ENV.items():          # the controller's own clients use the isolated services too
    os.environ[k] = v


def sh(*cmd: str, check: bool = True, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), check=check, capture_output=True, text=True, **kw)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_listing(root: Path) -> str:
    """`shasum -a 256` style listing with ./-relative paths, sorted (Phase 0 format)."""
    lines = [f"{sha256_file(p)}  ./{p.relative_to(root)}"
             for p in sorted(root.rglob("*"), key=lambda q: str(q.relative_to(root)))
             if p.is_file() and not p.is_symlink()]
    return "\n".join(lines) + "\n"


# ── state fingerprint ────────────────────────────────────────────────────────
def redis_lines(url: str) -> dict[str, str]:
    import redis as _r
    r = _r.Redis.from_url(url)
    keys = sorted(k.decode("utf-8", "surrogateescape") for k in r.scan_iter(count=5000))
    out = {}
    for i in range(0, len(keys), 1000):
        ch = keys[i:i + 1000]; p = r.pipeline(transaction=False)
        for k in ch:
            p.dump(k)
        out.update({k: hashlib.sha256(d or b"").hexdigest() for k, d in zip(ch, p.execute())})
    return out


def neo4j_fp() -> dict:
    from neo4j import GraphDatabase
    from core.settings import get_settings
    s = get_settings()
    with GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)) as d, d.session() as ss:
        q = lambda c: [x.data() for x in ss.run(c)]
        nodes = q("MATCH (n) RETURN labels(n) AS l, properties(n) AS p")
        prop = hashlib.sha256("\n".join(sorted(json.dumps(x, sort_keys=True, default=str)
                                               for x in nodes)).encode()).hexdigest()
        return {
            "nodes": len(nodes),
            "rels": q("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"],
            "labels": {x["l"]: x["c"] for x in q("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c")},
            "rel_types": {x["t"]: x["c"] for x in q("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c")},
            "node_prop_digest": prop,      # descriptive only (may carry wall-clock stamps)
        }


def qdrant_fp() -> dict:
    from qdrant_client import QdrantClient
    qc = QdrantClient(url=ISO_ENV["QDRANT_URL"], timeout=30)
    return {c.name: qc.count(c.name).count for c in qc.get_collections().collections}


def state_fp(redis_base: dict[str, str] | None) -> dict:
    rl = redis_lines(ISO_ENV["REDIS_URL"])
    content = {k: v for k, v in rl.items() if not k.startswith(COUNTER_PREFIXES)}
    fp = {"redis_n_keys": len(rl),
          "redis_content_digest": hashlib.sha256("\n".join(f"{k}\t{v}" for k, v in sorted(content.items())).encode()).hexdigest(),
          "neo4j": neo4j_fp(), "qdrant": qdrant_fp()}
    if redis_base is not None:
        base_c = {k: v for k, v in redis_base.items() if not k.startswith(COUNTER_PREFIXES)}
        fp["redis_changed"] = sorted(k for k in content if k in base_c and content[k] != base_c[k])
        fp["redis_added"] = sorted(k for k in content if k not in base_c)
        fp["redis_removed"] = sorted(k for k in base_c if k not in content)
        fp["redis_changed_digests"] = {k: content[k] for k in fp["redis_changed"] + fp["redis_added"]}
    return fp


# ── restore ──────────────────────────────────────────────────────────────────
def _wait(fn, what: str, timeout: float = 120) -> None:
    t0 = time.time(); err = None
    while time.time() - t0 < timeout:
        try:
            fn(); return
        except Exception as e:      # noqa: BLE001 — readiness polling
            err = e; time.sleep(1)
    raise RuntimeError(f"{what} not ready after {timeout}s: {err}")


def restore_redis(work: Path) -> None:
    sh("redis-cli", "-p", str(PORTS["redis"]), "shutdown", "nosave", check=False)
    for _ in range(30):                 # wait until the old instance is really gone
        if sh("redis-cli", "-p", str(PORTS["redis"]), "ping", check=False).stdout.strip() != "PONG":
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("redis 6380 did not shut down")
    d = work / "redis"; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    shutil.copyfile(DUMP, d / "dump.rdb")
    sh("redis-server", "--port", str(PORTS["redis"]), "--bind", "127.0.0.1", "--dir", str(d),
       "--dbfilename", "dump.rdb", "--save", "", "--appendonly", "no", "--daemonize", "yes",
       "--logfile", str(d / "redis.log"))

    def ok():
        out = sh("redis-cli", "-p", str(PORTS["redis"]), "dbsize").stdout.strip()
        int(out)
    _wait(ok, "redis")


def restore_neo4j(work: Path) -> None:
    from core.settings import get_settings
    sh("docker", "rm", "-f", C_NEO, check=False)
    d = work / "neo4j"
    if d.exists():
        sh("chmod", "-R", "u+w", str(d)); shutil.rmtree(d)
    sh("cp", "-c", "-R", str(NEO4J_BASE), str(d))          # APFS clone of the read-only baseline
    sh("chmod", "-R", "u+w", str(d))
    s = get_settings()
    sh("docker", "run", "-d", "--name", C_NEO, "--restart", "no",
       "-p", f"127.0.0.1:{PORTS['neo4j']}:7687",
       "-e", f"NEO4J_AUTH={s.neo4j_user}/{s.neo4j_password}",
       "-e", "NEO4J_server_memory_heap_max__size=1G", "-e", "NEO4J_server_memory_pagecache_size=1G",
       "-e", 'NEO4J_PLUGINS=["apoc"]', "-e", "NEO4J_dbms_usage__report_enabled=false",
       "-v", f"{d}:/data", NEO4J_IMAGE)
    from neo4j import GraphDatabase
    def ok():
        with GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)) as drv, drv.session() as ss:
            ss.run("RETURN 1").consume()
    _wait(ok, "neo4j", 180)


def restore_qdrant(work: Path) -> None:
    sh("docker", "rm", "-f", C_QD, check=False)
    d = work / "qdrant"; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    sh("docker", "run", "-d", "--name", C_QD, "--restart", "no",
       "-p", f"127.0.0.1:{PORTS['qdrant']}:6333", "-e", "QDRANT__TELEMETRY_DISABLED=true",
       "-v", f"{d}:/qdrant/storage", QDRANT_IMAGE)
    _wait(qdrant_fp, "qdrant", 60)


# ── preflight / main-data guard ──────────────────────────────────────────────
def main_guard() -> dict:
    out = {}
    for sub in ("neo4j", "qdrant"):
        p = REPO / "data" / sub
        out[f"data_{sub}"] = hashlib.sha256(tree_listing(p).encode()).hexdigest() if p.exists() else None
    main = redis_lines("redis://127.0.0.1:6379")
    out["main_redis_n_keys"] = len(main)
    out["main_redis_content_digest"] = hashlib.sha256("\n".join(
        f"{k}\t{v}" for k, v in sorted(main.items()) if not k.startswith(COUNTER_PREFIXES)).encode()).hexdigest()
    return out


def preflight() -> dict:
    ps = sh("docker", "ps", "--format", "{{.Names}}").stdout.split()
    bad = [n for n in ps if n.startswith("dymam-")]
    if bad:
        raise SystemExit(f"STOP: main-data containers running: {bad}")
    for img in (NEO4J_IMAGE, QDRANT_IMAGE):
        if sh("docker", "image", "inspect", img, check=False).returncode != 0:
            raise SystemExit(f"STOP: image {img} not present locally (would need a pull)")
    if sha256_file(DUMP) != DUMP_SHA:
        raise SystemExit("STOP: Redis dump hash mismatch")
    agg = hashlib.sha256(tree_listing(NEO4J_BASE).encode()).hexdigest()
    git = {"head": sh("git", "-C", str(REPO), "rev-parse", "HEAD").stdout.strip(),
           # code only: the full worktree diff also carries ~500MB of staged archive moves
           "worktree_py_diff_sha256": hashlib.sha256(subprocess.run(
               ["git", "-C", str(REPO), "diff", "HEAD", "--", "*.py", "pyproject.toml", "poetry.lock"],
               check=True, capture_output=True).stdout).hexdigest()}
    return {"docker_running": ps, "neo4j_base_agg": agg, "neo4j_base_agg_expected": NEO4J_BASE_AGG,
            "neo4j_base_agg_match": agg == NEO4J_BASE_AGG, "redis_dump_sha256": DUMP_SHA,
            "neo4j_image": NEO4J_IMAGE, "qdrant_image": QDRANT_IMAGE, "git": git,
            "protocol_sha256": sha256_file(PROTOCOL), "cell_py_sha256": sha256_file(CELL_PY),
            "controller_sha256": sha256_file(Path(__file__))}


# ── comparison ───────────────────────────────────────────────────────────────
def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _report_wo_runid(bundle: Path):
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k != "run_id"}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return strip(json.loads((bundle / "rendered_report.json").read_text()))


def compare(a: Path, b: Path) -> dict:
    from experiments.exp_arcs.replay_equal import replay_equal
    ra, rb = json.loads((a / "bundle_ref.json").read_text()), json.loads((b / "bundle_ref.json").read_text())
    ba, bb = Path(ra["bundle_dir"]), Path(rb["bundle_dir"])
    eq, diffs = replay_equal(ba, bb)
    tool_ms = lambda d: Counter((c["tool"], c["cache_key"], c["kwargs_hash"], c["status"]) for c in _jsonl(d / "tool_calls.jsonl"))
    tool_seq = lambda d: [(c["tool"], c["cache_key"], c["status"]) for c in _jsonl(d / "tool_calls.jsonl")]
    llm_key = lambda c: (c.get("model"), c.get("role"), c.get("prompt_version"), c.get("cached"),
                         c.get("prompt_tokens"), c.get("completion_tokens"))
    llm_ms = lambda d: Counter(llm_key(c) for c in _jsonl(d / "llm_calls.jsonl"))
    miss = lambda d: Counter((m["kind"], m.get("detail") or m.get("key")) for m in _jsonl(d / "miss_log.jsonl"))
    sa, sb = (json.loads((x / "state_after.json").read_text()) for x in (a, b))
    gate_state = lambda s: (sorted(s.get("redis_changed", []) + s.get("redis_added", []) + s.get("redis_removed", [])),
                            {k: v for k, v in s["neo4j"].items() if k != "node_prop_digest"}, s["qdrant"])
    out = {
        "replay_equal": eq, "replay_equal_diffs": diffs[:5] if diffs else [],
        "bundle_fingerprint_equal": ra["bundle_fingerprint"] == rb["bundle_fingerprint"],
        "report_equal": _report_wo_runid(ba) == _report_wo_runid(bb),
        "tool_multiset_equal": tool_ms(a) == tool_ms(b),
        "tool_sequence_equal": tool_seq(a) == tool_seq(b),
        "llm_multiset_equal": llm_ms(a) == llm_ms(b),
        "miss_set_equal": miss(a) == miss(b),
        "state_change_equal": gate_state(sa) == gate_state(sb),
        "state_change_content_equal": sa.get("redis_changed_digests") == sb.get("redis_changed_digests")
                                      and sa["neo4j"].get("node_prop_digest") == sb["neo4j"].get("node_prop_digest"),
    }
    gate_keys = ["replay_equal", "bundle_fingerprint_equal", "report_equal", "tool_multiset_equal",
                 "llm_multiset_equal", "miss_set_equal", "state_change_equal"]
    out["R3_pass"] = all(out[k] for k in gate_keys)
    out["vs_frozen_A"] = ra["bundle_fingerprint"] == ra["frozen_bundle_fingerprint"]
    out["vs_frozen_B"] = rb["bundle_fingerprint"] == rb["frozen_bundle_fingerprint"]
    if not out["vs_frozen_A"] and ra.get("frozen_bundle_dir"):
        feq, fd = replay_equal(Path(ra["frozen_bundle_dir"]), ba)
        out["vs_frozen_diffs"] = fd[:5]
    return out


def explain_misses(run: Path, frozen: str | None) -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_misses", P0 / "replay/check_misses.py")
    cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
    ms = [m for m in _jsonl(run / "miss_log.jsonl") if m["kind"] == "tool_miss"]
    if not ms:
        return {"n": 0, "unexplained": 0, "by_tool": {}}
    idx = cm.evidence_index(Path(frozen)) if frozen else {}
    res = [cm.explain(m, idx) for m in ms]
    return {"n": len(ms), "unexplained": sum(not r["explained"] for r in res),
            "by_tool": dict(Counter(m["tool"] for m in ms))}


# ── driver ───────────────────────────────────────────────────────────────────
def cells_for(orgs: list[str]) -> list[tuple]:
    grid = json.loads(GRID.read_text())["cells"]
    out = []
    for org in orgs:
        cs = [c for c in grid if c["org"] == org]
        assert len(cs) == 9, (org, len(cs))
        out.append((org, "clean", None, cs[0]["clean_dir"]))
        out += [(org, c["cell"], c["inject_node"], c["faulted_dir"]) for c in cs]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("work_root"); ap.add_argument("result")
    ap.add_argument("--orgs", default="mozilla,reddit")
    ap.add_argument("--passes", default="A,B")
    ap.add_argument("--max-cells", type=int, default=0, help="harness shakedown only (0 = all)")
    args = ap.parse_args()
    work = Path(args.work_root).expanduser().resolve()
    if work.exists():
        raise SystemExit(f"STOP: {work} exists")
    work.mkdir(parents=True)
    result_p = Path(args.result)
    log = (work / "controller.log").open("a")
    say = lambda m: (print(m, flush=True), log.write(m + "\n"), log.flush())

    meta = preflight()
    say(f"preflight: neo4j_base_agg_match={meta['neo4j_base_agg_match']} head={meta['git']['head'][:8]} "
        f"protocol={meta['protocol_sha256'][:12]}")
    if not meta["neo4j_base_agg_match"]:
        raise SystemExit("STOP: Neo4j baseline aggregate hash mismatch")
    guard_before = main_guard()
    svc = work / "services"
    restore_redis(svc)
    redis_base = redis_lines(ISO_ENV["REDIS_URL"])
    restore_neo4j(svc); restore_qdrant(svc)
    base_fp = state_fp(redis_base)
    say(f"baseline: redis_keys={base_fp['redis_n_keys']} neo4j={base_fp['neo4j']['nodes']}/{base_fp['neo4j']['rels']} "
        f"qdrant={base_fp['qdrant']}")
    gate_base = lambda s: (s["redis_content_digest"], {k: v for k, v in s["neo4j"].items() if k != "node_prop_digest"},
                           s["qdrant"], s["neo4j"]["node_prop_digest"])

    runs: dict = {}
    child_env = {**os.environ, **ISO_ENV}
    for ps in args.passes.split(","):
        cells = cells_for(args.orgs.split(","))
        for org, cell, inode, frozen in cells[:args.max_cells or None]:
            rd = work / ps / org / cell
            t0 = time.time()
            restore_redis(svc); restore_neo4j(svc); restore_qdrant(svc)
            before = state_fp(redis_base)
            rec = {"pass": ps, "org": org, "cell": cell, "inject_node": inode, "frozen": frozen, "dir": str(rd),
                   "restore_ok": gate_base(before) == gate_base(base_fp), "restore_s": round(time.time() - t0, 1)}
            rd.mkdir(parents=True)
            (rd / "state_before.json").write_text(json.dumps(before, indent=1, sort_keys=True))
            if not rec["restore_ok"]:
                rec["exit"] = "restore_fail"; runs[(ps, org, cell)] = rec
                say(f"[{ps}] {org}/{cell}: RESTORE_FAIL"); continue
            with (rd / "console.txt").open("w") as con:
                p = subprocess.run([str(PY), str(CELL_PY), str(rd), org, cell, json.dumps(inode), frozen or "-"],
                                   cwd=REPO, env=child_env, stdout=con, stderr=subprocess.STDOUT, timeout=1800)
            rec["exit"] = p.returncode
            after = state_fp(redis_base)
            (rd / "state_after.json").write_text(json.dumps(after, indent=1, sort_keys=True))
            rec["llm_miss"] = sum(1 for m in _jsonl(rd / "miss_log.jsonl") if m["kind"] == "llm_miss")
            rec["tool_miss"] = explain_misses(rd, frozen)
            rec["redis_changed"] = after.get("redis_changed", []) + after.get("redis_added", [])
            rec["neo4j_delta"] = {"nodes": after["neo4j"]["nodes"] - before["neo4j"]["nodes"],
                                  "rels": after["neo4j"]["rels"] - before["neo4j"]["rels"]}
            rec["qdrant_after"] = after["qdrant"]
            if (rd / "usage_summary.json").exists():
                rec["usage"] = json.loads((rd / "usage_summary.json").read_text())
            if (rd / "bundle_ref.json").exists():
                ref = json.loads((rd / "bundle_ref.json").read_text())
                ref.update({"redis_dump_sha256": DUMP_SHA, "neo4j_baseline_agg": meta["neo4j_base_agg"],
                            "qdrant_baseline": "empty", "git": meta["git"], "protocol_sha256": meta["protocol_sha256"]})
                (rd / "bundle_ref.json").write_text(json.dumps(ref, indent=1, sort_keys=True))
                rec["vs_frozen"] = ref["bundle_fingerprint"] == ref["frozen_bundle_fingerprint"]
            rec["wall_s"] = round(time.time() - t0, 1)
            runs[(ps, org, cell)] = rec
            say(f"[{ps}] {org}/{cell}: exit={rec['exit']} llm_miss={rec['llm_miss']} "
                f"tool_miss={rec['tool_miss']['n']}/{rec['tool_miss']['unexplained']}unexp "
                f"vs_frozen={rec.get('vs_frozen')} redis_changed={len(rec['redis_changed'])} "
                f"neo4jΔ={rec['neo4j_delta']} {rec['wall_s']}s")

    sh("docker", "rm", "-f", C_NEO, C_QD, check=False)
    sh("redis-cli", "-p", str(PORTS["redis"]), "shutdown", "nosave", check=False)
    guard_after = main_guard()

    pairs = {}
    passes = args.passes.split(",")
    if len(passes) == 2:
        for (ps, org, cell), rec in runs.items():
            if ps != passes[0]:
                continue
            other = runs.get((passes[1], org, cell))
            if rec.get("exit") == 0 and other and other.get("exit") == 0:
                pairs[f"{org}/{cell}"] = compare(Path(rec["dir"]), Path(other["dir"]))
            else:
                pairs[f"{org}/{cell}"] = {"R3_pass": False, "reason": f"exit A={rec.get('exit')} B={other and other.get('exit')}"}
    allrec = list(runs.values())
    gates = {
        "R1_restore": all(r["restore_ok"] for r in allrec),
        "R2_llm_miss_zero": all(r.get("exit") != 10 and r.get("llm_miss", 0) == 0 for r in allrec),
        "R3_determinism": bool(pairs) and all(v["R3_pass"] for v in pairs.values()),
        "R4_isolation": guard_before == guard_after,
        "all_runs_exit0": all(r.get("exit") == 0 for r in allrec),
    }
    gates["PASS"] = all(gates.values())
    res = {"protocol": str(PROTOCOL.relative_to(REPO)), "meta": meta, "work_root": str(work),
           "baseline_state": {k: v for k, v in base_fp.items()},
           "main_guard_before": guard_before, "main_guard_after": guard_after,
           "gates": gates, "pairs": pairs,
           "runs": [{**r, "key": "/".join(k)} for k, r in runs.items()],
           "descriptive": {
               "D1_vs_frozen_equal": sum(1 for r in allrec if r.get("vs_frozen")),
               "D1_n": sum(1 for r in allrec if "vs_frozen" in r),
               "D2_tool_miss_unexplained": sum(r.get("tool_miss", {}).get("unexplained", 0) for r in allrec)}}
    result_p.write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    say(f"GATES {json.dumps(gates)}  D1 {res['descriptive']}")
    return 0 if gates["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
