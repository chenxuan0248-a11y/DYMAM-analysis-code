"""Revision v2 · step-1 audit B — provenance of the four local enrichment tools.

Protocol: experiments/exp_arcs/revision_v2/STEP1_AUDIT_PROTOCOL.md §B. Read-only: frozen E3
bundles + Redis GETs (no counters touched) + static inspection of agent code.

    .venv/bin/python experiments/exp_arcs/revision_v2/provenance_local_tools.py [--out-dir D] [--suffix v2]

D-b reads Redis at $AUDIT_REDIS_URL (default 127.0.0.1:6379). Set $AUDIT_REDIS_RDB to the RDB
file that instance was loaded from; its SHA-256, DBSIZE, server version and a content digest
are written to provenance_local_tools_env_<suffix>.json (v2 onward).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
HERE = REPO / "experiments/exp_arcs/revision_v2"
GRID = REPO / "experiments/exp_arcs/e3_grid_v1.json"
SPEC = REPO / "experiments/exp_arcs/cost_harm_spec.yaml"
LOCAL = ("cve_rag", "geolocation", "wappalyzer_match", "kev_epss")
LOCAL_SRC = ("maxmind", "cve_corpus", "cve_rag", "kev", "epss", "wappalyzer")
LOCAL_CLAIM = ("cve_rag:", "kev_epss:", "wappalyzer:", "geo")
INJECTED_FAMILIES = ("dns", "certificate", "active_observation", "registration")
EV_LINKS = ("supporting_evidence_event_ids", "contradicting_evidence_event_ids", "context_evidence_event_ids")
DISPLAY = {"nmap": "InternetDB (code name nmap)"}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def is_local(e: dict) -> bool:
    src = str(e.get("source") or "").lower()
    claim = str((e.get("payload") or {}).get("claim", "")).lower()
    return (e.get("tool_name") in LOCAL or any(s in src for s in LOCAL_SRC)
            or claim.startswith(LOCAL_CLAIM))


def bundle_digest(b: Path) -> str:
    h = hashlib.sha256()
    for fn in ("evidence_event.jsonl", "claim.jsonl", "decision.jsonl"):
        h.update(fn.encode()); h.update(sha256_file(b / fn).encode())
    return h.hexdigest()


def static_paths() -> list[dict]:
    """Where each local tool's output would go if it worked (static, not measured)."""
    import yaml
    from core.state import _source_family
    from experiments.exp_arcs.s2_evidence_redundancy import INDEP_FAMILIES
    from tools.cve_rag import cve_rag
    from tools.geolocation import geolocation
    from tools.kev_epss_lookup import kev_epss
    from tools.wappalyzer_match import wappalyzer_match
    f2a = yaml.safe_load(SPEC.read_text())["family_to_action"]
    sites = []
    for py in sorted((REPO / "agents").glob("*.py")):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            m = re.search(r"self\.tool\((\w+),", line)
            if m and m.group(1) in LOCAL:
                sites.append((m.group(1), f"{py.relative_to(REPO)}:{i}"))
            if 'source="cve_rag"' in line:
                sites.append(("cve_rag", f"{py.relative_to(REPO)}:{i} (explicit source=\"cve_rag\" on reranked CVE evidence)"))
    out = []
    for t, src in ((cve_rag, cve_rag.source), (cve_rag, "cve_rag"), (geolocation, geolocation.source),
                   (wappalyzer_match, wappalyzer_match.source), (kev_epss, kev_epss.source)):
        fam = _source_family(src)
        out.append({"tool": t.name, "evidence_source": src, "family": fam,
                    "repair_action": f2a.get(fam, "full_replay"),
                    "repairable_leaf": f2a.get(fam, "full_replay") != "full_replay",
                    "injected_family_in_E3": fam in INJECTED_FAMILIES,
                    "D1_independent_family": fam in INDEP_FAMILIES,
                    "call_sites": [s for n, s in sites if n == t.name]})
    return out


REDIS_URL = os.environ.get("AUDIT_REDIS_URL", "redis://127.0.0.1:6379")
COUNTER_PREFIXES = ("cache:stats:", "tool:stats:")


def redis_identity() -> dict:
    """Which Redis D-b read: server, DBSIZE, content digest (counters excluded), source RDB hash."""
    import redis
    r = redis.Redis.from_url(REDIS_URL)
    keys = sorted(k.decode("utf-8", "surrogateescape") for k in r.scan_iter(count=5000))
    content = [k for k in keys if not k.startswith(COUNTER_PREFIXES)]
    h = hashlib.sha256()
    for i in range(0, len(content), 1000):
        ch = content[i:i + 1000]; p = r.pipeline(transaction=False)
        for k in ch:
            p.dump(k)
        for k, d in zip(ch, p.execute()):
            h.update(f"{k}\t{hashlib.sha256(d or b'').hexdigest()}\n".encode())
    rdb = os.environ.get("AUDIT_REDIS_RDB")
    return {"redis_url": REDIS_URL, "redis_version": r.info("server")["redis_version"],
            "dbsize": r.dbsize(), "n_content_keys": len(content),
            "content_digest_sha256": h.hexdigest(),
            "content_digest_rule": "sha256 over sorted '<key>\\t<sha256(DUMP key)>\\n', counter keys excluded",
            "source_rdb": rdb, "source_rdb_sha256": sha256_file(Path(rdb)) if rdb else None}


def vuln_check(rows: list[dict]) -> tuple[dict, list[dict]]:
    """D-b: each vulnerability claim's CVE is in the cached InternetDB response of its supporting IP.
    Returns (summary, per-(claim, supporting IP) index)."""
    import redis
    from tools._base import _hash
    from tools.nmap_replay import nmap
    r = redis.Redis.from_url(REDIS_URL)
    cache: dict[str, dict | None] = {}
    c = Counter()
    index = []
    for row in rows:
        c["claims"] += 1
        ok_any = False
        for ip, raw_hash in row["_internetdb_support"]:
            key = nmap._cache_key(ip=ip)
            if ip not in cache:
                v = r.get(key)
                cache[ip] = json.loads(v)["data"] if v else None
            data = cache[ip]
            rec = {"org": row["org"], "claim_id": row["claim_id"], "cve": row["object"], "ip": ip,
                   "internetdb_cache_key": key, "artifact_raw_hash": raw_hash,
                   "redis_hash": _hash(data) if data is not None else "",
                   "hash_match": False, "cve_present": False}
            if data is None:
                c["support_cache_absent"] += 1; index.append(rec); continue
            rec["hash_match"] = rec["redis_hash"] == raw_hash
            rec["cve_present"] = row["object"] in data.get("vulns", [])
            c["support_hash_match" if rec["hash_match"] else "support_hash_mismatch"] += 1
            if rec["hash_match"] and rec["cve_present"]:
                ok_any = True
            index.append(rec)
        c["claims_cve_in_internetdb_vulns" if ok_any else "claims_not_verified"] += 1
    return dict(c), index


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--suffix", default="v1")
    args = ap.parse_args()
    out_dir, sfx = Path(args.out_dir), args.suffix
    grid = json.loads(GRID.read_text())["cells"]
    bundles: dict[str, dict] = {}
    for c in grid:
        bundles.setdefault(c["clean_dir"], {"org": c["org"], "role": "clean", "cell": "clean"})
        bundles[c["faulted_dir"]] = {"org": c["org"], "role": "faulted", "cell": c["cell"]}
    assert len(bundles) == 200, len(bundles)

    P1 = P2 = 0
    per_bundle, inputs = [], {}
    mat = {"clean": defaultdict(Counter), "all": defaultdict(Counter)}
    n_claims = {"clean": Counter(), "all": Counter()}
    ev_tools = {"clean": Counter(), "all": Counter()}
    csv_rows, vuln_rows = [], []
    for bdir, meta in sorted(bundles.items(), key=lambda kv: (kv[1]["org"], kv[1]["role"], kv[1]["cell"])):
        b = REPO / bdir if not os.path.isabs(bdir) else Path(bdir)
        inputs[b.name] = bundle_digest(b)
        ev = {e["evidence_event_id"]: e for e in jsonl(b / "evidence_event.jsonl")}
        claims = {c["claim_id"]: c for c in jsonl(b / "claim.jsonl")}
        local_ev = {i for i, e in ev.items() if is_local(e)}
        n_dec_local = 0
        scopes = ("all", "clean") if meta["role"] == "clean" else ("all",)
        for e in ev.values():
            for s in scopes:
                ev_tools[s][(e.get("tool_name"), e.get("source_family"))] += 1
        for d in jsonl(b / "decision.jsonl"):
            linked = {i for k in EV_LINKS for i in (d.get(k) or [])}
            if linked & local_ev:
                n_dec_local += 1
            c = claims[d["claim_id"]]; ct = c["claim_type"]
            sup = [ev[i] for i in (d.get("supporting_evidence_event_ids") or []) if i in ev]
            tools = sorted({e.get("tool_name") for e in sup})
            for s in scopes:
                n_claims[s][ct] += 1
                for t in tools:
                    mat[s][ct][t] += 1
            if meta["role"] == "clean":
                csv_rows.append({
                    "org": meta["org"], "claim_id": c["claim_id"], "claim_type": ct,
                    "predicate": c.get("predicate"), "object": json.dumps(c.get("object"), sort_keys=True)[:120],
                    "decision": d.get("decision"), "n_supporting": len(sup),
                    "supporting_tools": ";".join(t for t in tools if t),
                    "supporting_families": ";".join(sorted({e.get("source_family") for e in sup})),
                    "local_tool_linked": bool(linked & local_ev)})
                if ct == "vulnerability_applicability":
                    # evidence.raw_hash points at the artifact's sha256 (recorder); the tool
                    # output's own sha1 (_hash(data)) is kept inside the source artifact
                    vuln_rows.append({"org": meta["org"], "claim_id": c["claim_id"],
                                      "object": c.get("object"), "_internetdb_support": [
                        (str(e["payload"]["claim"]).split(":", 1)[1],
                         json.loads((b / "sources" / f"{e['raw_ref'].split(':', 1)[1]}.json").read_text())["raw_hash"])
                        for e in sup if e.get("tool_name") == "nmap"]})
        P1 += len(local_ev); P2 += n_dec_local
        per_bundle.append({"bundle": b.name, **meta, "n_evidence": len(ev), "n_local_evidence": len(local_ev),
                           "n_decisions_linked_to_local": n_dec_local})

    gates = {"P1_local_evidence_total": P1, "P2_decisions_linked_total": P2}
    gates["verdict"] = "GO" if P1 == 0 and P2 == 0 else "STOP"
    fmt = lambda m: {ct: dict(sorted(v.items(), key=lambda kv: -kv[1])) for ct, v in sorted(m.items())}
    res = {
        "protocol": "experiments/exp_arcs/revision_v2/STEP1_AUDIT_PROTOCOL.md §B",
        "inputs": {"grid": str(GRID.relative_to(REPO)), "grid_sha256": sha256_file(GRID),
                   "protocol_sha256": sha256_file(HERE / "STEP1_AUDIT_PROTOCOL.md"),
                   "script_sha256": sha256_file(Path(__file__)),
                   "n_bundles": len(bundles), "bundle_record_digests": inputs},
        "gates": gates,
        "evidence_by_tool_family": {s: {f"{t}|{f}": n for (t, f), n in sorted(v.items(), key=lambda kv: -kv[1])}
                                    for s, v in ev_tools.items()},
        "claims_by_type": {s: dict(v) for s, v in n_claims.items()},
        "claim_type_x_supporting_tool": {s: fmt(v) for s, v in mat.items()},
        "vulnerability_source_check_clean": None,
        "counterfactual_paths_static": static_paths(),
        "per_bundle": per_bundle,
    }
    vsum, vindex = vuln_check(vuln_rows)
    res["vulnerability_source_check_clean"] = vsum
    outputs = {}
    csv_p = out_dir / f"provenance_claims_clean_{sfx}.csv"
    with csv_p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
    if sfx != "v1":       # v1 predates these two outputs
        vi_p = out_dir / f"provenance_vuln_index_{sfx}.csv"
        with vi_p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(vindex[0])); w.writeheader(); w.writerows(vindex)
        env_p = out_dir / f"provenance_local_tools_env_{sfx}.json"
        env_p.write_text(json.dumps(redis_identity(), indent=1, sort_keys=True))
        outputs.update({"vuln_index": vi_p.name, "vuln_index_sha256": sha256_file(vi_p),
                        "env": env_p.name, "env_sha256": sha256_file(env_p)})

    # paper table: clean bundles, claims supported by each tool
    cols = ["nmap", "httpx", "dns_resolve", "cert_lookup", "subfinder", "rdap", "asn_lookup"]
    lines = [r"\begin{tabular}{l r " + "r" * len(cols) + " r}", r"\toprule",
             "claim type & $n$ & " + " & ".join(DISPLAY.get(c, c).replace("_", r"\_") for c in cols)
             + r" & local enrichment$^\dagger$ \\", r"\midrule"]
    for ct in sorted(n_claims["clean"]):
        row = [str(mat["clean"][ct].get(c, 0)) for c in cols]
        loc = sum(mat["clean"][ct].get(t, 0) for t in LOCAL)
        lines.append(ct.replace("_", r"\_") + f" & {n_claims['clean'][ct]} & " + " & ".join(row) + f" & {loc} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = ("% generated by experiments/exp_arcs/revision_v2/provenance_local_tools.py — do not edit\n"
           "% cells: number of claims (20 clean E3 bundles) with >=1 supporting evidence event from that tool;\n"
           "% dagger: cve_rag, geolocation, wappalyzer_match, kev_epss (0 evidence events in all 200 bundles)\n"
           + "\n".join(lines) + "\n")
    tex_p = out_dir / ("provenance_local_tools_table.tex" if sfx == "v1" else f"provenance_local_tools_table_{sfx}.tex")
    tex_p.write_text(tex)
    res["outputs"] = {"csv": csv_p.name, "csv_sha256": sha256_file(csv_p),
                      "tex": tex_p.name, "tex_sha256": sha256_file(tex_p), **outputs}
    js = out_dir / f"provenance_local_tools_{sfx}.json"
    js.write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    print(json.dumps({"gates": gates, "claims_clean": dict(n_claims["clean"]),
                      "vuln_check": res["vulnerability_source_check_clean"]}, default=str))
    for p in res["counterfactual_paths_static"]:
        print({k: p[k] for k in ("tool", "evidence_source", "family", "repairable_leaf", "injected_family_in_E3")})
    print(f"wrote {js} sha256={sha256_file(js)}")
    return 0 if gates["verdict"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
