"""Retrospective four-tier input-permission ablation; no external requests."""
from __future__ import annotations

import hashlib
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from experiments.exp_arcs.detector_eval import detector_localize  # noqa: E402
from experiments.exp_arcs.localizer_baselines import _truth  # noqa: E402
from experiments.exp_arcs.rq2_combined_topk import tie_aware_hit  # noqa: E402
from experiments.exp_arcs.revision_v2 import step2_baselines as s2  # noqa: E402

D = REPO / "experiments/exp_arcs"
CONTRACT = frozenset(("dns", "active_observation"))
TIERS = ("records", "schema", "manifest", "paired_clean_oracle")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


@lru_cache(maxsize=None)
def view(run_dir: str, include_manifest: bool) -> dict:
    b = Path(run_dir)
    fams: dict[str, set[str]] = defaultdict(set)
    signatures: dict[str, Counter] = defaultdict(Counter)
    for e in lines(b / "evidence_event.jsonl"):
        fams[e["target"]].add(e.get("source_family"))
        signatures[e["target"]][(e.get("source_family"), e.get("raw_hash"),
                                  e.get("observation_status"))] += 1
    claims = {c["claim_id"]: c.get("subject") for c in lines(b / "claim.jsonl")}
    present = set(fams) | {h for h in claims.values() if h}
    flagged = set(detector_localize(b)["flagged_hosts"])
    present |= flagged
    mp = b / "discovery_manifest.json"
    manifest = (set(json.loads(mp.read_text()).get("resolved_hosts", []))
                if include_manifest and mp.exists() else set())
    return {"fams": fams, "signatures": signatures, "present": present, "flagged": flagged,
            "manifest": manifest, "manifest_exists": include_manifest and mp.exists()}


def scores(faulted: dict, tier: str, clean: dict | None = None) -> dict[str, float]:
    """Only the oracle tier accepts a paired-clean view."""
    assert tier in TIERS
    assert (clean is not None) == (tier == "paired_clean_oracle")
    assert tier not in ("records", "schema") or not faulted["manifest_exists"]
    fams = faulted["fams"]
    n = len(fams)
    counts: dict[str, int] = defaultdict(int)
    for fs in fams.values():
        for f in fs:
            counts[f] += 1
    base = {f: count / n for f, count in counts.items()} if n else {}
    out = {h: max((rate for f, rate in base.items() if f not in fams.get(h, set())), default=0.0)
           for h in faulted["present"]}
    for h in faulted["flagged"]:
        out[h] = 1.0
    if tier != "records":
        for h in faulted["present"]:
            if CONTRACT - fams.get(h, set()):
                out[h] = 1.0
    if tier in ("manifest", "paired_clean_oracle"):
        for h in faulted["manifest"]:
            out[h] = max(out.get(h, 0.0), 1.0 if h not in fams else 0.0)
    if clean is not None:
        for h in clean["present"] | clean["manifest"]:
            if (clean["signatures"].get(h, Counter()) != faulted["signatures"].get(h, Counter())
                    or (h in clean["manifest"]) != (h in faulted["manifest"])):
                out[h] = 2.0
            else:
                out.setdefault(h, 0.0)
    return out


def summarize(rows: list[dict], clean_alerts: list[dict]) -> dict:
    by_org: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_org[r["org"]].append(r)
    org_mean = lambda field: round(st.mean(st.mean(r[field] for r in rr) for rr in by_org.values()), 4)
    return {
        "truth_pairs_covered": sum(r["truth_covered"] for r in rows),
        "truth_pairs": sum(r["n_truth"] for r in rows),
        "cells": len(rows), "detect_at_1_org_mean": org_mean("detect_at_1"),
        "success_org_mean": org_mean("success"),
        "cost_over_cstar_org_mean": org_mean("cost_over_cstar"),
        "clean_alerts_per_run": round(st.mean(r["alerts"] for r in clean_alerts), 4),
        "clean_host_alert_rate_macro": round(st.mean(r["alerts"] / r["candidates"] for r in clean_alerts), 4),
        "clean_runs_with_alert": sum(bool(r["alerts"]) for r in clean_alerts),
    }


def main(out: Path) -> None:
    grid_file = D / "e3_grid_v1.json"
    grid = json.loads(grid_file.read_text())["cells"]
    clean_dir = {c["org"]: c["clean_dir"] for c in grid}
    clean_views = {(o, t): view(d, t in ("manifest", "paired_clean_oracle"))
                   for o, d in clean_dir.items() for t in TIERS}
    rows: dict[str, list[dict]] = {t: [] for t in TIERS}
    clean_rows: dict[str, list[dict]] = {t: [] for t in TIERS}
    assertion_count = 0
    evidence_only_covered = 0
    evidence_manifest_covered = 0
    for o in sorted(clean_dir):
        for t in TIERS:
            cv = clean_views[(o, t)]
            sc = scores(cv, t, cv if t == "paired_clean_oracle" else None)
            clean_rows[t].append({"org": o, "alerts": sum(x >= 0.5 for x in sc.values()),
                                  "candidates": len(sc)})
    for c in grid:
        truth = _truth(c["inject_node"])[0]
        sets: list[set[str]] = []
        raw_view = view(c["faulted_dir"], True)
        evidence_only_covered += len(truth & set(raw_view["fams"]))
        evidence_manifest_covered += len(truth & (set(raw_view["fams"]) | raw_view["manifest"]))
        for t in TIERS:
            fv = view(c["faulted_dir"], t in ("manifest", "paired_clean_oracle"))
            cv = clean_views[(c["org"], t)]
            sc = scores(fv, t, cv if t == "paired_clean_oracle" else None)
            cand = set(sc)
            sets.append(cand)
            flagged = frozenset(h for h, score in sc.items() if score >= 0.5)
            repair = s2.repair(c, flagged)
            rows[t].append({"org": c["org"], "cell": c["cell"],
                            "n_truth": len(truth), "truth_covered": len(truth & cand),
                            "score_sha256": hashlib.sha256(json.dumps(sorted(sc.items())).encode()).hexdigest(),
                            "detect_at_1": tie_aware_hit(sc, truth, 1),
                            "alerts": len(flagged), "success": int(repair["success"]),
                            "cost_over_cstar": repair["cost"] / c["C_star"]})
        assert sets[0] == sets[1]
        assert all(a <= b for a, b in zip(sets, sets[1:]))
        assertion_count += 2
    assert len(grid) == 180 and evidence_only_covered == 129 and evidence_manifest_covered == 243
    assertion_count += 3
    result = {
        "analysis_status": "retrospective_sensitivity_not_preregistered",
        "input_hashes": {"grid": sha(grid_file), "script": sha(Path(__file__)),
                         "protocol": sha(HERE / "INFORMATION_ACCESS_PROTOCOL_20261013.md")},
        "clean_dag_use": "evaluation_and_oracle_mapped_repair_cost_only; not deployable execution cost",
        "n_assertions_passed": assertion_count,
        "legacy_evidence_only_covered": evidence_only_covered,
        "legacy_evidence_plus_manifest_covered": evidence_manifest_covered,
        "summary": {t: summarize(rows[t], clean_rows[t]) for t in TIERS},
        "cells": rows,
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: information_access_ablation.py OUT.json")
    main(Path(sys.argv[1]))
