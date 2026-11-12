"""experiments/exp_arcs/localizer_baselines_tieaware — tie-aware Detect@k for the named probe
baselines, so they can be compared with the combined detector on the same footing
(rq2_combined_topk_v1.json).

The frozen baselines (localizer_baselines.py -> e3_localizers_v1.json) rank hosts and break
score ties by host name. Where ties occur that order is arbitrary, so this script also reports
the EXACT expected hit rate under uniformly random tie-breaking (same formula as
rq2_combined_topk.tie_aware_hit).

Per baseline (localizer_baselines.py is imported, not modified):
  * first-flag (agentdebug_style): hosts in order of their first validator-flagged decision.
    Every listed host has a distinct position, so there are no score ties and the tie-aware
    rate equals the deterministic one. Hosts never flagged are not ranked.
  * provenance-backtrace (famas_style): host score = number of flagged outputs whose
    repairable ancestry contains the host (fallback: count of flagged decisions per subject).
  * counterfactual repair probe (agentrx_style): host score = max over the host's candidate
    invocations of n_outputs_restored.
For the two scored baselines the host scores are recomputed here and, for EVERY cell, the
deterministic ranking implied by them is asserted to equal the frozen function's ranking.

    python -m experiments.exp_arcs.localizer_baselines_tieaware [out.json]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from evaluation.stat_test import bootstrap_ci
from experiments.exp_arcs.contamination import contamination_closure
from experiments.exp_arcs.counterfactual import repair_and_check
from experiments.exp_arcs.candidates import candidate_invocations
from experiments.exp_arcs.dag import build_dag
from experiments.exp_arcs.localizer_baselines import (
    _ancestors, _claim_subject, _flagged_decisions, _score_host, _truth,
    agentdebug_style, agentrx_style, famas_style)
from experiments.exp_arcs.rq2_combined_topk import tie_aware_hit

D = Path("experiments/exp_arcs")
K = 3


def famas_host_scores(faulted_dir: Path) -> dict[str, float]:
    """Same host Counter as localizer_baselines.famas_style (lines duplicated, checked below)."""
    dag = build_dag(faulted_dir)
    subj = _claim_subject(faulted_dir)
    flagged_claims = {d.get("claim_id") for d in _flagged_decisions(faulted_dir)}
    out_ids = set()
    for ln in (faulted_dir / "report_item.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            if r.get("claim_id") in flagged_claims and r["report_item_id"] in dag.nodes:
                out_ids.add(r["report_item_id"])
    if not out_ids:
        out_ids = {c for c in flagged_claims if c in dag.nodes}
    host: Counter = Counter()
    for o in out_ids:
        seen_h = set()
        for l in _ancestors(dag, {o}):
            v = dag.nodes.get(l)
            if v and v.get("repairable"):
                seen_h.add(v.get("target"))
        for h in seen_h:
            host[h] += 1
    if not host:
        host = Counter(subj.get(d.get("claim_id")) for d in _flagged_decisions(faulted_dir)
                       if subj.get(d.get("claim_id")))
    return {h: float(c) for h, c in host.items()}


def agentrx_host_scores(clean_dir: Path, faulted_dir: Path) -> dict[str, float]:
    """Host score = max n_outputs_restored over the host's invocations (agentrx_style)."""
    dfx = contamination_closure(clean_dir, faulted_dir)
    cand = candidate_invocations(clean_dir, dfx)
    best: dict[str, float] = {}
    for (h, _f), leaves in cand["invocations"].items():
        n = repair_and_check(clean_dir, dfx, set(leaves))["n_outputs_restored"]
        best[h] = max(best.get(h, float("-inf")), float(n))
    return best


def ranked_by_name(scores: dict[str, float]) -> list[str]:
    return [h for h, _ in sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))]


def org_ci(rows, method, key):
    by_org: dict[str, list[float]] = {}
    for r in rows:
        by_org.setdefault(r["org"], []).append(r[method][key])
    means = [sum(v) / len(v) for _, v in sorted(by_org.items())]
    m, lo, hi = bootstrap_ci(means, B=10_000, seed=42)
    return {"org_mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else D / "e3_localizers_tieaware_v1.json"
    grid = json.loads((D / "e3_grid_v1.json").read_text())["cells"]
    rows = []
    for c in grid:
        clean, faulted = Path(c["clean_dir"]), Path(c["faulted_dir"])
        hosts, _ = _truth(c["inject_node"])
        row = {"org": c["org"], "cell": c["cell"]}

        ad = agentdebug_style(faulted)                       # no score ties by construction
        assert len(ad) == len(set(ad))
        d1, dk = _score_host(ad, hosts, K)
        row["first_flag"] = {"detect1_det": d1, "detectk_det": dk,
                             "detect1_tie": float(d1), "detectk_tie": float(dk), "n_ranked": len(ad)}

        fs = famas_host_scores(faulted)
        assert ranked_by_name(fs) == famas_style(faulted)["ranked_host"], ("famas mismatch", c["cell"])
        d1, dk = _score_host(ranked_by_name(fs), hosts, K)
        row["provenance_backtrace"] = {
            "detect1_det": d1, "detectk_det": dk,
            "detect1_tie": round(tie_aware_hit(fs, hosts, 1), 6),
            "detectk_tie": round(tie_aware_hit(fs, hosts, K), 6), "n_ranked": len(fs),
            "n_tied_at_top": sum(1 for v in fs.values() if fs and v == max(fs.values()))}

        rs = agentrx_host_scores(clean, faulted)
        assert ranked_by_name(rs) == agentrx_style(clean, faulted)["ranked_host"], ("agentrx mismatch", c["cell"])
        d1, dk = _score_host(ranked_by_name(rs), hosts, K)
        row["counterfactual_probe"] = {
            "detect1_det": d1, "detectk_det": dk,
            "detect1_tie": round(tie_aware_hit(rs, hosts, 1), 6),
            "detectk_tie": round(tie_aware_hit(rs, hosts, K), 6), "n_ranked": len(rs),
            "n_tied_at_top": sum(1 for v in rs.values() if rs and v == max(rs.values()))}
        rows.append(row)

    methods = ["first_flag", "provenance_backtrace", "counterfactual_probe"]
    keys = ["detect1_det", "detectk_det", "detect1_tie", "detectk_tie"]
    pooled = {m: {k: round(sum(r[m][k] for r in rows) / len(rows), 4) for k in keys} for m in methods}
    org = {m: {k: org_ci(rows, m, k) for k in keys} for m in methods}
    frozen = json.loads((D / "e3_localizers_v1.json").read_text())["summary"]
    consistency = {
        "first_flag_top1_matches_frozen": pooled["first_flag"]["detect1_det"] == frozen["host_top1"]["agentdebug"],
        "first_flag_topk_matches_frozen": pooled["first_flag"]["detectk_det"] == frozen["host_topk"]["agentdebug"],
        "provenance_backtrace_top1_matches_frozen": pooled["provenance_backtrace"]["detect1_det"] == frozen["host_top1"]["famas"],
        "provenance_backtrace_topk_matches_frozen": pooled["provenance_backtrace"]["detectk_det"] == frozen["host_topk"]["famas"],
        "counterfactual_probe_top1_matches_frozen": pooled["counterfactual_probe"]["detect1_det"] == frozen["host_top1"]["agentrx"],
        "counterfactual_probe_topk_matches_frozen": pooled["counterfactual_probe"]["detectk_det"] == frozen["host_topk"]["agentrx"],
    }
    summary = {"n_cells": len(rows), "k": K, "pooled": pooled, "org_level": org,
               "deterministic_matches_frozen_e3_localizers": consistency,
               "json_key_to_paper_name": {"agentdebug": "first_flag", "famas": "provenance_backtrace",
                                          "agentrx": "counterfactual_probe"}}
    out.write_text(json.dumps({"phase": "tie-aware Detect@k for named probe baselines (host grain, 180 cells)",
                               "source": "experiments/exp_arcs/e3_grid_v1.json",
                               "summary": summary, "cells": rows}, indent=2, sort_keys=True))
    print(json.dumps({"pooled": pooled, "consistency": consistency}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
