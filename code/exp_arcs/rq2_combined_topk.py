"""experiments/exp_arcs/rq2_combined_topk — RQ2 Detect@k for the combined reference-free
localizer (validator-flag + peer-anomaly + discovery-manifest), on the same footing as the
named probe baselines.

Detect@k = 1 if a true fault host is among the top-k ranked hosts of the cell (host grain).
It is a ranking metric, distinct from Repair@B (end-to-end repair success); the two are
reported separately (§4.3).

Method (no new experiment):
  * host score = detector_pr_roc.py's _host_scores (the score already used for
    AUC-ROC/AUC-PR: 1.0 if the validator flag or the discovery-manifest flags the host,
    else peer-anomaly's max missing-family base rate). The JSON key `self_diagnosis`
    elsewhere is the validator-flag signal.
  * deterministic Detect@k: hosts ranked by (-score, host_name), scored with
    localizer_baselines.py's _score_host — the convention used for the named baselines.
  * tie-aware Detect@k: many hosts share a score (e.g. several flagged at 1.0), so the
    alphabetical tie-break is arbitrary. We also report the EXACT expected hit rate when
    ties are broken uniformly at random: walking score groups in order, P(miss) for the
    group that straddles rank k is C(g-t, s) / C(g, s) (g = group size, t = true hosts in
    it, s = slots left); any true host in a fully-included group is a certain hit.
  * fault variant from the cell name (the grid has no fault_type field); family = F4/F10.
  * org-level means with percentile bootstrap CIs (B=10000, seed=42; 20 orgs).
  * reference numbers are read from the frozen baseline artifacts, not hard-coded.

    python -m experiments.exp_arcs.rq2_combined_topk [out.json]
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

from evaluation.stat_test import bootstrap_ci
from experiments.exp_arcs.detector_pr_roc import _host_scores
from experiments.exp_arcs.localizer_baselines import _score_host, _truth

D = Path("experiments/exp_arcs")
K = 3


def ranked_hosts(scores: dict[str, float]) -> list[str]:
    return [h for h, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def fault_variant(cell: str) -> str:
    return cell


def fault_family(cell: str) -> str:
    return cell.split("@")[0].split("-")[0]


def tie_aware_hit(scores: dict[str, float], truth: set[str], k: int) -> float:
    """Exact P(some true host within top-k) under uniform random tie-breaking."""
    groups: dict[float, list[str]] = {}
    for h, s in scores.items():
        groups.setdefault(s, []).append(h)
    slots = k
    for s in sorted(groups, reverse=True):
        members = groups[s]
        g, t = len(members), sum(1 for h in members if h in truth)
        if g <= slots:
            if t:
                return 1.0
            slots -= g
            if slots == 0:
                return 0.0
            continue
        # this group straddles the cut: choose `slots` of g uniformly
        return 1.0 - comb(g - t, slots) / comb(g, slots)
    return 0.0


def org_ci(rows: list[dict], key: str) -> dict:
    by_org: dict[str, list[float]] = {}
    for r in rows:
        by_org.setdefault(r["org"], []).append(r[key])
    means = [sum(v) / len(v) for _, v in sorted(by_org.items())]
    m, lo, hi = bootstrap_ci(means, B=10_000, seed=42)
    return {"org_mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "org_range": [round(min(means), 4), round(max(means), 4)], "n_orgs": len(means)}


def rate(rows: list[dict], key: str) -> float:
    return round(sum(r[key] for r in rows) / len(rows), 4)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else D / "rq2_combined_topk_v1.json"
    grid = json.loads((D / "e3_grid_v1.json").read_text())["cells"]
    rows = []
    for c in grid:
        hosts, _fam = _truth(c["inject_node"])
        sc = _host_scores(Path(c["faulted_dir"]))
        ranked = ranked_hosts(sc)
        top1, topk = _score_host(ranked, hosts, K)
        top_score = sc[ranked[0]] if ranked else None
        rows.append({
            "org": c["org"], "cell": c["cell"],
            "fault_variant": fault_variant(c["cell"]), "fault_family": fault_family(c["cell"]),
            "detect1_det": top1, "detectk_det": topk,
            "detect1_tie": round(tie_aware_hit(sc, hosts, 1), 6),
            "detectk_tie": round(tie_aware_hit(sc, hosts, K), 6),
            "n_candidates": len(ranked), "n_true_hosts": len(hosts),
            "n_tied_at_top": sum(1 for v in sc.values() if v == top_score) if ranked else 0,
        })

    def block(rs: list[dict]) -> dict:
        return {"n_cells": len(rs),
                "detect1_deterministic": rate(rs, "detect1_det"),
                "detectk_deterministic": rate(rs, "detectk_det"),
                "detect1_tie_aware": rate(rs, "detect1_tie"),
                "detectk_tie_aware": rate(rs, "detectk_tie")}

    by_variant = {v: block([r for r in rows if r["fault_variant"] == v])
                  for v in sorted({r["fault_variant"] for r in rows})}
    by_family = {f: block([r for r in rows if r["fault_family"] == f])
                 for f in sorted({r["fault_family"] for r in rows})}

    loc = json.loads((D / "e3_localizers_v1.json").read_text())["summary"]
    rnd = json.loads((D / "random_baseline_localizers_v1.json").read_text())["expected_random"]
    summary = {
        "n_cells": len(rows), "k": K,
        "metric": "Detect@k (host grain): a true fault host is within the top-k ranked hosts",
        "pooled": block(rows),
        "org_level": {key: org_ci(rows, key) for key in
                      ("detect1_det", "detectk_det", "detect1_tie", "detectk_tie")},
        "by_fault_variant": by_variant,
        "by_fault_family": by_family,
        "ties": {"cells_with_tie_at_top": sum(1 for r in rows if r["n_tied_at_top"] > 1),
                 "max_tied_at_top": max(r["n_tied_at_top"] for r in rows)},
        "reference_from_frozen_artifacts": {
            "source": ["e3_localizers_v1.json (summary)", "random_baseline_localizers_v1.json"],
            "oracle_top1": loc["reference_localizers"]["oracle_top1"],
            "naive_top1": loc["reference_localizers"]["naive_top1"],
            "first_self_report_top1": loc["host_top1"]["agentdebug"],
            "first_self_report_topk": loc["host_topk"]["agentdebug"],
            "provenance_backtrace_top1": loc["host_top1"]["famas"],
            "provenance_backtrace_topk": loc["host_topk"]["famas"],
            "counterfactual_repair_probe_top1": loc["host_top1"]["agentrx"],
            "counterfactual_repair_probe_topk": loc["host_topk"]["agentrx"],
            "expected_random_top1": rnd["pooled_mean_top1"],
            "expected_random_topk": rnd["pooled_mean_topk"],
        },
    }
    out.write_text(json.dumps({
        "phase": "RQ2 combined-detector Detect@1/Detect@3 (host grain, 180 cells), same "
                 "footing as the named probe baselines; deterministic and tie-aware",
        "source": "experiments/exp_arcs/e3_grid_v1.json",
        "summary": summary, "cells": rows,
    }, indent=2, sort_keys=True))
    print(json.dumps({k: summary[k] for k in ("pooled", "by_fault_family", "ties")}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
