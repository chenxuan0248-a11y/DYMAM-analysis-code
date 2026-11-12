"""e7_org_bootstrap_auc_fp — organization-level bootstrap CIs for the pooled
detector AUC-ROC / AUC-PR / clean false-alarm rate reported in
detector_pr_roc_v1.json / e3_detector_fp_v1.json (0.914 / 0.269 / 0.106).

Those are pooled point estimates over 18,903 (score, host) points (243
positives + 18,660 negatives) treated as independent -- but every point is
tagged to one of 20 organizations that share evidence structure within a run,
the same pseudoreplication concern §4.5 already flags for cost/harm. This
computes each metric PER ORGANIZATION (pooling only that org's 9 faulted
cells' scored hosts + its 1 clean run's hosts, same scoring function and same
trapezoid AUC integration as detector_pr_roc.py) and bootstraps the 20
org-level values with evaluation.stat_test.bootstrap_ci (one-sample: these are
not baseline-vs-ARCS comparisons, just point estimates needing an interval).

Feasibility check done first: per-org positive counts range 7-13 (reddit is
the thinnest at 7), negatives are in the hundreds to low thousands per org --
thin but not degenerate for a rank-based AUC.

No new pipeline run: reuses the same _host_scores computation as
detector_pr_roc.py against the existing e3_grid_v1.json bundles.

    poetry run python -m experiments.exp_arcs.e7_org_bootstrap_auc_fp
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

from evaluation.stat_test import bootstrap_ci
from experiments.exp_arcs.detector_pr_roc import _host_scores

D = Path("experiments/exp_arcs")
OUT = D / "e7_org_bootstrap_auc_fp_v1.json"
FP_THRESHOLD = 0.5


def _auc(xy: list[tuple[float, float]]) -> float:
    xy = sorted(xy)
    return round(sum((xy[i + 1][0] - xy[i][0]) * (xy[i + 1][1] + xy[i][1]) / 2
                     for i in range(len(xy) - 1)), 4)


def _org_auc(pts: list[tuple[float, bool]]) -> dict:
    n_pos = sum(1 for _, y in pts if y)
    n_neg = len(pts) - n_pos
    thr = sorted({s for s, _ in pts}, reverse=True)
    pr, roc = [], []
    for t in thr:
        tp = sum(1 for s, y in pts if s >= t and y)
        fp = sum(1 for s, y in pts if s >= t and not y)
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        pr.append((rec, prec))
        roc.append((fpr, rec))
    roc_full = sorted(set(roc) | {(0.0, 0.0), (1.0, 1.0)})
    return {"auc_roc": _auc(roc_full), "auc_pr": _auc(pr), "n_pos": n_pos, "n_neg": n_neg}


def main() -> int:
    grid = json.loads((D / "e3_grid_v1.json").read_text())["cells"]
    clean_by_org: dict[str, str] = {}
    for c in grid:
        clean_by_org.setdefault(c["org"], c["clean_dir"])

    pts_by_org: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    fa_by_org: dict[str, float] = {}

    for c in grid:
        inj = c.get("inject_node") or []
        if len(inj) < 2:
            continue
        true_hosts = set(inj[1]) if isinstance(inj[1], list) else {inj[1]}
        sc = _host_scores(Path(c["faulted_dir"]))
        for h, s in sc.items():
            pts_by_org[c["org"]].append((s, h in true_hosts))

    for org, cd in clean_by_org.items():
        sc = _host_scores(Path(cd))
        for h, s in sc.items():
            pts_by_org[org].append((s, False))
        n_flag = sum(1 for s in sc.values() if s >= FP_THRESHOLD)
        fa_by_org[org] = n_flag / len(sc) if sc else 0.0

    orgs = sorted(pts_by_org)
    per_org = {o: _org_auc(pts_by_org[o]) for o in orgs}

    auc_roc_vals = [per_org[o]["auc_roc"] for o in orgs]
    auc_pr_vals = [per_org[o]["auc_pr"] for o in orgs]
    fa_vals = [fa_by_org[o] for o in orgs]

    pooled = json.loads((D / "detector_pr_roc_v1.json").read_text())
    pooled_fp = json.loads((D / "e3_detector_fp_v1.json").read_text()
                           )["clean_run_false_alarms"]["combined_host_false_alarm_rate"]
    pooled_auc_roc, pooled_auc_pr = pooled["auc_roc"], pooled["auc_pr"]

    def summarize(vals, pooled_val, name):
        mean, lo, hi = bootstrap_ci(vals, B=10_000, seed=42)
        return {"org_level_mean": round(mean, 4), "ci_95": [round(lo, 4), round(hi, 4)],
                "pooled_point_estimate": pooled_val, "n_orgs": len(vals),
                "org_level_min": round(min(vals), 4), "org_level_max": round(max(vals), 4),
                "note": name}

    result = {
        "phase": "organization-level bootstrap for pooled detector AUC-ROC/AUC-PR/"
                 "clean-FA-rate (avoids pseudoreplication across ~945 scored "
                 "hosts/org)",
        "method": "evaluation.stat_test.bootstrap_ci (one-sample), B=10000, seed=42",
        "per_org_positive_negative_counts": {o: {"n_pos": per_org[o]["n_pos"],
                                                  "n_neg": per_org[o]["n_neg"]} for o in orgs},
        "auc_roc": summarize(auc_roc_vals, pooled_auc_roc, "AUC-ROC"),
        "auc_pr": summarize(auc_pr_vals, pooled_auc_pr, "AUC-PR"),
        "clean_false_alarm_rate_at_t0.5": summarize(fa_vals, pooled_fp, "clean FA rate @t=0.5"),
        "per_org": {o: {**per_org[o], "clean_fa_rate": round(fa_by_org[o], 4)} for o in orgs},
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    for key in ("auc_roc", "auc_pr", "clean_false_alarm_rate_at_t0.5"):
        s = result[key]
        print(f"{key}: org-level mean={s['org_level_mean']} CI95={s['ci_95']} "
              f"(pooled point est.={s['pooled_point_estimate']}) "
              f"range=[{s['org_level_min']},{s['org_level_max']}]")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
