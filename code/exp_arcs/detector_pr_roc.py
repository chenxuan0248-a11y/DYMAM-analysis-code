"""Reviewer item B (part 2) — detector PR / ROC curve + AUC.

Reframes "precision 0.29" as one operating point on a recall–precision knob. The
combined detector's ONLY tunable, and its ONLY source of false alarms (self=0,
manifest=0 on clean runs; e3_detector_fp_v1.json), is peer-anomaly's base-rate
threshold. A host's continuous, threshold-free suspicion score is:

    score(h) = 1.0                              if flagged by self-diagnosis or manifest
             = max_{f the host lacks} baserate(f)  otherwise   (peer severity)

Thresholding this score at t reproduces the combined detector at
peer_anomaly min_base_rate = t EXACTLY (a host is peer-flagged at t iff it lacks
some family with base rate >= t iff its max-missing-baserate >= t). So the frozen
0.5 point (precision 0.286 / recall 0.819 / clean-FP 0.106) must lie on the curve.

Two consistent views:
  (1) Deployable-knob curve — per-cell-averaged precision/recall/F1 + clean-run
      false-alarm rate as t sweeps (same host-grain definition as e3_detector_fp).
  (2) Pooled score PR / ROC + AUC — positives = injected hosts (faulted runs),
      negatives = every other host (faulted non-injected + ALL clean-run hosts).

Cheap: jsonl reads only, no DAG builds. End-to-end cost/success vs t is in
detector_threshold_sweep_v1.json (the recall->cost half of the story).

    poetry run python -m experiments.exp_arcs.detector_pr_roc
"""
from __future__ import annotations
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from experiments.exp_arcs.detector_eval import (
    detector_localize, manifest_dropped_hosts,
)

D = Path("experiments/exp_arcs")
OUT = D / "detector_pr_roc_v1.json"


def _fam_dist(run_dir: Path):
    """host->families, family->hosts, from evidence_event.jsonl."""
    host_fams: dict[str, set] = defaultdict(set)
    fam_hosts: dict[str, set] = defaultdict(set)
    for ln in (run_dir / "evidence_event.jsonl").read_text().splitlines():
        if ln.strip():
            e = json.loads(ln)
            host_fams[e["target"]].add(e.get("source_family"))
            fam_hosts[e.get("source_family")].add(e["target"])
    return host_fams, fam_hosts


def _host_scores(run_dir: Path) -> dict[str, float]:
    """Continuous reference-free suspicion score per host (see module docstring)."""
    host_fams, fam_hosts = _fam_dist(run_dir)
    n = len(host_fams)
    scores: dict[str, float] = {}
    if n:
        baserate = {f: len(hs) / n for f, hs in fam_hosts.items()}
        for h, fs in host_fams.items():
            missing = [baserate[f] for f in fam_hosts if f not in fs]
            scores[h] = max(missing) if missing else 0.0
    # self-diagnosis + manifest fire independent of the threshold -> max score
    for h in detector_localize(run_dir)["flagged_hosts"] | manifest_dropped_hosts(run_dir):
        scores[h] = 1.0
    # hosts that appear ONLY via manifest (no evidence) still get 1.0 above.
    return scores


def _flagged_at(scores: dict[str, float], t: float) -> set[str]:
    return {h for h, s in scores.items() if s >= t}


def main() -> int:
    grid = json.loads((D / "e3_grid_v1.json").read_text())["cells"]
    clean_by_org = {}
    for c in grid:
        clean_by_org.setdefault(c["org"], c["clean_dir"])

    # precompute scores once per run
    faulted = []
    for c in grid:
        inj = c.get("inject_node") or []
        if len(inj) < 2:
            continue
        true_hosts = set(inj[1]) if isinstance(inj[1], list) else {inj[1]}
        sc = _host_scores(Path(c["faulted_dir"]))
        faulted.append({"cell": c["cell"], "true": true_hosts, "scores": sc})
    clean_scores = {org: _host_scores(Path(cd)) for org, cd in clean_by_org.items()}

    # ---------- (1) deployable-knob curve (per-cell-averaged) ----------
    THR = [round(x / 100, 2) for x in range(5, 100, 5)]  # 0.05 .. 0.95
    knob = []
    for t in THR:
        precs, recs = [], []
        for fc in faulted:
            fl = _flagged_at(fc["scores"], t)
            tp = len(fc["true"] & fl)
            precs.append(tp / len(fl) if fl else 0.0)
            recs.append(tp / len(fc["true"]))
        P, R = st.mean(precs), st.mean(recs)
        fa = [len(_flagged_at(sc, t)) / max(1, len(sc)) for sc in clean_scores.values()]
        knob.append({"t": t, "precision": round(P, 3), "recall": round(R, 3),
                     "f1": round(2 * P * R / (P + R), 3) if (P + R) else 0.0,
                     "clean_false_alarm_rate": round(st.mean(fa), 3)})

    # faithfulness: t=0.5 must reproduce frozen 0.286 / 0.819 / 0.106
    p50 = next(k for k in knob if k["t"] == 0.5)
    frz = json.loads((D / "e3_detector_fp_v1.json").read_text())
    frz_p = frz["faulted_detection_host_grain"]
    frz_fa = frz["clean_run_false_alarms"]["combined_host_false_alarm_rate"]
    ok = (abs(p50["precision"] - frz_p["precision"]) <= 0.001
          and abs(p50["recall"] - frz_p["recall"]) <= 0.001
          and abs(p50["clean_false_alarm_rate"] - frz_fa) <= 0.001)

    # ---------- (2) pooled score PR / ROC + AUC ----------
    pts = []  # (score, is_positive)
    for fc in faulted:
        for h, s in fc["scores"].items():
            pts.append((s, h in fc["true"]))
    for sc in clean_scores.values():
        for h, s in sc.items():
            pts.append((s, False))  # clean run: every host is a true negative
    n_pos = sum(1 for _, y in pts if y)
    n_neg = len(pts) - n_pos
    # sweep unique thresholds high->low
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
    # AUCs (trapezoid; sort by x)
    def auc(xy):
        xy = sorted(xy)
        return round(sum((xy[i + 1][0] - xy[i][0]) * (xy[i + 1][1] + xy[i][1]) / 2
                         for i in range(len(xy) - 1)), 3)
    # ROC endpoints (0,0)&(1,1) for a clean integral
    roc_full = sorted(set(roc) | {(0.0, 0.0), (1.0, 1.0)})
    auc_roc = auc(roc_full)
    auc_pr = auc(pr)

    result = {
        "phase": "reviewer-B detector PR/ROC (host grain, 180 faulted + 20 clean runs)",
        "score_def": "1.0 if self/manifest flags host, else peer severity = max base-rate among missing families",
        "faithfulness_at_0.5": {"curve": p50, "frozen_precision": frz_p["precision"],
                                "frozen_recall": frz_p["recall"],
                                "frozen_clean_fa": frz_fa, "match": ok},
        "n_positives": n_pos, "n_negatives": n_neg,
        "auc_roc": auc_roc, "auc_pr": auc_pr,
        "baseline_pr": round(n_pos / (n_pos + n_neg), 3),
        "knob_curve": knob,
        "pr_curve": [[round(a, 3), round(b, 3)] for a, b in pr],
        "roc_curve": [[round(a, 3), round(b, 3)] for a, b in roc],
    }
    OUT.write_text(json.dumps(result, indent=2))

    print(f"faithfulness @0.5: curve P={p50['precision']} R={p50['recall']} "
          f"FA={p50['clean_false_alarm_rate']} | frozen P={frz_p['precision']} "
          f"R={frz_p['recall']} FA={frz_fa} | match={ok}\n")
    print(f"AUC-ROC={auc_roc}  AUC-PR={auc_pr}  (PR baseline={result['baseline_pr']}, "
          f"pos={n_pos}/neg={n_neg})\n")
    print(f"{'t':>5} {'prec':>6} {'recall':>7} {'F1':>6} {'clean_FA':>9}")
    for k in knob:
        print(f"{k['t']:>5} {k['precision']:>6} {k['recall']:>7} {k['f1']:>6} "
              f"{k['clean_false_alarm_rate']:>9}")
    print(f"\nwrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
