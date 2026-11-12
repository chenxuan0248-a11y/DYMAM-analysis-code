"""Revision v2 · step 2 — statistical baselines (robust-z / Isolation Forest / LOF) vs the
combined reference-free detector, at equal clean-run alert budget.

Protocol: experiments/exp_arcs/revision_v2/STEP2_STATISTICAL_BASELINES_PROTOCOL.md (frozen,
commit 8e1bd62). Read-only over the frozen E3 bundles; no Docker, no Redis.

    .venv/bin/python experiments/exp_arcs/revision_v2/step2_baselines.py OUT.json [--orgs a,b] [--tex T.tex]

--orgs restricts the EVALUATED organizations (smoke); threshold calibration always uses the
clean runs of all other organizations (protocol §6).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import sys
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
D = REPO / "experiments/exp_arcs"
HERE = D / "revision_v2"
PROTOCOL = HERE / "STEP2_STATISTICAL_BASELINES_PROTOCOL.md"

from evaluation.stat_test import bootstrap_ci  # noqa: E402
from experiments.exp_arcs.contamination import contamination_closure  # noqa: E402
from experiments.exp_arcs.dag import build_dag  # noqa: E402
from experiments.exp_arcs.detector_eval import (  # noqa: E402
    detector_localize, manifest_dropped_hosts, peer_anomaly_hosts)
from experiments.exp_arcs.detector_pr_roc import _host_scores  # noqa: E402
from experiments.exp_arcs.localizer_baselines import _truth  # noqa: E402
from experiments.exp_arcs.rq2_combined_topk import tie_aware_hit  # noqa: E402
from experiments.exp_arcs.select import evaluate_scope  # noqa: E402

FAMS = ("dns", "certificate", "active_observation", "registration", "other")
CTYPES = ("asset_identity", "organization_association", "service_observation",
          "technology_identification", "vulnerability_applicability")
DECS = ("report", "abstain", "reject", "revalidate")
FEATURES = ([f"E_{f}" for f in FAMS] + [f"P_{f}" for f in FAMS] + [f"C_{t}" for t in CTYPES]
            + [f"D_{d}" for d in DECS] + ["N_no_supporting_observation"])
ALPHAS = {"primary": 0.106, "secondary": 0.020}
COMBINED_T = {"primary": 0.5, "secondary": 0.9}
IF_SEEDS = (0, 1, 2, 3, 4)
MAIN_DETECTORS = ("robust_z", "iforest", "lof")
SENS_DETECTORS = ("lof_k10", "lof_k35")
K = 3


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ── features (protocol §3–4) ────────────────────────────────────────────────
@lru_cache(maxsize=None)
def features(run: str, manifest_aware: bool) -> tuple[tuple[str, ...], np.ndarray]:
    b = Path(run)
    ev = jsonl(b / "evidence_event.jsonl")
    hosts = {e["target"] for e in ev}
    if manifest_aware and (b / "discovery_manifest.json").exists():
        hosts |= set(json.loads((b / "discovery_manifest.json").read_text()).get("resolved_hosts", []))
    hosts = tuple(sorted(hosts))
    idx = {h: i for i, h in enumerate(hosts)}
    X = np.zeros((len(hosts), len(FEATURES)))
    col = {f: j for j, f in enumerate(FEATURES)}
    for e in ev:
        fam = e.get("source_family") if e.get("source_family") in FAMS else "other"
        X[idx[e["target"]], col[f"E_{fam}"]] += 1
    claims = {c["claim_id"]: c for c in jsonl(b / "claim.jsonl")}
    for c in claims.values():
        if c.get("subject") in idx and c["claim_type"] in CTYPES:
            X[idx[c["subject"]], col[f"C_{c['claim_type']}"]] += 1
    for d in jsonl(b / "decision.jsonl"):
        h = claims[d["claim_id"]].get("subject")
        if h not in idx:
            continue
        dv = d["decision"] if d["decision"] in DECS else "reject"
        X[idx[h], col[f"D_{dv}"]] += 1
        if "no_supporting_observation" in (d.get("reason_codes") or []):
            X[idx[h], col["N_no_supporting_observation"]] += 1
    for f in FAMS:
        X[:, col[f"P_{f}"]] = (X[:, col[f"E_{f}"]] > 0).astype(float)
    logcols = [j for j, f in enumerate(FEATURES) if not f.startswith("P_")]
    X[:, logcols] = np.log1p(X[:, logcols])
    assert np.isfinite(X).all()
    return hosts, X


# ── detectors (protocol §5) ─────────────────────────────────────────────────
def s_robust_z(X: np.ndarray) -> np.ndarray:
    med = np.median(X, axis=0)
    scale = 1.4826 * np.median(np.abs(X - med), axis=0)
    mad0 = scale == 0
    scale[mad0] = 1.2533 * np.mean(np.abs(X - med), axis=0)[mad0]
    keep = scale > 0
    if not keep.any():
        return np.zeros(len(X))
    return np.max(np.abs((X[:, keep] - med[keep]) / scale[keep]), axis=1)


def s_iforest(X: np.ndarray) -> np.ndarray:
    from sklearn.ensemble import IsolationForest
    out = np.zeros(len(X))
    for s in IF_SEEDS:
        m = IsolationForest(n_estimators=200, max_samples="auto", max_features=1.0,
                            bootstrap=False, random_state=s).fit(X)
        out += -m.score_samples(X)
    return out / len(IF_SEEDS)


def s_lof(X: np.ndarray, k: int = 20) -> np.ndarray:
    from sklearn.neighbors import LocalOutlierFactor
    med = np.median(X, axis=0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
    iqr[iqr == 0] = 1.0
    Z = (X - med) / iqr
    m = LocalOutlierFactor(n_neighbors=min(k, len(X) - 1), metric="euclidean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # duplicate points: sklearn warns, result is defined
        m.fit_predict(Z)
    return -m.negative_outlier_factor_


DET_FN = {"robust_z": s_robust_z, "iforest": s_iforest, "lof": s_lof,
          "lof_k10": lambda X: s_lof(X, 10), "lof_k35": lambda X: s_lof(X, 35)}


@lru_cache(maxsize=None)
def scores(run: str, det: str, manifest_aware: bool) -> tuple[tuple[str, float], ...]:
    hosts, X = features(run, manifest_aware)
    if len(hosts) < 3:
        return tuple((h, 0.0) for h in hosts)
    s = DET_FN[det](X)
    assert np.isfinite(s).all(), (run, det)
    return tuple(zip(hosts, (float(v) for v in s)))


# ── calibration (protocol §6) ───────────────────────────────────────────────
def calibrate(train: dict[str, dict[str, float]], alpha: float) -> float:
    """Smallest tau with macro clean-run alert rate over the training clean runs <= alpha."""
    arrs = [np.array(sorted(v.values())) for v in train.values()]
    cands = np.unique(np.concatenate(arrs))
    rate = lambda t: st.mean(float((a >= t).sum()) / len(a) for a in arrs)
    lo, hi = 0, len(cands) - 1                 # rate is non-increasing in tau
    if rate(cands[hi]) > alpha:
        return float(np.nextafter(cands[hi], math.inf))   # flag nothing
    while lo < hi:
        mid = (lo + hi) // 2
        if rate(cands[mid]) <= alpha:
            hi = mid
        else:
            lo = mid + 1
    return float(cands[lo])


# ── repair + metrics ────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def clean_dag(clean: str):
    return build_dag(Path(clean))


@lru_cache(maxsize=None)
def dfx_of(clean: str, faulted: str):
    return contamination_closure(Path(clean), Path(faulted))


def repair(cell: dict, flagged: frozenset) -> dict:
    """Host-grain mapper of detector_threshold_sweep_v2.score_cell, on an arbitrary flagged set."""
    dag = clean_dag(cell["clean_dir"])
    scope = {n for n, v in dag.nodes.items() if v.get("repairable") and v.get("target") in flagged}
    ev = evaluate_scope(Path(cell["clean_dir"]), dfx_of(cell["clean_dir"], cell["faulted_dir"]), scope, dag=dag)
    return {"success": bool(ev["success"]), "cost": float(ev["cost"])}


def pooled_auc(pts: list[tuple[float, bool]]) -> dict:
    """Same algorithm as detector_pr_roc.py main() part (2)."""
    n_pos = sum(1 for _, y in pts if y); n_neg = len(pts) - n_pos
    order = sorted(pts, key=lambda p: -p[0])
    pr, roc, tp, fp, i = [], [], 0, 0, 0
    while i < len(order):                       # sweep unique thresholds high -> low
        t = order[i][0]
        while i < len(order) and order[i][0] == t:
            tp += order[i][1]; fp += (not order[i][1]); i += 1
        pr.append((tp / n_pos if n_pos else 0.0, tp / (tp + fp) if (tp + fp) else 1.0))
        roc.append((fp / n_neg if n_neg else 0.0, tp / n_pos if n_pos else 0.0))

    def auc(xy):
        xy = sorted(xy)
        return sum((xy[i + 1][0] - xy[i][0]) * (xy[i + 1][1] + xy[i][1]) / 2 for i in range(len(xy) - 1))
    return {"auc_roc_raw": auc(roc + [(0.0, 0.0), (1.0, 1.0)]), "auc_pr_raw": auc(pr),
            "n_pos": n_pos, "n_neg": n_neg}


def org_mean_ci(rows: list[dict], key) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["org"]].append(key(r))
    means = [sum(v) / len(v) for _, v in sorted(by.items())]
    if len(means) < 2:
        return {"org_mean": round(means[0], 4) if means else None, "ci95": None, "n_orgs": len(means)}
    m, lo, hi = bootstrap_ci(means, B=10_000, seed=42)
    return {"org_mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "n_orgs": len(means)}


def paired_ci(rows_a: list[dict], rows_b: list[dict], key) -> dict:
    """Org-level paired difference a - b (same cells)."""
    da, db = defaultdict(list), defaultdict(list)
    for r in rows_a:
        da[r["org"]].append(key(r))
    for r in rows_b:
        db[r["org"]].append(key(r))
    diffs = [st.mean(da[o]) - st.mean(db[o]) for o in sorted(da)]
    if len(diffs) < 2:
        return {"diff_org_mean": round(diffs[0], 4) if diffs else None, "ci95": None}
    m, lo, hi = bootstrap_ci(diffs, B=10_000, seed=42)
    return {"diff_org_mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "n_orgs": len(diffs)}


def clean_views(flag_counts: list[tuple[int, int]]) -> dict:
    """flag_counts: per clean run (n_flagged, n_hosts)."""
    return {"macro_rate": round(st.mean(f / n for f, n in flag_counts), 4),
            "micro_rate": round(sum(f for f, _ in flag_counts) / sum(n for _, n in flag_counts), 4),
            "mean_alerted_hosts_per_run": round(st.mean(f for f, _ in flag_counts), 2),
            "frac_runs_with_alert": round(sum(1 for f, _ in flag_counts if f) / len(flag_counts), 4),
            "n_runs": len(flag_counts)}


def summarize(rows: list[dict], B0: float) -> dict:
    out = {"n_cells": len(rows),
           "detect1": org_mean_ci(rows, lambda r: r["detect1"]),
           "detect3": org_mean_ci(rows, lambda r: r["detect3"])}
    if rows and "success" in rows[0]:
        out.update({
            "success": org_mean_ci(rows, lambda r: float(r["success"])),
            "cost_over_Cstar_all": org_mean_ci(rows, lambda r: r["cost"] / r["C_star"]),
            "cost_over_Cstar_success_only": round(st.mean(r["cost"] / r["C_star"] for r in rows if r["success"]), 4)
            if any(r["success"] for r in rows) else None,
            "repair_at_B0": org_mean_ci(rows, lambda r: float(r["success"] and r["cost"] <= B0)),
            "precision": round(st.mean(r["precision"] for r in rows), 4),
            "recall": round(st.mean(r["recall"] for r in rows), 4),
            "pooled_success": round(sum(r["success"] for r in rows) / len(rows), 4)})
    fam = defaultdict(list)
    for r in rows:
        fam[r["cell"].split("@")[0].split("-")[0]].append(r)
    out["by_family"] = {f: {k: round(st.mean(float(x[k]) for x in rs), 4)
                            for k in ("detect1", "detect3", "success") if k in rs[0]}
                        for f, rs in sorted(fam.items())}
    var = defaultdict(list)
    for r in rows:
        var[r["cell"]].append(r)
    out["by_variant"] = {v: {k: round(st.mean(float(x[k]) for x in rs), 4)
                             for k in ("detect1", "detect3", "success") if k in rs[0]}
                         for v, rs in sorted(var.items())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--orgs", default="")
    ap.add_argument("--tex", default="")
    args = ap.parse_args()
    grid = json.loads((D / "e3_grid_v1.json").read_text())["cells"]
    clean_of = {}
    for c in grid:
        clean_of.setdefault(c["org"], c["clean_dir"])
    orgs_all = sorted(clean_of)
    eval_orgs = [o for o in args.orgs.split(",") if o] or orgs_all
    cells = [c for c in grid if c["org"] in eval_orgs]
    B0 = float(json.loads((D / "e3_oracle_stats_v1.json").read_text())["summary"]["B0_median_singlepoint_cost"])
    frz_det = {(c["org"], c["cell"]): c for c in json.loads((D / "e3_detector_v1.json").read_text())["cells"]}
    frz_top = {(c["org"], c["cell"]): c for c in json.loads((D / "rq2_combined_topk_v1.json").read_text())["cells"]}

    res: dict = {"protocol": str(PROTOCOL.relative_to(REPO)), "inputs": {
        "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
        "grid_sha256": sha256_file(D / "e3_grid_v1.json"), "eval_orgs": eval_orgs,
        "alphas": ALPHAS, "combined_t": COMBINED_T, "features": FEATURES, "B0": B0}}

    # ── combined reference (protocol §7) ──
    comb_scores = {c["faulted_dir"]: _host_scores(Path(c["faulted_dir"])) for c in cells}
    comb_clean = {o: _host_scores(Path(clean_of[o])) for o in orgs_all}
    combined = {}
    for tag, t in COMBINED_T.items():
        rows = []
        for c in cells:
            truth, _ = _truth(c["inject_node"])
            fd = Path(c["faulted_dir"])
            flagged = frozenset(detector_localize(fd)["flagged_hosts"]
                                | peer_anomaly_hosts(fd, min_base_rate=t) | manifest_dropped_hosts(fd))
            sc = comb_scores[c["faulted_dir"]]
            rows.append({"org": c["org"], "cell": c["cell"], "C_star": c["C_star"],
                         "detect1": tie_aware_hit(sc, truth, 1), "detect3": tie_aware_hit(sc, truth, K),
                         "precision": len(truth & flagged) / len(flagged) if flagged else 0.0,
                         "recall": len(truth & flagged) / len(truth), "n_flagged": len(flagged),
                         **repair(c, flagged)})
        clean_counts = []
        for o in eval_orgs:
            cd = Path(clean_of[o])
            fl = detector_localize(cd)["flagged_hosts"] | peer_anomaly_hosts(cd, min_base_rate=t) | manifest_dropped_hosts(cd)
            clean_counts.append((len(fl), len(comb_clean[o])))
        combined[tag] = {"t": t, "rows": rows, "summary": summarize(rows, B0), "clean_run_alerts": clean_views(clean_counts)}

    # faithfulness (per cell; plus the pooled numbers when all orgs are evaluated)
    mism = []
    for r in combined["primary"]["rows"]:
        k = (r["org"], r["cell"])
        if bool(frz_det[k]["combined"]["success"]) != r["success"]:
            mism.append((k, "success"))
        if abs(frz_top[k]["detect1_tie"] - r["detect1"]) > 1e-6 or abs(frz_top[k]["detectk_tie"] - r["detect3"]) > 1e-6:
            mism.append((k, "detect"))
    pts = [(s, h in _truth(c["inject_node"])[0]) for c in cells for h, s in comb_scores[c["faulted_dir"]].items()]
    pts += [(s, False) for o in eval_orgs for s in comb_clean[o].values()]
    comb_auc = pooled_auc(pts)
    faith = {"per_cell_mismatches": mism, "n_cells_checked": len(cells)}
    if eval_orgs == orgs_all:
        faith.update({"success_t05": combined["primary"]["summary"]["pooled_success"], "success_frozen": 0.806,
                      "auc_roc": round(comb_auc["auc_roc_raw"], 3), "auc_roc_frozen": 0.914,
                      "auc_pr": round(comb_auc["auc_pr_raw"], 3), "auc_pr_frozen": 0.269,
                      "clean_macro_t05": combined["primary"]["clean_run_alerts"]["macro_rate"], "clean_macro_frozen": 0.1055})
        faith["pass"] = (not mism and abs(faith["success_t05"] - 0.806) < 0.0006
                         and faith["auc_roc"] == 0.914 and faith["auc_pr"] == 0.269
                         and abs(faith["clean_macro_t05"] - 0.1055) < 0.0006)
    else:
        faith["pass"] = not mism
    res["faithfulness"] = faith
    res["combined"] = {tag: {k: v for k, v in blk.items() if k != "rows"} | {"auc": comb_auc}
                       for tag, blk in combined.items()}

    # ── baselines ──
    base: dict = {}
    for ma in (True, False):
        for det in MAIN_DETECTORS + (SENS_DETECTORS if ma else ()):
            key = f"{det}|{'manifest_aware' if ma else 'manifest_free'}"
            clean_sc = {o: dict(scores(clean_of[o], det, ma)) for o in orgs_all}
            blk = {"thresholds": {}, "by_alpha": {}}
            # threshold-free metrics
            tf_rows, pts = [], []
            for c in cells:
                truth, _ = _truth(c["inject_node"])
                sc = dict(scores(c["faulted_dir"], det, ma))
                tf_rows.append({"org": c["org"], "cell": c["cell"],
                                "detect1": tie_aware_hit(sc, truth, 1), "detect3": tie_aware_hit(sc, truth, K)})
                pts += [(s, h in truth) for h, s in sc.items()]
            pts += [(s, False) for o in eval_orgs for s in clean_sc[o].values()]
            a = pooled_auc(pts)
            blk["threshold_free"] = summarize(tf_rows, B0) | {"auc_roc": round(a["auc_roc_raw"], 4),
                                                             "auc_pr": round(a["auc_pr_raw"], 4)}
            for tag, alpha in ALPHAS.items():
                if det in SENS_DETECTORS and tag != "primary":
                    continue
                taus = {}
                for o in eval_orgs:
                    train = {o2: clean_sc[o2] for o2 in orgs_all if o2 != o}
                    assert o not in train and len(train) == len(orgs_all) - 1   # protocol §6 hold-out
                    taus[o] = calibrate(train, alpha)
                rows, clean_counts = [], []
                for o in eval_orgs:
                    cs = clean_sc[o]
                    clean_counts.append((sum(1 for s in cs.values() if s >= taus[o]), len(cs)))
                for c, tf in zip(cells, tf_rows):
                    truth, _ = _truth(c["inject_node"])
                    sc = dict(scores(c["faulted_dir"], det, ma))
                    flagged = frozenset(h for h, s in sc.items() if s >= taus[c["org"]])
                    rows.append({**tf, "C_star": c["C_star"], "n_flagged": len(flagged),
                                 "precision": len(truth & flagged) / len(flagged) if flagged else 0.0,
                                 "recall": len(truth & flagged) / len(truth), **repair(c, flagged)})
                blk["thresholds"][tag] = taus
                blk["by_alpha"][tag] = {"alpha": alpha, "summary": summarize(rows, B0),
                                        "clean_run_alerts": clean_views(clean_counts),
                                        "paired_vs_combined": {
                                            m: paired_ci(combined[tag]["rows"], rows, f)
                                            for m, f in (("success", lambda r: float(r["success"])),
                                                         ("detect1", lambda r: r["detect1"]),
                                                         ("detect3", lambda r: r["detect3"]))},
                                        "cells": rows}
            base[key] = blk
            print(f"{key:32s} AUC-ROC {blk['threshold_free']['auc_roc']:.3f} AUC-PR {blk['threshold_free']['auc_pr']:.3f} "
                  f"D@1 {blk['threshold_free']['detect1']['org_mean']} "
                  f"Succ@a1 {blk['by_alpha']['primary']['summary']['success']['org_mean']} "
                  f"clean-macro {blk['by_alpha']['primary']['clean_run_alerts']['macro_rate']}", flush=True)
    res["baselines"] = base

    # ── descriptive: robust-z vs peer-anomaly (protocol §9) ──
    from scipy.stats import spearmanr
    rho, jac = [], []
    rz = base["robust_z|manifest_aware"]
    for o in eval_orgs:
        cd = clean_of[o]
        peer = comb_clean[o]
        z = dict(scores(cd, "robust_z", True))
        common = sorted(set(peer) & set(z))
        if len(common) > 2 and len({peer[h] for h in common}) > 1 and len({z[h] for h in common}) > 1:
            rho.append(spearmanr([peer[h] for h in common], [z[h] for h in common]).statistic)
        fz = {h for h, s in z.items() if s >= rz["thresholds"]["primary"][o]}
        fp = peer_anomaly_hosts(Path(cd), min_base_rate=0.5)
        jac.append(len(fz & fp) / len(fz | fp) if (fz | fp) else 1.0)
    res["descriptive"] = {"robust_z_vs_peer": {
        "mean_spearman_clean_hosts": round(float(np.mean(rho)), 4) if rho else None, "n_runs_rho": len(rho),
        "mean_jaccard_flagged_clean_alpha_primary": round(float(np.mean(jac)), 4),
        "reimplementation_check": bool(np.mean(jac) >= 0.8)}}

    # best baseline by primary Success point estimate (manifest-aware, main detectors)
    cand = {d: base[f"{d}|manifest_aware"]["by_alpha"]["primary"]["summary"]["success"]["org_mean"] for d in MAIN_DETECTORS}
    best = max(sorted(cand), key=lambda d: cand[d])
    pc = base[f"{best}|manifest_aware"]["by_alpha"]["primary"]["paired_vs_combined"]["success"]
    verdict = None
    if pc.get("ci95"):
        verdict = ("combined_better" if pc["ci95"][0] > 0 else "baseline_better" if pc["ci95"][1] < 0 else "comparable")
    res["decision"] = {"best_baseline": best, "success_org_means": cand, "paired_success_combined_minus_best": pc,
                       "verdict_protocol_s11": verdict}
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=float))
    print(json.dumps({"faithfulness": {k: v for k, v in faith.items() if k != "per_cell_mismatches"},
                      "n_mismatch": len(mism), "decision": res["decision"], "descriptive": res["descriptive"]}, default=float))

    if args.tex:
        write_tex(Path(args.tex), res)
    return 0 if faith["pass"] else 1


def write_tex(p: Path, res: dict) -> None:
    def ci(x):
        return f"{x['org_mean']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]" if x.get("ci95") else f"{x['org_mean']:.3f}"
    lines = [r"\begin{tabular}{l rrrr rr}", r"\toprule",
             r"detector & AUC-ROC & AUC-PR & Detect@1 & Detect@3 & clean alert (macro) & end-to-end Success \\",
             r"\midrule"]
    c = res["combined"]["primary"]
    lines.append(r"combined (validator $\cup$ peer@0.5 $\cup$ manifest) & "
                 f"{c['auc']['auc_roc_raw']:.3f} & {c['auc']['auc_pr_raw']:.3f} & {ci(c['summary']['detect1'])} & "
                 f"{ci(c['summary']['detect3'])} & {c['clean_run_alerts']['macro_rate']:.3f} & {ci(c['summary']['success'])} \\\\")
    names = {"robust_z": "robust-$z$", "iforest": "Isolation Forest", "lof": "LOF ($k{=}20$)"}
    for ma in ("manifest_aware", "manifest_free"):
        lines.append(r"\midrule" + "\n" + rf"\multicolumn{{7}}{{l}}{{\emph{{{ma.replace('_', '-')} candidates}}}} \\")
        for d in MAIN_DETECTORS:
            b = res["baselines"][f"{d}|{ma}"]
            tf, op = b["threshold_free"], b["by_alpha"]["primary"]
            lines.append(f"{names[d]} & {tf['auc_roc']:.3f} & {tf['auc_pr']:.3f} & {ci(tf['detect1'])} & {ci(tf['detect3'])} & "
                         f"{op['clean_run_alerts']['macro_rate']:.3f} & {ci(op['summary']['success'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    p.write_text("% generated by experiments/exp_arcs/revision_v2/step2_baselines.py — do not edit\n"
                 "% thresholds: leave-one-organization-out on clean runs at the combined detector's macro\n"
                 "% clean-run alert rate (alpha=0.106); Detect@k tie-aware; [..] = org-level bootstrap 95% CI\n"
                 + "\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
