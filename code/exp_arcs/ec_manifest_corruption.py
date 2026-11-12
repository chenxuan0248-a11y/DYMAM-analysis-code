"""experiments/exp_arcs/ec_manifest_corruption — E-C: how much the combined detector depends on
the discovery manifest record (protocol: experiments/exp_arcs/protocols/EC_PROTOCOL.md, frozen
before any run). A simulation on the recorded manifest, NOT a live discovery early-abort.

Manifest flags are Mf = resolved_hosts - hosts_with_evidence, so deleting hosts X from the manifest
gives Mf' = Mf - X: deletions can only remove flags, and only the hosts in Mf matter. For each cell
we enumerate every surviving subset of the at-risk hosts and weight it exactly, so all figures are
exact expectations over the corruption (no Monte Carlo band).

    python -m experiments.exp_arcs.ec_manifest_corruption [out.json]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from evaluation.stat_test import bootstrap_ci
from experiments.exp_arcs.contamination import contamination_closure
from experiments.exp_arcs.dag import build_dag
from experiments.exp_arcs.detector_eval import (
    detector_localize, manifest_dropped_hosts, peer_anomaly_hosts)
from experiments.exp_arcs.detector_pr_roc import _fam_dist, _host_scores
from experiments.exp_arcs.localizer_baselines import _truth
from experiments.exp_arcs.rq2_combined_topk import tie_aware_hit
from experiments.exp_arcs.select import evaluate_scope

D = Path("experiments/exp_arcs")
PS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
MODES = ("drop_true", "drop_irrelevant", "random_censorship", "absent")


def peer_scores(run_dir: Path) -> dict[str, float]:
    host_fams, fam_hosts = _fam_dist(run_dir)
    n = len(host_fams)
    out: dict[str, float] = {}
    if n:
        base = {f: len(hs) / n for f, hs in fam_hosts.items()}
        for h, fs in host_fams.items():
            miss = [base[f] for f in fam_hosts if f not in fs]
            out[h] = max(miss) if miss else 0.0
    return out


def scores_with(peer: dict, V: set, Mf_kept: set) -> dict[str, float]:
    s = dict(peer)
    for h in V | Mf_kept:
        s[h] = 1.0
    return s


def subsets(at_risk: list[str]):
    for r in range(len(at_risk) + 1):
        for S in combinations(at_risk, r):
            yield frozenset(S)


def weight(n_risk: int, n_kept: int, p: float) -> float:
    return (1 - p) ** n_kept * p ** (n_risk - n_kept)


def at_risk_hosts(mode: str, Mf: set, T: set) -> list[str] | None:
    if mode == "drop_true":
        return sorted(Mf & T)
    if mode == "drop_irrelevant":
        return sorted(Mf - T)
    if mode == "random_censorship":
        return sorted(Mf)
    return None  # absent


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else D / "ec_manifest_corruption_v1.json"
    grid = json.loads((D / "e3_grid_v1.json").read_text())["cells"]
    det_frozen = {(c["org"], c["cell"]): c for c in json.loads((D / "e3_detector_v1.json").read_text())["cells"]}

    rows = []
    for c in grid:
        clean, faulted = Path(c["clean_dir"]), Path(c["faulted_dir"])
        dag = build_dag(clean)
        dfx = contamination_closure(clean, faulted)
        V = detector_localize(faulted)["flagged_hosts"]
        P = peer_anomaly_hosts(faulted)
        Mf = manifest_dropped_hosts(faulted)
        T, _ = _truth(c["inject_node"])
        peer = peer_scores(faulted)
        assert scores_with(peer, V, Mf) == _host_scores(faulted), ("score mismatch", c["org"], c["cell"])
        cache: dict = {}

        def metrics(kept: frozenset) -> dict:
            if kept not in cache:
                flagged = V | P | set(kept)
                scope = {n for n, v in dag.nodes.items() if v.get("repairable") and v.get("target") in flagged}
                ev = evaluate_scope(clean, dfx, scope, dag=dag)
                sc = scores_with(peer, V, set(kept))
                cache[kept] = {"success": float(bool(ev["success"])),
                               "cost_over_Cstar": ev["cost"] / c["C_star"] if c["C_star"] else None,
                               "detect1": tie_aware_hit(sc, T, 1), "detect3": tie_aware_hit(sc, T, 3)}
            return cache[kept]

        base = metrics(frozenset(Mf))
        fz = det_frozen[(c["org"], c["cell"])]["combined"]
        assert bool(base["success"]) == bool(fz["success"]), ("success mismatch vs e3_detector", c["org"], c["cell"])
        assert base["cost_over_Cstar"] * c["C_star"] == fz["cost"], ("cost mismatch", c["org"], c["cell"])

        row = {"org": c["org"], "cell": c["cell"], "family": c["cell"].split("@")[0].split("-")[0],
               "n_Mf": len(Mf), "n_Mf_true": len(Mf & T), "n_Mf_irrelevant": len(Mf - T), "results": {}}
        for mode in MODES:
            risk = at_risk_hosts(mode, Mf, T)
            for p in (PS if mode != "absent" else (None,)):
                acc = {"success": 0.0, "cost_over_Cstar": 0.0, "detect1": 0.0, "detect3": 0.0}
                if risk is None:
                    m = metrics(frozenset())
                    acc = dict(m)
                else:
                    safe = Mf - set(risk)
                    for S in subsets(risk):
                        w = weight(len(risk), len(S), p)
                        if w == 0.0:
                            continue
                        m = metrics(frozenset(safe | S))
                        for k in acc:
                            acc[k] += w * m[k]
                key = mode if p is None else f"{mode}|p={p}"
                row["results"][key] = {k: round(v, 10) for k, v in acc.items()}
        rows.append(row)

    # clean false-alarm rate under random censorship (deletions can only remove manifest flags)
    clean_by_org = {}
    for c in grid:
        clean_by_org.setdefault(c["org"], c["clean_dir"])
    clean_fa = {}
    for org, cd in clean_by_org.items():
        cdp = Path(cd)
        hosts = set(_fam_dist(cdp)[0])
        V = detector_localize(cdp)["flagged_hosts"]; P = peer_anomaly_hosts(cdp); Mf = manifest_dropped_hosts(cdp)
        per_p = {}
        for p in PS:
            e = 0.0
            for S in subsets(sorted(Mf)):
                w = weight(len(Mf), len(S), p)
                if w:
                    e += w * len(V | P | set(S)) / max(1, len(hosts))
            per_p[f"p={p}"] = round(e, 10)
        clean_fa[org] = {"n_hosts": len(hosts), "n_manifest_flags_clean": len(Mf), "fa_rate": per_p}

    def summarize(sel) -> dict:
        res = {}
        keys = sorted({k for r in rows for k in r["results"]})
        for key in keys:
            entry = {}
            for metric in ("success", "cost_over_Cstar", "detect1", "detect3"):
                by_org = defaultdict(list)
                for r in rows:
                    if sel(r):
                        by_org[r["org"]].append(r["results"][key][metric])
                if not by_org:
                    continue
                pooled = [v for vs in by_org.values() for v in vs]
                means = [sum(v) / len(v) for _, v in sorted(by_org.items())]
                m, lo, hi = bootstrap_ci(means, B=10_000, seed=42)
                entry[metric] = {"pooled": round(sum(pooled) / len(pooled), 4),
                                 "org_mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
            res[key] = entry
        return res

    variants = sorted({r["cell"] for r in rows})
    summary = {
        "all": summarize(lambda r: True),
        "by_family": {f: summarize(lambda r, f=f: r["family"] == f) for f in ("F4", "F10")},
        "by_variant": {v: summarize(lambda r, v=v: r["cell"] == v) for v in variants},
        "Mf_composition": {"cells_with_Mf": sum(1 for r in rows if r["n_Mf"]),
                           "total_Mf_true": sum(r["n_Mf_true"] for r in rows),
                           "total_Mf_irrelevant": sum(r["n_Mf_irrelevant"] for r in rows)},
        "clean_false_alarm": {f"p={p}": round(sum(v["fa_rate"][f"p={p}"] for v in clean_fa.values()) / len(clean_fa), 6)
                              for p in PS},
    }
    abl = json.loads((D / "e7_localization_ablation_v1.json").read_text())
    self_peer = [x["self+peer"]["success"] for x in abl["cells"]]
    rq2 = json.loads((D / "rq2_combined_topk_v1.json").read_text())["summary"]["pooled"]
    a = summary["all"]
    selfcheck = {
        "p0_success_all_modes_equal_e3_detector_combined": all(
            a[f"{m}|p=0.0"]["success"]["pooled"] == round(sum(float(det_frozen[(r["org"], r["cell"])]["combined"]["success"]) for r in rows) / len(rows), 4)
            for m in ("drop_true", "drop_irrelevant", "random_censorship")),
        "p0_detect1_equals_rq2_tie_aware": a["random_censorship|p=0.0"]["detect1"]["pooled"] == rq2["detect1_tie_aware"],
        "absent_success_equals_ablation_self_plus_peer": a["absent"]["success"]["pooled"] == round(sum(self_peer) / len(self_peer), 4),
        "random_p1_equals_absent": all(abs(r["results"]["random_censorship|p=1.0"][k] - r["results"]["absent"][k]) < 1e-12
                                       for r in rows for k in ("success", "detect1", "detect3")
                                       if r["results"]["absent"][k] is not None),
    }
    summary["selfcheck"] = selfcheck
    out.write_text(json.dumps({"phase": "E-C manifest-record corruption simulation (exact expectation)",
                               "protocol": "experiments/exp_arcs/protocols/EC_PROTOCOL.md",
                               "source": "experiments/exp_arcs/e3_grid_v1.json", "p_grid": PS,
                               "summary": summary, "clean_false_alarm_by_org": clean_fa, "cells": rows},
                              indent=1, sort_keys=True))
    print(json.dumps({"selfcheck": selfcheck, "Mf": summary["Mf_composition"],
                      "clean_FA": summary["clean_false_alarm"]}, indent=1))
    for mode in MODES:
        for p in (PS if mode != "absent" else (None,)):
            key = mode if p is None else f"{mode}|p={p}"
            fam = summary["by_family"]
            print(f"{key:28s} success all={a[key]['success']['pooled']:.4f} F4={fam['F4'][key]['success']['pooled']:.4f} "
                  f"F10={fam['F10'][key]['success']['pooled']:.4f} | detect1={a[key]['detect1']['pooled']:.4f} "
                  f"detect3={a[key]['detect3']['pooled']:.4f} | cost/C*={a[key]['cost_over_Cstar']['pooled']:.2f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
