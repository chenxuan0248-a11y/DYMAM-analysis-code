"""Revision v2 · step 2 — POST-HOC analyses, round 2 (after external review of STEP2_RESULT.md).
Not part of the frozen protocol; does not change its §11 verdict.

  D. Per-cell pre-coverage: for every faulted cell, is the ACTUAL injected host already in the
     method's alert set on that organization's CLEAN run (same threshold), and is it flagged on
     the faulted run? Success is broken down by pre-coverage.
  E. Repair@B (success with spec cost <= B) for every method on one budget grid, next to each
     method's realized clean-run alert views.
  F. Score identity check for the robust-z/peer descriptive (protocol §9): on clean runs the
     combined score equals the peer score iff validator and manifest flag no clean host.

    .venv/bin/python experiments/exp_arcs/revision_v2/step2_posthoc2.py [out.json]
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
HERE = REPO / "experiments/exp_arcs/revision_v2"
sys.argv = sys.argv[:2] if len(sys.argv) > 1 else sys.argv
import experiments.exp_arcs.revision_v2.step2_baselines as S  # noqa: E402

B_GRID = ("B0", 5, 10, 25, 50, 100, 200, 410)


def combined_flagged(run: Path, t: float) -> set[str]:
    return (S.detector_localize(run)["flagged_hosts"] | S.peer_anomaly_hosts(run, min_base_rate=t)
            | S.manifest_dropped_hosts(run))


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "step2_posthoc2_v1.json"
    bp = HERE / "step2_baselines_v1.json"
    r = json.loads(bp.read_text())
    grid = json.loads((S.D / "e3_grid_v1.json").read_text())["cells"]
    clean_of: dict = {}
    for c in grid:
        clean_of.setdefault(c["org"], c["clean_dir"])
    bd = {o: set(json.loads((REPO / f"data/gt/{o}_gt.json").read_text()).get("base_domains", [])) for o in clean_of}
    B0 = r["inputs"]["B0"]

    # per-method: flagged-set functions on clean / faulted, per-cell success+cost
    methods: dict = {}
    for tag, t in S.COMBINED_T.items():
        cl = {o: combined_flagged(Path(cd), t) for o, cd in clean_of.items()}
        rows = []
        for c in grid:
            fl = frozenset(combined_flagged(Path(c["faulted_dir"]), t))
            rows.append({"org": c["org"], "cell": c["cell"], "inject_node": c["inject_node"], "C_star": c["C_star"],
                         "flagged": fl, **S.repair(c, fl)})
        methods[f"combined_t{t}"] = {"clean": cl, "rows": rows,
                                     "clean_views": S.clean_views([(len(cl[o]), len(S._host_scores(Path(clean_of[o]))))
                                                                   for o in clean_of])}
        rows_bd = []
        for c, x in zip(grid, rows):
            fl = frozenset(x["flagged"] | bd[c["org"]])
            rows_bd.append({**x, "flagged": fl, **S.repair(c, fl)})
        cl_bd = {o: cl[o] | bd[o] for o in clean_of}
        methods[f"combined_t{t}+base_domains"] = {
            "clean": cl_bd, "rows": rows_bd,
            "clean_views": S.clean_views([(len(cl_bd[o]), len(S._host_scores(Path(clean_of[o])))) for o in clean_of])}
    for det in ("iforest", "robust_z", "lof"):
        for tag in ("primary", "secondary"):
            blk = r["baselines"][f"{det}|manifest_aware"]
            taus = blk["thresholds"][tag]
            cl = {o: {h for h, s in S.scores(cd, det, True) if s >= taus[o]} for o, cd in clean_of.items()}
            rows = []
            for c, x in zip(grid, blk["by_alpha"][tag]["cells"]):
                assert (c["org"], c["cell"]) == (x["org"], x["cell"])
                fl = frozenset(h for h, s in S.scores(c["faulted_dir"], det, True) if s >= taus[c["org"]])
                assert len(fl) == x["n_flagged"]
                rows.append({"org": c["org"], "cell": c["cell"], "inject_node": c["inject_node"], "C_star": c["C_star"],
                             "flagged": fl, "success": x["success"], "cost": x["cost"]})
            methods[f"{det}_alpha{S.ALPHAS[tag]}"] = {"clean": cl, "rows": rows,
                                                      "clean_views": blk["by_alpha"][tag]["clean_run_alerts"]}

    # D. per-cell pre-coverage of the actual injected host(s)
    D: dict = {}
    for name, m in methods.items():
        agg = defaultdict(lambda: defaultdict(int))
        for x in m["rows"]:
            truth, _ = S._truth(x["inject_node"])
            pre = bool(truth & m["clean"][x["org"]])            # any injected host already alerted on clean
            pre_all = truth <= m["clean"][x["org"]]
            fl = bool(truth & x["flagged"])
            for grp in (x["cell"], "ALL", "F4@cert+reg" if x["cell"] in ("F4@certificate", "F4@registration") else "other_7"):
                a = agg[grp]
                a["n"] += 1; a["pre_covered_any"] += pre; a["pre_covered_all"] += pre_all
                a["flagged_on_faulted"] += fl; a["success"] += x["success"]
                a["success_and_pre_covered"] += x["success"] and pre
                a["success_not_pre_covered"] += x["success"] and not pre
        D[name] = {g: dict(v) for g, v in sorted(agg.items())}

    # E. Repair@B on one grid
    def at(rows, B):
        by = defaultdict(list)
        for x in rows:
            by[x["org"]].append(float(x["success"] and x["cost"] <= B))
        return round(sum(sum(v) / len(v) for v in by.values()) / len(by), 4)
    E = {}
    for name, m in methods.items():
        E[name] = {"repair_at_B": {str(b): at(m["rows"], B0 if b == "B0" else b) for b in B_GRID},
                   "success": at(m["rows"], float("inf")),
                   "mean_cost_over_Cstar": round(sum(x["cost"] / x["C_star"] for x in m["rows"]) / len(m["rows"]), 4),
                   "clean_run_alerts": m["clean_views"]}

    # F. score identity on clean runs
    nv = sum(len(S.detector_localize(Path(cd))["flagged_hosts"]) for cd in clean_of.values())
    nm = sum(len(S.manifest_dropped_hosts(Path(cd))) for cd in clean_of.values())
    F = {"validator_flagged_clean_hosts": nv, "manifest_dropped_clean_hosts": nm,
         "combined_score_equals_peer_score_on_clean_hosts": nv == 0 and nm == 0,
         "note": "step2_baselines.py §9 correlates robust-z with detector_pr_roc._host_scores (combined score); "
                 "on clean runs it equals the peer score exactly when both counts are 0"}

    res = {"note": "POST-HOC round 2; not part of the frozen STEP2 protocol; does not change its §11 verdict",
           "inputs": {"step2_baselines_v1_sha256": hashlib.sha256(bp.read_bytes()).hexdigest(),
                      "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "B0": B0},
           "D_precoverage_of_injected_host": D, "E_repair_at_B": E, "F_score_identity": F}
    out.write_text(json.dumps(res, indent=1, sort_keys=True, default=float))
    for name in methods:
        d = D[name]
        print(f"{name:30s} cert+reg: pre-covered {d['F4@cert+reg']['pre_covered_any']}/{d['F4@cert+reg']['n']}, "
              f"success {d['F4@cert+reg']['success']} (of which pre-covered {d['F4@cert+reg']['success_and_pre_covered']}) | "
              f"other7 pre {d['other_7']['pre_covered_any']}/{d['other_7']['n']} | R@B {E[name]['repair_at_B']} "
              f"| clean macro {E[name]['clean_run_alerts']['macro_rate']}")
    print(json.dumps(F))
    return 0


if __name__ == "__main__":
    sys.exit(main())
