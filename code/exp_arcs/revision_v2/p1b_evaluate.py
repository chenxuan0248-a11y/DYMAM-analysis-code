"""P1-B new-site study — cell evaluation and statistics (STEP4_NEW_SITE_PROTOCOL_v1.1.md sections 5-7).

Reads the controller's result JSON (p1b_run.py), evaluates every valid cell against the
org's frozen clean bundle, and computes the frozen statistics. All detector configuration
comes from frozen artifacts (combined t = 0.5 / 0.9; Step 2 per-org IF / robust-z
thresholds); nothing is tuned on new-site results.

Self-check (always run first): the same cell-evaluation function applied to original grid
cells of the selfcheck orgs must reproduce the frozen per-cell values (combined: E-C p=0
and rq2_combined_topk; IF / robust-z: step2_baselines_v1.json). Any mismatch aborts.

    .venv/bin/python experiments/exp_arcs/revision_v2/p1b_evaluate.py <run_result.json> <out.json> \
        [--selfcheck-orgs mozilla,reddit] [--R 200] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import step2_baselines as s2  # noqa: E402
import step3_coverage as s3  # noqa: E402

from evaluation.stat_test import bootstrap_ci  # noqa: E402
from experiments.exp_arcs.contamination import contamination_closure  # noqa: E402
from experiments.exp_arcs.counterfactual import vf  # noqa: E402
from experiments.exp_arcs.detector_eval import detector_localize, manifest_dropped_hosts, peer_anomaly_hosts  # noqa: E402
from experiments.exp_arcs.detector_pr_roc import _host_scores  # noqa: E402
from experiments.exp_arcs.ilp_oracle import solve_oracle  # noqa: E402
from experiments.exp_arcs.rq2_combined_topk import tie_aware_hit  # noqa: E402

REPO, D = s2.REPO, s2.D
PROTOCOL = HERE / "STEP4_NEW_SITE_PROTOCOL_v1.1.md"
SITES = HERE / "p1b_site_list_v1.json"
K = s2.K
COMB_T = {"primary": 0.5, "secondary": 0.9}
GEN = ("iforest", "robust_z")
ORIG_VARIANT = {"dns_only": "F4@dns", "dns_multi": "F4@dns", "ao": "F4@active_observation"}
TESTS = {"P1": ("dns_multi", "detect1", "+"), "P2": ("ao", "detect1", "+"),
         "P3": ("dns_multi", "cost_over_Cstar", "-"), "P4": ("ao", "cost_over_Cstar", "-")}
BONF = 0.05 / len(TESTS)
MIN_ORGS = 12


def methods_for_cell(cell: dict, S2: dict) -> dict:
    """Flagged set, scores and metrics of every method on one cell (dirs absolute)."""
    fd, truth, org = Path(cell["faulted_dir"]), set(cell["truth"]), cell["org"]
    out = {}
    sc = _host_scores(fd)
    for tag, t in COMB_T.items():
        fl = frozenset(detector_localize(fd)["flagged_hosts"] | peer_anomaly_hosts(fd, min_base_rate=t)
                       | manifest_dropped_hosts(fd))
        out[f"combined|{tag}"] = (sc, fl)
    for det in GEN:
        gsc = dict(s2.scores(cell["faulted_dir"], det, True))
        for tag in ("primary", "secondary"):
            tau = S2["baselines"][f"{det}|manifest_aware"]["thresholds"][tag][org]
            out[f"{det}|{tag}"] = (gsc, frozenset(h for h, s in gsc.items() if s >= tau))
    res = {}
    for key, (scores, fl) in out.items():
        rep = s2.repair(cell, fl)
        res[key] = {"detect1": tie_aware_hit(scores, truth, 1), "detect3": tie_aware_hit(scores, truth, K),
                    "success": rep["success"], "cost": rep["cost"], "n_flagged": len(fl),
                    "ufn": "U" if not truth & set(scores) else ("F" if truth & fl else "N")}
    return res


def selfcheck(orgs: list[str], S2: dict) -> dict:
    grid = json.loads((D / "e3_grid_v1.json").read_text())["cells"]
    ec = {(c["org"], c["cell"]): c["results"]["drop_true|p=0.0"]
          for c in json.loads((D / "ec_manifest_corruption_v1.json").read_text())["cells"]}
    frz = {det: {(x["org"], x["cell"]): x for x in S2["baselines"][f"{det}|manifest_aware"]["by_alpha"]["primary"]["cells"]}
           for det in GEN}
    mism, n = [], 0
    for c in grid:
        if c["org"] not in orgs or c["cell"] not in ("F4@dns", "F4@active_observation", "F4-2host-dns"):
            continue
        from experiments.exp_arcs.localizer_baselines import _truth
        cell = {"org": c["org"], "clean_dir": c["clean_dir"], "faulted_dir": c["faulted_dir"],
                "truth": sorted(_truth(c["inject_node"])[0]), "C_star": c["C_star"]}
        m = methods_for_cell(cell, S2)
        k = (c["org"], c["cell"])
        n += 1
        e = ec[k]
        if bool(m["combined|primary"]["success"]) != bool(e["success"]) or abs(m["combined|primary"]["detect1"] - e["detect1"]) > 1e-9 \
                or abs(m["combined|primary"]["cost"] / c["C_star"] - e["cost_over_Cstar"]) > 1e-9:
            mism.append([*k, "combined"])
        for det in GEN:
            f = frz[det][k]
            if f["success"] != m[f"{det}|primary"]["success"] or abs(f["cost"] - m[f"{det}|primary"]["cost"]) > 1e-9 \
                    or abs(f["detect1"] - m[f"{det}|primary"]["detect1"]) > 1e-9 or f["n_flagged"] != m[f"{det}|primary"]["n_flagged"]:
                mism.append([*k, det])
    return {"orgs": orgs, "n_cells": n, "mismatches": mism, "pass": n > 0 and not mism}


def org_ci(by: dict, alpha: float = 0.05) -> dict:
    vals = [st.mean(v) for _, v in sorted(by.items())]
    if len(vals) < 2:
        return {"mean": vals[0] if vals else None, "ci": None, "n_orgs": len(vals)}
    m, lo, hi = bootstrap_ci(vals, B=10_000, alpha=alpha, seed=42)
    return {"mean": round(m, 4), "ci": [round(lo, 4), round(hi, 4)], "n_orgs": len(vals)}


def metric(r: dict, key: str, C_star: float) -> float:
    return r["cost"] / C_star if key == "cost_over_Cstar" else float(r[key])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_result"); ap.add_argument("out")
    ap.add_argument("--selfcheck-orgs", default="mozilla,reddit")
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    S2 = json.loads(s3.S2_JSON.read_text())
    sc = selfcheck(args.selfcheck_orgs.split(","), S2)
    print(json.dumps({"selfcheck": sc}))
    if not sc["pass"]:
        return 2

    run = json.loads(Path(args.run_result).read_text())
    site_doc = json.loads(SITES.read_text())
    site_idx = {s["site_id"]: i for i, s in enumerate(site_doc["sites"])}
    site_of = {s["site_id"]: s for s in site_doc["sites"]}
    clean_of = {}
    for c in json.loads((D / "e3_grid_v1.json").read_text())["cells"]:
        clean_of.setdefault(c["org"], str(REPO / c["clean_dir"]))

    rows, structural = [], []
    for rc in run["cells"]:
        s = site_of[rc["site_id"]]
        row = {"site_id": s["site_id"], "org": s["org"], "stratum": s["stratum"], "host": s["host"],
               "status": rc["status"], "is_root": s["is_root"], "cross_exposed": bool(s["used_by_original_variants"]),
               "in_clean_manifest": s["in_clean_manifest"], "n_collateral_changes": rc.get("n_collateral_changes")}
        if rc["status"] != "valid":
            rows.append(row); continue
        clean, faulted = clean_of[s["org"]], rc["bundle_dir"]
        dfx = contamination_closure(Path(clean), Path(faulted))
        cf = vf(clean, faulted)
        ilp = solve_oracle(clean, faulted)
        n_rep = cf["n_outputs_leaf_repairable"]
        row.update({"n_outputs_leaf_repairable": n_rep, "has_consequence": n_rep > 0,
                    "C_star": ilp["C_star"], "ilp_status": ilp["status"]})
        cell = {"org": s["org"], "cell": f"F4@{s['family']}", "clean_dir": clean, "faulted_dir": faulted,
                "truth": [s["host"]], "C_star": ilp["C_star"]}
        fi = s3.run_info(faulted)
        in_cand = s["host"] in s3.ER(fi)
        # S1/S2 (protocol 6.2): a dns_only host is expected outside E u R only if it is not in the
        # clean manifest (reddit's dns_only hosts are listed, so they stay covered)
        expect = s["stratum"] != "dns_only" or s["in_clean_manifest"]
        structural.append({"site_id": s["site_id"], "stratum": s["stratum"], "host_in_E_union_R": in_cand,
                           "in_clean_manifest": s["in_clean_manifest"], "expected": expect, "ok": in_cand == expect})
        if n_rep > 0:
            row["methods"] = methods_for_cell(cell, S2)
            # position: pre-alerted on the org's clean run
            ci = Path(clean)
            pre = {"combined|primary": s["host"] in (detector_localize(ci)["flagged_hosts"] | peer_anomaly_hosts(ci, min_base_rate=0.5)
                                                     | manifest_dropped_hosts(ci))}
            for det in GEN:
                cs = dict(s2.scores(clean, det, True))
                pre[f"{det}|primary"] = cs.get(s["host"], float("-inf")) >= S2["baselines"][f"{det}|manifest_aware"]["thresholds"]["primary"][s["org"]]
            row["pre_alerted_on_clean"] = pre
            # simple baselines T1, T3-T6
            roots = s3.base_domains(s["org"])
            E, R = fi["E"], fi["R"] or frozenset()
            ER = E | R
            contract = lambda hosts: frozenset(h for h in hosts if not all(f in fi["fams"].get(h, ()) for f in s3.FAMILIES_CONTRACT))
            rules = {"T1": (ER, fi["Mf"]), "T3": (ER, (roots & ER) | fi["Mf"]), "T4": (ER, roots & ER),
                     "T5": (E, contract(E)), "T6": (ER, contract(ER))}
            row["rules"] = {}
            for name, (cand, fl) in rules.items():
                scr = {h: (1.0 if h in fl else 0.0) for h in cand}
                rep = s2.repair(cell, frozenset(fl))
                row["rules"][name] = {"detect1": tie_aware_hit(scr, {s["host"]}, 1), "success": rep["success"],
                                      "cost": rep["cost"], "n_flagged": len(fl)}
            row["_cell"] = cell
        rows.append(row)

    # T2 matched-alert random (combined primary, IF primary)
    jobs, keys = [], []
    for r in rows:
        if "methods" not in r:
            continue
        for midx, mkey, cand in ((0, "combined|primary", set(_host_scores(Path(r["_cell"]["faulted_dir"])))),
                                 (1, "iforest|primary", {h for h, _ in s2.scores(r["_cell"]["faulted_dir"], "iforest", True)})):
            cl = sorted(cand)
            jobs.append((r["_cell"], cl, min(r["methods"][mkey]["n_flagged"], len(cl)), {r["host"]},
                         (42, site_idx[r["site_id"]], midx), args.R, 3.0))
            keys.append((r["site_id"], mkey))
    if args.workers > 1 and jobs:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            t2 = list(ex.map(s3.t2_cell, jobs, chunksize=1))
    else:
        t2 = [s3.t2_cell(j) for j in jobs]
    by_site = {r["site_id"]: r for r in rows}
    for (sid, mkey), res in zip(keys, t2):
        by_site[sid].setdefault("T2", {})[mkey] = {k: res[k] for k in ("success", "cost", "detect1", "n_flagged")}
    for r in rows:
        r.pop("_cell", None)

    # ── statistics ──
    stats: dict = {"status_counts": dict(Counter(r["status"] for r in rows)),
                   "structural": {"n": len(structural), "violations": [x for x in structural if not x["ok"]]}}
    for stratum in ("dns_only", "dns_multi", "ao"):
        rs = [r for r in rows if r["stratum"] == stratum]
        valid = [r for r in rs if r["status"] == "valid"]
        cons = [r for r in valid if r.get("has_consequence")]
        blk = {"n_sites": len(rs), "n_valid": len(valid), "n_with_consequence": len(cons),
               "orgs_with_consequence": len({r["org"] for r in cons}),
               "tests_descriptive_only": len(cons) < len(valid) / 2 if valid else True, "methods": {}}
        for mkey in ("combined|primary", "combined|secondary", "iforest|primary", "iforest|secondary",
                     "robust_z|primary", "robust_z|secondary"):
            m = {}
            for metr in ("detect1", "detect3", "success", "cost_over_Cstar"):
                by = defaultdict(list)
                for r in cons:
                    by[r["org"]].append(metric(r["methods"][mkey], metr, r["C_star"]))
                m[metr] = org_ci(by)
            m["ufn"] = dict(Counter(r["methods"][mkey]["ufn"] for r in cons))
            blk["methods"][mkey] = m
        stats[stratum] = blk

    def paired(stratum, metr, a, b, alpha):
        da, db = defaultdict(list), defaultdict(list)
        for r in rows:
            if r["stratum"] == stratum and r.get("has_consequence"):
                da[r["org"]].append(metric(r["methods"][a], metr, r["C_star"]))
                db[r["org"]].append(metric(r["methods"][b], metr, r["C_star"]))
        diffs = {o: [st.mean(da[o]) - st.mean(db[o])] for o in da}
        return org_ci(diffs, alpha)

    tests = {}
    for name, (stratum, metr, sign) in TESTS.items():
        p95 = paired(stratum, metr, "combined|primary", "iforest|primary", 0.05)
        pb = paired(stratum, metr, "combined|primary", "iforest|primary", BONF)
        enough = (pb["n_orgs"] or 0) >= MIN_ORGS and not stats[stratum]["tests_descriptive_only"]
        verdict = None
        if enough and pb["ci"]:
            lo, hi = pb["ci"]
            if (sign == "+" and lo > 0) or (sign == "-" and hi < 0):
                verdict = "supported"
            elif (sign == "+" and hi < 0) or (sign == "-" and lo > 0):
                verdict = "contrary"
            else:
                verdict = "undetermined"
        tests[name] = {"stratum": stratum, "metric": metr, "predicted_sign": sign, "ci95": p95,
                       f"ci_bonf_{len(TESTS)}": pb, "enough_orgs": enough,
                       "verdict": verdict if enough else "descriptive_only"}
    stats["tests"] = tests
    stats["descriptive_success_diff"] = {
        st_: {f"combined-{b}": paired(st_, "success", "combined|primary", f"{b}|primary", 0.05) for b in GEN}
        for st_ in ("dns_multi", "ao")}
    stats["robust_z_same_tests"] = {name: paired(stratum, metr, "combined|primary", "robust_z|primary", 0.05)
                                    for name, (stratum, metr, _) in TESTS.items()}

    # selection sensitivity: new-site org mean minus the original site's value (same org)
    ec = {(c["org"], c["cell"]): c["results"]["drop_true|p=0.0"]
          for c in json.loads((D / "ec_manifest_corruption_v1.json").read_text())["cells"]}
    orig = {"combined|primary": lambda o, v, metr: ec[(o, v)]["cost_over_Cstar"] if metr == "cost_over_Cstar" else float(ec[(o, v)][metr])}
    for det in GEN:
        cellsd = {(x["org"], x["cell"]): x for x in S2["baselines"][f"{det}|manifest_aware"]["by_alpha"]["primary"]["cells"]}
        orig[f"{det}|primary"] = (lambda cd: lambda o, v, metr: cd[(o, v)]["cost"] / cd[(o, v)]["C_star"]
                                  if metr == "cost_over_Cstar" else float(cd[(o, v)][metr]))(cellsd)
    sens = {}
    for stratum in ("dns_only", "dns_multi", "ao"):
        v = ORIG_VARIANT[stratum]
        for mkey, fn in orig.items():
            for metr in ("detect1", "success", "cost_over_Cstar"):
                by = defaultdict(list)
                for r in rows:
                    if r["stratum"] == stratum and r.get("has_consequence"):
                        by[r["org"]].append(metric(r["methods"][mkey], metr, r["C_star"]))
                diffs = {o: [st.mean(x) - fn(o, v, metr)] for o, x in by.items() if (o, v) in ec}
                sens[f"{stratum}|{mkey}|{metr}"] = org_ci(diffs)
    stats["selection_sensitivity_new_minus_original"] = sens

    out = {"protocol": str(PROTOCOL.relative_to(REPO)), "selfcheck": sc,
           "inputs": {"run_result_sha256": s3.sha(Path(args.run_result)), "protocol_sha256": s3.sha(PROTOCOL),
                      "site_list_sha256": s3.sha(SITES), "script_sha256": s3.sha(Path(__file__)),
                      "step2_json_sha256": s3.sha(s3.S2_JSON), "R": args.R,
                      "run_isolation_ok": run.get("isolation_ok"), "run_role": run["meta"].get("role")},
           "cells": rows, "stats": stats}
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    print(json.dumps({"status": stats["status_counts"], "structural_violations": len(stats["structural"]["violations"]),
                      "tests": {k: v["verdict"] for k, v in tests.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
