"""Revision v2 Step 3: candidate coverage, U/F/N decomposition, feature/score audit,
AUC on a unified evaluation set, and simple (score-free) baselines T1-T6.

Implements STEP3_CANDIDATE_COVERAGE_PROTOCOL.md v1.1 (frozen in commit 59546fd before
this script existed). Reuses Step 2's code paths unchanged (features, scores, repair
mapper, summaries) and Step 2's frozen per-fold thresholds; it never recalibrates.

Usage:
  poetry run python experiments/exp_arcs/revision_v2/step3_coverage.py OUT.json \
      [--orgs mozilla,reddit] [--R 200] [--tex-dir DIR]

Exit code 0 only if every protocol section 10 consistency assertion passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import step2_baselines as s2  # noqa: E402  (also puts the repo root on sys.path)

from evaluation.stat_test import bootstrap_ci  # noqa: E402
from experiments.exp_arcs.detector_eval import (  # noqa: E402
    _true_hosts, detector_localize, manifest_dropped_hosts, peer_anomaly_hosts)
from experiments.exp_arcs.detector_pr_roc import _host_scores  # noqa: E402
from experiments.exp_arcs.localizer_baselines import _truth  # noqa: E402
from experiments.exp_arcs.rq2_combined_topk import tie_aware_hit  # noqa: E402

REPO, D = s2.REPO, s2.D
PROTOCOL = HERE / "STEP3_CANDIDATE_COVERAGE_PROTOCOL.md"
S2_JSON = HERE / "step2_baselines_v1.json"
S2_SUMS = HERE / "STEP2_CHECKSUMS.sha256"
K = s2.K
FAMILIES_CONTRACT = ("dns", "active_observation")
GENERIC = ("robust_z", "iforest", "lof")
TAGS = ("primary", "secondary")
BONF_K = 6
ALPHA_BONF = 0.05 / BONF_K


# ── task-scope base domains: only the `base_domains` field may be read ──────────
class _Tracked(dict):
    accessed: set[str] = set()

    def __getitem__(self, k):
        _Tracked.accessed.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        _Tracked.accessed.add(k)
        return super().get(k, default)


def base_domains(org: str) -> frozenset[str]:
    doc = json.loads((REPO / f"data/gt/{org}_gt.json").read_text(), object_hook=_Tracked)
    return frozenset(doc.get("base_domains", []))


# ── per-run primitives ──────────────────────────────────────────────────────────
def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


_RUN: dict[str, dict] = {}


def run_info(run: str) -> dict:
    """E, R (None if no manifest), V, host->families, Mf for one bundle."""
    if run in _RUN:
        return _RUN[run]
    b = Path(run)
    fams: dict[str, set] = defaultdict(set)
    for e in jl(b / "evidence_event.jsonl"):
        fams[e["target"]].add(e.get("source_family"))
    mp = b / "discovery_manifest.json"
    R = frozenset(json.loads(mp.read_text()).get("resolved_hosts", [])) if mp.exists() else None
    info = {"E": frozenset(fams), "R": R, "fams": {h: frozenset(v) for h, v in fams.items()},
            "V": frozenset(detector_localize(b)["flagged_hosts"]),
            "Mf": frozenset(manifest_dropped_hosts(b))}
    _RUN[run] = info
    return info


def ER(info: dict) -> frozenset:
    return info["E"] | (info["R"] or frozenset())


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def r4(x):
    return None if x is None else round(float(x), 4)


def org_ci(vals_by_org: dict[str, list[float]], alpha: float = 0.05) -> dict:
    means = [st.mean(v) for _, v in sorted(vals_by_org.items())]
    if len(means) < 2:
        return {"org_mean": r4(means[0]) if means else None, "ci": None, "n_orgs": len(means)}
    m, lo, hi = bootstrap_ci(means, B=10_000, alpha=alpha, seed=42)
    return {"org_mean": r4(m), "ci": [r4(lo), r4(hi)], "n_orgs": len(means)}


def paired(rows_a: list[dict], rows_b: list[dict], key: str) -> dict:
    """Org-level paired difference a - b over the same cells; 95% and Bonferroni intervals."""
    ka = {(r["org"], r["cell"]): r[key] for r in rows_a}
    kb = {(r["org"], r["cell"]): r[key] for r in rows_b}
    assert ka.keys() == kb.keys()
    by = defaultdict(list)
    for k in sorted(ka):
        by[k[0]].append(float(ka[k]) - float(kb[k]))
    out = {"ci95": org_ci(by, 0.05), f"ci_bonf_{BONF_K}": org_ci(by, ALPHA_BONF)}
    out["diff_org_mean"] = out["ci95"]["org_mean"]
    return out


def clean_views(counts: list[tuple[int, int]]) -> dict:
    return s2.clean_views(counts)


def t2_cell(job: tuple) -> dict:
    """One cell of T2 (matched-alert-count random). Top-level so a process pool can run it.
    The RNG is seeded per (cell, matched method), so the result does not depend on which
    process runs it or in what order; the caller reassembles rows in cell order."""
    cell, cand, mcount, truth, seed, R, B0 = job
    rng = np.random.default_rng(list(seed))
    succ, cost, rb0, prec, rec = [], [], [], [], []
    cache: dict = {}
    for _ in range(R):
        fl = frozenset(rng.choice(cand, size=mcount, replace=False).tolist()) if mcount else frozenset()
        if fl not in cache:
            cache[fl] = s2.repair(cell, fl)
        rr = cache[fl]
        succ.append(float(rr["success"])); cost.append(rr["cost"])
        rb0.append(float(rr["success"] and rr["cost"] <= B0))
        prec.append(len(truth & fl) / len(fl) if fl else 0.0); rec.append(len(truth & fl) / len(truth))
    zero = {h: 0.0 for h in cand}
    return {"org": cell["org"], "cell": cell["cell"], "C_star": cell["C_star"],
            "detect1": tie_aware_hit(zero, truth, 1), "detect3": tie_aware_hit(zero, truth, K),
            "success": st.mean(succ), "cost": st.mean(cost), "repair_at_B0": st.mean(rb0),
            "precision": st.mean(prec), "recall": st.mean(rec), "n_flagged": mcount}


# ── main ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--orgs", default="")
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1,
                    help="processes for T2; results are identical for any value (per-cell seeds)")
    ap.add_argument("--tex-dir", default="")
    args = ap.parse_args()
    t_start = time.time()

    s2_sum_expected = next(l.split()[0] for l in S2_SUMS.read_text().splitlines()
                           if l.strip().endswith("step2_baselines_v1.json"))
    assertions: dict[str, bool] = {"step2_json_sha_before": sha(S2_JSON) == s2_sum_expected}

    grid = json.loads((D / "e3_grid_v1.json").read_text())["cells"]
    grid_idx = {(c["org"], c["cell"]): i for i, c in enumerate(grid)}
    clean_of: dict[str, str] = {}
    for c in grid:
        clean_of.setdefault(c["org"], c["clean_dir"])
    orgs_all = sorted(clean_of)
    eval_orgs = [o for o in args.orgs.split(",") if o] or orgs_all
    full = eval_orgs == orgs_all
    cells = [c for c in grid if c["org"] in eval_orgs]
    B0 = float(json.loads((D / "e3_oracle_stats_v1.json").read_text())["summary"]["B0_median_singlepoint_cost"])
    S2 = json.loads(S2_JSON.read_text())
    frz_det = {(c["org"], c["cell"]): c for c in json.loads((D / "e3_detector_v1.json").read_text())["cells"]}
    frz_top = {(c["org"], c["cell"]): c for c in json.loads((D / "rq2_combined_topk_v1.json").read_text())["cells"]}
    ed = json.loads((D / "ed_variant_signal_matrix_v1.json").read_text())
    roots = {o: base_domains(o) for o in orgs_all}
    truth_of = {i: _truth(c["inject_node"])[0] for i, c in enumerate(cells)}
    pairs = [(i, h) for i in range(len(cells)) for h in sorted(truth_of[i])]
    n_pairs = len(pairs)
    if full:
        assertions["n_truth_pairs_260"] = n_pairs == 260

    res: dict = {"protocol": str(PROTOCOL.relative_to(REPO)), "inputs": {
        "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
        "grid_sha256": sha(D / "e3_grid_v1.json"), "step2_json_sha256": sha(S2_JSON),
        "step2_script_sha256": sha(HERE / "step2_baselines.py"),
        "e3_detector_sha256": sha(D / "e3_detector_v1.json"),
        "rq2_topk_sha256": sha(D / "rq2_combined_topk_v1.json"),
        "ed_matrix_sha256": sha(D / "ed_variant_signal_matrix_v1.json"),
        "eval_orgs": eval_orgs, "B0": B0, "R": args.R, "n_truth_pairs": n_pairs}}

    # ── §3 candidate sets per method x setting ────────────────────────────────────
    V_outside = []
    for i, c in enumerate(cells):
        fi = run_info(c["faulted_dir"])
        extra = sorted(fi["V"] - ER(fi))
        if extra:
            V_outside.append({"org": c["org"], "cell": c["cell"], "hosts": extra})
    res["V_outside_E_union_R"] = V_outside

    # ── §4 coverage and reasons ──────────────────────────────────────────────────
    cov_rows = []
    for i, h in pairs:
        c = cells[i]
        fi, ci_ = run_info(c["faulted_dir"]), run_info(c["clean_dir"])
        E, R, V = fi["E"], fi["R"], fi["V"]
        row = {"org": c["org"], "cell": c["cell"], "host": h,
               "covered_E": h in E, "covered_ER": h in ER(fi), "covered_ERV": h in (ER(fi) | V)}
        if not row["covered_ER"]:
            if R is None:
                reason = "A_no_manifest"
            elif h not in ci_["E"] and h not in (ci_["R"] or frozenset()):
                reason = "B_absent_in_clean"
            elif h in ci_["E"] and h not in (ci_["R"] or frozenset()) and h not in R:
                reason = "C1_never_in_manifest"
            elif h in (ci_["R"] or frozenset()) and h not in R:
                reason = "C2_lost_from_manifest"
            else:
                reason = "D_other"
            row["reason_ER"] = reason
        if not row["covered_E"]:
            if h not in ci_["E"] and h not in (ci_["R"] or frozenset()):
                reason = "B_absent_in_clean"
            elif c["cell"].startswith("F10"):
                reason = "F10"
            elif c["cell"].startswith("F4") and not fi["fams"].get(h):
                reason = "F4_all_evidence_removed"
            else:
                reason = "other"
            row["reason_E"] = reason
        if not (row["covered_E"] and row["covered_ER"]):
            row.update({"in_true_hosts": h in _true_hosts(Path(c["clean_dir"]), s2.dfx_of(c["clean_dir"], c["faulted_dir"])),
                        "clean_families": sorted(f for f in ci_["fams"].get(h, ()) if f),
                        "in_clean_manifest": h in (ci_["R"] or frozenset()),
                        "is_root": h in roots[c["org"]]})
        cov_rows.append(row)

    def cov_summary(key: str) -> dict:
        by_org = defaultdict(list)
        per_cell = defaultdict(list)
        for r in cov_rows:
            per_cell[(r["org"], r["cell"])].append(float(r[key]))
        for (o, _), v in per_cell.items():
            by_org[o].append(st.mean(v))
        by_var = defaultdict(lambda: [0, 0])
        for r in cov_rows:
            by_var[r["cell"]][0] += r[key]
            by_var[r["cell"]][1] += 1
        by_fam, by_o = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        for r in cov_rows:
            f = "F10" if r["cell"].startswith("F10") else "F4"
            by_fam[f][0] += r[key]; by_fam[f][1] += 1
            by_o[r["org"]][0] += r[key]; by_o[r["org"]][1] += 1
        n_cov = sum(r[key] for r in cov_rows)
        return {"covered": n_cov, "uncovered": n_pairs - n_cov, "org_level": org_ci(by_org),
                "by_family": {f: {"covered": a, "n": n} for f, (a, n) in sorted(by_fam.items())},
                "by_variant": {v: {"covered": a, "n": n} for v, (a, n) in sorted(by_var.items())},
                "by_org": {o: {"covered": a, "n": n} for o, (a, n) in sorted(by_o.items())}}

    res["coverage"] = {k: cov_summary(k) for k in ("covered_E", "covered_ER", "covered_ERV")}
    for k in ("covered_E", "covered_ER", "covered_ERV"):
        assertions[f"coverage_sum_{k}"] = res["coverage"][k]["covered"] + res["coverage"][k]["uncovered"] == n_pairs
    res["uncovered_reasons"] = {
        "ER": dict(sorted(Counter(r["reason_ER"] for r in cov_rows if "reason_ER" in r).items())),
        "E": dict(sorted(Counter(r["reason_E"] for r in cov_rows if "reason_E" in r).items()))}
    res["uncovered_pairs"] = [r for r in cov_rows if not (r["covered_E"] and r["covered_ER"])]

    # ── method evaluation (combined + generic) with frozen thresholds ────────────
    methods: dict[str, dict] = {}   # key -> {"rows": [...], "flagged": {i: set}, "scores": {i: dict}, "cand": {i: set}}
    mism: list = []

    comb_sc = {i: _host_scores(Path(c["faulted_dir"])) for i, c in enumerate(cells)}
    comb_clean_sc = {o: _host_scores(Path(clean_of[o])) for o in orgs_all}
    for tag, t in s2.COMBINED_T.items():
        key = f"combined|t={t}"
        rows, flagged_of = [], {}
        for i, c in enumerate(cells):
            fd = Path(c["faulted_dir"])
            fl = frozenset(detector_localize(fd)["flagged_hosts"] | peer_anomaly_hosts(fd, min_base_rate=t)
                           | manifest_dropped_hosts(fd))
            truth, sc = truth_of[i], comb_sc[i]
            rows.append({"org": c["org"], "cell": c["cell"], "C_star": c["C_star"],
                         "detect1": tie_aware_hit(sc, truth, 1), "detect3": tie_aware_hit(sc, truth, K),
                         "precision": len(truth & fl) / len(fl) if fl else 0.0,
                         "recall": len(truth & fl) / len(truth), "n_flagged": len(fl), **s2.repair(c, fl)})
            flagged_of[i] = fl
        clean_counts = []
        for o in eval_orgs:
            cd = Path(clean_of[o])
            fl = detector_localize(cd)["flagged_hosts"] | peer_anomaly_hosts(cd, min_base_rate=t) | manifest_dropped_hosts(cd)
            clean_counts.append((len(fl), len(comb_clean_sc[o])))
        methods[key] = {"rows": rows, "flagged": flagged_of, "scores": comb_sc, "clean_scores": comb_clean_sc,
                        "cand": {i: frozenset(comb_sc[i]) for i in comb_sc}, "tau": {o: t for o in orgs_all},
                        "clean_alerts": clean_views(clean_counts)}
        if tag == "primary":
            for r in rows:
                k = (r["org"], r["cell"])
                if bool(frz_det[k]["combined"]["success"]) != r["success"]:
                    mism.append([*k, key, "success"])
                if abs(frz_top[k]["detect1_tie"] - r["detect1"]) > 1e-6 or abs(frz_top[k]["detectk_tie"] - r["detect3"]) > 1e-6:
                    mism.append([*k, key, "detect"])

    for ma in (True, False):
        setting = "manifest_aware" if ma else "manifest_free"
        for det in GENERIC:
            frz = S2["baselines"][f"{det}|{setting}"]
            sc_of = {i: dict(s2.scores(c["faulted_dir"], det, ma)) for i, c in enumerate(cells)}
            clean_sc = {o: dict(s2.scores(clean_of[o], det, ma)) for o in orgs_all}
            for tag in TAGS:
                taus = frz["thresholds"][tag]
                fcells = {(x["org"], x["cell"]): x for x in frz["by_alpha"][tag]["cells"]}
                key = f"{det}|{setting}|{tag}"
                rows, flagged_of = [], {}
                for i, c in enumerate(cells):
                    truth, sc = truth_of[i], sc_of[i]
                    fl = frozenset(h for h, s in sc.items() if s >= taus[c["org"]])
                    r = {"org": c["org"], "cell": c["cell"], "C_star": c["C_star"],
                         "detect1": tie_aware_hit(sc, truth, 1), "detect3": tie_aware_hit(sc, truth, K),
                         "precision": len(truth & fl) / len(fl) if fl else 0.0,
                         "recall": len(truth & fl) / len(truth), "n_flagged": len(fl), **s2.repair(c, fl)}
                    f = fcells[(c["org"], c["cell"])]
                    for fld in ("success", "n_flagged"):
                        if f[fld] != r[fld]:
                            mism.append([c["org"], c["cell"], key, fld])
                    for fld in ("cost", "detect1", "detect3"):
                        if abs(float(f[fld]) - float(r[fld])) > 1e-6:
                            mism.append([c["org"], c["cell"], key, fld])
                    rows.append(r)
                    flagged_of[i] = fl
                clean_counts = [(sum(1 for s in clean_sc[o].values() if s >= taus[o]), len(clean_sc[o])) for o in eval_orgs]
                methods[key] = {"rows": rows, "flagged": flagged_of, "scores": sc_of, "clean_scores": clean_sc,
                                "cand": {i: frozenset(sc_of[i]) for i in sc_of}, "tau": taus,
                                "clean_alerts": clean_views(clean_counts), "ma": ma, "det": det}
                if ma:   # manifest-aware candidate set must equal E ∪ R
                    for i, c in enumerate(cells):
                        if methods[key]["cand"][i] != ER(run_info(c["faulted_dir"])):
                            mism.append([c["org"], c["cell"], key, "candidate_set_not_E_union_R"])
    assertions["recompute_matches_step2_and_frozen"] = not mism
    res["recompute_mismatches"] = mism

    # ── §5 U/F/N and cell-level cross tables ─────────────────────────────────────
    ufn, cross = {}, {}
    for key, m in methods.items():
        cnt = Counter()
        xt = Counter()
        for i, c in enumerate(cells):
            truth, fl, cand = truth_of[i], m["flagged"][i], m["cand"][i]
            for h in sorted(truth):
                cnt["U" if h not in cand else ("F" if h in fl else "N")] += 1
            nf = len(truth & fl)
            cat = "all_flagged" if nf == len(truth) else ("some_flagged" if nf else "none_flagged")
            xt[(cat, "success" if m["rows"][i]["success"] else "fail")] += 1
        ufn[key] = {k: cnt[k] for k in "UFN"}
        cross[key] = {f"{a}|{b}": xt[(a, b)] for a in ("all_flagged", "some_flagged", "none_flagged")
                      for b in ("success", "fail")}
        assertions[f"UFN_sum_{key}"] = sum(cnt.values()) == n_pairs
    res["ufn"], res["cell_cross"] = ufn, cross

    # ── §6 feature and score audit for N pairs ───────────────────────────────────
    audit = []
    for key, m in methods.items():
        for i, c in enumerate(cells):
            for h in sorted(truth_of[i]):
                if h not in m["cand"][i] or h in m["flagged"][i]:
                    continue
                sc = m["scores"][i]
                rank = 1 + sum(1 for v in sc.values() if v > sc[h])
                ties = sum(1 for v in sc.values() if v == sc[h]) - 1
                rec = {"method": key, "org": c["org"], "cell": c["cell"], "host": h,
                       "score_faulted": r4(sc[h]), "tau": r4(m["tau"][c["org"]]), "rank": rank, "n_tied": ties,
                       "n_candidates": len(sc), "score_clean": r4(m["clean_scores"][c["org"]].get(h))}
                if key.startswith("combined"):
                    fi, ci_ = run_info(c["faulted_dir"]), run_info(c["clean_dir"])
                    rec["families_clean"] = sorted(f for f in ci_["fams"].get(h, ()) if f)
                    rec["families_faulted"] = sorted(f for f in fi["fams"].get(h, ()) if f)
                    rec["host_input_identical"] = ci_["fams"].get(h) == fi["fams"].get(h)
                    rec["full_input_identical"] = (ci_["fams"] == fi["fams"] and ci_["V"] == fi["V"]
                                                   and ci_["Mf"] == fi["Mf"])
                else:
                    hc, Xc = s2.features(c["clean_dir"], m["ma"])
                    hf, Xf = s2.features(c["faulted_dir"], m["ma"])
                    vf = Xf[hf.index(h)]
                    vc = Xc[hc.index(h)] if h in hc else None
                    rec["host_input_identical"] = vc is not None and bool(np.array_equal(vc, vf))
                    rec["changed_features"] = ([s2.FEATURES[j] for j in np.flatnonzero(vc != vf)]
                                               if vc is not None else None)
                    rec["full_input_identical"] = hc == hf and bool(np.array_equal(Xc, Xf))
                audit.append(rec)
    res["feature_audit"] = {
        "pairs": audit,
        "summary": {key: {"n_N": sum(1 for a in audit if a["method"] == key),
                          "host_input_identical": sum(1 for a in audit if a["method"] == key and a["host_input_identical"]),
                          "full_input_identical": sum(1 for a in audit if a["method"] == key and a["full_input_identical"])}
                    for key in methods}}

    # ── §7 AUC on the unified evaluation set ─────────────────────────────────────
    auc = {}
    for key, m in methods.items():
        if not (key.startswith("combined|t=0.5") or key.endswith("|primary")):
            continue   # threshold-free: one entry per scorer
        name = key.split("|t=")[0] if key.startswith("combined") else key.rsplit("|", 1)[0]
        pts_c, pts_u = [], []
        for i, c in enumerate(cells):
            sc = m["scores"][i]
            pts_c += [(s, h in truth_of[i]) for h, s in sc.items()]
            pts_u += [(s, h in truth_of[i]) for h, s in sc.items()]
            pts_u += [(-math.inf, True) for h in truth_of[i] if h not in sc]
        neg = [(s, False) for o in eval_orgs for s in m["clean_scores"][o].values()]
        a_c, a_u = s2.pooled_auc(pts_c + neg), s2.pooled_auc(pts_u + neg)
        n_cov = sum(1 for _, y in pts_c if y)
        ident = abs(a_u["auc_roc_raw"] - (n_cov / a_u["n_pos"]) * a_c["auc_roc_raw"]) < 1e-9
        assertions[f"auc_identity_{name}"] = ident
        auc[name] = {"auc_roc_cand": r4(a_c["auc_roc_raw"]), "auc_pr_cand": r4(a_c["auc_pr_raw"]),
                     "auc_roc_U": r4(a_u["auc_roc_raw"]), "auc_pr_U": r4(a_u["auc_pr_raw"]),
                     "n_pos_cand": a_c["n_pos"], "n_pos_U": a_u["n_pos"], "n_neg": a_c["n_neg"]}
        if full:
            if name == "combined":
                assertions["auc_combined_matches_frozen"] = abs(
                    a_c["auc_roc_raw"] - S2["combined"]["primary"]["auc"]["auc_roc_raw"]) < 1e-9
                # protocol s10: positives in AUC_cand = Step 2 n_pos (stored only for combined;
                # for the generic detectors Step 2 did not store n_pos, so it is recorded above)
                assertions["auc_combined_n_pos_matches_frozen"] = (
                    a_c["n_pos"] == S2["combined"]["primary"]["auc"]["n_pos"])
            else:
                det, setting = name.split("|")
                frz_tf = S2["baselines"][f"{det}|{setting}"]["threshold_free"]
                assertions[f"auc_{name}_matches_frozen"] = round(a_c["auc_roc_raw"], 4) == frz_tf["auc_roc"]
    res["auc"] = auc

    # ── §8 simple baselines T1-T6 ────────────────────────────────────────────────
    rep_cache: dict = {}

    def rep(i: int, fl: frozenset) -> dict:
        k = (i, fl)
        if k not in rep_cache:
            rep_cache[k] = s2.repair(cells[i], fl)
        return rep_cache[k]

    def rule_row(i: int, cand: frozenset, fl: frozenset) -> dict:
        c, truth = cells[i], truth_of[i]
        sc = {h: (1.0 if h in fl else 0.0) for h in cand}
        return {"org": c["org"], "cell": c["cell"], "C_star": c["C_star"],
                "detect1": tie_aware_hit(sc, truth, 1), "detect3": tie_aware_hit(sc, truth, K),
                "precision": len(truth & fl) / len(fl) if fl else 0.0,
                "recall": len(truth & fl) / len(truth), "n_flagged": len(fl), **rep(i, fl)}

    def contract(info: dict, hosts: frozenset) -> frozenset:
        return frozenset(h for h in hosts if not all(f in info["fams"].get(h, ()) for f in FAMILIES_CONTRACT))

    rule_defs = {
        "T1_manifest_diff": (lambda fi, o: ER(fi), lambda fi, o: fi["Mf"]),
        "T3_root_plus_manifest": (lambda fi, o: ER(fi), lambda fi, o: (roots[o] & ER(fi)) | fi["Mf"]),
        "T4_always_root": (lambda fi, o: ER(fi), lambda fi, o: roots[o] & ER(fi)),
        "T5_contract_A": (lambda fi, o: fi["E"], lambda fi, o: contract(fi, fi["E"])),
        "T6_contract_B": (lambda fi, o: ER(fi), lambda fi, o: contract(fi, ER(fi))),
    }
    rules: dict[str, dict] = {}
    for rname, (cand_fn, flag_fn) in rule_defs.items():
        rows = []
        for i, c in enumerate(cells):
            fi = run_info(c["faulted_dir"])
            rows.append(rule_row(i, frozenset(cand_fn(fi, c["org"])), frozenset(flag_fn(fi, c["org"]))))
        clean_counts = []
        for o in eval_orgs:
            ci_ = run_info(clean_of[o])
            clean_counts.append((len(flag_fn(ci_, o)), len(cand_fn(ci_, o))))
        rules[rname] = {"rows": rows, "clean_alerts": clean_views(clean_counts)}

    # T2: matched-alert-count random within the matched method's own candidate set
    t2_targets = [("combined|t=0.5", 0), ("iforest|manifest_aware|primary", 1), ("robust_z|manifest_aware|primary", 2)]
    t2_time = {}
    for mkey, midx in t2_targets:
        m = methods[mkey]
        t0 = time.time()
        jobs = []
        for i, c in enumerate(cells):
            cand = sorted(m["cand"][i])
            jobs.append((c, cand, min(m["rows"][i]["n_flagged"], len(cand)), truth_of[i],
                         (42, grid_idx[(c["org"], c["cell"])], midx), args.R, B0))
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                rows = list(ex.map(t2_cell, jobs, chunksize=1))   # map preserves cell order
        else:
            rows = [t2_cell(j) for j in jobs]
        t2_time[mkey] = round((time.time() - t0) / max(1, len(cells)), 3)   # printed only: keeps JSON deterministic
        rules[f"T2_random_matched[{mkey}]"] = {"rows": rows, "clean_alerts": None}

    if full:
        ed_m = ed["matrix"]
        t1 = defaultdict(list)
        for r in rules["T1_manifest_diff"]["rows"]:
            t1[r["cell"]].append(float(r["success"]))
        assertions["T1_matches_ED_manifest_column"] = all(
            abs(st.mean(v) - ed_m[cell]["manifest"]["success_rate"]) < 1e-4 for cell, v in t1.items())

    def rule_summary(blk: dict) -> dict:
        rows = blk["rows"]
        is_t2 = "repair_at_B0" in rows[0]
        if is_t2:
            by = lambda k: org_ci(group(rows, k))
            out = {"n_cells": len(rows), "success": by("success"), "detect1": by("detect1"), "detect3": by("detect3"),
                   "cost_over_Cstar_all": org_ci(group(rows, None)), "repair_at_B0": by("repair_at_B0"),
                   "mean_flagged": r4(st.mean(r["n_flagged"] for r in rows))}
        else:
            sm = s2.summarize(rows, B0)
            out = {k: sm[k] for k in ("n_cells", "success", "detect1", "detect3", "cost_over_Cstar_all",
                                      "cost_over_Cstar_success_only", "repair_at_B0", "by_variant")}
            out["mean_flagged"] = r4(st.mean(r["n_flagged"] for r in rows))
        out["clean_alerts"] = blk["clean_alerts"]
        return out

    def group(rows, k):
        by = defaultdict(list)
        for r in rows:
            by[r["org"]].append(float(r[k]) if k else r["cost"] / r["C_star"])
        return by

    rule_ufn = {}
    for rname, blk in rules.items():
        if rname.startswith("T2"):
            continue
        cnt = Counter()
        for i, r in enumerate(blk["rows"]):
            fi = run_info(cells[i]["faulted_dir"])
            cand = rule_defs[rname][0](fi, cells[i]["org"])
            fl = rule_defs[rname][1](fi, cells[i]["org"])
            for h in sorted(truth_of[i]):
                cnt["U" if h not in cand else ("F" if h in fl else "N")] += 1
        rule_ufn[rname] = {k: cnt[k] for k in "UFN"}
    res["rules"] = {rname: rule_summary(blk) | ({"ufn": rule_ufn[rname]} if rname in rule_ufn else {})
                    for rname, blk in rules.items()}
    res["rules_cells"] = {rname: blk["rows"] for rname, blk in rules.items()}

    # ── method summaries (for the tables) ────────────────────────────────────────
    res["methods"] = {key: {"summary": {k: v for k, v in s2.summarize(m["rows"], B0).items()
                                        if k in ("success", "detect1", "detect3", "cost_over_Cstar_all", "repair_at_B0")},
                            "clean_alerts": m["clean_alerts"], "mean_flagged": r4(st.mean(r["n_flagged"] for r in m["rows"]))}
                      for key, m in methods.items()}

    # ── §9 comparison family (K = 6) ─────────────────────────────────────────────
    comb, ifo = methods["combined|t=0.5"]["rows"], methods["iforest|manifest_aware|primary"]["rows"]
    t2c = rules["T2_random_matched[combined|t=0.5]"]["rows"]
    t2i = rules["T2_random_matched[iforest|manifest_aware|primary]"]["rows"]
    t3, t6 = rules["T3_root_plus_manifest"]["rows"], rules["T6_contract_B"]["rows"]
    fam = {"1_combined_minus_T2[combined]": (comb, t2c), "2_IF_minus_T2[IF]": (ifo, t2i),
           "3_IF_minus_T3": (ifo, t3), "4_combined_minus_T3": (comb, t3),
           "5_combined_minus_T6": (comb, t6), "6_IF_minus_T6": (ifo, t6)}
    res["comparisons"] = {name: {"success": paired(a, b, "success"),
                                 "descriptive": {k: paired(a, b, k)["ci95"] for k in ("detect1", "detect3")}}
                          for name, (a, b) in fam.items()}

    # ── closing assertions and output ────────────────────────────────────────────
    assertions["gt_only_base_domains_read"] = _Tracked.accessed <= {"base_domains"}
    assertions["step2_json_sha_after"] = sha(S2_JSON) == s2_sum_expected
    res["assertions"] = assertions
    res["all_assertions_pass"] = all(assertions.values())
    res["runtime_seconds"] = None   # wall time is printed, not stored, so outputs stay byte-identical
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")
    print(json.dumps({"assertions": assertions, "n_mismatch": len(mism), "t2_s_per_cell": t2_time,
                      "wall_s": round(time.time() - t_start, 1)}, indent=1))
    if args.tex_dir:
        write_tex(Path(args.tex_dir), res)
    return 0 if res["all_assertions_pass"] else 1


def write_tex(d: Path, res: dict) -> None:
    """Rendered only for the formal run (D5); kept here so tables come from one script."""
    d.mkdir(parents=True, exist_ok=True)
    cov = res["coverage"]
    variants = sorted(cov["covered_E"]["by_variant"])
    lines = [r"\begin{tabular}{lrrrr}\toprule",
             r"variant & truth pairs & covered by $E$ & by $E\cup R$ & by $E\cup R\cup V$\\\midrule"]
    for v in variants:
        n = cov["covered_E"]["by_variant"][v]["n"]
        cells = " & ".join(str(cov[k]["by_variant"][v]["covered"]) for k in ("covered_E", "covered_ER", "covered_ERV"))
        lines.append(f"{v.replace('_', chr(92) + '_')} & {n} & {cells}\\\\")
    lines += [r"\bottomrule\end{tabular}"]
    (d / "candidate_coverage_table.tex").write_text(
        "% generated by experiments/exp_arcs/revision_v2/step3_coverage.py -- do not edit\n" + "\n".join(lines) + "\n")
    names = ["T1_manifest_diff", "T2_random_matched[combined|t=0.5]", "T2_random_matched[iforest|manifest_aware|primary]",
             "T3_root_plus_manifest", "T4_always_root", "T5_contract_A", "T6_contract_B"]
    lines = [r"\begin{tabular}{lrrrrr}\toprule",
             r"rule & Success & Detect@1 & $c/C^\ast$ & flagged/cell & clean alert (macro)\\\midrule"]
    for n in names:
        r = res["rules"][n]
        ca = r["clean_alerts"]["macro_rate"] if r["clean_alerts"] else float("nan")
        lines.append(f"{n.replace('_', chr(92) + '_')} & {r['success']['org_mean']:.3f} & {r['detect1']['org_mean']:.3f} & "
                     f"{r['cost_over_Cstar_all']['org_mean']:.2f} & {r['mean_flagged']:.1f} & {ca:.3f}\\\\")
    lines += [r"\bottomrule\end{tabular}"]
    (d / "step3_baselines_table.tex").write_text(
        "% generated by experiments/exp_arcs/revision_v2/step3_coverage.py -- do not edit\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
