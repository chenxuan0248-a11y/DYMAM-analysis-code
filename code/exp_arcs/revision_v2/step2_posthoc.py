"""Revision v2 · step 2 — POST-HOC analyses (not in the frozen protocol; run after seeing
step2_baselines_v1.json). They diagnose why the generic baselines beat the combined detector
on end-to-end Success and do not change the protocol's §11 verdict.

  A. Base-domain confound: are all F4@certificate / F4@registration injection hosts a base
     domain, and do the baselines flag a base domain on every clean run (i.e. regardless of
     any fault)?
  B. Paired Success / Detect on the 140 cells outside F4@certificate and F4@registration.
  C. A trivial deployment rule: combined detector (t=0.5 / 0.9) UNION "always flag the
     organization's base domains" — its Success, cost and clean-run alert burden.

    .venv/bin/python experiments/exp_arcs/revision_v2/step2_posthoc.py [out.json]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
HERE = REPO / "experiments/exp_arcs/revision_v2"
sys.argv = sys.argv[:2] if len(sys.argv) > 1 else sys.argv
import experiments.exp_arcs.revision_v2.step2_baselines as S  # noqa: E402

CR = ("F4@certificate", "F4@registration")


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "step2_posthoc_v1.json"
    base_p = HERE / "step2_baselines_v1.json"
    r = json.loads(base_p.read_text())
    grid = json.loads((S.D / "e3_grid_v1.json").read_text())["cells"]
    clean_of: dict = {}
    for c in grid:
        clean_of.setdefault(c["org"], c["clean_dir"])
    bd = {o: set(json.loads((REPO / f"data/gt/{o}_gt.json").read_text()).get("base_domains", [])) for o in clean_of}
    B0 = r["inputs"]["B0"]

    # A
    inj = [(c["org"], c["inject_node"][1]) for c in grid if c["cell"] in CR]
    A = {"cert_reg_injection_on_base_domain": f"{sum(h in bd[o] for o, h in inj)}/{len(inj)}"}
    for det in ("robust_z", "iforest", "lof"):
        taus = r["baselines"][f"{det}|manifest_aware"]["thresholds"]["primary"]
        fl = top = 0
        for o, cd in clean_of.items():
            sc = dict(S.scores(cd, det, True))
            fl += bool({h for h, v in sc.items() if v >= taus[o]} & bd[o])
            top += max(sc, key=sc.get) in bd[o]
        A[det] = {"clean_runs_base_domain_flagged": f"{fl}/20", "clean_runs_base_domain_top_scored": f"{top}/20"}
    n = 0
    for o, cd in clean_of.items():
        p = Path(cd)
        n += bool((S.detector_localize(p)["flagged_hosts"] | S.peer_anomaly_hosts(p, min_base_rate=0.5)
                   | S.manifest_dropped_hosts(p)) & bd[o])
    A["combined_t0.5"] = {"clean_runs_base_domain_flagged": f"{n}/20"}

    # B
    comb_rows = {}
    for tag, t in S.COMBINED_T.items():
        rows = []
        for c in grid:
            fd = Path(c["faulted_dir"])
            flagged = frozenset(S.detector_localize(fd)["flagged_hosts"] | S.peer_anomaly_hosts(fd, min_base_rate=t)
                                | S.manifest_dropped_hosts(fd))
            truth, _ = S._truth(c["inject_node"])
            sc = S._host_scores(fd)
            rows.append({"org": c["org"], "cell": c["cell"], "C_star": c["C_star"], "flagged": flagged,
                         "detect1": S.tie_aware_hit(sc, truth, 1), "detect3": S.tie_aware_hit(sc, truth, S.K),
                         **S.repair(c, flagged)})
        comb_rows[tag] = rows
    Bres = {}
    for det in ("robust_z", "iforest"):
        for tag in ("primary", "secondary"):
            brows = r["baselines"][f"{det}|manifest_aware"]["by_alpha"][tag]["cells"]
            keep = lambda rs: [x for x in rs if x["cell"] not in CR]
            Bres[f"{det}|{tag}"] = {
                "n_cells": len(keep(brows)),
                "combined_success": S.org_mean_ci(keep(comb_rows[tag]), lambda x: float(x["success"])),
                "baseline_success": S.org_mean_ci(keep(brows), lambda x: float(x["success"])),
                "paired_combined_minus_baseline": {
                    m: S.paired_ci(keep(comb_rows[tag]), keep(brows), f)
                    for m, f in (("success", lambda x: float(x["success"])), ("detect1", lambda x: x["detect1"]),
                                 ("detect3", lambda x: x["detect3"]))}}

    # C
    Cres = {}
    for tag, t in S.COMBINED_T.items():
        rows = []
        for c, cr in zip(grid, comb_rows[tag]):
            flagged = frozenset(cr["flagged"] | bd[c["org"]])
            rows.append({"org": c["org"], "cell": c["cell"], "C_star": c["C_star"],
                         "detect1": cr["detect1"], "detect3": cr["detect3"], "precision": 0.0, "recall": 0.0,
                         **S.repair(c, flagged)})
        counts = []
        for o, cd in clean_of.items():
            p = Path(cd)
            fl = (S.detector_localize(p)["flagged_hosts"] | S.peer_anomaly_hosts(p, min_base_rate=t)
                  | S.manifest_dropped_hosts(p) | bd[o])
            counts.append((len(fl), len(S._host_scores(p))))
        s = S.summarize(rows, B0)
        Cres[f"combined_t{t}_plus_base_domains"] = {
            "success": s["success"], "cost_over_Cstar_all": s["cost_over_Cstar_all"], "by_variant": s["by_variant"],
            "clean_run_alerts": S.clean_views(counts),
            "paired_vs_combined_alone": S.paired_ci(rows, comb_rows[tag], lambda x: float(x["success"]))}

    res = {"note": "POST-HOC; not part of the frozen STEP2 protocol; does not change its §11 verdict",
           "inputs": {"step2_baselines_v1_sha256": hashlib.sha256(base_p.read_bytes()).hexdigest(),
                      "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
           "A_base_domain_confound": A, "B_excluding_cert_reg": Bres, "C_combined_plus_base_domain_rule": Cres}
    out.write_text(json.dumps(res, indent=1, sort_keys=True, default=float))
    print(json.dumps(res, indent=1, default=float)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
