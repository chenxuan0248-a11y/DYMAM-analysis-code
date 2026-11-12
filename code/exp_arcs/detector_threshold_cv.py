"""Reviewer item ① (EMSE) — leave-one-organization-out CV of the peer-anomaly
threshold, to rule out test-set threshold tuning.

Concern an EMSE reviewer raises: the t=0.9 operating point (detector_pr_roc) was read
off the SAME 180 cells it is evaluated on -> possible test-set leakage. This does
proper LOOO: for each held-out org, select t* = argmax detection-F1 on the OTHER 19
orgs, then evaluate precision/recall/F1 and the fault-free false-alarm rate on the
HELD-OUT org only (never used to pick t*). If held-out performance ~= the pooled
full-grid curve and t* is stable across folds, the operating point is not a tuning
artifact.

Reuses detector_pr_roc's continuous per-host score (self/manifest -> 1.0, else peer
severity = max missing-family base-rate); thresholding at t == peer_anomaly
min_base_rate = t. Detection-side only (jsonl reads, no DAG builds).

    poetry run python -m experiments.exp_arcs.detector_threshold_cv
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

from experiments.exp_arcs.detector_pr_roc import _host_scores, _flagged_at

D = Path("experiments/exp_arcs")
OUT = D / "detector_threshold_cv_v1.json"
GRID = [round(x / 100, 2) for x in range(5, 100, 5)] + [0.97, 0.99]


def _org_detection(faulted_cells, clean_scores_by_org, org, t):
    """precision/recall/F1 (mean over org's faulted cells) + clean FA rate at t."""
    precs, recs = [], []
    for fc in faulted_cells:
        if fc["org"] != org:
            continue
        fl = _flagged_at(fc["scores"], t)
        tp = len(fc["true"] & fl)
        precs.append(tp / len(fl) if fl else 0.0)
        recs.append(tp / len(fc["true"]))
    P, R = st.mean(precs), st.mean(recs)
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    sc = clean_scores_by_org[org]
    fa = len(_flagged_at(sc, t)) / max(1, len(sc))
    return {"precision": P, "recall": R, "f1": F1, "clean_fa": fa}


def main() -> int:
    grid = json.loads((D / "e3_grid_v1.json").read_text())["cells"]
    # precompute scores
    faulted = []
    clean_by_org = {}
    for c in grid:
        inj = c.get("inject_node") or []
        if len(inj) < 2:
            continue
        true_hosts = set(inj[1]) if isinstance(inj[1], list) else {inj[1]}
        faulted.append({"org": c["org"], "true": true_hosts,
                        "scores": _host_scores(Path(c["faulted_dir"]))})
        clean_by_org.setdefault(c["org"], c["clean_dir"])
    clean_scores = {o: _host_scores(Path(cd)) for o, cd in clean_by_org.items()}
    orgs = sorted(clean_by_org)

    # per-org, per-t detection metrics (so folds just re-aggregate)
    per = {o: {t: _org_detection(faulted, clean_scores, o, t) for t in GRID} for o in orgs}

    folds = []
    for held in orgs:
        train = [o for o in orgs if o != held]
        # select t* = argmax mean training-org F1 (tie -> higher t for precision/FA)
        best_t, best_f1 = None, -1.0
        for t in GRID:
            mf1 = st.mean(per[o][t]["f1"] for o in train)
            if mf1 > best_f1 + 1e-12 or (abs(mf1 - best_f1) <= 1e-12 and (best_t is None or t > best_t)):
                best_f1, best_t = mf1, t
        ho = per[held][best_t]
        folds.append({"held_out": held, "selected_t": best_t,
                      "train_mean_f1": round(best_f1, 4),
                      "heldout_precision": round(ho["precision"], 4),
                      "heldout_recall": round(ho["recall"], 4),
                      "heldout_f1": round(ho["f1"], 4),
                      "heldout_clean_fa": round(ho["clean_fa"], 4)})

    sel_ts = [f["selected_t"] for f in folds]
    ho_f1 = [f["heldout_f1"] for f in folds]
    ho_p = [f["heldout_precision"] for f in folds]
    ho_r = [f["heldout_recall"] for f in folds]
    ho_fa = [f["heldout_clean_fa"] for f in folds]
    # pooled full-grid reference F1 at the globally-best t (for comparison)
    pooled_best_t, pooled_best_f1 = max(
        ((t, st.mean(per[o][t]["f1"] for o in orgs)) for t in GRID), key=lambda kv: kv[1])

    result = {
        "phase": "reviewer-① LOOO threshold cross-validation (detection, host grain)",
        "objective": "argmax mean detection-F1 on the 19 training orgs; tie-break higher t",
        "grid": GRID, "n_folds": len(folds),
        "selected_t": {"values": sel_ts, "min": min(sel_ts), "max": max(sel_ts),
                       "mode": max(set(sel_ts), key=sel_ts.count),
                       "unique": sorted(set(sel_ts))},
        "heldout_mean": {"precision": round(st.mean(ho_p), 4),
                         "recall": round(st.mean(ho_r), 4),
                         "f1": round(st.mean(ho_f1), 4),
                         "clean_fa": round(st.mean(ho_fa), 4)},
        "heldout_f1_std": round(st.pstdev(ho_f1), 4),
        "pooled_reference": {"best_t": pooled_best_t, "pooled_f1": round(pooled_best_f1, 4)},
        "verdict": ("No test-set tuning: t* is selected only on training orgs; held-out "
                    "mean F1 is compared to the pooled full-grid F1 at its own best t. "
                    "Stable t* across folds + heldout≈pooled ⇒ the operating point "
                    "generalizes."),
        "folds": folds,
    }
    OUT.write_text(json.dumps(result, indent=2))

    print(f"{'held_out':12} {'t*':>5} {'ho_P':>6} {'ho_R':>6} {'ho_F1':>6} {'ho_FA':>6}")
    for f in folds:
        print(f"{f['held_out']:12} {f['selected_t']:>5} {f['heldout_precision']:>6} "
              f"{f['heldout_recall']:>6} {f['heldout_f1']:>6} {f['heldout_clean_fa']:>6}")
    hm = result["heldout_mean"]
    print(f"\nselected t* across 20 folds: unique={result['selected_t']['unique']} "
          f"mode={result['selected_t']['mode']}")
    print(f"held-out mean: P={hm['precision']} R={hm['recall']} F1={hm['f1']} "
          f"FA={hm['clean_fa']} (F1 std={result['heldout_f1_std']})")
    print(f"pooled full-grid best t={pooled_best_t}, pooled F1={round(pooled_best_f1,4)}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
