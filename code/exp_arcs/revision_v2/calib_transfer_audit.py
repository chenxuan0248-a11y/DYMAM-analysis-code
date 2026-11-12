"""Step 5 clean-only calibration transfer audit (protocol v1).

Read-only over the frozen E3 bundles and Step-2 artifact. It never runs the pipeline.

    .venv/bin/python experiments/exp_arcs/revision_v2/calib_transfer_audit.py \
        experiments/exp_arcs/revision_v2/calib_transfer_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import step2_baselines as s2  # noqa: E402
from experiments.exp_arcs.detector_eval import (  # noqa: E402
    detector_localize, manifest_dropped_hosts, peer_anomaly_hosts)
from experiments.exp_arcs.detector_pr_roc import _host_scores  # noqa: E402
from experiments.exp_arcs.localizer_baselines import _truth  # noqa: E402
from experiments.exp_arcs.rq2_combined_topk import tie_aware_hit  # noqa: E402
from experiments.exp_arcs.counterfactual import coverage_map  # noqa: E402
from experiments.exp_arcs.dag import scope_cost  # noqa: E402

PROTOCOL = HERE / "STEP5_CALIB_TRANSFER_PROTOCOL.md"
STEP2 = HERE / "step2_baselines_v1.json"
GRID = s2.D / "e3_grid_v1.json"
ORACLE = s2.D / "e3_oracle_stats_v1.json"
T_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97)
METHODS = ("combined", "iforest", "robust_z", "lof")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combined_flags(run: str, t: float) -> frozenset[str]:
    path = Path(run)
    return frozenset(detector_localize(path)["flagged_hosts"]
                     | peer_anomaly_hosts(path, min_base_rate=t)
                     | manifest_dropped_hosts(path))


def rate(flags: set[str] | frozenset[str], scores: dict[str, float]) -> float:
    if not scores:
        return math.nan
    return len(flags) / len(scores)


def dist(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q3": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def mean_metrics(rows: list[dict]) -> dict:
    keys = ("success", "detect1", "detect3", "cost_over_Cstar", "repair_at_B0")
    return {key: float(st.mean(float(row[key]) for row in rows)) for key in keys}


@lru_cache(maxsize=None)
def repair_inputs(clean: str, faulted: str):
    """Cache the expensive graph/diff work once per cell across all methods."""
    dag = s2.clean_dag(clean)
    cm = coverage_map(Path(clean), s2.dfx_of(clean, faulted))
    return dag, cm


def fast_repair(cell: dict, flagged: frozenset[str]) -> dict:
    dag, cm = repair_inputs(cell["clean_dir"], cell["faulted_dir"])
    scope = {node for node, attrs in dag.nodes.items()
             if attrs.get("repairable") and attrs.get("target") in flagged}
    broken = [out for out, required in cm["coverage_map"].items()
              if not set(required) <= scope]
    return {
        # Match select.evaluate_scope exactly: the frozen Step-2 metric treats a
        # scope as successful when no leaf-repairable output remains broken.
        "success": not broken,
        "cost": float(scope_cost(dag, scope)["cost"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    args = parser.parse_args()

    grid = json.loads(GRID.read_text())["cells"]
    step2 = json.loads(STEP2.read_text())
    b0 = float(json.loads(ORACLE.read_text())["summary"]["B0_median_singlepoint_cost"])
    clean_of: dict[str, str] = {}
    cells_by_org: dict[str, list[dict]] = defaultdict(list)
    for cell in grid:
        clean_of.setdefault(cell["org"], cell["clean_dir"])
        cells_by_org[cell["org"]].append(cell)
    orgs = sorted(clean_of)
    assert len(orgs) == 20 and all(len(cells_by_org[o]) == 9 for o in orgs)

    combined_clean_scores = {o: _host_scores(Path(clean_of[o])) for o in orgs}
    combined_clean_flags = {
        o: {str(t): combined_flags(clean_of[o], t) for t in T_GRID} for o in orgs
    }
    generic_clean_scores = {
        det: {o: dict(s2.scores(clean_of[o], det, True)) for o in orgs}
        for det in METHODS if det != "combined"
    }

    folds = []
    for tag, alpha in s2.ALPHAS.items():
        for heldout in orgs:
            train_orgs = [o for o in orgs if o != heldout]
            for method in METHODS:
                selected_t = None
                no_feasible = False
                if method == "combined":
                    train_rates = {
                        str(t): st.mean(rate(combined_clean_flags[o][str(t)], combined_clean_scores[o])
                                        for o in train_orgs)
                        for t in T_GRID
                    }
                    selected_t = next((t for t in T_GRID if train_rates[str(t)] <= alpha), None)
                    no_feasible = selected_t is None
                    if no_feasible:
                        folds.append({"alpha_tag": tag, "alpha": alpha, "heldout_org": heldout,
                                      "method": method, "no_feasible": True,
                                      "training_rates_by_t": train_rates})
                        continue
                    clean_flags = combined_clean_flags[heldout][str(selected_t)]
                    heldout_clean_rate = rate(clean_flags, combined_clean_scores[heldout])
                    training_clean_rate = train_rates[str(selected_t)]
                    idx = T_GRID.index(selected_t)
                    adjacent = {}
                    for direction, pos in (("lower", idx - 1), ("higher", idx + 1)):
                        if 0 <= pos < len(T_GRID):
                            adj_t = T_GRID[pos]
                            adjacent[direction] = {
                                "t": adj_t,
                                "same_alert_set": clean_flags == combined_clean_flags[heldout][str(adj_t)],
                            }
                    score_getter = lambda c: _host_scores(Path(c["faulted_dir"]))
                    flag_getter = lambda c: combined_flags(c["faulted_dir"], selected_t)
                else:
                    tau = float(step2["baselines"][f"{method}|manifest_aware"]
                                ["thresholds"][tag][heldout])
                    training_rates_by_org = {
                        o: rate(frozenset(h for h, v in generic_clean_scores[method][o].items() if v >= tau),
                                generic_clean_scores[method][o])
                        for o in train_orgs
                    }
                    training_clean_rate = st.mean(training_rates_by_org.values())
                    held_scores = generic_clean_scores[method][heldout]
                    clean_flags = frozenset(h for h, v in held_scores.items() if v >= tau)
                    heldout_clean_rate = rate(clean_flags, held_scores)
                    adjacent = None
                    score_getter = lambda c, m=method: dict(s2.scores(c["faulted_dir"], m, True))
                    flag_getter = lambda c, m=method, th=tau: frozenset(
                        h for h, v in s2.scores(c["faulted_dir"], m, True) if v >= th)

                rows = []
                for cell in cells_by_org[heldout]:
                    truth, _ = _truth(cell["inject_node"])
                    scores = score_getter(cell)
                    flags = flag_getter(cell)
                    repaired = fast_repair(cell, flags)
                    rows.append({
                        "cell": cell["cell"],
                        "success": repaired["success"],
                        "detect1": tie_aware_hit(scores, truth, 1),
                        "detect3": tie_aware_hit(scores, truth, s2.K),
                        "cost_over_Cstar": repaired["cost"] / cell["C_star"],
                        "repair_at_B0": bool(repaired["success"] and repaired["cost"] <= b0),
                    })
                excess = heldout_clean_rate - alpha
                fold = {
                    "alpha_tag": tag,
                    "alpha": alpha,
                    "heldout_org": heldout,
                    "method": method,
                    "no_feasible": no_feasible,
                    "training_clean_alert_rate": training_clean_rate,
                    "heldout_clean_alert_rate": heldout_clean_rate,
                    "exceed_amount": excess,
                    "exceeds_alpha": bool(excess > 0),
                    "heldout_metrics": mean_metrics(rows),
                }
                if method == "combined":
                    fold.update({"selected_t": selected_t, "adjacent_on_heldout_clean": adjacent})
                else:
                    fold["threshold"] = tau
                folds.append(fold)

    summary = {}
    for tag, alpha in s2.ALPHAS.items():
        summary[tag] = {}
        for method in METHODS:
            rows = [f for f in folds if f["alpha_tag"] == tag and f["method"] == method]
            feasible = [f for f in rows if not f["no_feasible"]]
            excesses = [f["exceed_amount"] for f in feasible if f["exceeds_alpha"]]
            block = {
                "alpha": alpha,
                "n_folds": len(rows),
                "no_feasible_folds": sum(f["no_feasible"] for f in rows),
                "exceed_folds": sum(f["exceeds_alpha"] for f in feasible),
                "exceed_amount_mean_positive": float(st.mean(excesses)) if excesses else 0.0,
                "exceed_amount_max": max(excesses, default=0.0),
                "heldout_clean_alert_rate_distribution": dist(
                    [f["heldout_clean_alert_rate"] for f in feasible]),
                "heldout_metrics_mean": {
                    key: float(st.mean(f["heldout_metrics"][key] for f in feasible))
                    for key in ("success", "detect1", "detect3", "cost_over_Cstar", "repair_at_B0")
                },
            }
            if method == "combined":
                block["selected_t_counts"] = {
                    str(t): sum(f.get("selected_t") == t for f in feasible) for t in T_GRID
                }
                block["same_as_adjacent_counts"] = {
                    direction: sum(bool(f["adjacent_on_heldout_clean"].get(direction, {})
                                          .get("same_alert_set")) for f in feasible)
                    for direction in ("lower", "higher")
                }
            summary[tag][method] = block

    out = {
        "protocol": str(PROTOCOL.relative_to(REPO)),
        "identity": "descriptive audit; no hypothesis tests",
        "inputs": {
            "protocol_sha256": sha256_file(PROTOCOL),
            "script_sha256": sha256_file(Path(__file__)),
            "step2_sha256": sha256_file(STEP2),
            "grid_sha256": sha256_file(GRID),
            "oracle_sha256": sha256_file(ORACLE),
            "alphas": s2.ALPHAS,
            "combined_t_grid": T_GRID,
            "B0": b0,
        },
        "folds": folds,
        "summary": summary,
    }
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
