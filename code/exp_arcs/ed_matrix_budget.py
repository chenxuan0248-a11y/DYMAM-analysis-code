"""experiments/exp_arcs/ed_matrix_budget — E-D: variant x signal repair-success matrix and
budget-success step curves (protocol: experiments/exp_arcs/protocols/ED_PROTOCOL.md, frozen
before any run). Pure re-aggregation of frozen per-cell success/cost; reads no bundle.

    python -m experiments.exp_arcs.ed_matrix_budget [out.json]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from evaluation.stat_test import bootstrap_ci

D = Path("experiments/exp_arcs")
BUDGET_POINTS_CI = ("B0", 10, 50, 100, 200, 410)
GRID_MAX = 410


def load() -> list[dict]:
    det = {(c["org"], c["cell"]): c for c in json.loads((D / "e3_detector_v1.json").read_text())["cells"]}
    arcs = {(c["org"], c["cell"]): c for c in json.loads((D / "e3_arcs_v1.json").read_text())["cells"]}
    grid = json.loads((D / "e3_grid_v1.json").read_text())["cells"]
    rows = []
    for g in grid:
        k = (g["org"], g["cell"])
        d, a = det[k], arcs[k]["methods"]
        m = {
            "validator_flag": d["self_diagnosis"], "peer_anomaly": d["peer_anomaly"], "manifest": d["manifest"],
            "combined_host": d["combined"], "combined_family": d["combined_family_grain"],
            "oracle_arcs": a["arcs"], "oracle_single_point": a["single_point_oracle"], "global": a["global"],
        }
        rows.append({"org": g["org"], "cell": g["cell"], "C_star": g["C_star"],
                     "m": {name: {"success": bool(v["success"]), "cost": float(v["cost"])} for name, v in m.items()}})
    return rows


MATRIX_COLS = ("validator_flag", "peer_anomaly", "manifest", "combined_host", "combined_family",
               "oracle_arcs", "oracle_single_point")
CURVE_METHODS = ("oracle_arcs", "oracle_single_point", "global", "combined_host", "combined_family")


def cell_stats(rs: list[dict], col: str) -> dict:
    n = len(rs)
    succ = [r["m"][col]["success"] for r in rs]
    ratio = [r["m"][col]["cost"] / r["C_star"] for r in rs]
    ratio_s = [x for x, s in zip(ratio, succ) if s]
    total_cost = sum(r["m"][col]["cost"] for r in rs)
    return {"n": n, "success_rate": round(sum(succ) / n, 4),
            "mean_cost_over_Cstar_all_cells": round(sum(ratio) / n, 4),
            "mean_cost_over_Cstar_success_only": round(sum(ratio_s) / len(ratio_s), 4) if ratio_s else None,
            "cost_per_successful_repair": round(total_cost / sum(succ), 4) if sum(succ) else None}


def org_boot(rs: list[dict], f) -> dict:
    by = defaultdict(list)
    for r in rs:
        by[r["org"]].append(f(r))
    means = [sum(v) / len(v) for _, v in sorted(by.items())]
    m, lo, hi = bootstrap_ci(means, B=10_000, seed=42)
    return {"org_mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def repair_at(rs: list[dict], method: str, B: float) -> float:
    return sum(1 for r in rs if r["m"][method]["success"] and r["m"][method]["cost"] <= B) / len(rs)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else D / "ed_variant_signal_matrix_v1.json"
    rows = load()
    B0 = float(json.loads((D / "e3_oracle_stats_v1.json").read_text())["summary"]["B0_median_singlepoint_cost"])
    variants = sorted({r["cell"] for r in rows})

    matrix = {v: {col: cell_stats([r for r in rows if r["cell"] == v], col) for col in MATRIX_COLS} for v in variants}
    matrix["ALL"] = {col: cell_stats(rows, col) for col in MATRIX_COLS}
    matrix_ci = {col: {"success_rate": org_boot(rows, lambda r, c=col: float(r["m"][c]["success"])),
                       "mean_cost_over_Cstar_all_cells": org_boot(rows, lambda r, c=col: r["m"][c]["cost"] / r["C_star"])}
                 for col in MATRIX_COLS}

    curves = {}
    for meth in CURVE_METHODS:
        knots = sorted({0.0} | {r["m"][meth]["cost"] for r in rows})
        curves[meth] = {
            "step_knots": [[b, round(repair_at(rows, meth, b), 6)] for b in knots],
            "integer_grid": [[b, round(repair_at(rows, meth, b), 6)] for b in range(GRID_MAX + 1)],
            "at_points": {str(bp): {"B": (B0 if bp == "B0" else float(bp)),
                                    "repair_at_B": round(repair_at(rows, meth, B0 if bp == "B0" else bp), 4),
                                    **org_boot(rows, lambda r, m=meth, b=(B0 if bp == "B0" else bp):
                                               float(r["m"][m]["success"] and r["m"][m]["cost"] <= b))}
                          for bp in BUDGET_POINTS_CI},
            "overall_success": round(sum(r["m"][meth]["success"] for r in rows) / len(rows), 4),
        }

    detsum = json.loads((D / "e3_detector_v1.json").read_text())["summary"]
    selfcheck = {
        "combined_host_success_eq_e3_detector": matrix["ALL"]["combined_host"]["success_rate"] == round(detsum["detector_only_success"]["combined"], 4),
        "combined_family_success_eq_e3_detector": matrix["ALL"]["combined_family"]["success_rate"] == round(detsum["detector_only_success"]["combined_family_grain"], 4),
        "combined_host_success_only_cost_eq_7.12": round(matrix["ALL"]["combined_host"]["mean_cost_over_Cstar_success_only"], 2) == 7.12,
        "combined_family_success_only_cost_eq_1.01": round(matrix["ALL"]["combined_family"]["mean_cost_over_Cstar_success_only"], 2) == 1.01,
        "oracle_arcs_success_1": matrix["ALL"]["oracle_arcs"]["success_rate"] == 1.0,
        "oracle_arcs_mean_cost_4.15": round(sum(r["m"]["oracle_arcs"]["cost"] for r in rows) / len(rows), 2) == 4.15,
        "global_mean_cost_321.75": round(sum(r["m"]["global"]["cost"] for r in rows) / len(rows), 2) == 321.75,
        "curves_monotone_and_end_at_overall": all(
            all(a[1] <= b[1] for a, b in zip(c["integer_grid"], c["integer_grid"][1:]))
            and abs(c["integer_grid"][-1][1] - c["overall_success"]) < 5e-5 for c in curves.values()),
    }
    out.write_text(json.dumps({
        "phase": "E-D variant x signal repair-success matrix + budget-success step curves",
        "protocol": "experiments/exp_arcs/protocols/ED_PROTOCOL.md",
        "sources": ["e3_detector_v1.json", "e3_arcs_v1.json", "e3_oracle_stats_v1.json", "e3_grid_v1.json"],
        "B0": {"value": B0, "definition": "median single-point (Oracle) cost, spec units (e3_oracle_stats_v1.json)"},
        "json_key_to_paper_name": {"validator_flag": "self_diagnosis (e3_detector)", "combined_host": "flagged-host full re-collection"},
        "matrix": matrix, "matrix_org_ci_ALL": matrix_ci, "budget_curves": curves, "selfcheck": selfcheck,
    }, indent=1, sort_keys=True))
    print(json.dumps(selfcheck, indent=1))
    hdr = "variant".ljust(24) + "".join(c[:15].rjust(16) for c in MATRIX_COLS)
    print(hdr)
    for v in variants + ["ALL"]:
        print(v.ljust(24) + "".join(f"{matrix[v][c]['success_rate']:.3f}/{matrix[v][c]['mean_cost_over_Cstar_all_cells']:.2f}".rjust(16) for c in MATRIX_COLS))
    print("Repair@B at B0=%.1f:" % B0, {m: curves[m]["at_points"]["B0"]["repair_at_B"] for m in CURVE_METHODS})
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
