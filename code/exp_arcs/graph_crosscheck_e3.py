"""experiments/exp_arcs/graph_crosscheck_e3 — reviewer-H post-hoc: run
contamination.py's graph_crosscheck over the full E3 main grid (20 orgs x 9
fault variants = 180 cells), not just the 5-org pilot it was originally run on
(dfs_pilot_v1.json).

graph_crosscheck validates the EMPIRICAL D(f) (clean-vs-faulted diff,
contamination_closure) against an INDEPENDENT structural expectation: every
contaminated derived record (claim/decision/report_item) should be reachable,
via the clean bundle's static DAG edges, from some contaminated repairable
leaf. It is the one check in this codebase that does not share
contamination_closure's own diff logic as its verification path.

On the 5-org pilot, 3/20 org x fault_type combinations showed a nonzero
residual (mozilla/F10 12/1028, wikimedia/F10 18/498, wikimedia/F4 5/630) that
was never investigated. This script settles whether that residual is present
on the actual 180-cell corpus the paper's numbers are drawn from.

Pure offline: reuses the already-recorded bundles referenced by e3_grid_v1.json
(clean_dir/faulted_dir) — no pipeline re-run, no LLM calls. e3_grid_v1.json only
stores per-type contamination COUNTS, not the id lists graph_crosscheck needs,
so contamination_closure is recomputed fresh from each bundle pair (cheap: pure
JSONL diffing, no interceptor / no run_pipeline).

    poetry run python -m experiments.exp_arcs.graph_crosscheck_e3 \
        [e3_grid_v1.json] [graph_crosscheck_e3_v1.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from experiments.exp_arcs.contamination import contamination_closure, graph_crosscheck

DEFAULT_GRID = Path(__file__).with_name("e3_grid_v1.json")
DEFAULT_OUT = Path(__file__).with_name("graph_crosscheck_e3_v1.json")


def _fault_type(cell_name: str) -> str:
    return "F10" if "F10" in cell_name else "F4"


def run(grid_path: Path) -> dict[str, Any]:
    grid = json.loads(grid_path.read_text())
    rows: list[dict[str, Any]] = []
    for c in grid["cells"]:
        clean_dir, faulted_dir = Path(c["clean_dir"]), Path(c["faulted_dir"])
        if not clean_dir.exists() or not faulted_dir.exists():
            rows.append({"org": c["org"], "cell": c["cell"], "error": "bundle missing"})
            continue
        dfx = contamination_closure(clean_dir, faulted_dir)
        gc = graph_crosscheck(clean_dir, dfx)
        rows.append({"org": c["org"], "cell": c["cell"],
                      "fault_type": _fault_type(c["cell"]),
                      "n_contaminated": dfx["n_contaminated"], **gc})
        flag = "" if gc["n_unexplained"] == 0 else "  <-- UNEXPLAINED"
        print(f"{c['org']:12} {c['cell']:16} coverage={gc['coverage']:.4f} "
              f"unexplained={gc['n_unexplained']:3d}/{gc['n_derived_in_graph']:4d}{flag}")

    n_with_gap = sum(1 for r in rows if r.get("n_unexplained", 0) > 0)
    n_errors = sum(1 for r in rows if "error" in r)
    by_fault_gap: dict[str, list] = {}
    for r in rows:
        if r.get("n_unexplained", 0) > 0:
            by_fault_gap.setdefault(r["fault_type"], []).append(
                (r["org"], r["cell"], r["n_unexplained"], r["n_derived_in_graph"]))

    return {
        "phase": "reviewer-H graph_crosscheck backfill onto the full E3 grid",
        "note": ("dfs_pilot_v1.json (5 orgs) showed n_unexplained>0 on 3/20 "
                 "org x fault_type combinations (mozilla/F10, wikimedia/F10, "
                 "wikimedia/F4); this run checks whether that residual is "
                 "present on the 180-cell corpus e3_grid_v1.json actually "
                 "reports results from."),
        "source": str(grid_path),
        "n_cells": len(rows),
        "n_cells_with_gap": n_with_gap,
        "n_bundle_errors": n_errors,
        "gaps_by_fault_type": by_fault_gap,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    grid_path = Path(argv[0]) if len(argv) > 0 else DEFAULT_GRID
    out_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT

    summary = run(grid_path)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{summary['n_cells_with_gap']}/{summary['n_cells']} cells have "
          f"n_unexplained > 0 ({summary['n_bundle_errors']} bundle errors)")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
