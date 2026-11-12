"""dfx_claimtype_breakdown — for each of the 180 E3 cells, break the contaminated
claim ids (from contamination_closure's D(f)) down by claim_type, using the same
asset-class / substantive-class grouping as the pre-registered D1 rule (s2_d1_
frozen_v1.json). contamination_closure already reports contaminated counts by
RECORD TYPE (source_artifact/evidence_event/claim/decision/report_item); it does
not look inside `claim` to see WHICH claim_type was hit. This closes that gap to
answer: across the main grid, how often does D(f) actually touch a substantive
(single-source, Coverage_2+=0.000) claim, vs. only the highly-redundant asset
claims (Coverage_2+=0.996) whose loss is invisible at the output? This is the
factual basis for whether a "the repair also matters for outputs users would
notice" claim needs a new fault variant, or can already be made from this grid.

    poetry run python -m experiments.exp_arcs.dfx_claimtype_breakdown
"""
from __future__ import annotations
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from experiments.exp_arcs.contamination import contamination_closure

OUT = Path("experiments/exp_arcs/dfx_claimtype_breakdown_v1.json")

ASSET_TYPES = {"asset_identity", "organization_association"}
SUBSTANTIVE_TYPES = {"service_observation", "technology_identification",
                      "vulnerability_applicability"}


def _claim_types(run_dir: Path, claim_ids: set[str]) -> Counter:
    types: Counter = Counter()
    path = run_dir / "claim.jsonl"
    if not path.exists() or not claim_ids:
        return types
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        if row["claim_id"] in claim_ids:
            types[row["claim_type"]] += 1
    return types


def score_cell(cell: dict) -> dict:
    clean, faulted = Path(cell["clean_dir"]), Path(cell["faulted_dir"])
    dfx = contamination_closure(clean, faulted)
    claim_ids = {i.split(":", 1)[1] for i in dfx["contaminated_ids"] if i.startswith("claim:")}
    # ids are content-addressed, so a contaminated id exists in exactly one of
    # clean/faulted (removed vs. added) -- summing both lookups double-counts
    # nothing.
    types = _claim_types(clean, claim_ids)
    types.update(_claim_types(faulted, claim_ids))
    n_asset = sum(v for k, v in types.items() if k in ASSET_TYPES)
    n_substantive = sum(v for k, v in types.items() if k in SUBSTANTIVE_TYPES)
    n_other = sum(v for k, v in types.items() if k not in ASSET_TYPES | SUBSTANTIVE_TYPES)
    return {
        "org": cell["org"], "cell": cell["cell"],
        "n_contaminated_claims": len(claim_ids),
        "n_asset_claims_hit": n_asset,
        "n_substantive_claims_hit": n_substantive,
        "n_other_claims_hit": n_other,
        "claim_types_hit": dict(types),
    }


def main() -> int:
    grid = json.loads(Path("experiments/exp_arcs/e3_grid_v1.json").read_text())["cells"]
    rows = [score_cell(c) for c in grid]
    n_cells_with_any_claim_damage = sum(1 for r in rows if r["n_contaminated_claims"] > 0)
    n_cells_hitting_substantive = sum(1 for r in rows if r["n_substantive_claims_hit"] > 0)
    n_cells_hitting_only_asset = sum(
        1 for r in rows if r["n_asset_claims_hit"] > 0 and r["n_substantive_claims_hit"] == 0)
    total_substantive_hit = sum(r["n_substantive_claims_hit"] for r in rows)
    total_asset_hit = sum(r["n_asset_claims_hit"] for r in rows)

    by_cell_type: dict[str, list[int]] = {}
    for r in rows:
        by_cell_type.setdefault(r["cell"], []).append(r["n_substantive_claims_hit"])

    summary = {
        "n_cells": len(rows),
        "n_cells_with_any_claim_damage": n_cells_with_any_claim_damage,
        "n_cells_hitting_a_substantive_claim": n_cells_hitting_substantive,
        "n_cells_hitting_only_asset_claims": n_cells_hitting_only_asset,
        "total_substantive_claims_hit_across_grid": total_substantive_hit,
        "total_asset_claims_hit_across_grid": total_asset_hit,
        "substantive_hits_by_fault_variant": {
            k: {"n_cells_gt0": sum(1 for x in v if x > 0), "total": sum(v)}
            for k, v in by_cell_type.items()},
        "reading": (
            "If n_cells_hitting_a_substantive_claim is 0 (or small) while asset "
            "hits dominate, the current 180-cell grid structurally cannot show a "
            "case where repair changes a claim type without Coverage_2+ "
            "redundancy protecting it -- confirming the 'practical value' gap "
            "flagged in review, and motivating a new fault variant targeted at a "
            "single-source substantive claim (a positive control, analogous to "
            "the existing F2 negative control)."),
    }
    result = {"phase": "D(f) claim-type breakdown vs D1 asset/substantive grouping",
              "summary": summary, "cells": rows}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
