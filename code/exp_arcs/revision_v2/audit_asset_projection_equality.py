"""Read-only audit of clean/faulted recorded asset projection equality.

Reads the existing E3 grid and report_item records; never rewrites frozen inputs.
The output is an auxiliary manuscript audit, not a new experiment.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from experiments.exp_arcs.e8_delta_f1 import assets_from_report


ROOT = Path(__file__).resolve().parents[3]
GRID = ROOT / "experiments/exp_arcs/e3_grid_v1.json"
OUT = ROOT / "experiments/exp_arcs/revision_v2/asset_projection_equality_v1.json"


def main() -> None:
    raw = GRID.read_bytes()
    cells = json.loads(raw)["cells"]
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"equal": 0, "different": 0})
    for cell in cells:
        clean = assets_from_report(ROOT / cell["clean_dir"])
        faulted = assets_from_report(ROOT / cell["faulted_dir"])
        if not clean or not faulted:
            raise ValueError(f"empty recorded asset projection: {cell['org']}/{cell['cell']}")
        key = "equal" if clean == faulted else "different"
        counts[cell["cell"]][key] += 1
    result = {
        "definition": "Exact equality of the E8 assets_from_report structured projection (hosts, IPs, ports, CPEs); not equality of natural-language reports or raw report_item files.",
        "grid_sha256": hashlib.sha256(raw).hexdigest(),
        "total_cells": len(cells),
        "by_variant": dict(sorted(counts.items())),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
