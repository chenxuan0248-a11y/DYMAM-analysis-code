"""experiments/exp_arcs/record_dfs — record-layer ground-truth D(f) (Step 1.4, P0-c).

The live faults (F2/F8/F10/F4) get their D(f) empirically in ``contamination.py``
by diffing clean-vs-fault bundles. The six record-layer faults
(F1/F3/F5/F6/F7/F9 in ``experiments/mechanism_v1/fault_suite.py``) are
id-invariant validator faults applied to a *recorded* bundle rather than to the
running pipeline, so an empirical clean-vs-fault run diff does not apply. Their
D(f) is therefore the plan's structural definition, computed directly on the DAG:

    D(f) = {inject_node} ∪ Descendants(inject_node)   (∩ Relevant(G) = the DAG)

``inject_node`` is the record each mutator tampers. Matching the mutators
(``fault_suite`` uses ``_jsonl(path)[0]``), it is the FIRST row of the target
ledger:

    F1 delete_source / F3 tamper_source   → source_artifact[0]
    F5 swap_snapshot / F6 swap_scope /
    F9 forge_parent                       → evidence_event[0]
    F7 tamper_decision                    → decision[0]

Computed on the same fixture bundle the record-layer suite runs against
(``fault_suite._build_base`` — the infra_v1 adapter fixture, no pipeline run), so
these are fixture-scoped labels, complementing the corpus-scoped live D(f).

    poetry run python -m experiments.exp_arcs.record_dfs [--out record_dfs.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from experiments.exp_arcs.dag import build_dag
from experiments.exp_arcs.faults import FAULT_REGISTRY
from experiments.mechanism_v1.fault_suite import _build_base

_REPAIRABLE = ("source_artifact", "evidence_event")

# record-layer fault_id → (target ledger file, id field) the mutator tampers.
# The mutators (fault_suite) all hit the FIRST row of these files.
_TARGETS: dict[str, tuple[str, str]] = {
    "F1": ("source_artifact.jsonl", "artifact_id"),   # _delete_source
    "F3": ("source_artifact.jsonl", "artifact_id"),   # _tamper_source
    "F5": ("evidence_event.jsonl", "evidence_event_id"),  # _swap_snapshot
    "F6": ("evidence_event.jsonl", "evidence_event_id"),  # _swap_scope
    "F7": ("decision.jsonl", "decision_id"),          # _tamper_decision
    "F9": ("evidence_event.jsonl", "evidence_event_id"),  # _fake_parent
}


def _first_row_id(run_dir: Path, filename: str, id_field: str) -> str | None:
    path = run_dir / filename
    if not path.exists():
        return None
    for ln in path.read_text().splitlines():
        if ln.strip():
            return str(json.loads(ln)[id_field])
    return None


def record_closure(run_dir: str | Path, fault_id: str) -> dict[str, Any]:
    """D(f) = {inject_node} ∪ Descendants(inject_node) for one record-layer fault."""
    run_dir = Path(run_dir)
    filename, id_field = _TARGETS[fault_id]
    seed = _first_row_id(run_dir, filename, id_field)
    dag = build_dag(run_dir)

    closure: set[str] = set()
    if seed is not None and seed in dag.nodes:
        closure = {seed} | dag.descendants(seed)

    by_type: dict[str, list[str]] = {}
    for nid in closure:
        by_type.setdefault(dag.nodes[nid]["type"], []).append(nid)

    all_ids = [f"{t}:{i}" for t, ids in by_type.items() for i in sorted(ids)]
    repairable = [f"{t}:{i}" for t in _REPAIRABLE for i in sorted(by_type.get(t, []))]
    return {
        "inject_node": FAULT_REGISTRY[fault_id].inject_node,
        "inject_node_id": seed,
        "fault_name": FAULT_REGISTRY[fault_id].name,
        "contaminated_by_type": {t: len(ids) for t, ids in by_type.items()},
        "n_contaminated": len(all_ids),
        "n_repairable": len(repairable),
        "contaminated_ids": all_ids,
        "repairable_ids": repairable,
    }


def compute_record_dfs(faults: list[str], fixture_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "bundle": "mechanism_v1.fault_suite._build_base (fixture)",
        "bundle_run_id": fixture_dir.name,
        "faults": {},
    }
    for f in faults:
        dfx = record_closure(fixture_dir, f)
        out["faults"][f] = dfx
        print(f"  {f} @ {dfx['inject_node']} (id={dfx['inject_node_id']}): "
              f"D(f) = {dfx['n_contaminated']} records "
              f"({dfx['n_repairable']} repairable) {dfx['contaminated_by_type']}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faults", default=",".join(_TARGETS),
                        help="record-layer faults (default: all six)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    faults = [f.strip() for f in args.faults.split(",") if f.strip()]
    fixture_dir = _build_base(Path(tempfile.mkdtemp(prefix="arcs_record_df_")) / "base")
    print(f"fixture: {fixture_dir}")
    result = compute_record_dfs(faults, fixture_dir)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
