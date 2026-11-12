"""experiments/exp_arcs/contamination — ground-truth contamination closure D(f).

Step 1.4. ``D(f) = Descendants(inject_node) ∩ Relevant(G)`` — the records a fault
actually contaminates. The referenced ``exp_ep_error_propagation/run.py`` is an
empty stub, so D(f) is computed EMPIRICALLY: run the pipeline clean and under
fault f (both deterministic — Phase 0.3 G1), record both bundles, and take the
records that differ (added, removed, or content-changed by normalized content).
This is exactly the set of contaminated records, and it needs no fragile
static-descendants analysis of a live injection point.

Output per fault: the contaminated record ids (all types) and the subset that is
directly repairable (source_artifact / evidence_event — the nodes ARCS may
choose to re-run). These are the scope precision/recall labels for Phase 2/6.

    poetry run python -m experiments.exp_arcs.contamination \
        --org mozilla --faults F2,F8,F10,F4 [--out dfs.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.graph import run_pipeline
from experiments.exp_arcs.dag import build_dag
from experiments.exp_arcs.faults import FAULT_REGISTRY, LIVE_FAULTS
from experiments.exp_arcs.interceptors import inject
from experiments.exp_arcs.recorder import record_run

_RECORD_FILES = {
    "source_artifact": ("source_artifact.jsonl", "artifact_id"),
    "evidence_event": ("evidence_event.jsonl", "evidence_event_id"),
    "claim": ("claim.jsonl", "claim_id"),
    "decision": ("decision.jsonl", "decision_id"),
    "report_item": ("report_item.jsonl", "report_item_id"),
}
_REPAIRABLE = ("source_artifact", "evidence_event")
_VOLATILE = frozenset({"run_id"})


def _index(run_dir: Path, filename: str, id_field: str) -> dict[str, str]:
    """id → normalized content (run_id stripped) for one record file."""
    path = run_dir / filename
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        out[row[id_field]] = json.dumps(
            {k: v for k, v in row.items() if k not in _VOLATILE}, sort_keys=True)
    return out


def contamination_closure(clean_dir: Path, faulted_dir: Path) -> dict[str, Any]:
    """D(f): contaminated record ids (added / removed / modified) per type."""
    contaminated: dict[str, list[str]] = {}
    for rtype, (filename, id_field) in _RECORD_FILES.items():
        a = _index(clean_dir, filename, id_field)
        b = _index(faulted_dir, filename, id_field)
        changed = {i for i in a.keys() | b.keys() if a.get(i) != b.get(i)}
        if changed:
            contaminated[rtype] = sorted(changed)
    all_ids = [f"{t}:{i}" for t, ids in contaminated.items() for i in ids]
    repairable = [f"{t}:{i}" for t in _REPAIRABLE for i in contaminated.get(t, [])]
    return {
        "contaminated_by_type": {t: len(ids) for t, ids in contaminated.items()},
        "n_contaminated": len(all_ids),
        "n_repairable": len(repairable),
        "contaminated_ids": all_ids,
        "repairable_ids": repairable,
    }


_DERIVED_TYPES = ("claim", "decision", "report_item")


def graph_crosscheck(clean_dir: Path, dfx: dict[str, Any]) -> dict[str, Any]:
    """Validate empirical D(f) against DAG reachability (Step 1.2 ↔ 1.4).

    The plan defines D(f) = ``Descendants(inject_node) ∩ Relevant(G)``, where G is
    the *original* (clean) evidence DAG. We keep the empirical clean-vs-fault diff
    as the authoritative D(f) (it is the actual contamination), and use G as a
    consistency check: every empirically contaminated *derived* record
    (claim / decision / report_item) that exists in G should be reachable in G
    from a contaminated *leaf* (source_artifact / evidence_event).

    The DAG is built from the CLEAN bundle, because record ids are
    content-addressed: a fault that changes or drops a record re-derives (or
    removes) its id, so contaminated clean-side ids do not exist in the faulted
    DAG. Contaminated derived ids that are *not* in G are fault-side records the
    fault ADDED (``n_derived_off_graph``); they have no clean ancestor by
    construction and are reported separately rather than counted as misses.

    ``coverage`` (over in-graph derived records) ≈ 1 ⇒ contamination is pure
    data-flow propagation and G's edges are complete. ``coverage < 1`` with
    ``n_seed_leaves = 0`` ⇒ the fault acts directly on derived nodes (a gate-logic
    fault such as F8) — the signal a leaf-re-run repair model does not cover it.
    """
    g = build_dag(clean_dir)

    def _bare(types: tuple[str, ...]) -> set[str]:
        return {i.split(":", 1)[1] for i in dfx["contaminated_ids"]
                if i.split(":", 1)[0] in types}

    seeds = {s for s in _bare(_REPAIRABLE) if s in g.nodes}
    derived_all = _bare(_DERIVED_TYPES)
    derived = {d for d in derived_all if d in g.nodes}       # expressible in G
    reachable: set[str] = set()
    for s in seeds:
        reachable |= g.descendants(s)
    explained = derived & reachable
    return {
        "n_seed_leaves": len(seeds),
        "n_derived_in_graph": len(derived),
        "n_derived_off_graph": len(derived_all - derived),   # fault-added ids
        "n_explained": len(explained),
        "n_unexplained": len(derived - reachable),
        "coverage": (len(explained) / len(derived)) if derived else 1.0,
    }


def compute_dfs(org: str, faults: list[str], raw_root: Path) -> dict[str, Any]:
    def rec(fault: str | None) -> tuple[Path, dict[str, Any]]:
        with inject(fault) as meta:
            return record_run(run_pipeline(organization=org), raw_root=raw_root), meta

    clean, _ = rec(None)
    out: dict[str, Any] = {"org": org, "faults": {}}
    for f in faults:
        faulted, meta = rec(f)
        dfx = contamination_closure(clean, faulted)
        dfx["inject_node"] = meta["inject_node"]          # P0-a: OUTPUT (inject_node, D(f))
        dfx["fault_name"] = FAULT_REGISTRY[f].name
        dfx["graph_crosscheck"] = graph_crosscheck(clean, dfx)  # P0-b (Relevant(G)=clean)
        out["faults"][f] = dfx
        cc = dfx["graph_crosscheck"]
        print(f"  {org}/{f} @ {dfx['inject_node']}: D(f) = {dfx['n_contaminated']} "
              f"records ({dfx['n_repairable']} repairable) {dfx['contaminated_by_type']} "
              f"| graph coverage {cc['coverage']:.2f} ({cc['n_unexplained']} unexplained)")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org", default="mozilla",
                        help="one org, or comma-separated orgs (e.g. the pilot set)")
    parser.add_argument("--faults", default=",".join(LIVE_FAULTS))
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    faults = [f.strip() for f in args.faults.split(",") if f.strip()]
    orgs = [o.strip() for o in args.org.split(",") if o.strip()]
    raw_root = Path(args.raw_root) if args.raw_root else Path(
        tempfile.mkdtemp(prefix="arcs_df_"))

    if len(orgs) == 1:
        result: dict[str, Any] = compute_dfs(orgs[0], faults, raw_root)
    else:
        result = {"orgs": {}, "skipped": {}}
        for org in orgs:
            print(f"{org}:")
            try:  # one failing org must not kill the pilot batch
                result["orgs"][org] = compute_dfs(org, faults, raw_root)["faults"]
            except Exception as exc:  # noqa: BLE001
                print(f"  SKIPPED ({type(exc).__name__}: {exc})")
                result["skipped"][org] = f"{type(exc).__name__}: {exc}"

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
