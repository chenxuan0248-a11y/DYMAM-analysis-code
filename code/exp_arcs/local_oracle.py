"""Phase 2 (local): mozilla x {F4-family-scoped, F10-local} ILP oracle S*/C*.

Mirrors ilp_oracle_pilot.py, but injects LOCAL single-target faults via
interceptors_local.inject_local. The local live repair grid:

  - F4@<family>  for each source_family the corpus actually has evidence in
    (target = the host with the MOST evidence of that family; deterministic).
    Drops only that host's evidence of that one family -> small, well-defined I(f).
  - F10-local    (drop the max-active_observation host from discovery).

F2 stays a GLOBAL negative control (frozen ilp_oracle_pilot_v1.json, cited, NOT
recomputed here); F8 stays the gate-logic non-leaf oracle. Record-layer F1/F3/F5/
F6/F7/F9 are EXCLUDED from the repair grid (id-invariant -> D(f) trivial under
contamination_closure; they are the taxonomy's detection sideline).

Each cell: solve_oracle (ILP S*,C*) + vf cross-check + GATE + rho = C*/global.
"""
from __future__ import annotations
import json, sys, hashlib
from collections import Counter
from pathlib import Path

from core.graph import run_pipeline
from experiments.exp_arcs.recorder import record_run
from experiments.exp_arcs.interceptors_local import inject_local
from experiments.exp_arcs.contamination import (
    contamination_closure, _index, _RECORD_FILES)
from experiments.exp_arcs.ilp_oracle import solve_oracle, load_config
from experiments.exp_arcs.counterfactual import vf
from experiments.exp_arcs.dag import build_dag, scope_cost

ORGS = ["cloudflare", "github", "mozilla", "redhat", "wikimedia"]  # pilot_orgs_FROZEN
FAMILIES = ["dns", "active_observation", "certificate", "registration"]

RAW_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/arcs_local_bundles")
RAW_ROOT.mkdir(parents=True, exist_ok=True)
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "experiments/exp_arcs/local_oracle_v1.json")


def bundle_fingerprint(run_dir: Path) -> str:
    h = hashlib.sha256()
    for rtype, (fn, idf) in sorted(_RECORD_FILES.items()):
        idx = _index(run_dir, fn, idf)
        h.update(f"@{rtype}:{len(idx)}".encode())
        for i in sorted(idx):
            h.update(i.encode()); h.update(b"="); h.update(idx[i].encode())
    return h.hexdigest()


def output_fingerprint(dfx: dict, ilp: dict) -> str:
    payload = json.dumps({
        "contaminated_ids": sorted(dfx["contaminated_ids"]),
        "repairable_ids": sorted(dfx["repairable_ids"]),
        "selected_invocations": sorted(str(i) for i in ilp["_selected"]),
        "C_star": ilp["C_star"], "H_star": ilp["H_star"],
        "coverage_map": {k: sorted(v) for k, v in sorted(ilp["_coverage_map"].items())},
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def pick_targets(clean: Path) -> dict[str, str]:
    """Deterministic per-family target = host with most evidence of that family."""
    evs = [json.loads(l) for l in (clean / "evidence_event.jsonl").read_text().splitlines() if l.strip()]
    per_fam: dict[str, Counter] = {}
    for e in evs:
        per_fam.setdefault(e.get("source_family"), Counter())[e["target"]] += 1
    targets = {}
    for fam in FAMILIES:
        c = per_fam.get(fam)
        if not c:
            continue
        targets[fam] = max(sorted(c), key=lambda h: c[h])
    # F10-local target: max active_observation host (the proven local F10 target)
    ao = per_fam.get("active_observation", Counter())
    targets["_f10"] = max(sorted(ao), key=lambda h: ao[h]) if ao else None
    return targets


def make_cell(clean: Path, cdag, gcost: int, org: str, name: str, faulted: Path,
              inject_node) -> dict:
    dfx = contamination_closure(clean, faulted)
    ilp = solve_oracle(clean, faulted)
    cf = vf(clean, faulted)
    c_ilp, c_min = ilp["C_star"], cf["minimal_cost"]["cost"]
    gate_optimal = ilp["status"] == "Optimal"
    gate_cost_match = (c_ilp == c_min)
    gate_covers_all = ilp["n_outputs_covered"] == cf["n_outputs_leaf_repairable"]
    gate_pass = gate_optimal and gate_cost_match and gate_covers_all
    cell = {
        "org": org, "cell": name, "inject_node": inject_node,
        "D_f_contaminated": dfx["n_contaminated"],
        "D_f_repairable": dfx["n_repairable"],
        "D_f_contaminated_by_type": dfx["contaminated_by_type"],
        "ilp_status": ilp["status"],
        "C_star": c_ilp, "H_star": ilp["H_star"],
        "n_selected_invocations": ilp["n_selected_invocations"],
        "n_outputs_covered": ilp["n_outputs_covered"],
        "n_outputs_unrepairable_by_leaf": ilp["n_outputs_unrepairable_by_leaf"],
        "crosscheck_minimal_cost": c_min,
        "crosscheck_minimal_scope_size": cf["minimal_repair_scope_size"],
        "global_recollect_cost": gcost,
        "rho_C_vs_global": round(c_ilp / gcost, 4) if gcost else None,
        "gate": {"optimal": gate_optimal,
                 "cost_matches_counterfactual": gate_cost_match,
                 "covers_all_leaf_repairable_outputs": gate_covers_all,
                 "PASS": gate_pass},
        "fingerprint": {"clean_bundle": bundle_fingerprint(clean),
                        "faulted_bundle": bundle_fingerprint(faulted),
                        "output": output_fingerprint(dfx, ilp)},
        "clean_dir": str(clean), "faulted_dir": str(faulted),
    }
    print(f"   {org}/{name:22} inode={inject_node} D(f)={dfx['n_contaminated']} "
          f"rep={dfx['n_repairable']} C*={c_ilp} (min={c_min}) "
          f"rho={cell['rho_C_vs_global']} cover={ilp['n_outputs_covered']} "
          f"GATE={'PASS' if gate_pass else 'FAIL'}", flush=True)
    return cell


def main() -> int:
    cells = []
    global_cost: dict[str, int] = {}
    targets_by_org: dict[str, dict] = {}
    for org in ORGS:
        print(f"== {org}: clean ==", flush=True)
        clean = record_run(run_pipeline(organization=org), raw_root=RAW_ROOT)
        cdag = build_dag(clean)
        gcost = scope_cost(cdag, [n for n, v in cdag.nodes.items() if v.get("repairable")])["cost"]
        global_cost[org] = gcost
        targets = pick_targets(clean)
        targets_by_org[org] = targets
        print(f"   global_recollect_cost={gcost}  targets={json.dumps(targets)}", flush=True)

        for fam in FAMILIES:
            tgt = targets.get(fam)
            if not tgt:
                print(f"   [skip] {org}/F4@{fam}: no evidence of this family", flush=True)
                continue
            with inject_local("F4", tgt, family=fam):
                faulted = record_run(run_pipeline(organization=org), raw_root=RAW_ROOT)
            cells.append(make_cell(clean, cdag, gcost, org, f"F4@{fam}", faulted,
                                   ["target", tgt, fam]))

        if targets.get("_f10"):
            tgt = targets["_f10"]
            with inject_local("F10", tgt):
                faulted = record_run(run_pipeline(organization=org), raw_root=RAW_ROOT)
            cells.append(make_cell(clean, cdag, gcost, org, "F10-local", faulted,
                                   ["target", tgt]))

    result = {
        "phase": "2.local",
        "spec": "5 pilot orgs x {F4-family-scoped, F10-local} local ILP oracle S*/C*",
        "orgs": ORGS, "families": FAMILIES,
        "global_recollect_cost_per_org": global_cost,
        "targets_by_org": targets_by_org,
        "grid_note": ("local live repair grid; F2=global negative control "
                      "(cited from ilp_oracle_pilot_v1.json, not recomputed); "
                      "F8=gate-logic non-leaf oracle; record-layer F1/F3/F5/F6/F7/F9 "
                      "excluded (id-invariant, detection sideline)."),
        "n_cells": len(cells),
        "n_gate_pass": sum(c["gate"]["PASS"] for c in cells),
        "all_gate_pass": all(c["gate"]["PASS"] for c in cells),
        "aggregate_fingerprint": hashlib.sha256(
            "".join(c["fingerprint"]["output"] for c in cells).encode()).hexdigest(),
        "cells": cells,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nwrote {OUT}  cells={len(cells)}  gate_pass={result['n_gate_pass']}/{len(cells)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
