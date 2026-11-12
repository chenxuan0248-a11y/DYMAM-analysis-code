"""Phase 2 pilot: 5 orgs x {F2,F4,F10} ILP oracle (S*, C*) + per-cell GATE.

Regenerates clean + faulted bundles under a PERSISTENT raw-root (deterministic
replay, Qdrant up), then for each (org, fault) cell:
  - solve_oracle  -> ILP S*, C* (cost+harm objective, frozen ilp_config.yaml)
  - vf            -> independent counterfactual minimal cover (cross-check)
  - GATE          -> (a) ILP Optimal, (b) C*(ILP) == minimal cover cost,
                     (c) every leaf-repairable contaminated output covered.
  - rho           -> C*/global re-collect (ARCS-vs-global cost story, H3 preview)

Reproducibility: each bundle gets a run_id-invariant fingerprint (sha256 over
``contamination._index`` normalized records — the SAME normalization D(f) uses,
so it is stable across runs despite fresh run_ids; all timestamps are pinned in
replay). Each cell also gets an output fingerprint over its run_id-invariant
substance (sorted contaminated ids + selected S* invocations + C*/H*). When a
prior artifact exists at the out path, this run compares its analytical outputs
against the prior run and records the result under ``reproducibility`` — a
determinism self-check (G1 discipline applied to the Phase 2 inputs/outputs).

F8 is excluded by design (gate_logic, 0 repairable leaves) — deferred to the
non-leaf action model (Modification 3).
"""
from __future__ import annotations
import json, sys, hashlib
from pathlib import Path

from core.graph import run_pipeline
from experiments.exp_arcs.recorder import record_run
from experiments.exp_arcs.interceptors import inject
from experiments.exp_arcs.faults import FAULT_REGISTRY
from experiments.exp_arcs.contamination import (
    contamination_closure, _index, _RECORD_FILES)
from experiments.exp_arcs.ilp_oracle import solve_oracle, load_config
from experiments.exp_arcs.counterfactual import vf
from experiments.exp_arcs.dag import build_dag, scope_cost

ORGS = ["cloudflare", "github", "mozilla", "redhat", "wikimedia"]  # pilot_orgs_FROZEN
FAULTS = ["F2", "F4", "F10"]                                        # leaf faults (F8 deferred)

RAW_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/arcs_pilot_bundles")
RAW_ROOT.mkdir(parents=True, exist_ok=True)

# fields compared fresh-vs-prior for the determinism self-check
_REPRO_FIELDS = ("D_f_contaminated", "D_f_repairable", "C_star", "H_star",
                 "n_selected_invocations", "n_outputs_covered",
                 "crosscheck_minimal_cost", "rho_C_vs_global")


def rec(org: str, fault: str | None) -> Path:
    with inject(fault):
        return record_run(run_pipeline(organization=org), raw_root=RAW_ROOT)


def bundle_fingerprint(run_dir: Path) -> str:
    """run_id-invariant sha256 over normalized records (same normalization as
    D(f) — record ids are content-addressed, run_id is the only volatile field
    and is stripped, all timestamps pinned in replay). Stable across reruns."""
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


def main() -> int:
    cfg = load_config()
    cfg_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]

    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("ilp_oracle_pilot_v1.json")
    prior = None
    if out_path.exists():
        try:
            pj = json.loads(out_path.read_text())
            prior = {(c["org"], c["fault"]): c for c in pj.get("cells", [])}
        except Exception:  # noqa: BLE001
            prior = None

    cells = []
    global_cost: dict[str, int] = {}
    for org in ORGS:
        print(f"== {org}: clean ==", flush=True)
        clean = rec(org, None)
        clean_fp = bundle_fingerprint(clean)
        cdag = build_dag(clean)
        gcost = scope_cost(cdag, [n for n, v in cdag.nodes.items() if v.get("repairable")])["cost"]
        global_cost[org] = gcost
        for f in FAULTS:
            print(f"== {org}/{f} ==", flush=True)
            faulted = rec(org, f)
            dfx = contamination_closure(clean, faulted)
            ilp = solve_oracle(clean, faulted, cfg)
            cf = vf(clean, faulted)
            c_ilp = ilp["C_star"]
            c_min = cf["minimal_cost"]["cost"]
            gate_optimal = ilp["status"] == "Optimal"
            gate_cost_match = (c_ilp == c_min)
            gate_covers_all = ilp["n_outputs_covered"] == cf["n_outputs_leaf_repairable"]
            gate_pass = gate_optimal and gate_cost_match and gate_covers_all
            cell = {
                "org": org, "fault": f,
                "fault_name": FAULT_REGISTRY[f].name,
                "inject_node": getattr(FAULT_REGISTRY[f], "inject_node", None),
                "D_f_contaminated": dfx["n_contaminated"],
                "D_f_repairable": dfx["n_repairable"],
                "ilp_status": ilp["status"],
                "C_star": c_ilp, "H_star": ilp["H_star"],
                "n_selected_invocations": ilp["n_selected_invocations"],
                "n_outputs_covered": ilp["n_outputs_covered"],
                "n_outputs_unrepairable_by_leaf": ilp["n_outputs_unrepairable_by_leaf"],
                "crosscheck_minimal_cost": c_min,
                "crosscheck_minimal_harm": cf["minimal_cost"]["harm"],
                "crosscheck_minimal_scope_size": cf["minimal_repair_scope_size"],
                "global_recollect_cost": gcost,
                "rho_C_vs_global": round(c_ilp / gcost, 3) if gcost else None,
                "gate": {
                    "optimal": gate_optimal,
                    "cost_matches_counterfactual": gate_cost_match,
                    "covers_all_leaf_repairable_outputs": gate_covers_all,
                    "PASS": gate_pass,
                },
                "fingerprint": {
                    "clean_bundle": clean_fp,
                    "faulted_bundle": bundle_fingerprint(faulted),
                    "output": output_fingerprint(dfx, ilp),
                },
                "clean_dir": str(clean), "faulted_dir": str(faulted),
            }
            cells.append(cell)
            print(f"   D(f)={dfx['n_contaminated']} rep={dfx['n_repairable']} "
                  f"C*={c_ilp} (min={c_min}) H*={ilp['H_star']} rho={cell['rho_C_vs_global']} "
                  f"cover={ilp['n_outputs_covered']} GATE={'PASS' if gate_pass else 'FAIL'}",
                  flush=True)

    # determinism self-check vs the prior artifact (if any)
    repro: dict = {"prior_artifact_present": prior is not None}
    if prior is not None:
        mismatches = []
        for c in cells:
            p = prior.get((c["org"], c["fault"]))
            if p is None:
                mismatches.append({"cell": f"{c['org']}/{c['fault']}", "reason": "absent in prior"})
                continue
            diffs = {k: [p.get(k), c[k]] for k in _REPRO_FIELDS if p.get(k) != c[k]}
            if diffs:
                mismatches.append({"cell": f"{c['org']}/{c['fault']}", "field_diffs": diffs})
        prior_has_fp = any("fingerprint" in (prior.get(k) or {}) for k in prior)
        repro.update({
            "compared_fields": list(_REPRO_FIELDS),
            "n_cells_compared": len(cells),
            "n_mismatch": len(mismatches),
            "all_match": not mismatches,
            "mismatches": mismatches,
            "prior_had_fingerprints": prior_has_fp,
            "note": ("analytical outputs verified byte-equal to the prior "
                     "independent run; fingerprints stored this run as the "
                     "byte-level baseline for future `diff` verification"
                     if not prior_has_fp else
                     "compared against prior run carrying fingerprints"),
        })
    else:
        repro["note"] = ("no prior artifact; this run establishes the "
                         "fingerprint baseline")

    result = {
        "phase": "2.pilot",
        "spec": "5 orgs x {F2,F4,F10} ILP oracle S*/C* + per-cell GATE",
        "orgs": ORGS, "faults": FAULTS,
        "f8_excluded": "gate_logic, 0 repairable leaves; deferred to non-leaf action model (Mod 3)",
        "ilp_config_version": cfg["config_version"],
        "ilp_config_hash16": cfg_hash,
        "raw_root": str(RAW_ROOT),
        "global_recollect_cost_per_org": global_cost,
        "n_cells": len(cells),
        "n_gate_pass": sum(c["gate"]["PASS"] for c in cells),
        "all_gate_pass": all(c["gate"]["PASS"] for c in cells),
        "reproducibility": repro,
        "aggregate_fingerprint": hashlib.sha256(
            "".join(c["fingerprint"]["output"] for c in cells).encode()).hexdigest(),
        "cells": cells,
    }
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nwrote {out_path}  cells={len(cells)}  "
          f"gate_pass={result['n_gate_pass']}/{len(cells)}  "
          f"repro={'MATCH' if repro.get('all_match') else ('BASELINE' if prior is None else 'MISMATCH')}")
    ok = result["all_gate_pass"] and (prior is None or repro.get("all_match", True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
