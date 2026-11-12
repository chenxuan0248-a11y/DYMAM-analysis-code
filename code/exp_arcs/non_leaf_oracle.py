"""Phase 2 non-leaf repair oracle — gate-logic faults F8 (live) + F7 (record).

Implements `non_leaf_oracle_FROZEN.md` (frozen 2026-09-17, BEFORE any S*/C*).

Gate-logic faults contaminate only decision / report_item (n_repairable = 0), so
the leaf clean-DAG coverage frame is unsound for them (F8 over-admission adds
faulted-side records absent from the clean DAG). This module instead:

  - identifies the contaminated NON-LEAF set:
      F8 (live)   -> empirical clean-vs-faulted diff at the logical state_slot grain
      F7 (record) -> structural closure {inject}∪Descendants on the fixture DAG
  - S* = FORCED cover: re_evaluate every contaminated decision + re_render every
    contaminated report_item (no substitution freedom → S* deterministic; NL-B.3)
  - C* = |re_evaluate|*cost + |re_render|*cost   (cost=1, harm=0, idempotent; NL-A)
  - GATE (NL-C): (1) forced cover computed; (2) the slot-grain diff and the
    id-grain contamination_closure identify the SAME set (two independent comps);
    (3) A1 success — contamination confined to {decision, report_item}, so applying
    S* restores clean (no other record type is left contaminated).
  - fingerprints + determinism self-check, reused from ilp_oracle_pilot.

cost_harm_spec.yaml is left byte-identical (I-1); the non-leaf action costs live in
the frozen NL_ACTIONS block below with their own nl_spec_hash.
"""
from __future__ import annotations
import json, sys, hashlib, tempfile
from pathlib import Path

from core.graph import run_pipeline
from experiments.exp_arcs.recorder import record_run
from experiments.exp_arcs.interceptors import inject
from experiments.exp_arcs.faults import FAULT_REGISTRY
from experiments.exp_arcs.contamination import (
    contamination_closure, _index, _RECORD_FILES, _VOLATILE)
from experiments.exp_arcs.record_dfs import record_closure
from experiments.mechanism_v1.fault_suite import _build_base

# --- frozen non-leaf action block (non_leaf_oracle_FROZEN.md NL-A) ---
NL_ACTIONS = {
    "re_evaluate": {"cost": 1, "reversibility": "idempotent", "harm": 0},
    "re_render":   {"cost": 1, "reversibility": "idempotent", "harm": 0},
}
NL_SPEC_HASH = hashlib.sha256(
    json.dumps(NL_ACTIONS, sort_keys=True).encode()).hexdigest()[:16]
C_RE_EVAL = NL_ACTIONS["re_evaluate"]["cost"]
C_RE_REND = NL_ACTIONS["re_render"]["cost"]

F8_ORGS = ["cloudflare", "github", "mozilla", "redhat", "wikimedia"]  # cloudflare = no-op


def rec(org: str, fault: str | None, raw_root: Path) -> Path:
    with inject(fault):
        return record_run(run_pipeline(organization=org), raw_root=raw_root)


def _slot_index(run_dir: Path, filename: str) -> dict[str, list[str]]:
    """state_slot_key -> sorted normalized (run_id-stripped) record contents."""
    out: dict[str, list[str]] = {}
    path = run_dir / filename
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        k = row.get("state_slot_key")
        out.setdefault(k, []).append(
            json.dumps({x: v for x, v in row.items() if x not in _VOLATILE},
                       sort_keys=True))
    return {k: sorted(v) for k, v in out.items()}


def _changed_slots(clean: Path, faulted: Path, filename: str) -> set[str]:
    a = _slot_index(clean, filename)
    b = _slot_index(faulted, filename)
    return {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}


def _id_to_slot(run_dir: Path, filename: str, id_field: str) -> dict[str, str]:
    out: dict[str, str] = {}
    path = run_dir / filename
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        if ln.strip():
            row = json.loads(ln)
            out[row[id_field]] = row.get("state_slot_key")
    return out


def bundle_fingerprint(run_dir: Path) -> str:
    h = hashlib.sha256()
    for rtype, (fn, idf) in sorted(_RECORD_FILES.items()):
        idx = _index(run_dir, fn, idf)
        h.update(f"@{rtype}:{len(idx)}".encode())
        for i in sorted(idx):
            h.update(i.encode()); h.update(b"="); h.update(idx[i].encode())
    return h.hexdigest()


def f8_cell(org: str, raw_root: Path) -> dict:
    clean = rec(org, None, raw_root)
    faulted = rec(org, "F8", raw_root)
    dfx = contamination_closure(clean, faulted)

    # (i) slot-grain forced cover
    dec_slots = _changed_slots(clean, faulted, "decision.jsonl")
    rep_slots = _changed_slots(clean, faulted, "report_item.jsonl")

    # (ii) independent id-grain set from contamination_closure, mapped to slots
    id2slot = {}
    for side in (clean, faulted):
        id2slot.update(_id_to_slot(side, "decision.jsonl", "decision_id"))
        id2slot.update(_id_to_slot(side, "report_item.jsonl", "report_item_id"))
    dec_slots_id, rep_slots_id = set(), set()
    for cid in dfx["contaminated_ids"]:
        typ, bare = cid.split(":", 1)
        if typ == "decision":
            dec_slots_id.add(id2slot.get(bare))
        elif typ == "report_item":
            rep_slots_id.add(id2slot.get(bare))

    n_re_eval = len(dec_slots)
    n_re_rend = len(rep_slots)
    c_star = n_re_eval * C_RE_EVAL + n_re_rend * C_RE_REND
    # GATE
    g_forced = True  # forced cover computed by construction
    g_crosscheck = (dec_slots == dec_slots_id and rep_slots == rep_slots_id)
    # A1: contamination confined to non-leaf types => S* restores clean
    contaminated_types = set(dfx["contaminated_by_type"])
    g_a1 = contaminated_types <= {"decision", "report_item"}
    no_op = dfx["n_contaminated"] == 0
    gate_pass = g_forced and g_crosscheck and (g_a1 or no_op)

    out_fp = hashlib.sha256(json.dumps({
        "dec_slots": sorted(dec_slots), "rep_slots": sorted(rep_slots),
        "C_star": c_star,
    }, sort_keys=True).encode()).hexdigest()

    return {
        "fault": "F8", "org": org, "track": "live",
        "inject_node": getattr(FAULT_REGISTRY["F8"], "inject_node", None),
        "no_op": no_op,
        "D_f_contaminated": dfx["n_contaminated"],
        "D_f_repairable": dfx["n_repairable"],
        "contaminated_by_type": dfx["contaminated_by_type"],
        "n_re_evaluate": n_re_eval, "n_re_render": n_re_rend,
        "C_star": c_star, "H_star": 0,
        "gate": {
            "forced_cover_computed": g_forced,
            "slot_vs_id_crosscheck": g_crosscheck,
            "a1_confined_to_non_leaf": g_a1,
            "PASS": gate_pass,
        },
        "fingerprint": {
            "clean_bundle": bundle_fingerprint(clean),
            "faulted_bundle": bundle_fingerprint(faulted),
            "output": out_fp,
        },
        "clean_dir": str(clean), "faulted_dir": str(faulted),
    }


def f7_cell(raw_root: Path) -> dict:
    fixture = _build_base(raw_root / "f7_fixture" / "base")
    rc = record_closure(fixture, "F7")
    cbt = rc["contaminated_by_type"]
    n_re_eval = cbt.get("decision", 0)
    n_re_rend = cbt.get("report_item", 0)
    c_star = n_re_eval * C_RE_EVAL + n_re_rend * C_RE_REND
    contaminated_types = set(cbt)
    g_a1 = contaminated_types <= {"decision", "report_item"}
    gate_pass = g_a1
    out_fp = hashlib.sha256(json.dumps({
        "closure_ids": sorted(rc["contaminated_ids"]), "C_star": c_star,
    }, sort_keys=True).encode()).hexdigest()
    return {
        "fault": "F7", "org": None, "track": "record",
        "inject_node": rc["inject_node"], "no_op": rc["n_contaminated"] == 0,
        "D_f_contaminated": rc["n_contaminated"],
        "D_f_repairable": rc["n_repairable"],
        "contaminated_by_type": cbt,
        "n_re_evaluate": n_re_eval, "n_re_render": n_re_rend,
        "C_star": c_star, "H_star": 0,
        "gate": {
            "forced_cover_computed": True,
            "slot_vs_id_crosscheck": None,   # structural closure, single fixture
            "a1_confined_to_non_leaf": g_a1,
            "PASS": gate_pass,
        },
        "fingerprint": {
            "fixture_bundle": bundle_fingerprint(fixture),
            "output": out_fp,
        },
        "fixture_dir": str(fixture),
    }


_REPRO_FIELDS = ("D_f_contaminated", "n_re_evaluate", "n_re_render", "C_star")


def main() -> int:
    raw_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/arcs_nl_bundles")
    raw_root.mkdir(parents=True, exist_ok=True)
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("non_leaf_oracle_v1.json")

    prior = None
    if out_path.exists():
        try:
            pj = json.loads(out_path.read_text())
            prior = {(c["fault"], c["org"]): c for c in pj.get("cells", [])}
        except Exception:  # noqa: BLE001
            prior = None

    cells = []
    for org in F8_ORGS:
        print(f"== F8/{org} ==", flush=True)
        c = f8_cell(org, raw_root)
        cells.append(c)
        print(f"   D(f)={c['D_f_contaminated']} type={c['contaminated_by_type']} "
              f"re_eval={c['n_re_evaluate']} re_render={c['n_re_render']} "
              f"C*={c['C_star']} no_op={c['no_op']} "
              f"GATE={'PASS' if c['gate']['PASS'] else 'FAIL'}", flush=True)
    print("== F7 (record fixture) ==", flush=True)
    c7 = f7_cell(raw_root)
    cells.append(c7)
    print(f"   D(f)={c7['D_f_contaminated']} type={c7['contaminated_by_type']} "
          f"re_eval={c7['n_re_evaluate']} re_render={c7['n_re_render']} "
          f"C*={c7['C_star']} GATE={'PASS' if c7['gate']['PASS'] else 'FAIL'}", flush=True)

    repro: dict = {"prior_artifact_present": prior is not None}
    if prior is not None:
        mism = []
        for c in cells:
            p = prior.get((c["fault"], c["org"]))
            if p is None:
                mism.append({"cell": f"{c['fault']}/{c['org']}", "reason": "absent in prior"})
                continue
            diffs = {k: [p.get(k), c[k]] for k in _REPRO_FIELDS if p.get(k) != c[k]}
            if diffs:
                mism.append({"cell": f"{c['fault']}/{c['org']}", "field_diffs": diffs})
        repro.update({"compared_fields": list(_REPRO_FIELDS), "n_cells_compared": len(cells),
                      "n_mismatch": len(mism), "all_match": not mism, "mismatches": mism})
    else:
        repro["note"] = "no prior artifact; establishes baseline"

    result = {
        "phase": "2.non_leaf_oracle",
        "spec": "gate-logic faults F8 (live) + F7 (record): forced non-leaf cover S*/C* + GATE",
        "freeze": "non_leaf_oracle_FROZEN.md (2026-09-17)",
        "nl_actions": NL_ACTIONS, "nl_spec_hash16": NL_SPEC_HASH,
        "f8_orgs": F8_ORGS,
        "raw_root": str(raw_root),
        "n_cells": len(cells),
        "n_gate_pass": sum(c["gate"]["PASS"] for c in cells),
        "all_gate_pass": all(c["gate"]["PASS"] for c in cells),
        "reproducibility": repro,
        "aggregate_fingerprint": hashlib.sha256(
            "".join(c["fingerprint"]["output"] for c in cells).encode()).hexdigest(),
        "cells": cells,
    }
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nwrote {out_path}  cells={len(cells)}  gate_pass={result['n_gate_pass']}/{len(cells)}  "
          f"repro={'MATCH' if repro.get('all_match') else ('BASELINE' if prior is None else 'MISMATCH')}")
    ok = result["all_gate_pass"] and (prior is None or repro.get("all_match", True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
