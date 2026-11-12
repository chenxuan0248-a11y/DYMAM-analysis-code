"""experiments/exp_arcs/run_e3 — Phase 6 E3 Stage A: record the 20-org repair grid.

Records the full E3 repair grid — 20 orgs × 9 local fault types — FRESH under one
Qdrant/cache state (the E3 recording-consistency discipline in PHASE6_DESIGN.md:
all cells new-recorded, pilot bundles NOT reused). Each cell is gate-checked ILP
oracle (C\*, D(f)) via `local_oracle.make_cell`, producing the SAME schema as
`local_oracle_all_v1.json`, so the existing scorers run unchanged on the output:

    poetry run python -m experiments.exp_arcs.arcs_pilot   e3_grid_v1.json e3_arcs_v1.json
    poetry run python -m experiments.exp_arcs.pareto        e3_grid_v1.json e3_pareto_v1.json
    poetry run python -m experiments.exp_arcs.phase5_pilot  e3_grid_v1.json e3_oracle_stats_v1.json

Fault grid per org (same as the frozen pilot grid, `arcs_grid_v2_STATUS.md`):
  single-invocation : F4@{dns, active_observation, certificate, registration}, F10-local
  multi-invocation  : F4-host, F4-2host-dns, F10-2host, F10-3host

F2 (global negative control) and F8 (non-leaf) are separate artifacts per the domain
freeze — not in this repair grid. Record-layer F1/F3/F5/F6/F7/F9 are the detection
sideline (excluded).

RESUMABLE: writes the checkpoint after every org; on restart, orgs already present
in the output are skipped (bundles are re-recorded only for unfinished orgs). A
single failed cell is recorded as an error and does not abort the run.

    poetry run python -m experiments.exp_arcs.run_e3 [raw_root] [out.json]
"""
from __future__ import annotations

import hashlib
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from core.graph import run_pipeline
from experiments.exp_arcs.dag import build_dag, scope_cost
from experiments.exp_arcs.interceptors_local import inject_local
from experiments.exp_arcs.local_oracle import make_cell, pick_targets
from experiments.exp_arcs.recorder import record_run

# Plan's 20 main-experiment orgs (ARCS_MANIFEST). Filtered to those with a GT file.
ORGS = ["akamai", "apache", "canonical", "cloudflare", "eff", "fastly", "github",
        "gitlab", "ibm", "linkedin", "microsoft", "mozilla", "netflix", "oracle",
        "reddit", "redhat", "salesforce", "spotify", "twitch", "wikimedia"]

RAW_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/arcs_e3_bundles")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("experiments/exp_arcs/e3_grid_v1.json")
GT_DIR = Path("data/gt")


def _pick_multi(clean: Path) -> dict:
    """Multi-invocation targets (mirrors local_oracle_multi.pick)."""
    evs = [json.loads(l) for l in (clean / "evidence_event.jsonl").read_text().splitlines() if l.strip()]
    total = Counter(e["target"] for e in evs)
    dns = Counter(e["target"] for e in evs if e.get("source_family") == "dns")
    fams: dict[str, set] = defaultdict(set)
    for e in evs:
        fams[e["target"]].add(e.get("source_family"))
    multi_fam = {h for h, fs in fams.items() if len(fs) >= 2}
    f4host = max(sorted(multi_fam), key=lambda h: (len(fams[h]), total[h])) if multi_fam else None
    top_dns2 = [h for h, _ in sorted(dns.items(), key=lambda kv: (-kv[1], kv[0]))[:2]]
    top_ev = [h for h, _ in sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))]
    return {"f4host": f4host, "dns2": top_dns2, "ev2": top_ev[:2], "ev3": top_ev[:3]}


def _org_specs(clean: Path):
    """(cell_name, inject_local kwargs, inject_node) for one org's 9 cells."""
    tg = pick_targets(clean)
    specs = []
    for fam in ["dns", "active_observation", "certificate", "registration"]:
        if tg.get(fam):
            specs.append((f"F4@{fam}", dict(fault_id="F4", target=tg[fam], family=fam),
                          ["target", tg[fam], fam]))
    if tg.get("_f10"):
        specs.append(("F10-local", dict(fault_id="F10", target=tg["_f10"]),
                      ["target", tg["_f10"]]))
    p = _pick_multi(clean)
    if p["f4host"]:
        specs.append(("F4-host", dict(fault_id="F4", targets=[p["f4host"]]),
                      ["targets", [p["f4host"]]]))
    if len(p["dns2"]) == 2:
        specs.append(("F4-2host-dns", dict(fault_id="F4", targets=p["dns2"], family="dns"),
                      ["targets", sorted(p["dns2"]), "dns"]))
    if len(p["ev2"]) == 2:
        specs.append(("F10-2host", dict(fault_id="F10", targets=p["ev2"]),
                      ["targets", sorted(p["ev2"])]))
    if len(p["ev3"]) == 3:
        specs.append(("F10-3host", dict(fault_id="F10", targets=p["ev3"]),
                      ["targets", sorted(p["ev3"])]))
    return specs


def _load_checkpoint() -> tuple[list, dict]:
    if OUT.exists():
        prior = json.loads(OUT.read_text())
        return prior.get("cells", []), prior.get("global_recollect_cost_per_org", {})
    return [], {}


def _write(cells: list, gcost: dict, orgs_done: list, orgs_avail: list):
    result = {
        "phase": "6.E3.grid",
        "spec": "20 orgs x {F4@{dns,ao,cert,reg}, F10-local, F4-host, F4-2host-dns, "
                "F10-2host, F10-3host} local ILP oracle (fresh-recorded, one state)",
        "orgs_target": ORGS, "orgs_available": orgs_avail, "orgs_recorded": orgs_done,
        "global_recollect_cost_per_org": gcost,
        "n_cells": len(cells),
        "n_gate_pass": sum(1 for c in cells if c.get("gate", {}).get("PASS")),
        "all_gate_pass": all(c.get("gate", {}).get("PASS") for c in cells) if cells else False,
        "aggregate_fingerprint": hashlib.sha256(
            "".join(c.get("fingerprint", {}).get("output", "") for c in cells).encode()).hexdigest(),
        "cells": cells,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))


def main() -> int:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    orgs_avail = [o for o in ORGS if (GT_DIR / f"{o}_gt.json").exists()]
    skipped = [o for o in ORGS if o not in orgs_avail]
    if skipped:
        print(f"[skip] no GT file: {skipped}", flush=True)

    cells, gcost = _load_checkpoint()
    done = {c["org"] for c in cells}
    print(f"resume: {len(done)} orgs already recorded ({sorted(done)})", flush=True)

    for org in orgs_avail:
        if org in done:
            continue
        print(f"\n== {org}: clean ==", flush=True)
        try:
            clean = record_run(run_pipeline(organization=org), raw_root=RAW_ROOT)
        except Exception as e:
            print(f"   [FAIL clean] {org}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        cdag = build_dag(clean)
        g = scope_cost(cdag, [n for n, v in cdag.nodes.items() if v.get("repairable")])["cost"]
        gcost[org] = g
        print(f"   global_recollect_cost={g}", flush=True)

        org_cells = []
        for name, kw, inode in _org_specs(clean):
            try:
                with inject_local(**kw):
                    faulted = record_run(run_pipeline(organization=org), raw_root=RAW_ROOT)
                org_cells.append(make_cell(clean, cdag, g, org, name, faulted, inode))
            except Exception as e:
                print(f"   [FAIL cell] {org}/{name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                traceback.print_exc()
        cells.extend(org_cells)
        done.add(org)
        _write(cells, gcost, sorted(done), orgs_avail)   # checkpoint after each org
        print(f"   [checkpoint] {org}: +{len(org_cells)} cells, total {len(cells)}", flush=True)

    _write(cells, gcost, sorted(done), orgs_avail)
    npass = sum(1 for c in cells if c.get("gate", {}).get("PASS"))
    print(f"\nwrote {OUT}  orgs={len(done)}/{len(orgs_avail)}  cells={len(cells)}  "
          f"gate_pass={npass}/{len(cells)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
