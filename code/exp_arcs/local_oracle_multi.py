"""experiments/exp_arcs/local_oracle_multi — MULTI-INVOCATION local fault grid.

Extends the single-invocation local grid (`local_oracle_v1.json`) with faults whose
leaf-repairable impact spans ≥2 (host,family) INVOCATIONS, so a single-point repair
structurally under-covers and ARCS's adaptive scope is exercised across MORE than
the one F10 fault type. All go through `solve_oracle` (GATE-checked). Pure replay.

Fault types (per org, deterministic targets):
  - F4-host      : drop ALL evidence of the host with the most DISTINCT families
                   (≥2) → multi-family, multi-invocation (normalize layer).
  - F4-2host-dns : drop dns evidence for the top-2 dns hosts → 2 invocations
                   across 2 hosts (a multi-POINT fault).
  - F10-2host    : premature-terminate the top-2 hosts by evidence → multi-point.
  - F10-3host    : premature-terminate the top-3 hosts.

    poetry run python -m experiments.exp_arcs.local_oracle_multi [raw_root] [out.json]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from core.graph import run_pipeline
from experiments.exp_arcs.recorder import record_run
from experiments.exp_arcs.interceptors_local import inject_local
from experiments.exp_arcs.dag import build_dag, scope_cost
from experiments.exp_arcs.local_oracle import make_cell

ORGS = ["cloudflare", "github", "mozilla", "redhat", "wikimedia"]

RAW_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/arcs_multi_bundles")
RAW_ROOT.mkdir(parents=True, exist_ok=True)
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "experiments/exp_arcs/local_oracle_multi_v1.json")


def pick(clean: Path) -> dict:
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


def main() -> int:
    cells = []
    global_cost: dict[str, int] = {}
    picks: dict[str, dict] = {}
    for org in ORGS:
        print(f"== {org}: clean ==", flush=True)
        clean = record_run(run_pipeline(organization=org), raw_root=RAW_ROOT)
        cdag = build_dag(clean)
        gcost = scope_cost(cdag, [n for n, v in cdag.nodes.items() if v.get("repairable")])["cost"]
        global_cost[org] = gcost
        p = pick(clean); picks[org] = p
        print(f"   gcost={gcost} picks={json.dumps(p)}", flush=True)

        specs = []
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

        for name, kw, inode in specs:
            with inject_local(**kw):
                faulted = record_run(run_pipeline(organization=org), raw_root=RAW_ROOT)
            cells.append(make_cell(clean, cdag, gcost, org, name, faulted, inode))

    result = {
        "phase": "2.local.multi",
        "spec": "5 orgs x {F4-host, F4-2host-dns, F10-2host, F10-3host} multi-invocation local ILP oracle",
        "orgs": ORGS,
        "global_recollect_cost_per_org": global_cost,
        "picks_by_org": picks,
        "n_cells": len(cells),
        "n_gate_pass": sum(c["gate"]["PASS"] for c in cells),
        "all_gate_pass": all(c["gate"]["PASS"] for c in cells),
        "cells": cells,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nwrote {OUT}  cells={len(cells)}  gate_pass={result['n_gate_pass']}/{len(cells)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
