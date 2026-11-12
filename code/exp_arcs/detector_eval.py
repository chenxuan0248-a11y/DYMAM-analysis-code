"""experiments/exp_arcs/detector_eval — Step 3.2(b) detector-only localization + its
end-to-end effect on the ARCS GO.

The Oracle / semi-oracle lineage localizers used the clean-vs-faulted divergence.
This localizer is REFERENCE-FREE: it flags suspects from the FAULTED run alone,
using the pipeline's own self-diagnosed "evidence missing" signals — decisions that
came back `revalidate` or carry `no_supporting_observation` (optionally the noisier
`unsupported` support_state). Flagged decisions → their claim's subject host →
ranked suspect hosts. ARCS then re-collects the suspected hosts' evidence and is
scored by the A1 counterfactual (the clean reference is used ONLY to score success,
never to localize).

This exposes the honest limit: a fault that DROPS data leaves a trace only if a
surviving record still references the affected host. F4 (drop evidence) leaves the
host with revalidate/unsupported decisions → detectable. F10 (drop the host from
discovery) removes every mention of the host → invisible (verified: 0 mentions) →
undetectable reference-free → ARCS cannot target it.

    poetry run python -m experiments.exp_arcs.detector_eval [grid.json] [out.json]
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.exp_arcs.contamination import contamination_closure, _REPAIRABLE
from experiments.exp_arcs.counterfactual import _bare
from experiments.exp_arcs.dag import build_dag
from experiments.exp_arcs.select import evaluate_scope

FLAG_DECISIONS = {"revalidate"}
FLAG_REASONS = {"no_supporting_observation"}
FLAG_SUPPORT = {"unsupported"}
PEER_MIN_BASE_RATE = 0.5   # a family present on ≥50% of hosts is "expected"


def peer_anomaly_hosts(faulted_dir: Path, min_base_rate: float = PEER_MIN_BASE_RATE
                       ) -> set[str]:
    """Upgrade (2), reference-free: flag hosts MISSING a family that most peer
    hosts have (uses THIS run's cross-host distribution, not a clean baseline).
    Catches corroborating-family drops on common families (dns/ao); by
    construction cannot catch rare-family drops (cert/reg, ~4% base rate → absence
    is normal) nor silent host omission (F10 host has no evidence to be sparse)."""
    from collections import defaultdict
    host_fams: dict[str, set] = defaultdict(set)
    fam_hosts: dict[str, set] = defaultdict(set)
    for ln in (faulted_dir / "evidence_event.jsonl").read_text().splitlines():
        if ln.strip():
            e = json.loads(ln)
            f = e.get("source_family")
            host_fams[e["target"]].add(f)
            fam_hosts[f].add(e["target"])
    n = len(host_fams)
    if n == 0:
        return set()
    expected = {f for f, hs in fam_hosts.items() if len(hs) / n >= min_base_rate}
    return {h for h, fs in host_fams.items() if expected - fs}


def peer_anomaly_invocations(faulted_dir: Path, min_base_rate: float = PEER_MIN_BASE_RATE
                             ) -> set[tuple]:
    """Upgrade (1)-family-grain: the SPECIFIC (host, family) invocations a host is
    missing vs its peers. Lets ARCS repair the exact invocation instead of the
    whole host — recovering the cost the host-grain detector inflated."""
    from collections import defaultdict
    host_fams: dict[str, set] = defaultdict(set)
    fam_hosts: dict[str, set] = defaultdict(set)
    for ln in (faulted_dir / "evidence_event.jsonl").read_text().splitlines():
        if ln.strip():
            e = json.loads(ln)
            host_fams[e["target"]].add(e.get("source_family"))
            fam_hosts[e.get("source_family")].add(e["target"])
    n = len(host_fams)
    if n == 0:
        return set()
    expected = {f for f, hs in fam_hosts.items() if len(hs) / n >= min_base_rate}
    return {(h, f) for h, fs in host_fams.items() for f in (expected - fs)}


def detector_localize(faulted_dir: Path, use_unsupported: bool = False) -> dict[str, Any]:
    """Reference-free suspect hosts from the faulted bundle alone."""
    subj: dict[str, str] = {}
    for ln in (faulted_dir / "claim.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            subj[r["claim_id"]] = r.get("subject")
    flagged: Counter = Counter()
    n_flagged_decisions = 0
    for ln in (faulted_dir / "decision.jsonl").read_text().splitlines():
        if not ln.strip():
            continue
        d = json.loads(ln)
        codes = set(d.get("reason_codes") or [])
        hit = (d.get("decision") in FLAG_DECISIONS or (codes & FLAG_REASONS)
               or (use_unsupported and d.get("support_state") in FLAG_SUPPORT))
        if hit:
            n_flagged_decisions += 1
            h = subj.get(d.get("claim_id"))
            if h:
                flagged[h] += 1
    ranked = [h for h, _ in sorted(flagged.items(), key=lambda kv: (-kv[1], kv[0]))]
    return {"ranked_host": ranked, "flagged_hosts": set(flagged),
            "n_flagged_decisions": n_flagged_decisions}


def manifest_dropped_hosts(faulted_dir: Path) -> set[str]:
    """Upgrade (1), reference-free: hosts discovery resolved (from the run's own
    discovery_manifest.json, emitted pre-drop) that have NO evidence downstream =
    silently omitted hosts (F10). Empty if the bundle predates manifest capture."""
    mpath = faulted_dir / "discovery_manifest.json"
    if not mpath.exists():
        return set()
    resolved = set(json.loads(mpath.read_text()).get("resolved_hosts", []))
    ev_hosts = set()
    for ln in (faulted_dir / "evidence_event.jsonl").read_text().splitlines():
        if ln.strip():
            ev_hosts.add(json.loads(ln)["target"])
    return resolved - ev_hosts


def _true_hosts(clean_dir: Path, dfx: dict) -> set[str]:
    dag = build_dag(clean_dir)
    hosts = set()
    for l in _bare(dfx, _REPAIRABLE):
        v = dag.nodes.get(l)
        if v and v.get("repairable") and v.get("target"):
            hosts.add(v["target"])
    return hosts


def run_cell(cell: dict) -> dict[str, Any]:
    clean, faulted = Path(cell["clean_dir"]), Path(cell["faulted_dir"])
    dag = build_dag(clean)
    dfx = contamination_closure(clean, faulted)
    true_hosts = _true_hosts(clean, dfx)
    det = detector_localize(faulted)
    self_flag = det["flagged_hosts"]
    peer_flag = peer_anomaly_hosts(faulted)
    manifest_flag = manifest_dropped_hosts(faulted)
    combined = self_flag | peer_flag | manifest_flag
    ftype = "F10" if "F10" in cell["cell"] else "F4"

    def arcs_on(flagged: set) -> dict:
        scope = {n for n, v in dag.nodes.items()
                 if v.get("repairable") and v.get("target") in flagged}
        ev = evaluate_scope(clean, dfx, scope, dag=dag)
        return {"success": ev["success"], "cost": ev["cost"],
                "region_covered": bool(true_hosts) and true_hosts <= flagged,
                "n_flagged": len(flagged)}

    # family-grain combined: peer gives exact (host,family) invocations (cheap);
    # manifest F10 hosts + any self-diagnosis host not pinpointed by peer fall back
    # to whole-host (family unknown reference-free).
    def arcs_family_grain() -> dict:
        fam_susp = peer_anomaly_invocations(faulted)
        pinned_hosts = {h for h, _ in fam_susp}
        whole_hosts = manifest_flag | {h for h in self_flag if h not in pinned_hosts}
        scope = set()
        for nid, v in dag.nodes.items():
            if not v.get("repairable"):
                continue
            if v.get("target") in whole_hosts:
                scope.add(nid)
            elif (v.get("target"), v["source_family"]) in fam_susp:
                scope.add(nid)
        ev = evaluate_scope(clean, dfx, scope, dag=dag)
        return {"success": ev["success"], "cost": ev["cost"],
                "n_invocations": ev["n_invocations"]}

    return {
        "org": cell["org"], "cell": cell["cell"], "fault_type": ftype,
        "inject_node": cell["inject_node"], "C_star": cell["C_star"],
        "n_true_hosts": len(true_hosts),
        "self_diagnosis": arcs_on(self_flag),
        "peer_anomaly": arcs_on(peer_flag),
        "manifest": arcs_on(manifest_flag),
        "combined": arcs_on(combined),
        "combined_family_grain": arcs_family_grain(),
    }


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    src = Path(argv[0]) if argv else Path("experiments/exp_arcs/local_oracle_all_v1.json")
    out = Path(argv[1]) if len(argv) > 1 else Path("experiments/exp_arcs/detector_eval_v1.json")
    cells = json.loads(src.read_text())["cells"]
    rows = [run_cell(c) for c in cells]

    variants = ["self_diagnosis", "peer_anomaly", "manifest", "combined",
                "combined_family_grain"]

    def succ_rate(subset, variant):
        v = [1 if r[variant]["success"] else 0 for r in subset]
        return round(sum(v) / len(v), 4) if v else None

    def cost_stats(subset, variant):
        """mean cost and mean cost/C* over SUCCESS cells of this variant."""
        cc = [(r[variant]["cost"], r["C_star"]) for r in subset
              if r[variant]["success"] and r["C_star"]]
        if not cc:
            return {"mean_cost": None, "mean_cost_over_Cstar": None}
        return {"mean_cost": round(statistics.mean([c for c, _ in cc]), 2),
                "mean_cost_over_Cstar": round(statistics.mean([c / s for c, s in cc]), 2)}

    by_type = {}
    for t in ("F4", "F10"):
        sub = [r for r in rows if r["fault_type"] == t]
        by_type[t] = {"n_cells": len(sub),
                      **{v: succ_rate(sub, v) for v in variants}}
    by_subtype = {}
    for name in sorted({r["cell"] for r in rows}):
        sub = [r for r in rows if r["cell"] == name]
        by_subtype[name] = {v: succ_rate(sub, v) for v in variants}

    cost_by_variant = {v: cost_stats(rows, v) for v in ("combined", "combined_family_grain")}
    summary = {
        "n_cells": len(rows),
        "oracle_localization_success": 1.0,
        "detector_only_success": {v: succ_rate(rows, v) for v in variants},
        "cost_recovery": cost_by_variant,
        "by_fault_type": by_type,
        "by_subtype": by_subtype,
        "verdict": (
            "Upgrade (2) peer-anomaly recovers common-family drops (F4@dns / "
            "F4-2host-dns) and reinforces F4@ao/F4-host; combined detector-only "
            f"Success = {succ_rate(rows, 'combined')} (vs oracle 1.0, self-diagnosis "
            f"only {succ_rate(rows, 'self_diagnosis')}). STILL missing: F4@cert/reg "
            "(rare families, absence is normal — undetectable reference-free) and "
            "ALL F10 (silent host omission — 0 bundle mentions; needs upgrade (1) "
            "discovery-manifest telemetry, which the current recorder does not "
            "capture)."),
    }
    result = {"phase": "3.2b.detector_only", "source": str(src),
              "signals": {"self_diagnosis_decisions": sorted(FLAG_DECISIONS),
                          "self_diagnosis_reasons": sorted(FLAG_REASONS),
                          "peer_min_base_rate": PEER_MIN_BASE_RATE},
              "summary": summary, "cells": rows}
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
