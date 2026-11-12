"""experiments/exp_arcs/e3_detector_fp — detector detection-quality (⑥).

The reformed primary contribution is reference-free localization, but detector
quality was only ever reported as END-TO-END repair success (which conflates
detection + repair). This closes the gap a reviewer will hit first: the
reference-free detectors (self-diagnosis, peer-anomaly, discovery-manifest) FLAG
hosts/invocations — what is their FALSE-ALARM rate on a FAULT-FREE run, and their
precision/recall on faulted runs, as detectors in their own right?

Method (offline, reuses the recorded E3 bundles; jsonl reads only, no DAG builds):
  * CLEAN runs (20, one per org): the run is fault-free, so EVERY flag is a false
    positive. Per detector we count flagged hosts / (host,family) invocations and
    divide by the run's universe → false-alarm rate.
  * FAULTED runs (180 cells): true host = inject_node's host. Per cell, host-grain
    precision = |flag∩true|/|flag|, recall = |flag∩true|/|true|; aggregate to F1.

    poetry run python -m experiments.exp_arcs.e3_detector_fp
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
from collections import OrderedDict, Counter

from experiments.exp_arcs.detector_eval import (
    detector_localize, peer_anomaly_hosts, peer_anomaly_invocations,
    manifest_dropped_hosts,
)

D = Path("experiments/exp_arcs")

def _universe(run_dir: Path):
    """Hosts with evidence + distinct (host,family) invocations in the run."""
    hosts, invs = set(), set()
    p = run_dir / "evidence_event.jsonl"
    for ln in p.read_text().splitlines():
        if not ln.strip():
            continue
        e = json.loads(ln)
        h = e.get("target"); f = e.get("source_family") or e.get("family")
        if h:
            hosts.add(h)
            if f:
                invs.add((h, f))
    return hosts, invs

def _combined_hosts(run_dir: Path):
    return (detector_localize(run_dir)["flagged_hosts"]
            | peer_anomaly_hosts(run_dir)
            | manifest_dropped_hosts(run_dir))

def main() -> int:
    grid = json.loads((D/"e3_grid_v1.json").read_text())["cells"]
    # unique clean dir per org
    clean_by_org = OrderedDict()
    for c in grid:
        clean_by_org.setdefault(c["org"], c["clean_dir"])

    # ---- CLEAN-run false alarms (every flag is a FP) ----
    clean_fp = {"self": [], "peer_host": [], "manifest": [], "combined_host": [], "peer_inv": []}
    clean_rate = {"combined_host": [], "peer_inv": []}
    for org, cd in clean_by_org.items():
        cdp = Path(cd)
        hosts, invs = _universe(cdp)
        self_f = detector_localize(cdp)["flagged_hosts"]
        peer_f = peer_anomaly_hosts(cdp)
        man_f = manifest_dropped_hosts(cdp)
        comb = self_f | peer_f | man_f
        peer_i = peer_anomaly_invocations(cdp)
        clean_fp["self"].append(len(self_f))
        clean_fp["peer_host"].append(len(peer_f))
        clean_fp["manifest"].append(len(man_f))
        clean_fp["combined_host"].append(len(comb))
        clean_fp["peer_inv"].append(len(peer_i))
        clean_rate["combined_host"].append(len(comb) / max(1, len(hosts)))
        clean_rate["peer_inv"].append(len(peer_i) / max(1, len(invs)))

    # ---- FAULTED-run detection precision/recall (host grain) ----
    precs, recs = [], []
    per_type = {}
    for c in grid:
        inj = c.get("inject_node") or []
        if len(inj) < 2:
            continue
        # inj[0]=="target" → inj[1] is a host string; "targets" → inj[1] is a host list
        true_hosts = set(inj[1]) if isinstance(inj[1], list) else {inj[1]}
        if not true_hosts:
            continue
        fl = _combined_hosts(Path(c["faulted_dir"]))
        tp = len(true_hosts & fl)
        prec = tp / len(fl) if fl else 0.0
        rec = tp / len(true_hosts)
        precs.append(prec); recs.append(rec)
        per_type.setdefault(c["cell"], {"prec": [], "rec": []})
        per_type[c["cell"]]["prec"].append(prec)
        per_type[c["cell"]]["rec"].append(rec)

    P = st.mean(precs); R = st.mean(recs)
    F1 = 2*P*R/(P+R) if (P+R) else 0.0

    def sm(xs): return round(st.mean(xs), 3)
    result = {
        "phase": "7/⑥ — detector detection quality (clean-run false alarms + faulted P/R/F1)",
        "n_clean_runs": len(clean_by_org), "n_faulted_cells": len(precs),
        "clean_run_false_alarms": {
            "note": "fault-free runs → every flag is a false positive; mean count per run over 20 orgs",
            "self_diagnosis_hosts": sm(clean_fp["self"]),
            "peer_anomaly_hosts": sm(clean_fp["peer_host"]),
            "manifest_hosts": sm(clean_fp["manifest"]),
            "combined_hosts": sm(clean_fp["combined_host"]),
            "peer_anomaly_invocations": sm(clean_fp["peer_inv"]),
            "combined_host_false_alarm_rate": sm(clean_rate["combined_host"]),
            "peer_invocation_false_alarm_rate": sm(clean_rate["peer_inv"]),
        },
        "faulted_detection_host_grain": {
            "precision": round(P, 3), "recall": round(R, 3), "f1": round(F1, 3),
            "note": "combined detector (self|peer|manifest); true host = inject_node host",
        },
        "by_fault_type": {t: {"precision": sm(v["prec"]), "recall": sm(v["rec"])}
                          for t, v in sorted(per_type.items())},
    }
    out = D/"e3_detector_fp_v1.json"
    out.write_text(json.dumps(result, indent=2))

    fa = result["clean_run_false_alarms"]
    print(f"⑥ DETECTOR DETECTION QUALITY (n_clean={len(clean_by_org)}, n_faulted={len(precs)})\n")
    print("CLEAN-run false alarms (mean per fault-free run; any flag = FP):")
    print(f"  self-diagnosis   hosts : {fa['self_diagnosis_hosts']}")
    print(f"  peer-anomaly     hosts : {fa['peer_anomaly_hosts']}  (rate {fa['peer_invocation_false_alarm_rate']} at invocation grain)")
    print(f"  manifest         hosts : {fa['manifest_hosts']}  (expect ~0: no host dropped on a clean run)")
    print(f"  COMBINED         hosts : {fa['combined_hosts']}  (false-alarm rate {fa['combined_host_false_alarm_rate']} of all hosts)")
    fd = result["faulted_detection_host_grain"]
    print(f"\nFAULTED-run detection (host grain, combined): precision {fd['precision']}  recall {fd['recall']}  F1 {fd['f1']}")
    print(f"\nwrote {out}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
