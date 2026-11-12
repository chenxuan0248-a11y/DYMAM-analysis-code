"""Reviewer item ② (EMSE) — representativeness of the injected faults.

An EMSE reviewer asks: are the F4 (evidence-family omission) and F10 (host omission)
faults we inject representative of what happens naturally, or are they artificial? We
answer offline on the 20 CLEAN (un-injected) runs by measuring how often the SAME
STRUCTURAL PATTERN occurs without any injection:

  * F4-analog (family omission): a host naturally missing an evidence family that
    >= 50% of its peers have -- exactly the structure F4 injects. We report the rate
    per clean run and the per-family breakdown, mapped to the families we inject
    (dns / active_observation / certificate / registration).
  * F10-analog (host omission): a host the discovery manifest resolved but with no
    downstream evidence -- the structure F10 injects.

Honesty: a naturally family-sparse host on a clean run is a pattern of the same FORM
as F4, not necessarily a confirmed bug (we have no per-host GT to certify it). We
claim structural representativeness, not that every natural instance is a fault.
Reuses the peer / manifest family distributions; jsonl reads only, no DAG builds.

    poetry run python -m experiments.exp_arcs.natural_fault_prevalence
"""
from __future__ import annotations
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from experiments.exp_arcs.detector_eval import manifest_dropped_hosts

D = Path("experiments/exp_arcs")
OUT = D / "natural_fault_prevalence_v1.json"
BASE = 0.5  # "expected" family = present on >= 50% of the run's hosts
INJECTED_FAMILIES = ["dns", "active_observation", "certificate", "registration"]


def _fam_dist(run_dir: Path):
    host_fams: dict[str, set] = defaultdict(set)
    fam_hosts: dict[str, set] = defaultdict(set)
    for ln in (run_dir / "evidence_event.jsonl").read_text().splitlines():
        if ln.strip():
            e = json.loads(ln)
            host_fams[e["target"]].add(e.get("source_family"))
            fam_hosts[e.get("source_family")].add(e["target"])
    return host_fams, fam_hosts


def main() -> int:
    grid = json.loads((D / "e3_grid_v1.json").read_text())["cells"]
    clean_by_org = {}
    for c in grid:
        clean_by_org.setdefault(c["org"], c["clean_dir"])

    per_org = []
    fam_natural_missing = defaultdict(list)   # family -> per-org rate of hosts missing it while expected
    fam_baserate = defaultdict(list)
    for org, cd in sorted(clean_by_org.items()):
        p = Path(cd)
        host_fams, fam_hosts = _fam_dist(p)
        n = len(host_fams)
        if n == 0:
            continue
        baserate = {f: len(hs) / n for f, hs in fam_hosts.items()}
        expected = {f for f, r in baserate.items() if r >= BASE}
        # F4-analog: hosts missing >=1 expected family
        f4_hosts = {h for h, fs in host_fams.items() if expected - fs}
        # per injected family: is it expected here, and how many hosts naturally lack it
        for fam in INJECTED_FAMILIES:
            br = baserate.get(fam, 0.0)
            fam_baserate[fam].append(br)
            if br >= BASE:
                miss = sum(1 for h, fs in host_fams.items() if fam not in fs)
                fam_natural_missing[fam].append(miss / n)
        # F10-analog: resolved-but-no-evidence hosts (manifest) on a clean run
        f10 = len(manifest_dropped_hosts(p))
        per_org.append({"org": org, "n_hosts": n,
                        "f4_analog_hosts": len(f4_hosts),
                        "f4_analog_rate": round(len(f4_hosts) / n, 4),
                        "f10_analog_hosts": f10})

    f4_rates = [r["f4_analog_rate"] for r in per_org]
    f10_total = sum(r["f10_analog_hosts"] for r in per_org)
    result = {
        "phase": "reviewer-② injected-fault representativeness (20 clean runs)",
        "base_rate_threshold": BASE,
        "n_clean_runs": len(per_org),
        "F4_family_omission_analog": {
            "mean_rate_hosts_missing_expected_family": round(st.mean(f4_rates), 4),
            "range": [round(min(f4_rates), 4), round(max(f4_rates), 4)],
            "reading": ("F4's structure (a host lacking a peer-common evidence family) "
                        "occurs naturally on clean runs at this host rate -> the "
                        "injected F4 mechanism is structurally representative."),
        },
        "F4_by_injected_family": {
            fam: {"mean_base_rate": round(st.mean(fam_baserate[fam]), 3),
                  "expected_in_n_orgs": len(fam_natural_missing[fam]),
                  "mean_natural_missing_rate_when_expected":
                      round(st.mean(fam_natural_missing[fam]), 4) if fam_natural_missing[fam] else None}
            for fam in INJECTED_FAMILIES},
        "F10_host_omission_analog": {
            "total_manifest_dropped_hosts_on_clean_runs": f10_total,
            "reading": ("Resolved-but-unused hosts on clean runs = natural F10 analogs. "
                        "If 0, F10 has NO natural analog in this corpus (clean discovery "
                        "+ evidence is complete) -> F10 is a constructed worst-case "
                        "stressor, reported honestly as such."),
        },
        "per_org": sorted(per_org, key=lambda r: r["n_hosts"]),
    }
    OUT.write_text(json.dumps(result, indent=2))

    f4 = result["F4_family_omission_analog"]
    print(f"F4 family-omission analog (clean runs): mean {f4['mean_rate_hosts_missing_expected_family']*100:.1f}% "
          f"of hosts naturally miss a peer-common family (range {f4['range']})")
    print("\nby injected family (base rate | expected in #orgs | natural-missing rate when expected):")
    for fam in INJECTED_FAMILIES:
        b = result["F4_by_injected_family"][fam]
        print(f"  {fam:20} base={b['mean_base_rate']:.3f}  expected_orgs={b['expected_in_n_orgs']:2}  "
              f"nat_miss={b['mean_natural_missing_rate_when_expected']}")
    print(f"\nF10 host-omission analog on clean runs: {f10_total} total resolved-but-unused hosts")
    print(f"  => {'has natural analog' if f10_total>0 else 'NO natural analog (F10 is a constructed stressor)'}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
