"""experiments/exp_arcs/s2_evidence_redundancy — Phase 0.4 S2 survey + D1.

For every claim ``c`` we count ``R(c)`` = the number of *independent source
families* whose evidence supports it (the plan's rule: DNS#1/#2/#3 = 1 family,
DNS+CT+RDAP = 3). Support comes from the decision ledger's
``supporting_evidence_event_ids``; each evidence event carries a
``source_family``. We report P(R=1), P(R=2), P(R≥3) and Coverage₂₊ = P(R≥2),
overall and per claim type. No ground truth is read (avoids selection bias).

Independence matters: a family that is not an *independent external observation*
inflates redundancy (F9 pseudo-redundancy). We therefore score two ways:

  * ``R_all``   — every distinct source_family (liberal upper bound).
  * ``R_indep`` — only INDEP_FAMILIES (conservative; D1 is frozen on this).

INDEP_FAMILIES = {dns, certificate, active_observation, registration}: genuine
independent external observations of an asset/service fact. Excluded:
``llm_inference`` (derived), ``temporal_memory`` (the system's own past output),
``vulnerability_context`` (CVE metadata, not an independent observation of the
asset), ``other``.

    poetry run python -m experiments.exp_arcs.s2_evidence_redundancy \
        [--orgs a,b,c | --raw-root DIR] [--out result.json]

GATE / DECISION D1 (pre-registered, frozen 2026-09-15 BEFORE looking at E3):
  The primary metric is Coverage₂₊ over the SUBSTANTIVE claim types
  (service_observation / technology_identification / vulnerability_applicability)
  — the ones ARCS repairs. Redundancy in asset_identity /
  organization_association does not make a service / tech / CVE finding
  substitutable, so the overall metric (which those dominate) would overstate
  H2; it is reported as context only.
  substantive Coverage₂₊ ≥ 0.30                       → H2 Core
  substantive Coverage₂₊ < 0.30                       → H2 Opportunistic
  pooled < 0.30 but some substantive type ≥ 0.30      → H2 Conditional
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

INDEP_FAMILIES = frozenset({"dns", "certificate", "active_observation", "registration"})
D1_THRESHOLD = 0.30


def _rows(run_dir: Path, name: str) -> list[dict[str, Any]]:
    path = run_dir / name
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _org_of(run_dir: Path) -> str:
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text()).get("snapshot_id", run_dir.name)
    return run_dir.name


def claim_redundancy(run_dir: Path) -> list[dict[str, Any]]:
    """Per-claim R_all / R_indep for one recorded bundle."""
    fam = {e["evidence_event_id"]: e.get("source_family", "other")
           for e in _rows(run_dir, "evidence_event.jsonl")}
    ctype = {c["claim_id"]: c["claim_type"] for c in _rows(run_dir, "claim.jsonl")}
    out: list[dict[str, Any]] = []
    for dec in _rows(run_dir, "decision.jsonl"):
        fams = {fam[e] for e in dec.get("supporting_evidence_event_ids", []) if e in fam}
        out.append({
            "claim_id": dec["claim_id"],
            "claim_type": ctype.get(dec["claim_id"], "unknown"),
            "r_all": len(fams),
            "r_indep": len(fams & INDEP_FAMILIES),
            "families": sorted(fams),
        })
    return out


def _dist(values: list[int]) -> dict[str, float]:
    n = len(values) or 1
    buckets = {0: 0, 1: 0, 2: 0, 3: 0}  # 3 == "≥3"
    for v in values:
        buckets[min(v, 3)] += 1
    return {
        "n": len(values),
        "P(R=1)": round(buckets[1] / n, 4),
        "P(R=2)": round(buckets[2] / n, 4),
        "P(R>=3)": round(buckets[3] / n, 4),
        "coverage_2plus": round(sum(1 for v in values if v >= 2) / n, 4),
    }


def survey(run_dirs: list[Path]) -> dict[str, Any]:
    per_claim: list[dict[str, Any]] = []
    for d in run_dirs:
        org = _org_of(d)
        for row in claim_redundancy(d):
            row["org"] = org
            per_claim.append(row)

    all_indep = [r["r_indep"] for r in per_claim]
    all_liberal = [r["r_all"] for r in per_claim]

    by_type: dict[str, list[int]] = defaultdict(list)
    for r in per_claim:
        by_type[r["claim_type"]].append(r["r_indep"])
    per_type = {t: _dist(v) for t, v in sorted(by_type.items())}

    overall = _dist(all_indep)

    # D1 is frozen on the SUBSTANTIVE claim types — the ones ARCS actually
    # repairs. H2 (evidence substitution) is only useful where a substantive
    # claim has an independent second family; asset_identity /
    # organization_association redundancy does not help repair a service / tech /
    # CVE finding, so counting it (as the overall metric does) inflates H2.
    substantive = ("service_observation", "technology_identification",
                   "vulnerability_applicability")
    substantive_indep = [r["r_indep"] for r in per_claim
                         if r["claim_type"] in substantive]
    substantive_dist = _dist(substantive_indep)
    cov = substantive_dist["coverage_2plus"]
    types_ge_sub = [t for t in substantive
                    if per_type.get(t, {}).get("coverage_2plus", 0.0) >= D1_THRESHOLD]

    if cov >= D1_THRESHOLD:
        verdict = "H2_Core"
    elif types_ge_sub:
        verdict = "H2_Conditional"
    else:
        verdict = "H2_Opportunistic"

    return {
        "phase": "0.4_S2_evidence_redundancy",
        "indep_families": sorted(INDEP_FAMILIES),
        "d1_threshold": D1_THRESHOLD,
        "d1_primary_metric": "coverage_2plus over substantive claim types "
                             f"{substantive} (frozen 2026-09-15)",
        "orgs": sorted({_org_of(d) for d in run_dirs}),
        "n_claims": len(per_claim),
        "d1_substantive_indep": substantive_dist,
        "overall_indep": overall,
        "overall_liberal": _dist(all_liberal),
        "per_claim_type_indep": per_type,
        "d1_verdict": verdict,
        "d1_substantive_types_at_or_above_threshold": types_ge_sub,
        "context_overall_coverage_2plus": overall["coverage_2plus"],
    }


def _resolve_dirs(args: argparse.Namespace) -> list[Path]:
    if args.raw_root:
        root = Path(args.raw_root)
        by_org: dict[str, Path] = {}
        for d in sorted(root.glob("run_*")):
            if (d / "run_manifest.json").exists():
                by_org.setdefault(_org_of(d), d)  # one bundle per org
        return list(by_org.values())
    # else: run fresh
    from core.graph import run_pipeline
    from experiments.exp_arcs.recorder import record_run
    root = Path(tempfile.mkdtemp(prefix="arcs_s2_"))
    orgs = [o.strip() for o in args.orgs.split(",") if o.strip()]
    dirs = []
    for org in orgs:
        print(f"  running {org} ...")
        dirs.append(record_run(run_pipeline(organization=org), raw_root=root))
    return dirs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orgs", default="mozilla,cloudflare,wikimedia,akamai,github")
    parser.add_argument("--raw-root", default=None,
                        help="reuse existing bundles here (one per org) instead of running")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    dirs = _resolve_dirs(args)
    result = survey(dirs)

    o = result["overall_indep"]
    print("=" * 66)
    print(f"S2 redundancy over {len(result['orgs'])} orgs, {result['n_claims']} claims "
          f"(independent families = {result['indep_families']})")
    print(f"  overall: P(R=1)={o['P(R=1)']}  P(R=2)={o['P(R=2)']}  "
          f"P(R>=3)={o['P(R>=3)']}  Coverage2+={o['coverage_2plus']}")
    sub = result["d1_substantive_indep"]
    substantive = ("service_observation", "technology_identification",
                   "vulnerability_applicability")
    print("  per claim type (Coverage2+):")
    for t, d in result["per_claim_type_indep"].items():
        mark = "  <-- substantive" if t in substantive else ""
        print(f"     {t:28s} n={d['n']:5d}  cov2+={d['coverage_2plus']}{mark}")
    print(f"  D1 primary (substantive pooled): n={sub['n']} "
          f"Coverage2+={sub['coverage_2plus']} "
          f"(context: overall={result['context_overall_coverage_2plus']})")
    print(f"D1 VERDICT (frozen on substantive types, threshold {D1_THRESHOLD}): "
          f"{result['d1_verdict']}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
