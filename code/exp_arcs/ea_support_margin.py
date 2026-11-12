"""experiments/exp_arcs/ea_support_margin — E-A: evidence support margin before vs after each
injected fault (protocol: experiments/exp_arcs/protocols/EA_PROTOCOL.md, frozen before any run).

For every E3 grid cell, align clean vs faulted claims on K = (claim_type, subject, predicate,
canonical object) and compare each claim's set of independent supporting families (D1 rule:
supporting_evidence_event_ids -> source_family ∩ {dns, certificate, active_observation,
registration}). Categories: intact / degraded / omitted_unsupported / omitted_absent / altered
(+ R=0 stratum, + new keys). Cross-tabulated against dfx (contamination_closure) per claim and
against the combined detector's success per cell. Offline, deterministic.

    python -m experiments.exp_arcs.ea_support_margin [out.json]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from evaluation.stat_test import bootstrap_ci
from experiments.exp_arcs.contamination import contamination_closure
from experiments.exp_arcs.s2_evidence_redundancy import INDEP_FAMILIES

D = Path("experiments/exp_arcs")
ASSET = ("asset_identity", "organization_association")
SUBSTANTIVE = ("service_observation", "technology_identification", "vulnerability_applicability")
CATS = ("intact", "degraded", "omitted_unsupported", "omitted_absent", "altered",
        "r0_present", "r0_absent")


def _rows(d: Path, name: str) -> list[dict]:
    with (d / name).open() as fh:
        return [json.loads(l) for l in fh if l.strip()]


def claim_index(bundle: Path) -> dict:
    """K -> {claim_type, fams, semantic_keys, claim_ids, decision_ids, n_dup}."""
    fam = {e["evidence_event_id"]: e.get("source_family") for e in _rows(bundle, "evidence_event.jsonl")}
    claims = {c["claim_id"]: c for c in _rows(bundle, "claim.jsonl")}
    idx: dict = {}
    seen_claims = set()
    for dec in _rows(bundle, "decision.jsonl"):
        c = claims[dec["claim_id"]]
        seen_claims.add(c["claim_id"])
        k = (c["claim_type"], c["subject"], c["predicate"], json.dumps(c["object"], sort_keys=True))
        e = idx.setdefault(k, {"claim_type": c["claim_type"], "fams": set(), "semantic_keys": set(),
                               "claim_ids": set(), "decision_ids": set(), "n_dup": 0})
        e["fams"] |= {fam[x] for x in dec.get("supporting_evidence_event_ids", []) if x in fam} & INDEP_FAMILIES
        e["semantic_keys"].add(c.get("semantic_key"))
        e["claim_ids"].add(c["claim_id"]); e["decision_ids"].add(dec["decision_id"])
        e["n_dup"] += 1
    # claims without a decision (should not happen; reported)
    for cid, c in claims.items():
        if cid not in seen_claims:
            k = (c["claim_type"], c["subject"], c["predicate"], json.dumps(c["object"], sort_keys=True))
            e = idx.setdefault(k, {"claim_type": c["claim_type"], "fams": set(), "semantic_keys": set(),
                                   "claim_ids": set(), "decision_ids": set(), "n_dup": 0})
            e["semantic_keys"].add(c.get("semantic_key")); e["claim_ids"].add(cid); e["n_dup"] += 1
    return idx


def categorize(a: dict, b: dict | None) -> str:
    R = len(a["fams"])
    if R == 0:
        return "r0_present" if b is not None else "r0_absent"
    if b is None:
        return "omitted_absent"
    Rp = len(b["fams"])
    if b["fams"] == a["fams"]:
        return "intact"
    if Rp == 0:
        return "omitted_unsupported"
    if b["fams"] < a["fams"] and 0 < Rp < R:
        return "degraded"
    return "altered"


def r_stratum(R: int) -> str:
    return "R>=3" if R >= 3 else f"R={R}"


def klass(ct: str) -> str:
    return "asset" if ct in ASSET else ("substantive" if ct in SUBSTANTIVE else "other")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else D / "ea_support_margin_v1.json"
    grid = json.loads((D / "e3_grid_v1.json").read_text())["cells"]
    det = {(c["org"], c["cell"]): c for c in json.loads((D / "e3_detector_v1.json").read_text())["cells"]}

    claim_tab: Counter = Counter()          # (variant, class, claim_type, stratum, category) -> n
    xtab: Counter = Counter()               # (variant, category, dfx_harmed) -> n
    cells, checks = [], Counter()
    for c in grid:
        clean, faulted = Path(c["clean_dir"]), Path(c["faulted_dir"])
        A, B = claim_index(clean), claim_index(faulted)
        dfx = contamination_closure(clean, faulted)
        contaminated = set(dfx["contaminated_ids"])
        cat_counts: Counter = Counter()
        for k, a in A.items():
            b = B.get(k)
            cat = categorize(a, b)
            cls = klass(a["claim_type"])
            claim_tab[(c["cell"], cls, a["claim_type"], r_stratum(len(a["fams"])), cat)] += 1
            harmed = any(f"claim:{i}" in contaminated for i in a["claim_ids"]) or \
                any(f"decision:{i}" in contaminated for i in a["decision_ids"])
            xtab[(c["cell"], cat, harmed)] += 1
            cat_counts[(cls, cat)] += 1
            if b is not None and a["semantic_keys"] != b["semantic_keys"]:
                checks["semantic_key_mismatch"] += 1
            if a["n_dup"] > 1:
                checks["dup_keys_clean"] += 1
        new_keys = len(B.keys() - A.keys())
        checks["new_keys_total"] += new_keys
        n_asset = sum(v for (cls, _), v in cat_counts.items() if cls == "asset")
        n_sub = sum(v for (cls, _), v in cat_counts.items() if cls == "substantive")
        dc = det[(c["org"], c["cell"])]
        cells.append({
            "org": c["org"], "cell": c["cell"], "inject_node": c["inject_node"], "new_keys": new_keys,
            "counts": {f"{cls}:{cat}": v for (cls, cat), v in sorted(cat_counts.items())},
            "n_clean_asset": n_asset, "n_clean_substantive": n_sub,
            "has_degraded_asset": cat_counts[("asset", "degraded")] > 0,
            "has_degraded_substantive": cat_counts[("substantive", "degraded")] > 0,
            "has_omitted_substantive": (cat_counts[("substantive", "omitted_absent")]
                                        + cat_counts[("substantive", "omitted_unsupported")]) > 0,
            "frac_asset_degraded": round(cat_counts[("asset", "degraded")] / n_asset, 6) if n_asset else None,
            "frac_substantive_omitted": round((cat_counts[("substantive", "omitted_absent")]
                                               + cat_counts[("substantive", "omitted_unsupported")]) / n_sub, 6) if n_sub else None,
            "detector_combined_success": bool(dc["combined"]["success"]),
        })

    variants = sorted({c["cell"] for c in cells})

    def org_macro(variant: str, key: str) -> dict:
        by_org = defaultdict(list)
        for r in cells:
            if r["cell"] == variant and r[key] is not None:
                by_org[r["org"]].append(r[key])
        means = [sum(v) / len(v) for _, v in sorted(by_org.items())]
        if not means:
            return {"org_mean": None}
        m, lo, hi = bootstrap_ci(means, B=10_000, seed=42)
        return {"org_mean": round(m, 6), "ci95": [round(lo, 6), round(hi, 6)], "n_orgs": len(means)}

    per_variant = {}
    for v in variants:
        rs = [r for r in cells if r["cell"] == v]
        per_variant[v] = {
            "n_cells": len(rs),
            "cells_with_degraded_asset": sum(r["has_degraded_asset"] for r in rs),
            "cells_with_degraded_substantive": sum(r["has_degraded_substantive"] for r in rs),
            "cells_with_omitted_substantive": sum(r["has_omitted_substantive"] for r in rs),
            "claims_by_class_category": {
                f"{cls}:{cat}": sum(n for (vv, cl, _ct, _s, ca), n in claim_tab.items()
                                    if vv == v and cl == cls and ca == cat)
                for cls in ("asset", "substantive") for cat in CATS},
            "org_macro_frac_asset_degraded": org_macro(v, "frac_asset_degraded"),
            "org_macro_frac_substantive_omitted": org_macro(v, "frac_substantive_omitted"),
            "dfx_crosstab": {f"{cat}|{'harmed' if h else 'unharmed'}": n
                             for (vv, cat, h), n in sorted(xtab.items()) if vv == v},
            "detector_link": {
                f"subst_damaged={sd}|detector_success={ds}":
                    sum(1 for r in rs if (r["has_omitted_substantive"] or r["has_degraded_substantive"]) == sd
                        and r["detector_combined_success"] == ds)
                for sd in (True, False) for ds in (True, False)},
        }
    strata = defaultdict(Counter)
    for (v, cls, ct, st, cat), n in claim_tab.items():
        strata[f"{v}|{cls}|{st}"][cat] += n
    result = {
        "phase": "E-A support margin (protocol EA_PROTOCOL.md, frozen before run)",
        "source": "experiments/exp_arcs/e3_grid_v1.json",
        "definitions": {"indep_families": sorted(INDEP_FAMILIES), "asset_types": ASSET,
                        "substantive_types": SUBSTANTIVE, "categories": CATS},
        "checks": dict(checks),
        "per_variant": per_variant,
        "strata_variant_class_R": {k: dict(v) for k, v in sorted(strata.items())},
        "claims_by_variant_class_type_R_category": [
            {"variant": v, "class": cls, "claim_type": ct, "R": st, "category": cat, "n": n}
            for (v, cls, ct, st, cat), n in sorted(claim_tab.items())],
        "cells": cells,
    }
    out.write_text(json.dumps(result, indent=1, sort_keys=True, default=list))
    print(json.dumps({"checks": dict(checks)}, indent=1))
    for v in variants:
        p = per_variant[v]
        cc = p["claims_by_class_category"]
        print(f"{v:24s} asset deg={cc['asset:degraded']:4d} omit_abs={cc['asset:omitted_absent']:4d} | "
              f"subst deg={cc['substantive:degraded']:3d} omit_abs={cc['substantive:omitted_absent']:4d} "
              f"omit_unsup={cc['substantive:omitted_unsupported']:3d} altered={cc['substantive:altered']:3d} | "
              f"cells subst-omitted={p['cells_with_omitted_substantive']}/{p['n_cells']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
