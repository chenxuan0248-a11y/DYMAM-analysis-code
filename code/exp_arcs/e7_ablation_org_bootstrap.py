"""e7_ablation_org_bootstrap — organization-level paired bootstrap CIs for the
signal-ablation success drops reported in e7_localization_ablation_v1.json.

That artifact's leave_one_out_success_drop (drop_self=0.0112, drop_peer=0.4112,
drop_manifest=0.3389) is a POOLED point estimate over the 180 (org, fault) cells
-- treating them as 180 independent samples, which §4.5's own stated discipline
(aggregate within org first, avoid pseudoreplication) says not to do for any
other paired comparison in the paper. This closes that gap for the ablation
margins specifically, reusing the same evaluation.stat_test.paired_bootstrap
machinery (and the same B=10_000, seed=42 convention) already used for the
cost/harm CIs in phase5_pilot.py / e3_oracle_stats_v1.json.

Per organization (9 cells each), compute the success RATE for "combined" and
for each leave-one-out variant (peer+manifest, self+manifest, self+peer), then
paired-bootstrap the 20 matched org-level rates. No new pipeline run: reuses
e7_localization_ablation_v1.json's already-computed per-cell success booleans.

    poetry run python -m experiments.exp_arcs.e7_ablation_org_bootstrap
"""
from __future__ import annotations
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from evaluation.stat_test import cliffs_delta, paired_bootstrap

OUT = Path("experiments/exp_arcs/e7_ablation_org_bootstrap_v1.json")

# signal -> the leave-one-out combo whose success rate excludes it
LOO_VARIANT = {"self": "peer+manifest", "peer": "self+manifest", "manifest": "self+peer"}


def _org_rates(cells: list[dict], key: str) -> dict[str, float]:
    by_org = defaultdict(list)
    for c in cells:
        by_org[c["org"]].append(1.0 if c[key]["success"] else 0.0)
    return {org: st.mean(v) for org, v in by_org.items()}


def main() -> int:
    src = json.loads(Path("experiments/exp_arcs/e7_localization_ablation_v1.json").read_text())
    cells = src["cells"]
    orgs = sorted({c["org"] for c in cells})
    n_cells_per_org = {o: sum(1 for c in cells if c["org"] == o) for o in orgs}
    assert set(n_cells_per_org.values()) == {9}, "expected exactly 9 cells per org"

    combined_rates = _org_rates(cells, "combined")

    results = {}
    for signal, variant_key in LOO_VARIANT.items():
        variant_rates = _org_rates(cells, variant_key)
        a = [combined_rates[o] for o in orgs]
        b = [variant_rates[o] for o in orgs]
        pb = paired_bootstrap(a, b, B=10_000, seed=42)
        cd = cliffs_delta(a, b)
        results[f"drop_{signal}"] = {
            "org_level_mean_drop": round(pb.effect, 4),
            "pooled_180cell_drop_for_comparison": src["leave_one_out_success_drop"][f"drop_{signal}"],
            "ci_95": [round(pb.ci_low, 4), round(pb.ci_high, 4)],
            "p_value": pb.p_value,
            "n_orgs": pb.n,
            "cliffs_delta": round(cd["delta"], 4),
            "cliffs_delta_magnitude": cd["magnitude"],
            "org_level_combined_success_rates": {o: round(combined_rates[o], 4) for o in orgs},
            "org_level_variant_success_rates": {o: round(variant_rates[o], 4) for o in orgs},
        }

    out = {
        "phase": "organization-level paired bootstrap for the e7 signal-ablation "
                 "success drops (avoids pseudoreplication across the 9 cells/org)",
        "method": "evaluation.stat_test.paired_bootstrap, B=10000, seed=42 "
                   "(same convention as phase5_pilot.py's cost/harm CIs)",
        "n_orgs": len(orgs),
        "results": results,
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    for k, v in results.items():
        print(f"{k}: org-level mean={v['org_level_mean_drop']} "
              f"(pooled-180-cell={v['pooled_180cell_drop_for_comparison']}) "
              f"CI95={v['ci_95']} p={v['p_value']} cliffs_delta={v['cliffs_delta']} "
              f"({v['cliffs_delta_magnitude']})")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
