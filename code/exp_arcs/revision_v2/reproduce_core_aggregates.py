"""Offline, stdlib-only verification of selected EMSE main-result aggregates.

This is not a raw-bundle or whole-pipeline replay. It verifies frozen JSON inputs and
rebuilds a small human-readable main-result table without network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parent
INPUTS = {
    "coverage": HERE / "candidate_coverage_v1.json",
    "access": HERE / "information_access_v1.json",
    "detector": D / "e3_detector_v1.json",
    "localizer": D / "e3_localizers_tieaware_v1.json",
    "no_clean_plan": HERE / "repair_plan_no_clean_eval_v1.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    docs = {k: json.loads(p.read_text()) for k, p in INPUTS.items()}
    cov = docs["coverage"]["coverage"]
    acc = docs["access"]
    methods = docs["coverage"]["methods"]
    loc = docs["localizer"]["summary"]["org_level"]
    det = docs["detector"]["summary"]["detector_only_success"]
    assert docs["coverage"]["all_assertions_pass"] is True
    assert acc["n_assertions_passed"] == 363
    assert (cov["covered_E"]["covered"], cov["covered_ER"]["covered"]) == (129, 243)
    assert (acc["legacy_evidence_only_covered"], acc["legacy_evidence_plus_manifest_covered"]) == (129, 243)
    assert (acc["summary"]["records"]["truth_pairs_covered"],
            acc["summary"]["manifest"]["truth_pairs_covered"],
            acc["summary"]["paired_clean_oracle"]["truth_pairs_covered"]) == (163, 254, 260)
    assert abs(loc["first_flag"]["detect1_tie"]["org_mean"] - 0.0556) < 0.0001
    assert abs(loc["provenance_backtrace"]["detect1_tie"]["org_mean"] - 0.0131) < 0.0001
    assert abs(det["combined"] - 0.8056) < 0.0001
    plan = docs["no_clean_plan"]
    assert (plan["n_cells"], plan["n_orgs"], plan["canonical_scores_matched"],
            plan["planner_zero_file_open_cells"]) == (180, 20, 180, 180)
    expected_plan = {
        "no_action": (0, 0),
        "selective": (119, 36.9167),
        "fixed_four_family": (145, 69.4333),
        "global_four_family": (177, 567.4),
    }
    for key, (successes, spec_cost) in expected_plan.items():
        entry = plan["summary"][key]
        assert entry["offline_success_cells"] == successes
        assert abs(entry["tool_action_spec_cost_org_mean"] - spec_cost) < 0.0001
    for variant in ("F4@certificate", "F4@registration"):
        subset = [cell for cell in plan["cells"] if cell["cell"] == variant]
        assert len(subset) == 20
        assert sum(cell["policies"]["selective"]["offline_counterfactual_success"]
                   for cell in subset) == 0
    assert sum(not cell["policies"]["global_four_family"]["offline_counterfactual_success"]
               for cell in plan["cells"] if cell["org"] == "reddit"
               and cell["cell"].startswith("F10")) == 3
    rows = []
    for label, key in (("Combined", "combined|t=0.5"),
                       ("Isolation Forest", "iforest|manifest_aware|primary"),
                       ("Robust z", "robust_z|manifest_aware|primary")):
        m = methods[key]
        s = m["summary"]
        rows.append((label, s["success"]["org_mean"], s["detect1"]["org_mean"],
                     m["clean_alerts"]["macro_rate"], s["cost_over_Cstar_all"]["org_mean"]))
    lines = [
        "# Offline aggregate verification",
        "",
        "This report is reconstructed from frozen aggregate JSON; it does not replay raw collection, LLMs, or online repair.",
        "",
        "| Method | Success | Detect@1 | Clean alert | Cost/C* |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [f"| {name} | {success:.4f} | {detect:.4f} | {alert:.4f} | {cost:.4f} |"
              for name, success, detect, alert, cost in rows]
    lines += ["", "Evidence-only coverage: 129/260; evidence+manifest: 243/260.",
              "All-surviving-record coverage: 163/260; with manifest: 254/260; paired-clean oracle: 260/260.",
              "Validator-flag probe Detect@1: 0.0556; positive-lineage probe, tie-aware Detect@1: 0.0131.",
              "", "## No-clean-DAG plan: retrospective offline evaluation", "",
              "| Policy | Offline recoveries | Planned tool-action spec units |",
              "|---|---:|---:|",
              *[f"| {label} | {plan['summary'][key]['offline_success_cells']}/180 | "
                f"{plan['summary'][key]['tool_action_spec_cost_org_mean']:.2f} |"
                for label, key in (("No action", "no_action"),
                                   ("Selective family", "selective"),
                                   ("Alerted hosts, four families", "fixed_four_family"),
                                   ("All visible hosts, four families", "global_four_family"))],
              "", "Selective recoveries: F4@certificate 0/20; F4@registration 0/20.",
              "All-visible misses: three Reddit F10 cells. These are plans, not executed repairs.",
              "", "## Input SHA-256", ""]
    lines += [f"- `{p.name}`: `{hashlib.sha256(p.read_bytes()).hexdigest()}`"
              for p in INPUTS.values()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"Verified {len(INPUTS)} aggregate inputs; wrote {args.output}")


if __name__ == "__main__":
    main()
