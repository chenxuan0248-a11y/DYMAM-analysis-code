"""Check the local, aggregate-only release candidate against retained results.

This script needs restricted local inputs; it is not part of the public
candidate and does not grant rights to redistribute those inputs.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RELEASE = EXP / "release_candidate_emse_aggregate_v1" / "aggregate_results.json"
PLANNER = HERE / "common_planner_eval_20261020.json"


def main() -> None:
    candidate = json.loads(RELEASE.read_text())
    coverage = json.loads((HERE / "candidate_coverage_v1.json").read_text())
    access = json.loads((HERE / "information_access_v1.json").read_text())
    detector = json.loads((EXP / "e3_detector_v1.json").read_text())
    localizer = json.loads((EXP / "e3_localizers_tieaware_v1.json").read_text())
    no_clean = json.loads((HERE / "repair_plan_no_clean_eval_v1.json").read_text())
    common = json.loads(PLANNER.read_text())
    assert common["n_cells"] == candidate["population"]["fault_cells"] == 180
    assert common["n_orgs"] == candidate["population"]["organizations"] == 20
    assert candidate["coverage"] == {
        "evidence_only": coverage["coverage"]["covered_E"]["covered"],
        "evidence_and_manifest": coverage["coverage"]["covered_ER"]["covered"],
        "surviving_records": access["summary"]["records"]["truth_pairs_covered"],
        "surviving_records_and_manifest": access["summary"]["manifest"]["truth_pairs_covered"],
        "paired_clean_oracle": access["summary"]["paired_clean_oracle"]["truth_pairs_covered"],
    }
    method_keys = ("combined|t=0.5", "iforest|manifest_aware|primary",
                   "robust_z|manifest_aware|primary")
    for row, key in zip(candidate["methods"], method_keys, strict=True):
        source = coverage["methods"][key]
        summary = source["summary"]
        assert row["success"] == summary["success"]["org_mean"]
        assert row["detect_at_1"] == summary["detect1"]["org_mean"]
        assert row["clean_alert"] == source["clean_alerts"]["macro_rate"]
        assert row["cost_over_cstar"] == summary["cost_over_Cstar_all"]["org_mean"]
    assert candidate["probes"] == {
        "validator_flag_detect_at_1": round(
            localizer["summary"]["org_level"]["first_flag"]["detect1_tie"]["org_mean"], 4
        ),
        "positive_lineage_tie_aware_detect_at_1": round(
            localizer["summary"]["org_level"]["provenance_backtrace"]["detect1_tie"]["org_mean"], 4
        ),
    }
    assert detector["summary"]["detector_only_success"]["combined"] == 0.8056
    policy_keys = ("no_action", "selective", "fixed_four_family",
                   "global_four_family")
    for row, key in zip(candidate["retrospective_no_clean_dag_plans"],
                        policy_keys, strict=True):
        source = no_clean["summary"][key]
        assert row["offline_recoveries"] == source["offline_success_cells"]
        assert row["planned_tool_action_spec_units"] == round(
            source["tool_action_spec_cost_org_mean"], 2
        )
    for label, variant in (("certificate", "F4@certificate"),
                           ("registration", "F4@registration")):
        cells = [cell for cell in no_clean["cells"] if cell["cell"] == variant]
        assert candidate["rare_family_selective_recoveries"][label] == {
            "recovered": sum(
                bool(cell["policies"]["selective"]["offline_counterfactual_success"])
                for cell in cells
            ),
            "cells": len(cells),
        }
    assert candidate["all_visible_f10_misses"] == sum(
        not cell["policies"]["global_four_family"]["offline_counterfactual_success"]
        for cell in no_clean["cells"]
        if cell["org"] == "reddit" and cell["cell"].startswith("F10")
    )
    assert set(candidate["same_planner"]) == set(common["summary"])

    fields = ("detect1_org_mean", "truth_host_alert_recall_org_mean",
              "clean_alert_macro_rate")
    policy_fields = ("recovery_cells", "planned_cost_org_mean",
                     "alerted_hosts_org_mean")
    for method, source in common["summary"].items():
        released = candidate["same_planner"][method]
        for field in fields:
            assert released[field] == source[field], (method, field)
        assert set(released["policies"]) == set(source["policies"])
        for policy, source_policy in source["policies"].items():
            for field in policy_fields:
                assert released["policies"][policy][field] == source_policy[field], (
                    method, policy, field
                )
    print("All curated aggregate fields match the retained frozen JSON inputs.")
    print("This verifies selected saved summaries, not raw-response reproducibility.")


if __name__ == "__main__":
    main()
