"""Reviewer item G — non-leaf (gate-logic) fault triage & escalation.

ARCS repairs by re-collecting leaf evidence. Gate-logic faults (F7 evidence-gate
bypass, F8 authorization-gate bypass) corrupt decisions/reports WITHOUT corrupting
any leaf evidence, so their contamination closure D(f) has ZERO repairable leaves:
leaf re-collection cannot cover them. Silently running ARCS there would "succeed"
by selecting the empty scope while leaving the fault in place -- a false pass.

This makes the boundary explicit: ARCS triages every localized fault by whether its
contaminated outputs are leaf-repairable. If D(f) contains contaminated
decision/report outputs but NO repairable leaf covers them, ARCS declines the
scoped repair and ESCALATES (to global replay / human review). The experiment shows
the triage correctly separates the 2 non-leaf faults from the 8 leaf-repairable
ones, using only the frozen D(f) artifacts (no new fault runs).

    poetry run python -m experiments.exp_arcs.triage_nonleaf
"""
from __future__ import annotations
import json
from pathlib import Path

LIVE = Path("experiments/exp_arcs/dfs_v2.json")          # F2/F4/F8/F10 (mozilla)
REC = Path("experiments/exp_arcs/record_dfs_v1.json")    # F1/F3/F5/F6/F7/F9
OUT = Path("experiments/exp_arcs/triage_nonleaf_v1.json")

OUTPUT_TYPES = {"decision", "report_item"}   # the outputs a fault ultimately damages


def _rep(fx):
    return fx.get("n_repairable", len(fx.get("repairable_ids", [])))


def classify(fname, fx):
    cbt = fx.get("contaminated_by_type", {})
    n_out = sum(cbt.get(t, 0) for t in OUTPUT_TYPES)
    n_rep = _rep(fx)
    n_cont = fx.get("n_contaminated", len(fx.get("contaminated_ids", [])))
    # leaf-repairable iff there is at least one repairable leaf to re-collect.
    # non-leaf (escalate) iff the fault damaged outputs but no repairable leaf exists.
    leaf_repairable = n_rep > 0
    action = "arcs_scoped_repair" if leaf_repairable else "escalate_to_global"
    # what leaf-only ARCS would recover on this fault:
    #  - leaf-repairable: covers its repairable outputs (=ARCS success, shown in E3)
    #  - non-leaf: selects empty scope -> recovers 0 of the damaged outputs
    leaf_output_recovery = None if leaf_repairable else 0
    return {
        "fault": fname,
        "n_contaminated": n_cont,
        "n_damaged_outputs": n_out,
        "n_repairable_leaves": n_rep,
        "class": "leaf-repairable" if leaf_repairable else "non-leaf (gate-logic)",
        "arcs_action": action,
        "leaf_only_output_recovery_on_nonleaf": leaf_output_recovery,
    }


def main() -> int:
    rows = []
    live = json.loads(LIVE.read_text())["faults"]
    for fname in ("F2", "F4", "F8", "F10"):
        if fname in live:
            rows.append(classify(fname, live[fname]))
    rec = json.loads(REC.read_text())["faults"]
    for fname in ("F1", "F3", "F5", "F6", "F7", "F9"):
        if fname in rec:
            rows.append(classify(fname, rec[fname]))
    rows.sort(key=lambda r: r["fault"])

    escalate = [r["fault"] for r in rows if r["arcs_action"] == "escalate_to_global"]
    scoped = [r["fault"] for r in rows if r["arcs_action"] == "arcs_scoped_repair"]
    result = {
        "phase": "reviewer-G non-leaf fault triage",
        "note": ("Uses frozen D(f): dfs_v2.json (live F2/F4/F8/F10, mozilla) + "
                 "record_dfs_v1.json (F1/F3/F5/F6/F7/F9 fixture). A fault with damaged "
                 "decision/report outputs but 0 repairable leaves cannot be fixed by "
                 "leaf re-collection; ARCS escalates instead of falsely passing."),
        "triage": {"escalate_to_global": escalate, "arcs_scoped_repair": scoped},
        "finding": (
            f"{len(escalate)}/10 faults are non-leaf gate-logic ({', '.join(escalate)}): "
            "0 repairable leaves, so leaf-only ARCS would recover 0 damaged outputs "
            "(false pass) -- ARCS instead escalates. The other "
            f"{len(scoped)}/10 are leaf-repairable and handled by scoped repair."),
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    hdr = f"{'fault':6} {'class':22} {'damaged_out':>11} {'rep_leaves':>10} {'action':>20}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['fault']:6} {r['class']:22} {r['n_damaged_outputs']:>11} "
              f"{r['n_repairable_leaves']:>10} {r['arcs_action']:>20}")
    print("\n" + result["finding"])
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
