"""experiments/exp_arcs/localizer_baselines — Step 3.1 named localizer baselines.

The plan names AgentRx / AgentDebug / FAMAS-style lineage as fault-localization
baselines. Those papers are not vendored here, so these are faithful **style
adaptations** of each method's core principle to this domain, clearly labelled as
such — NOT exact ports. They are scored on the same footing as the Oracle /
semi-oracle-lineage / naive localizers (top-1 / top-k vs the recorded inject_node),
over the 45-cell local grid.

Methods (mechanisms deliberately distinct):

  AgentDebug-style — critical-step attribution, REFERENCE-FREE. AgentDebug isolates
    the single earliest critical error in a trajectory. Adaptation: scan decisions
    in execution (append) order; the critical step is the FIRST decision that
    self-reports failure (decision=revalidate, or reason_code
    no_supporting_observation, or support_state=unsupported). Rank hosts by first
    appearance of such a decision.

  FAMAS-style lineage — failure attribution ("who/when") by lineage, REFERENCE-FREE.
    Seed = self-diagnosis-flagged outputs; trace ancestors on the FAULTED run's own
    DAG (no clean reference); rank (host, family) invocations by responsibility mass
    (how many flagged outputs each is an ancestor of).

  AgentRx-style — diagnostic ablation ("prescription"), SEMI-ORACLE probe. AgentRx
    diagnoses by testing which change fixes the outcome. Adaptation: over the
    candidate invocations in I(f), score each by counterfactual repair restoration
    (repair_and_check with just that invocation → #leaf-repairable outputs
    restored); rank by restoration. Uses the A1 counterfactual, so it is a
    semi-oracle probe, comparable to the semi-oracle-lineage ceiling.

    poetry run python -m experiments.exp_arcs.localizer_baselines [grid.json] [out.json]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.exp_arcs.candidates import candidate_invocations
from experiments.exp_arcs.contamination import contamination_closure, _REPAIRABLE
from experiments.exp_arcs.counterfactual import _ancestors, _bare, _OUTPUT_TYPE, repair_and_check
from experiments.exp_arcs.dag import build_dag

FLAG_DECISIONS = {"revalidate"}
FLAG_REASONS = {"no_supporting_observation"}
FLAG_SUPPORT = {"unsupported"}


def _claim_subject(run_dir: Path) -> dict[str, str]:
    m = {}
    for ln in (run_dir / "claim.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            m[r["claim_id"]] = r.get("subject")
    return m


def _flagged_decisions(run_dir: Path) -> list[dict]:
    """Decisions self-reporting failure, in file (execution) order."""
    out = []
    for ln in (run_dir / "decision.jsonl").read_text().splitlines():
        if not ln.strip():
            continue
        d = json.loads(ln)
        codes = set(d.get("reason_codes") or [])
        if (d.get("decision") in FLAG_DECISIONS or (codes & FLAG_REASONS)
                or d.get("support_state") in FLAG_SUPPORT):
            out.append(d)
    return out


# ─────────────────────────── AgentDebug-style ───────────────────────────
def agentdebug_style(faulted_dir: Path) -> list[str]:
    subj = _claim_subject(faulted_dir)
    ranked: list[str] = []
    for d in _flagged_decisions(faulted_dir):          # execution order
        h = subj.get(d.get("claim_id"))
        if h and h not in ranked:
            ranked.append(h)                            # earliest-first
    return ranked


# ─────────────────────────── FAMAS-style lineage ───────────────────────────
def famas_style(faulted_dir: Path) -> dict[str, list]:
    dag = build_dag(faulted_dir)                        # faulted run's OWN DAG
    subj = _claim_subject(faulted_dir)
    flagged_claims = {d.get("claim_id") for d in _flagged_decisions(faulted_dir)}
    # flagged outputs = report_items of flagged claims (fall back to claim nodes)
    out_ids = set()
    for ln in (faulted_dir / "report_item.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            if r.get("claim_id") in flagged_claims and r["report_item_id"] in dag.nodes:
                out_ids.add(r["report_item_id"])
    if not out_ids:
        out_ids = {c for c in flagged_claims if c in dag.nodes}
    tf: Counter = Counter(); host: Counter = Counter()
    for o in out_ids:
        anc = _ancestors(dag, {o})
        seen_tf = set(); seen_h = set()
        for l in anc:
            v = dag.nodes.get(l)
            if v and v.get("repairable"):
                seen_tf.add((v.get("target"), v["source_family"])); seen_h.add(v.get("target"))
        for k in seen_tf:
            tf[k] += 1
        for h in seen_h:
            host[h] += 1
    # fall back to the flagged decisions' subjects if lineage is empty (dropped ev.)
    if not host:
        host = Counter(subj.get(d.get("claim_id")) for d in _flagged_decisions(faulted_dir)
                       if subj.get(d.get("claim_id")))
    ranked_tf = [k for k, _ in sorted(tf.items(), key=lambda kv: (-kv[1], str(kv[0])))]
    ranked_host = [h for h, _ in sorted(host.items(), key=lambda kv: (-kv[1], str(kv[0])))]
    return {"ranked_tf": ranked_tf, "ranked_host": ranked_host}


# ─────────────────────────── AgentRx-style ───────────────────────────
def agentrx_style(clean_dir: Path, faulted_dir: Path) -> dict[str, list]:
    dfx = contamination_closure(clean_dir, faulted_dir)
    cand = candidate_invocations(clean_dir, dfx)
    scores: list[tuple[tuple, int]] = []
    for inv, leaves in cand["invocations"].items():
        r = repair_and_check(clean_dir, dfx, set(leaves))
        scores.append((inv, r["n_outputs_restored"]))
    scores.sort(key=lambda kv: (-kv[1], str(kv[0])))
    ranked_tf = [inv for inv, _ in scores]
    ranked_host: list[str] = []
    for (h, _f), _ in scores:
        if h not in ranked_host:
            ranked_host.append(h)
    return {"ranked_tf": ranked_tf, "ranked_host": ranked_host}


# ─────────────────────────── scoring ───────────────────────────
def _truth(inject_node: list) -> tuple[set[str], str | None]:
    if inject_node[0] == "targets":
        hosts = set(inject_node[1])
        fam = inject_node[2] if len(inject_node) >= 3 else None
    else:
        hosts = {inject_node[1]}
        fam = inject_node[2] if len(inject_node) >= 3 else None
    return hosts, fam


def _score_host(ranked: list[str], hosts: set[str], k: int) -> tuple[int, int]:
    top1 = int(bool(ranked) and ranked[0] in hosts)
    topk = int(any(h in hosts for h in ranked[:k]))
    return top1, topk


def _score_tf(ranked_tf: list, hosts: set[str], fam: str | None, k: int) -> tuple[int, int]:
    if fam is None:
        return (0, 0)
    want = {(h, fam) for h in hosts}
    top1 = int(bool(ranked_tf) and tuple(ranked_tf[0]) in want)
    topk = int(any(tuple(t) in want for t in ranked_tf[:k]))
    return top1, topk


def run_cell(cell: dict, k: int = 3) -> dict[str, Any]:
    clean, faulted = Path(cell["clean_dir"]), Path(cell["faulted_dir"])
    hosts, fam = _truth(cell["inject_node"])
    ad = agentdebug_style(faulted)
    fa = famas_style(faulted)
    rx = agentrx_style(clean, faulted)
    out = {"org": cell["org"], "cell": cell["cell"], "inject_node": cell["inject_node"]}
    ad1, adk = _score_host(ad, hosts, k)
    fa1, fak = _score_host(fa["ranked_host"], hosts, k)
    rx1, rxk = _score_host(rx["ranked_host"], hosts, k)
    out["agentdebug"] = {"top1_host": ad1, "topk_host": adk}
    out["famas"] = {"top1_host": fa1, "topk_host": fak,
                    "top1_tf": _score_tf(fa["ranked_tf"], hosts, fam, k)[0]}
    out["agentrx"] = {"top1_host": rx1, "topk_host": rxk,
                      "top1_tf": _score_tf(rx["ranked_tf"], hosts, fam, k)[0]}
    out["has_family_truth"] = fam is not None
    return out


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    src = Path(argv[0]) if argv else Path("experiments/exp_arcs/local_oracle_all_v1.json")
    out = Path(argv[1]) if len(argv) > 1 else Path("experiments/exp_arcs/localization_baselines_v1.json")
    K = 3
    cells = json.loads(src.read_text())["cells"]
    rows = [run_cell(c, K) for c in cells]

    def rate(key_path):
        vals = []
        for r in rows:
            m, key = key_path
            if key.endswith("_tf") and not r["has_family_truth"]:
                continue
            vals.append(r[m][key])
        return round(sum(vals) / len(vals), 4) if vals else None

    methods = ["agentdebug", "famas", "agentrx"]
    summary = {
        "n_cells": len(rows), "k": K,
        "note": ("style adaptations, not exact ports; AgentDebug/FAMAS are "
                 "reference-free, AgentRx is a semi-oracle counterfactual probe"),
        "reference_localizers": {"oracle_top1": 1.0, "semi_oracle_lineage_top1": 1.0,
                                 "naive_top1": 0.6},
        "host_top1": {m: rate((m, "top1_host")) for m in methods},
        "host_topk": {m: rate((m, "topk_host")) for m in methods},
        "host_family_top1": {m: rate((m, "top1_tf")) for m in ("famas", "agentrx")},
    }
    result = {"phase": "3.1.named_baselines", "source": str(src),
              "summary": summary, "cells": rows}
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
