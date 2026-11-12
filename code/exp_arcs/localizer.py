"""experiments/exp_arcs/localizer — Step 3.1/3.2 fault localization.

Predicts the LOCAL fault's inject_node = (target host[, source_family]) from a
faulted run, and scores top-1/top-k against the recorded truth. Three localizers:

  - oracle          : returns the recorded inject_node. Top-1 == 1.0 by
                      construction — the paradigm anchor (Step 3.2a): its role is
                      to feed the TRUE node to Phase 4 so ARCS's *scope selection*
                      is isolated from detector error, NOT to demonstrate skill.

  - lineage_blame   : SEMI-ORACLE (Step 3.1 honest upper bound). Uses the
                      clean-vs-faulted divergence (contamination_closure) as the
                      detector signal, then does real lineage work: for each
                      divergent OUTPUT (report_item), walk `_ancestors` on the
                      clean DAG to the contaminated leaves that explain it, and
                      blame each culprit leaf's (target, source_family). Rank
                      locations by how many divergent outputs they explain. Uses
                      the clean reference (hence semi-oracle), so it is the CEILING
                      the future detector-only localizer (Step 3.2b) is measured
                      against — not the end-to-end number.

  - naive_output    : baseline WITHOUT lineage — blame the hosts whose report_items
                      changed (report_item.subject), ranked by count. Included to
                      show lineage is NECESSARY on base-domain-fanout faults
                      (cert/registration), where the changed outputs are spread
                      across the corpus but the culprit is one hub host.

    poetry run python -m experiments.exp_arcs.localizer \
        [local_oracle_v1.json] [localization_v1.json]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.exp_arcs.contamination import (
    contamination_closure, _REPAIRABLE, _index, _RECORD_FILES)
from experiments.exp_arcs.counterfactual import _ancestors, _bare, _OUTPUT_TYPE
from experiments.exp_arcs.dag import build_dag


def _true(inject_node: list) -> tuple[str, str | None]:
    """(host, family|None) from inject_node = ['target', host[, family]]."""
    host = inject_node[1]
    family = inject_node[2] if len(inject_node) >= 3 else None
    return host, family


def lineage_blame(clean_dir: Path, faulted_dir: Path) -> dict[str, Any]:
    """Rank (target, family) and host by lineage blame mass over divergent outputs."""
    dfx = contamination_closure(clean_dir, faulted_dir)
    dag = build_dag(clean_dir)
    cont_leaves = {l for l in _bare(dfx, _REPAIRABLE) if l in dag.nodes}
    outputs = {o for o in _bare(dfx, (_OUTPUT_TYPE,)) if o in dag.nodes}

    tf_score: Counter = Counter()   # (target, family) -> #outputs explained
    host_score: Counter = Counter()
    explained_outputs = 0
    for o in outputs:
        culprits = _ancestors(dag, {o}) & cont_leaves
        if not culprits:
            continue
        explained_outputs += 1
        tfs = {(dag.nodes[l].get("target"), dag.nodes[l]["source_family"])
               for l in culprits}
        hosts = {dag.nodes[l].get("target") for l in culprits}
        for tf in tfs:
            tf_score[tf] += 1
        for h in hosts:
            host_score[h] += 1

    # fallback when no output has a leaf ancestor (e.g. all divergence is
    # decision-level): blame the contaminated leaves directly by count.
    if not tf_score and cont_leaves:
        for l in cont_leaves:
            tf = (dag.nodes[l].get("target"), dag.nodes[l]["source_family"])
            tf_score[tf] += 1
            host_score[dag.nodes[l].get("target")] += 1

    ranked_tf = [tf for tf, _ in sorted(
        tf_score.items(), key=lambda kv: (-kv[1], str(kv[0])))]
    ranked_host = [h for h, _ in sorted(
        host_score.items(), key=lambda kv: (-kv[1], str(kv[0])))]
    return {"ranked_tf": ranked_tf, "ranked_host": ranked_host,
            "n_outputs_explained": explained_outputs,
            "n_cont_leaves": len(cont_leaves),
            "n_candidate_locations": len(tf_score)}


def naive_output(clean_dir: Path, faulted_dir: Path) -> dict[str, Any]:
    """Baseline: rank hosts by count of changed report_items (by subject)."""
    dfx = contamination_closure(clean_dir, faulted_dir)
    changed = _bare(dfx, (_OUTPUT_TYPE,))
    fn, idf = _RECORD_FILES[_OUTPUT_TYPE]
    subj: dict[str, str] = {}
    for d in (clean_dir, faulted_dir):
        for ln in (d / fn).read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                subj[r[idf]] = r.get("subject")
    host_score = Counter(subj.get(i) for i in changed if subj.get(i))
    ranked_host = [h for h, _ in sorted(
        host_score.items(), key=lambda kv: (-kv[1], str(kv[0])))]
    return {"ranked_host": ranked_host, "n_changed_outputs": len(changed)}


def _hit(ranked: list, true, k: int) -> tuple[bool, bool]:
    """(top1, topk) membership of `true` in `ranked`."""
    return (bool(ranked) and ranked[0] == true), (true in ranked[:k])


def score_cell(cell: dict, k: int = 3) -> dict[str, Any]:
    host, family = _true(cell["inject_node"])
    clean, faulted = Path(cell["clean_dir"]), Path(cell["faulted_dir"])
    lb = lineage_blame(clean, faulted)
    nv = naive_output(clean, faulted)

    lb_h1, lb_hk = _hit(lb["ranked_host"], host, k)
    nv_h1, nv_hk = _hit(nv["ranked_host"], host, k)
    out = {
        "org": cell["org"], "cell": cell["cell"],
        "inject_node": cell["inject_node"],
        "oracle_top1_host": True,          # by construction
        "lineage_top1_host": lb_h1, "lineage_topk_host": lb_hk,
        "naive_top1_host": nv_h1, "naive_topk_host": nv_hk,
        "lineage_pred_host": lb["ranked_host"][:k],
        "naive_pred_host": nv["ranked_host"][:k],
        "n_candidate_locations": lb["n_candidate_locations"],
        "n_outputs_explained": lb["n_outputs_explained"],
    }
    if family is not None:   # F4 family-scoped cells: also score (host, family)
        tf_true = (host, family)
        lb_tf1, lb_tfk = _hit(lb["ranked_tf"], tf_true, k)
        out["oracle_top1_tf"] = True
        out["lineage_top1_tf"] = lb_tf1
        out["lineage_topk_tf"] = lb_tfk
        out["lineage_pred_tf"] = [list(t) for t in lb["ranked_tf"][:k]]
    return out


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    src = Path(argv[0]) if len(argv) > 0 else Path("experiments/exp_arcs/local_oracle_v1.json")
    out_path = Path(argv[1]) if len(argv) > 1 else Path("experiments/exp_arcs/localization_v1.json")
    K = 3
    cells = json.loads(src.read_text())["cells"]

    scored = [score_cell(c, K) for c in cells]

    def rate(key: str, subset: list) -> float:
        vals = [s[key] for s in subset if key in s]
        return round(sum(vals) / len(vals), 4) if vals else None

    tf_cells = [s for s in scored if "lineage_top1_tf" in s]
    summary = {
        "k": K, "n_cells": len(scored),
        "host_level": {
            "oracle_top1": 1.0,
            "lineage_top1": rate("lineage_top1_host", scored),
            "lineage_topk": rate("lineage_topk_host", scored),
            "naive_top1": rate("naive_top1_host", scored),
            "naive_topk": rate("naive_topk_host", scored),
        },
        "host_family_level_F4_only": {
            "n_cells": len(tf_cells),
            "oracle_top1": 1.0,
            "lineage_top1": rate("lineage_top1_tf", tf_cells),
            "lineage_topk": rate("lineage_topk_tf", tf_cells),
        },
    }
    result = {
        "phase": "3.localization",
        "spec": "Oracle + semi-oracle lineage blame + naive baseline on the 25 local cells",
        "source": str(src), "summary": summary, "cells": scored,
    }
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(json.dumps(summary, indent=2))
    print("\nfanout cells (cert/registration) — lineage vs naive top-1 host:")
    for s in scored:
        if "certificate" in s["cell"] or "registration" in s["cell"]:
            print(f"  {s['org']:11}/{s['cell']:18} true={s['inject_node'][1]:24} "
                  f"lineage_top1={s['lineage_top1_host']} naive_top1={s['naive_top1_host']} "
                  f"naive_pred={s['naive_pred_host'][:1]}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
