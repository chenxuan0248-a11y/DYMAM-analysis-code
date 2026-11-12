"""random_baseline_localizers — pool-size-normalized random-chance top-1/top-k
hit rate, to give a denominator for the "AgentDebug/FAMAS top-1 is near-random"
claim in RQ1 (currently unsupported: 0.056/0.033 were never compared to what
chance actually predicts for each org's candidate pool size).

For each of the 180 E3 cells, the candidate pool is the clean run's resolved
host set (`discovery_manifest.json::resolved_hosts`, the same set a host-grain
localizer ranks over). Expected random top-1 hit rate for a single true host is
1/N; expected random top-k (k=3, sampling without replacement) is
min(k, N)/N. Aggregated as a mean over cells, and as a mean of per-org means
(each org contributes 9 cells, so cross-checked against the pooled mean).

    poetry run python -m experiments.exp_arcs.random_baseline_localizers
"""
from __future__ import annotations
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path("experiments/exp_arcs/random_baseline_localizers_v1.json")
K = 3


def main() -> int:
    grid = json.loads(Path("experiments/exp_arcs/e3_grid_v1.json").read_text())["cells"]
    localizers = json.loads(Path("experiments/exp_arcs/e3_localizers_v1.json").read_text())

    by_key = {(c["org"], c["cell"]): c for c in grid}
    rows = []
    for c in localizers["cells"]:
        key = (c["org"], c["cell"])
        g = by_key.get(key)
        if g is None:
            continue
        manifest = json.loads((Path(g["clean_dir"]) / "discovery_manifest.json").read_text())
        n_hosts = len(manifest["resolved_hosts"])
        rows.append({
            "org": c["org"], "cell": c["cell"], "n_hosts": n_hosts,
            "random_top1": 1 / n_hosts, "random_topk": min(K, n_hosts) / n_hosts,
        })

    pooled_top1 = st.mean(r["random_top1"] for r in rows)
    pooled_topk = st.mean(r["random_topk"] for r in rows)

    by_org = defaultdict(list)
    for r in rows:
        by_org[r["org"]].append(r)
    org_means_top1 = [st.mean(x["random_top1"] for x in v) for v in by_org.values()]
    org_means_topk = [st.mean(x["random_topk"] for x in v) for v in by_org.values()]

    observed = {
        "agentdebug": {"top1": localizers["summary"]["host_top1"]["agentdebug"],
                       "topk": localizers["summary"]["host_topk"]["agentdebug"]},
        "famas": {"top1": localizers["summary"]["host_top1"]["famas"],
                  "topk": localizers["summary"]["host_topk"]["famas"]},
    }

    result = {
        "phase": "random-baseline pool-size normalization for RQ1 'near-random' claim",
        "k": K,
        "n_cells": len(rows),
        "n_hosts_range": [min(r["n_hosts"] for r in rows), max(r["n_hosts"] for r in rows)],
        "expected_random": {
            "pooled_mean_top1": round(pooled_top1, 4),
            "pooled_mean_topk": round(pooled_topk, 4),
            "org_level_mean_top1": round(st.mean(org_means_top1), 4),
            "org_level_mean_topk": round(st.mean(org_means_topk), 4),
        },
        "observed_named_baselines": observed,
        "ratio_vs_random": {
            "agentdebug_top1_over_random": round(observed["agentdebug"]["top1"] / pooled_top1, 3),
            "famas_top1_over_random": round(observed["famas"]["top1"] / pooled_top1, 3),
            "agentdebug_topk_over_random": round(observed["agentdebug"]["topk"] / pooled_topk, 3),
            "famas_topk_over_random": round(observed["famas"]["topk"] / pooled_topk, 3),
        },
        "cells": rows,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
