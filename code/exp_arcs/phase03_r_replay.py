"""experiments/exp_arcs/phase03_r_replay — Phase 0.3 S1b replay robustness (G1).

Measures ``R_replay`` = fraction of replays that reproduce the reference run
exactly, over ``orgs × replays``. "Exactly" is the plan's structural definition
(event ids, event order, normalized inputs, decision order) — implemented by
``replay_equal`` over the recorded infra_v1 bundle, which is content-addressed,
so it ignores the volatile run_id and wall-clock while catching any substantive
divergence.

    poetry run python -m experiments.exp_arcs.phase03_r_replay \
        --orgs mozilla,cloudflare,wikimedia --replays 3 [--out result.json]

GATE G1: R_replay ≥ 0.95. Failure means the record-replay harness leaks a
non-deterministic call — stop and fix the harness before building on it.

Fault dimension: each condition is ``clean`` plus the Phase 1.1 live faults
(``experiments/exp_arcs/interceptors.py``: F2 wrong-tool / F8 gate-bypass / F10
premature-termination), injected into the running pipeline so the recorded
bundle actually differs from clean. A faulted cell's R_replay tests that the
*faulted* trajectory reproduces across replays — the real robustness question.
(The record-layer ``mechanism_v1.fault_suite.FAULTS`` are NOT usable here: they
tamper a recorded bundle while leaving content-addressed ids intact, so
``replay_equal`` cannot see them.)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.graph import run_pipeline
from evaluation.stat_test import bootstrap_ci
from experiments.exp_arcs.faults import LIVE_FAULTS
from experiments.exp_arcs.interceptors import inject
from experiments.exp_arcs.recorder import record_run
from experiments.exp_arcs.replay_equal import replay_equal

DEFAULT_ORGS = ("mozilla", "cloudflare", "wikimedia")
G1_THRESHOLD = 0.95


def _record_replays(org: str, condition: str, replays: int, raw_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for r in range(replays):
        with inject(None if condition == "clean" else condition):
            state = run_pipeline(organization=org)
        dirs.append(record_run(state, raw_root=raw_root))
    return dirs


def _score_cell(dirs: list[Path]) -> tuple[int, int, list[dict[str, Any]]]:
    """Exact replays / comparisons for one (org, condition) cell vs replay 0."""
    ref = dirs[0]
    exact = 0
    mismatches: list[dict[str, Any]] = []
    for r in range(1, len(dirs)):
        equal, diffs = replay_equal(ref, dirs[r])
        if equal:
            exact += 1
        else:
            mismatches.append({
                "replay": r,
                "diffs": [f"[{d.record_type}] {d.kind}: {d.detail}" for d in diffs],
            })
    return exact, len(dirs) - 1, mismatches


def measure(
    orgs: list[str], conditions: list[str], replays: int, raw_root: Path,
    *, B: int, seed: int,
) -> dict[str, Any]:
    """R_replay over orgs × conditions × replays (each cell scored vs its replay 0).

    A faulted cell tests that the *faulted* trajectory reproduces across replays
    — the real Phase 0.3 robustness question. The org is the clustering unit
    (pooled across its conditions) for the bootstrap CI, per the plan.
    """
    cells: list[dict[str, Any]] = []
    per_org_pool: dict[str, list[int]] = {}     # org -> [exact, cmp]
    per_condition: dict[str, list[int]] = {c: [0, 0] for c in conditions}
    total_exact = total_cmp = 0
    scored_orgs: set[str] = set()

    for org in orgs:
        print(f"  {org}:")
        for cond in conditions:
            try:
                dirs = _record_replays(org, cond, replays, raw_root)
            except Exception as exc:  # noqa: BLE001 — one bad cell shouldn't kill the run
                print(f"    {cond}: SKIPPED ({type(exc).__name__}: {exc})")
                cells.append({"org": org, "condition": cond, "status": "skipped",
                              "error": str(exc)})
                continue
            exact, n_cmp, mism = _score_cell(dirs)
            total_exact += exact
            total_cmp += n_cmp
            per_org_pool.setdefault(org, [0, 0])
            per_org_pool[org][0] += exact
            per_org_pool[org][1] += n_cmp
            per_condition[cond][0] += exact
            per_condition[cond][1] += n_cmp
            scored_orgs.add(org)
            cells.append({"org": org, "condition": cond, "status": "ok",
                          "exact": exact, "comparisons": n_cmp,
                          "r_replay": exact / n_cmp if n_cmp else 1.0,
                          "mismatches": mism})
            print(f"    {cond:6s} R_replay = {exact}/{n_cmp}")

    org_scores = [e / c for e, c in per_org_pool.values() if c]
    r_replay = (total_exact / total_cmp) if total_cmp else 0.0
    mean, lo, hi = bootstrap_ci(org_scores, B=B, seed=seed) if org_scores else (0.0, 0.0, 0.0)
    passed = len(scored_orgs) >= 3 and len(conditions) >= 3 and r_replay >= G1_THRESHOLD

    return {
        "phase": "0.3_S1b_replay_robustness",
        "conditions": conditions,
        "orgs_scored": sorted(scored_orgs),
        "replays_per_condition": replays,
        "r_replay": r_replay,
        "r_replay_org_clustered_ci": {"mean": mean, "ci_low": lo, "ci_high": hi,
                                      "B": B, "alpha": 0.05, "seed": seed},
        "total_exact": total_exact,
        "total_comparisons": total_cmp,
        "per_condition_r_replay": {
            c: {"exact": e, "comparisons": n, "r_replay": (e / n if n else None)}
            for c, (e, n) in per_condition.items()},
        "g1_threshold": G1_THRESHOLD,
        "g1_pass": passed,
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orgs", default=",".join(DEFAULT_ORGS),
                        help="comma-separated org list (≥3 for a valid G1 verdict)")
    parser.add_argument("--faults", default=",".join(LIVE_FAULTS),
                        help="live faults to add as conditions (≥3 for the plan's "
                             "fault dimension); 'none' for clean-only")
    parser.add_argument("--replays", type=int, default=3, help="replays per cell (≥3)")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--out", default=None, help="write result JSON here")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    orgs = [o.strip() for o in args.orgs.split(",") if o.strip()]
    faults = [] if args.faults.strip().lower() in ("", "none") else \
        [f.strip() for f in args.faults.split(",") if f.strip()]
    conditions = ["clean"] + faults
    raw_root = Path(args.raw_root) if args.raw_root else Path(
        tempfile.mkdtemp(prefix="arcs_phase03_"))
    print(f"raw_root: {raw_root}\nconditions: {conditions}")

    result = measure(orgs, conditions, args.replays, raw_root,
                     B=args.bootstrap, seed=args.seed)

    print("=" * 64)
    ci = result["r_replay_org_clustered_ci"]
    print(f"R_replay = {result['r_replay']:.4f} "
          f"({result['total_exact']}/{result['total_comparisons']}) | "
          f"org-clustered 95% CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")
    print(f"  orgs scored = {len(result['orgs_scored'])}, conditions = {conditions}")
    for c, d in result["per_condition_r_replay"].items():
        print(f"    {c:6s} R_replay = {d['exact']}/{d['comparisons']}")
    verdict = "PASS" if result["g1_pass"] else "FAIL"
    print(f"GATE G1 (R_replay ≥ {G1_THRESHOLD}, ≥3 orgs, ≥3 conditions): {verdict}")
    if not result["g1_pass"]:
        for c in result["cells"]:
            for m in c.get("mismatches", []):
                print(f"  [{c['org']}/{c['condition']}] replay {m['replay']}: {m['diffs']}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"wrote {args.out}")

    return 0 if result["g1_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
