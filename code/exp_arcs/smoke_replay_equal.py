"""experiments/exp_arcs/smoke_replay_equal — Phase 0.2 smoke (all record slices).

Health check (NOT a paper result — that is Phase 0.3). For one org it verifies:
(1) two CLEAN replays are ``replay_equal`` (the bridge round-trips, ids stable);
(2) a live fault perturbs the recorded bundle (the fault fires) yet its own two
replays are still ``replay_equal`` (a faulted trajectory reproduces).

Plan deviations (documented, honest): the plan asked for fault F6 and the
``phase7_replay_v1/pipeline.py`` driver. F6 is a record-layer, id-invariant
validator fault (``mechanism_v1.fault_suite``) that ``replay_equal`` cannot see,
so a live fault (default F8) is used instead. ``phase7_replay_v1/pipeline.py`` is
hard-coupled to the frozen phase7 design (15 orgs / 60 cells / frozen protocol)
and cannot drive arbitrary recon bundles; ``record_run`` + ``replay_equal`` is
the equivalent driver for ARCS bundles.

    poetry run python -m experiments.exp_arcs.smoke_replay_equal [--org mozilla] [--raw-root DIR]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from core.graph import run_pipeline
from experiments.exp_arcs.interceptors import inject
from experiments.exp_arcs.recorder import record_run
from experiments.exp_arcs.replay_equal import replay_equal

_RECORD_FILES = (
    "source_artifact.jsonl", "evidence_event.jsonl", "claim.jsonl",
    "decision.jsonl", "report_item.jsonl",
)


def _rec(org: str, condition: str | None, raw_root: Path) -> Path:
    with inject(condition):
        return record_run(run_pipeline(organization=org), raw_root=raw_root)


def _counts(run_dir: Path) -> str:
    def n(name: str) -> int:
        p = run_dir / name
        return sum(1 for ln in p.read_text().splitlines() if ln.strip()) if p.exists() else 0
    return ", ".join(f"{f.split('.')[0]}={n(f)}" for f in _RECORD_FILES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org", default="mozilla")
    parser.add_argument("--fault", default="F8",
                        help="live fault to demonstrate (plan asked for F6, but "
                             "F6 is a record-layer id-invariant fault invisible "
                             "to replay_equal; a live fault is used instead)")
    parser.add_argument("--raw-root", default=None)
    args = parser.parse_args(argv)

    raw_root = Path(args.raw_root) if args.raw_root else Path(
        tempfile.mkdtemp(prefix="arcs_smoke_"))
    print(f"raw_root: {raw_root}")

    # (1) clean determinism: two clean replays must be replay-equal.
    clean_a = _rec(args.org, None, raw_root)
    clean_b = _rec(args.org, None, raw_root)
    print(f"[clean A] {clean_a.name}: {_counts(clean_a)}")
    print(f"[clean B] {clean_b.name}: {_counts(clean_b)}")
    clean_equal, clean_diffs = replay_equal(clean_a, clean_b)

    # (2) fault visibility + determinism: a faulted trajectory must differ from
    #     clean (the fault fires) yet still reproduce across its own replays.
    fault_a = _rec(args.org, args.fault, raw_root)
    fault_b = _rec(args.org, args.fault, raw_root)
    print(f"[{args.fault} A] {fault_a.name}: {_counts(fault_a)}")
    print(f"[{args.fault} B] {fault_b.name}: {_counts(fault_b)}")
    fault_fires = not replay_equal(clean_a, fault_a)[0]
    fault_determ, fault_diffs = replay_equal(fault_a, fault_b)

    print("-" * 60)
    ok = clean_equal and fault_fires and fault_determ
    print(f"clean replays equal   : {clean_equal}")
    print(f"{args.fault} perturbs bundle : {fault_fires}")
    print(f"{args.fault} replays equal   : {fault_determ}")
    if ok:
        print("PASS: clean is deterministic; the fault is visible and reproducible")
        return 0
    print("FAIL:")
    for label, diffs in (("clean", clean_diffs), (args.fault, fault_diffs)):
        for d in diffs:
            print(f"  [{label}/{d.record_type}] {d.kind}: {d.detail}")
    if not fault_fires:
        print(f"  {args.fault} did not perturb the bundle (fault is a no-op)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
