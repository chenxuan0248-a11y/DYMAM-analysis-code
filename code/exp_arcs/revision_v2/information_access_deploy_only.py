"""Recompute deployable score hashes while denying reads from paired-clean bundles."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from information_access_ablation import D, TIERS, scores, view


def main(result_path: Path, out: Path | None = None) -> None:
    result = json.loads(result_path.read_text())
    grid = json.loads((D / "e3_grid_v1.json").read_text())["cells"]
    clean_roots = {Path(c["clean_dir"]).resolve() for c in grid}
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes
    blocked = 0
    current_tier = "records"

    def deny(path: Path) -> None:
        nonlocal blocked
        p = path.resolve()
        if any(p == root or root in p.parents for root in clean_roots):
            blocked += 1
            raise AssertionError(f"deploy-only path attempted paired-clean read: {p}")
        if current_tier in ("records", "schema") and p.name == "discovery_manifest.json":
            blocked += 1
            raise AssertionError(f"{current_tier} attempted manifest read: {p}")

    def guarded_text(path: Path, *args, **kwargs):
        deny(path)
        return original_read_text(path, *args, **kwargs)

    def guarded_bytes(path: Path, *args, **kwargs):
        deny(path)
        return original_read_bytes(path, *args, **kwargs)

    Path.read_text = guarded_text
    Path.read_bytes = guarded_bytes
    checks = 0
    try:
        for t in TIERS[:3]:
            current_tier = t
            expected = {(r["org"], r["cell"]): r["score_sha256"] for r in result["cells"][t]}
            for c in grid:
                fv = view(c["faulted_dir"], t == "manifest")
                sc = scores(fv, t)
                digest = hashlib.sha256(json.dumps(sorted(sc.items())).encode()).hexdigest()
                assert digest == expected[(c["org"], c["cell"])], (t, c["org"], c["cell"])
                checks += 1
    finally:
        Path.read_text = original_read_text
        Path.read_bytes = original_read_bytes
    assert checks == 540 and blocked == 0
    verdict = {"score_hashes_matched": checks, "forbidden_read_attempts": blocked,
               "tiers": list(TIERS[:3]),
               "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if out is not None:
        out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: information_access_deploy_only.py RESULT.json [OUT.json]")
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) == 3 else None)
