"""experiments/exp_arcs/phase0_diff — Phase 0 equivalence comparator (PHASE0_PROTOCOL v1 + R1 + R2).

Compares frozen result JSONs with re-derived ones under the frozen protocol:
  * bool / str / None / int-vs-int: exact; any float involved: |a-b| <= 1e-9*max(1,|a|,|b|); NaN == NaN
  * dict: key sets must match; list: same length, compared IN ORDER (no sort keys registered)
  * ignorable (R1.1): ONLY top-level $.generated_at, $.timestamp, $.run_id
  * N1 (R1.3): `$.source` of the 9 registered files -> repo-relative path before comparing
  * N3 (R1.2): any key ending in `_dir` whose value is a bundle path -> (basename, bundle_content_hash)
    looked up in audit_inventory_v1.json; unknown bundle -> raw comparison
  * every difference is written (no truncation) to <out_dir>/<target>.diffs.jsonl

Verdicts: MATCH | MATCH-NORMALIZED (differences only removed by N1/N3/ignore) | MISMATCH | MISSING.

    python experiments/exp_arcs/phase0_diff.py --frozen DIR --rederived DIR --registry PHASE0_COMMANDS.tsv \
        --inventory audit_inventory_v1.json --repo-roots ROOT1,ROOT2 --out-dir DIR [--label A]
    python experiments/exp_arcs/phase0_diff.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REL_TOL = 1e-9
TOP_LEVEL_IGNORABLE = {"generated_at", "timestamp", "run_id"}
N1_SOURCE_FILES = {
    "e3_arcs_v1.json", "e3_detector_v1.json", "e3_localizers_v1.json", "e3_oracle_stats_v1.json",
    "e3_pareto_v1.json", "e4_safety_v2.json", "e7_localization_ablation_v1.json",
    "e7a_fixed_topk_v1.json", "graph_crosscheck_e3_v1.json",
}


class Ctx:
    def __init__(self, target: str, repo_roots: list[str], bundle_hash: dict[str, str]):
        self.target = target
        self.roots = sorted({r.rstrip("/") + "/" for r in repo_roots if r}, key=len, reverse=True)
        self.bundle_hash = bundle_hash
        self.diffs: list[dict] = []
        self.normalized: list[dict] = []

    def rel(self, p: str) -> str:
        for r in self.roots:
            if p.startswith(r):
                return p[len(r):]
        return p

    def bundle_key(self, p: str):
        name = Path(p).name
        h = self.bundle_hash.get(name)
        return (name, h) if h is not None else None


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _float_eq(a: float, b: float) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= REL_TOL * max(1.0, abs(a), abs(b))


def cmp(a: Any, b: Any, path: str, key: str | None, depth: int, ctx: Ctx) -> None:
    # --- N1: registered top-level $.source
    if depth == 1 and key == "source" and ctx.target in N1_SOURCE_FILES and isinstance(a, str) and isinstance(b, str):
        if a == b:
            return
        if ctx.rel(a) == ctx.rel(b):
            ctx.normalized.append({"path": path, "rule": "N1", "frozen": a, "rederived": b})
        else:
            ctx.diffs.append({"path": path, "kind": "value", "rule": "N1-failed", "frozen": a, "rederived": b})
        return
    # --- N3: *_dir bundle paths
    if key is not None and key.endswith("_dir") and isinstance(a, str) and isinstance(b, str):
        if a == b:
            return
        ka, kb = ctx.bundle_key(a), ctx.bundle_key(b)
        if ka is not None and ka == kb:
            ctx.normalized.append({"path": path, "rule": "N3", "frozen": a, "rederived": b})
        else:
            ctx.diffs.append({"path": path, "kind": "value", "rule": "N3-failed", "frozen": a, "rederived": b,
                              "frozen_key": ka, "rederived_key": kb})
        return
    # --- structural comparison
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a), set(b)
        for k in sorted(ka ^ kb, key=str):
            p = f"{path}.{k}"
            entry = {"path": p, "kind": "key_set", "in_frozen": k in ka, "in_rederived": k in kb}
            if depth == 0 and k in TOP_LEVEL_IGNORABLE:
                ctx.normalized.append({**entry, "rule": "ignore-top-level"})
            else:
                ctx.diffs.append(entry)
        for k in sorted(ka & kb, key=str):
            p = f"{path}.{k}"
            if depth == 0 and k in TOP_LEVEL_IGNORABLE:
                if a[k] != b[k]:
                    ctx.normalized.append({"path": p, "rule": "ignore-top-level", "frozen": a[k], "rederived": b[k]})
                continue
            cmp(a[k], b[k], p, k, depth + 1, ctx)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            ctx.diffs.append({"path": path, "kind": "len", "frozen": len(a), "rederived": len(b)})
            return
        for i, (x, y) in enumerate(zip(a, b)):
            cmp(x, y, f"{path}[{i}]", None, depth + 1, ctx)
        return
    if _is_num(a) and _is_num(b):
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                ctx.diffs.append({"path": path, "kind": "int", "frozen": a, "rederived": b})
        elif not _float_eq(float(a), float(b)):
            ctx.diffs.append({"path": path, "kind": "float", "frozen": a, "rederived": b,
                              "delta": float(b) - float(a)})
        return
    if type(a) is not type(b) or a != b:
        ctx.diffs.append({"path": path, "kind": "value" if type(a) is type(b) else "type",
                          "frozen": a, "rederived": b})


def compare_docs(target: str, frozen: Any, rederived: Any, roots: list[str], bundle_hash: dict) -> dict:
    ctx = Ctx(target, roots, bundle_hash)
    cmp(frozen, rederived, "$", None, 0, ctx)
    verdict = "MISMATCH" if ctx.diffs else ("MATCH-NORMALIZED" if ctx.normalized else "MATCH")
    return {"verdict": verdict, "diffs": ctx.diffs, "normalized": ctx.normalized}


def load_registry(path: Path) -> list[dict]:
    lines = [l for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    header = lines[0].split("\t")
    rows = [dict(zip(header, l.split("\t"))) for l in lines[1:]]
    return [r for r in rows if r["status"] == "TARGET"]


def run(args) -> int:
    inv = json.loads(Path(args.inventory).read_text())
    bundle_hash = {name: b["bundle_content_hash"] for name, b in inv["bundles"].items()}
    roots = [r for r in args.repo_roots.split(",") if r]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)   # never overwrite a previous comparison
    summary = {"label": args.label, "frozen_dir": args.frozen, "rederived_dir": args.rederived,
               "registry": args.registry, "files": {}}
    for row in load_registry(Path(args.registry)):
        t = row["target"]
        f, r = Path(args.frozen) / t, Path(args.rederived) / t
        if not r.exists():
            summary["files"][t] = {"verdict": "MISSING", "layer": row["layer"]}
            continue
        res = compare_docs(t, json.loads(f.read_text()), json.loads(r.read_text()), roots, bundle_hash)
        with (out_dir / f"{t}.diffs.jsonl").open("w") as fh:
            for d in res["diffs"]:
                fh.write(json.dumps(d, sort_keys=True, default=str) + "\n")
        with (out_dir / f"{t}.normalized.jsonl").open("w") as fh:
            for d in res["normalized"]:
                fh.write(json.dumps(d, sort_keys=True, default=str) + "\n")
        summary["files"][t] = {"verdict": res["verdict"], "layer": row["layer"],
                               "n_diffs": len(res["diffs"]), "n_normalized": len(res["normalized"]),
                               "first_diff_paths": [d["path"] for d in res["diffs"][:10]]}
    verdicts = [v["verdict"] for v in summary["files"].values()]
    summary["counts"] = {k: verdicts.count(k) for k in ["MATCH", "MATCH-NORMALIZED", "MISMATCH", "MISSING"]}
    summary["n_targets"] = len(verdicts)
    (out_dir / "reproduce_equivalence_v1.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps(summary["counts"]), "n_targets", len(verdicts))
    return 0 if summary["counts"]["MISMATCH"] == 0 and summary["counts"]["MISSING"] == 0 else 1


# ---------------------------------------------------------------- self-test
def selftest() -> int:
    R1, R2 = "/repo/dymam", "/shadow/x"
    bh = {"run_A": "h1", "run_B": "h2"}
    T = "e3_detector_v1.json"
    cases = [
        ("identical", T, {"a": 1, "b": [1.0, "x", None, True]}, {"a": 1, "b": [1.0, "x", None, True]}, "MATCH"),
        ("float within tol", T, {"x": 0.30000000000000004}, {"x": 0.3}, "MATCH"),
        ("float beyond tol", T, {"x": 0.806}, {"x": 0.8056}, "MISMATCH"),
        ("int differs", T, {"n": 180}, {"n": 179}, "MISMATCH"),
        ("int vs float equal value", T, {"n": 1}, {"n": 1.0}, "MATCH"),
        ("bool vs int", T, {"s": True}, {"s": 1}, "MISMATCH"),
        ("key added", T, {"a": 1}, {"a": 1, "b": 2}, "MISMATCH"),
        ("list reordered", T, {"l": [1, 2]}, {"l": [2, 1]}, "MISMATCH"),
        ("list length", T, {"l": [1, 2]}, {"l": [1]}, "MISMATCH"),
        ("NaN both", T, {"x": float("nan")}, {"x": float("nan")}, "MATCH"),
        ("N1 abs vs rel (registered)", "graph_crosscheck_e3_v1.json",
         {"source": R1 + "/experiments/exp_arcs/e3_grid_v1.json"},
         {"source": "experiments/exp_arcs/e3_grid_v1.json"}, "MATCH-NORMALIZED"),
        ("N1 shadow vs repo prefix", T, {"source": R1 + "/experiments/exp_arcs/e3_grid_v1.json"},
         {"source": R2 + "/experiments/exp_arcs/e3_grid_v1.json"}, "MATCH-NORMALIZED"),
        ("N1 different file", T, {"source": "experiments/exp_arcs/e3_grid_v1.json"},
         {"source": "experiments/exp_arcs/local_oracle_all_v1.json"}, "MISMATCH"),
        ("N1 unregistered file", "e8_delta_f1_v1.json", {"source": R1 + "/a.json"}, {"source": "a.json"}, "MISMATCH"),
        ("N1 nested source not covered", T, {"c": {"source": R1 + "/a.json"}}, {"c": {"source": "a.json"}}, "MISMATCH"),
        ("N3 same bundle, other prefix", T, {"cells": [{"clean_dir": R1 + "/fb/run_A"}]},
         {"cells": [{"clean_dir": R2 + "/fb/run_A"}]}, "MATCH-NORMALIZED"),
        ("N3 different bundle", T, {"cells": [{"faulted_dir": "fb/run_A"}]},
         {"cells": [{"faulted_dir": "fb/run_B"}]}, "MISMATCH"),
        ("N3 unknown bundle", T, {"x_dir": "/a/run_Z"}, {"x_dir": "/b/run_Z"}, "MISMATCH"),
        ("top-level timestamp ignored", T, {"timestamp": "t1", "a": 1}, {"timestamp": "t2", "a": 1}, "MATCH-NORMALIZED"),
        ("top-level run_id only one side", T, {"run_id": "r", "a": 1}, {"a": 1}, "MATCH-NORMALIZED"),
        ("nested timestamp NOT ignored", T, {"c": {"timestamp": "t1"}}, {"c": {"timestamp": "t2"}}, "MISMATCH"),
        ("type change str vs num", T, {"x": "1"}, {"x": 1}, "MISMATCH"),
    ]
    fails = 0
    for name, tgt, a, b, want in cases:
        got = compare_docs(tgt, a, b, [R1, R2], bh)["verdict"]
        ok = got == want
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: want {want}, got {got}")
    print(f"selftest: {len(cases) - fails}/{len(cases)} passed")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--frozen"); ap.add_argument("--rederived"); ap.add_argument("--registry")
    ap.add_argument("--inventory"); ap.add_argument("--repo-roots", default="")
    ap.add_argument("--out-dir"); ap.add_argument("--label", default="")
    args = ap.parse_args()
    return selftest() if args.selftest else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
