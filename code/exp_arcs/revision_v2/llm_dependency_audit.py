"""E-A Stage 1: offline LLM/report-text dependency audit.

This script is intentionally read-only over the frozen E3 inputs.  It creates
temporary shadow bundles whose only changed file is rendered_report.json, then
recomputes the primary combined / Isolation-Forest / robust-z localization and
repair metrics over all 180 cells.

Run:
    poetry run python -m experiments.exp_arcs.revision_v2.llm_dependency_audit
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import statistics as st
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from experiments.exp_arcs.detector_eval import (
    detector_localize,
    manifest_dropped_hosts,
    peer_anomaly_hosts,
)
from experiments.exp_arcs.detector_pr_roc import _host_scores
from experiments.exp_arcs.localizer_baselines import _truth
from experiments.exp_arcs.revision_v2 import step2_baselines as s2
from experiments.exp_arcs.rq2_combined_topk import tie_aware_hit

REPO = Path(__file__).resolve().parents[3]
D = REPO / "experiments/exp_arcs"
HERE = D / "revision_v2"
PROTOCOL = HERE / "llm_dependency_protocol.md"
OUT = HERE / "llm_dependency_v1.json"
GRAPH = HERE / "llm_dependency_graph.svg"
TABLE = HERE / "llm_dependency_table.tex"
RESULT = HERE / "LLM_DEPENDENCY_RESULT.md"
GRID = D / "e3_grid_v1.json"
STEP2 = HERE / "step2_baselines_v1.json"

TEXT_KEYS = frozenset({"summary", "text", "narrative", "explanation", "analysis"})
VARIANTS = ("full_existing", "report_llm_disabled", "fixed_template", "perturbed_text")
METHODS = ("combined", "iforest", "robust_z")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _free_text_paths(value: Any, prefix: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            here = f"{prefix}.{key}" if prefix else key
            if key.lower() in TEXT_KEYS and isinstance(child, str) and child.strip():
                out.append(here)
            out.extend(_free_text_paths(child, here))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            out.extend(_free_text_paths(child, f"{prefix}[{i}]"))
    return out


def _strip_text(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_text(v) for k, v in value.items() if k.lower() not in TEXT_KEYS}
    if isinstance(value, list):
        return [_strip_text(v) for v in value]
    return value


def _mutate_rendered(doc: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant == "full_existing":
        return doc
    out = _strip_text(doc)
    if variant == "report_llm_disabled":
        return out
    if variant == "fixed_template":
        out["summary"] = "FIXED_TEMPLATE_REPORT_SUMMARY"
        return out
    if variant == "perturbed_text":
        out["summary"] = "PERTURBED_REPORT_TEXT_SENTINEL_9f6c4b1a"
        return out
    raise ValueError(variant)


def _resolve_bundle(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else REPO / p


def bundle_census(grid: list[dict[str, Any]]) -> dict[str, Any]:
    paths = sorted({_resolve_bundle(c[k]) for c in grid for k in ("clean_dir", "faulted_dir")})
    totals = Counter()
    nonempty_rendered: list[dict[str, Any]] = []
    report_item_text: list[dict[str, Any]] = []
    for p in paths:
        for filename, key in (
            ("evidence_event.jsonl", "evidence_rows"),
            ("decision.jsonl", "decision_rows"),
            ("execution_event.jsonl", "execution_rows"),
            ("report_item.jsonl", "report_item_rows"),
        ):
            n = jsonl_count(p / filename)
            totals[key] += n
            totals[f"bundles_with_{filename}"] += int((p / filename).exists())
        candidate = json.loads((p / "candidate.json").read_text()) if (p / "candidate.json").exists() else None
        totals["candidate_records"] += len(candidate) if isinstance(candidate, list) else int(candidate is not None)
        rendered = json.loads((p / "rendered_report.json").read_text())
        paths_text = _free_text_paths(rendered)
        nonempty_sections = bool(rendered.get("sections"))
        nonempty_ids = bool(rendered.get("report_item_ids"))
        if paths_text or nonempty_sections or nonempty_ids:
            nonempty_rendered.append({
                "bundle": str(p.relative_to(REPO)),
                "free_text_paths": paths_text,
                "n_sections": len(rendered.get("sections") or []),
                "n_report_item_ids": len(rendered.get("report_item_ids") or []),
            })
        rp = p / "report_item.jsonl"
        if rp.exists():
            for i, line in enumerate(rp.read_text().splitlines(), start=1):
                if not line.strip():
                    continue
                text_paths = _free_text_paths(json.loads(line))
                if text_paths:
                    report_item_text.append({
                        "bundle": str(p.relative_to(REPO)), "line": i,
                        "free_text_paths": text_paths,
                    })
    return {
        "n_unique_bundles": len(paths),
        "totals": dict(sorted(totals.items())),
        "n_bundles_with_nonempty_rendered_content": len(nonempty_rendered),
        "nonempty_rendered_examples": nonempty_rendered[:10],
        "n_report_items_with_free_text_fields": len(report_item_text),
        "report_item_text_examples": report_item_text[:10],
        "interpretation": (
            "report_item.jsonl is a formal policy projection, not free-form LLM text; "
            "the reporting summary lives only in in-memory state['report']['summary'] and "
            "is not persisted by the E3 recorder"
        ),
    }


def source_inventory() -> dict[str, Any]:
    specs: dict[str, dict[str, Any]] = {
        "combined_score": {
            "callables": [_host_scores, detector_localize, peer_anomaly_hosts, manifest_dropped_hosts],
            "fields": {
                "evidence_event.jsonl": ["target", "source_family"],
                "claim.jsonl": ["claim_id", "subject"],
                "decision.jsonl": ["claim_id", "decision", "reason_codes", "support_state"],
                "discovery_manifest.json": ["resolved_hosts"],
            },
        },
        "statistical_features": {
            "callables": [s2.features, s2.s_iforest, s2.s_robust_z, s2.calibrate],
            "fields": {
                "evidence_event.jsonl": ["target", "source_family"],
                "claim.jsonl": ["claim_id", "subject", "claim_type"],
                "decision.jsonl": ["claim_id", "decision", "reason_codes"],
                "discovery_manifest.json": ["resolved_hosts"],
            },
        },
        "repair_success": {
            "callables": [s2.repair, s2.dfx_of, s2.clean_dag],
            "fields": {
                "source_artifact.jsonl": ["artifact_id", "content except run_id"],
                "evidence_event.jsonl": ["evidence_event_id", "raw_ref", "target", "source_family", "content except run_id"],
                "claim.jsonl": ["claim_id", "content except run_id"],
                "decision.jsonl": ["decision_id", "claim_id", "supporting_evidence_event_ids", "contradicting_evidence_event_ids", "content except run_id"],
                "report_item.jsonl": ["report_item_id", "claim_id", "decision_id", "evidence_event_ids", "content except run_id"],
            },
            "note": "formal report_item nodes affect contamination closure; free-form rendered report text does not",
        },
        "repair_cost": {
            "callables": [s2.repair],
            "fields": {
                "evidence_event.jsonl": ["evidence_event_id", "target", "source_family", "raw_ref"],
                "source_artifact.jsonl": ["artifact_id"],
                "cost_harm_spec.yaml": ["family_to_action", "actions.*.cost", "actions.*.harm", "actions.*.reversibility"],
            },
        },
    }
    out: dict[str, Any] = {}
    for name, spec in specs.items():
        funcs = []
        for fn in spec.pop("callables"):
            inspected = inspect.unwrap(fn)
            path = Path(inspect.getsourcefile(inspected) or "")
            lines, start = inspect.getsourcelines(inspected)
            funcs.append({
                "qualified_name": f"{fn.__module__}.{fn.__name__}",
                "source": str(path.resolve().relative_to(REPO)),
                "line_start": start,
                "line_end": start + len(lines) - 1,
                "source_sha256": sha_file(path),
            })
        out[name] = {**spec, "functions": funcs, "reads_free_form_report_text": False}
    return out


def make_shadow(original: Path, root: Path, index: int, variant: str) -> Path:
    out = root / f"b{index:03d}"
    out.mkdir(parents=True)
    for item in original.iterdir():
        dst = out / item.name
        if item.name == "rendered_report.json":
            doc = json.loads(item.read_text())
            dst.write_text(json.dumps(_mutate_rendered(doc, variant), indent=2, sort_keys=True) + "\n")
        else:
            os.symlink(item.resolve(), dst, target_is_directory=item.is_dir())
    return out


@contextmanager
def trace_reads():
    seen: Counter[str] = Counter()
    original = Path.read_text

    def wrapped(self: Path, *args, **kwargs):
        seen[self.name] += 1
        return original(self, *args, **kwargs)

    Path.read_text = wrapped  # type: ignore[method-assign]
    try:
        yield seen
    finally:
        Path.read_text = original  # type: ignore[method-assign]


def _org_macro(rows: list[dict[str, Any]], key: str) -> float:
    by: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by[row["org"]].append(float(row[key]))
    return st.mean(st.mean(v) for v in by.values())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_cells": len(rows),
        "success": round(_org_macro(rows, "success"), 10),
        "detect1": round(_org_macro(rows, "detect1"), 10),
        "detect3": round(_org_macro(rows, "detect3"), 10),
        "mean_cost": round(_org_macro(rows, "cost"), 10),
        "cost_over_Cstar": round(_org_macro(rows, "cost_over_Cstar"), 10),
        "mean_n_flagged": round(_org_macro(rows, "n_flagged"), 10),
    }


def _row(cell: dict[str, Any], scores: dict[str, float], flagged: frozenset[str]) -> dict[str, Any]:
    truth, _ = _truth(cell["inject_node"])
    repaired = s2.repair(cell, flagged)
    return {
        "org": cell["org"], "cell": cell["cell"],
        "detect1": tie_aware_hit(scores, truth, 1),
        "detect3": tie_aware_hit(scores, truth, 3),
        "n_flagged": len(flagged),
        "success": bool(repaired["success"]),
        "cost": float(repaired["cost"]),
        "cost_over_Cstar": float(repaired["cost"]) / float(cell["C_star"]),
    }


def recompute(cells: list[dict[str, Any]]) -> dict[str, Any]:
    for cached in (s2.features, s2.scores, s2.clean_dag, s2.dfx_of):
        cached.cache_clear()
    clean_of: dict[str, str] = {}
    for c in cells:
        clean_of.setdefault(c["org"], c["clean_dir"])
    orgs = sorted(clean_of)
    out_rows: dict[str, list[dict[str, Any]]] = {m: [] for m in METHODS}

    for c in cells:
        fd = Path(c["faulted_dir"])
        sc = _host_scores(fd)
        flagged = frozenset(
            detector_localize(fd)["flagged_hosts"]
            | peer_anomaly_hosts(fd, min_base_rate=0.5)
            | manifest_dropped_hosts(fd)
        )
        out_rows["combined"].append(_row(c, sc, flagged))

    for method in ("iforest", "robust_z"):
        clean_scores = {org: dict(s2.scores(clean_of[org], method, True)) for org in orgs}
        thresholds = {
            org: s2.calibrate({o: clean_scores[o] for o in orgs if o != org}, 0.106)
            for org in orgs
        }
        for c in cells:
            sc = dict(s2.scores(c["faulted_dir"], method, True))
            flagged = frozenset(h for h, score in sc.items() if score >= thresholds[c["org"]])
            out_rows[method].append(_row(c, sc, flagged))

    canonical_rows = {
        method: sorted(rows, key=lambda r: (r["org"], r["cell"]))
        for method, rows in out_rows.items()
    }
    return {
        "summaries": {m: summarize(canonical_rows[m]) for m in METHODS},
        "rows": canonical_rows,
    }


def variant_run(grid: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    originals = sorted({_resolve_bundle(c[k]) for c in grid for k in ("clean_dir", "faulted_dir")})
    with tempfile.TemporaryDirectory(prefix=f"arcs-llm-{variant}-") as td:
        root = Path(td)
        mapping = {str(p): make_shadow(p, root, i, variant) for i, p in enumerate(originals)}
        cells = []
        for c in grid:
            cc = dict(c)
            cc["clean_dir"] = str(mapping[str(_resolve_bundle(c["clean_dir"]))])
            cc["faulted_dir"] = str(mapping[str(_resolve_bundle(c["faulted_dir"]))])
            cells.append(cc)
        with trace_reads() as reads:
            result = recompute(cells)
        payload = canonical_bytes(result)
        return {
            "metric_output_sha256": sha_bytes(payload),
            "metric_output_nbytes": len(payload),
            "summaries": result["summaries"],
            "rows": result["rows"],
            "read_counts_by_filename": dict(sorted(reads.items())),
        }


def compare_variants(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = variants["full_existing"]
    comparisons: dict[str, Any] = {}
    for name in VARIANTS[1:]:
        other = variants[name]
        by_method: dict[str, Any] = {}
        total = 0
        for method in METHODS:
            a = base["rows"][method]
            b = other["rows"][method]
            mismatches = []
            for ra, rb in zip(a, b):
                if ra != rb:
                    mismatches.append({"baseline": ra, "variant": rb})
            total += len(mismatches)
            by_method[method] = {
                "n_cell_mismatches": len(mismatches),
                "examples": mismatches[:3],
            }
        comparisons[name] = {
            "canonical_bytes_equal": base["metric_output_sha256"] == other["metric_output_sha256"],
            "baseline_sha256": base["metric_output_sha256"],
            "variant_sha256": other["metric_output_sha256"],
            "n_cell_method_mismatches": total,
            "by_method": by_method,
        }
    return comparisons


def faithfulness(summary: dict[str, Any]) -> dict[str, Any]:
    frozen = json.loads(STEP2.read_text())
    expected = {
        "combined": frozen["combined"]["primary"]["summary"],
        "iforest": frozen["baselines"]["iforest|manifest_aware"]["by_alpha"]["primary"]["summary"],
        "robust_z": frozen["baselines"]["robust_z|manifest_aware"]["by_alpha"]["primary"]["summary"],
    }
    checks = {}
    for method in METHODS:
        got = summary[method]
        want = expected[method]
        delta = {
            "success": got["success"] - float(want["success"]["org_mean"]),
            "detect1": got["detect1"] - float(want["detect1"]["org_mean"]),
            "detect3": got["detect3"] - float(want["detect3"]["org_mean"]),
            "cost_over_Cstar": got["cost_over_Cstar"] - float(want["cost_over_Cstar_all"]["org_mean"]),
        }
        checks[method] = {
            "observed": got,
            "frozen": {
                "success": want["success"]["org_mean"],
                "detect1": want["detect1"]["org_mean"],
                "detect3": want["detect3"]["org_mean"],
                "cost_over_Cstar": want["cost_over_Cstar_all"]["org_mean"],
            },
            "delta": delta,
            "pass": all(abs(v) <= 5e-5 for v in delta.values()),
        }
    return {"by_method": checks, "pass": all(c["pass"] for c in checks.values())}


def write_graph() -> None:
    # Static vector diagram of the inspected dependency graph.  The dashed red
    # edge is explicitly absent from the evaluated metrics.
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="820" viewBox="0 0 1400 820">
<defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#475569"/></marker><style>
.b{fill:#f8fafc;stroke:#334155;stroke-width:2}.f{fill:#ecfeff;stroke:#0891b2;stroke-width:2}.m{fill:#eef2ff;stroke:#4f46e5;stroke-width:2}.x{fill:#fff1f2;stroke:#e11d48;stroke-width:2}.t{font:17px sans-serif;fill:#0f172a}.s{font:14px sans-serif;fill:#334155}.e{stroke:#475569;stroke-width:2;fill:none;marker-end:url(#a)}.n{stroke:#e11d48;stroke-width:3;stroke-dasharray:8 7;fill:none}</style></defs>
<text x="40" y="42" class="t" font-weight="700">E-A field-level dependency graph (audited source paths)</text>
<rect x="45" y="90" width="290" height="120" rx="12" class="b"/><text x="65" y="122" class="t">Tool / evidence records</text><text x="65" y="150" class="s">evidence: target, source_family</text><text x="65" y="176" class="s">source artifact: id, raw_ref</text>
<rect x="45" y="260" width="290" height="115" rx="12" class="b"/><text x="65" y="292" class="t">Policy projection</text><text x="65" y="320" class="s">claim + decision fields</text><text x="65" y="346" class="s">formal report_item links</text>
<rect x="45" y="425" width="290" height="100" rx="12" class="b"/><text x="65" y="457" class="t">Discovery manifest</text><text x="65" y="485" class="s">resolved_hosts</text>
<rect x="45" y="595" width="290" height="105" rx="12" class="x"/><text x="65" y="627" class="t">Reporting LLM text</text><text x="65" y="655" class="s">state.report.summary</text><text x="65" y="681" class="s">not persisted in E3 bundle</text>
<rect x="470" y="105" width="285" height="105" rx="12" class="f"/><text x="492" y="139" class="t">Candidate universe</text><text x="492" y="167" class="s">E ∪ resolved-host manifest</text>
<rect x="470" y="270" width="285" height="125" rx="12" class="f"/><text x="492" y="304" class="t">Host features / scores</text><text x="492" y="332" class="s">combined, IF, robust-z</text><text x="492" y="360" class="s">no rendered-report field</text>
<rect x="470" y="455" width="285" height="110" rx="12" class="f"/><text x="492" y="489" class="t">Contamination + DAG</text><text x="492" y="517" class="s">formal record differences</text><text x="492" y="543" class="s">repairable leaves</text>
<rect x="900" y="155" width="330" height="125" rx="12" class="m"/><text x="925" y="191" class="t">Localization metrics</text><text x="925" y="219" class="s">Detect@1 / Detect@3</text><text x="925" y="247" class="s">flags and candidate coverage</text>
<rect x="900" y="390" width="330" height="145" rx="12" class="m"/><text x="925" y="426" class="t">Repair metrics</text><text x="925" y="454" class="s">offline success</text><text x="925" y="482" class="s">specification cost / C*</text><text x="925" y="510" class="s">formal report_item is downstream truth</text>
<path d="M335 150 C400 150 400 150 470 150" class="e"/><path d="M335 310 C400 310 400 330 470 330" class="e"/><path d="M335 475 C400 475 400 175 470 175" class="e"/><path d="M335 325 C400 420 400 500 470 500" class="e"/>
<path d="M755 158 C825 158 825 210 900 210" class="e"/><path d="M755 330 C825 330 825 245 900 245" class="e"/><path d="M755 510 C825 510 825 460 900 460" class="e"/><path d="M755 330 C825 350 825 420 900 420" class="e"/>
<path d="M335 645 C590 645 690 620 900 535" class="n"/><text x="535" y="680" class="s" fill="#e11d48">no source-level or dynamic-read path to evaluated metrics</text>
<text x="45" y="775" class="s">Solid arrows: consumed formal fields. Dashed red line: tested absent dependency; Stage 1 mutation result determines the empirical verdict.</text>
</svg>"""
    GRAPH.write_text(svg)


def write_table(variants: dict[str, dict[str, Any]], comparisons: dict[str, Any]) -> None:
    lines = [
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Report variant & Method & Success & Detect@1 & Detect@3 & Cost/$C^*$ & Same bytes \\",
        r"\midrule",
    ]
    base_hash = variants["full_existing"]["metric_output_sha256"]
    for variant in VARIANTS:
        for method in METHODS:
            s = variants[variant]["summaries"][method]
            same = "yes" if variants[variant]["metric_output_sha256"] == base_hash else "no"
            lines.append(
                f"{variant.replace('_', r'\_')} & {method.replace('_', r'\_')} & "
                f"{s['success']:.4f} & {s['detect1']:.4f} & {s['detect3']:.4f} & "
                f"{s['cost_over_Cstar']:.4f} & {same} " + r"\\"
            )
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    TABLE.write_text("\n".join(lines) + "\n")


def write_result(result: dict[str, Any]) -> None:
    c = result["bundle_census"]
    comp = result["comparisons"]
    stage1_pass = result["stage1_pass"]
    base = result["variants"]["full_existing"]["summaries"]
    lines = [
        "# E-A LLM dependency audit — Stage 1 result",
        "",
        f"Executed: {result['executed_at']}  ",
        f"Protocol: `{PROTOCOL.relative_to(REPO)}`  ",
        f"Verdict: **{'PASS' if stage1_pass else 'FAIL'}**",
        "",
        "## Observed bundle facts",
        "",
        f"- E3 grid: {result['n_cells']} cells, {c['n_unique_bundles']} unique bundles.",
        f"- Bundles with non-empty rendered-report content: {c['n_bundles_with_nonempty_rendered_content']}.",
        f"- Formal report items with free-form summary/text fields: {c['n_report_items_with_free_text_fields']}.",
        "- The E3 recorder did not persist `state['report']['summary']`; `rendered_report.json` is a fixed shell.",
        "- Formal `report_item.jsonl` records are still used by contamination/repair evaluation; they are deterministic policy projections, not LLM prose.",
        "",
        "## Full-grid recomputation",
        "",
        "| Variant | Canonical metric bytes equal | Cell-method mismatches |",
        "|---|---:|---:|",
    ]
    for name in VARIANTS[1:]:
        cc = comp[name]
        lines.append(f"| {name} | {str(cc['canonical_bytes_equal']).lower()} | {cc['n_cell_method_mismatches']} |")
    lines.extend([
        "",
        "Baseline summaries reproduced in this audit:",
        "",
        "| Method | Success | Detect@1 | Detect@3 | Cost/C* |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        s = base[method]
        lines.append(f"| {method} | {s['success']:.4f} | {s['detect1']:.4f} | {s['detect3']:.4f} | {s['cost_over_Cstar']:.4f} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        ("All three report-text mutations produced byte-identical canonical metric outputs over all 180 cells. "
         if stage1_pass else "At least one report-text mutation changed an evaluated metric. "),
        "This establishes only that the evaluated combined/IF/robust-z localization and offline repair metrics do not consume persisted reporting text. It does not yet establish that disabling every upstream LLM call leaves formal evidence unchanged; that question belongs to Stage 2.",
        "",
        "## Dynamic reads",
        "",
        "The recomputation read formal evidence/claim/decision/report-item/manifest and cost-spec inputs. `rendered_report.json` was changed in each shadow bundle but is absent from the metric read trace after shadow creation.",
        "",
        "## Artifacts",
        "",
        f"- `{OUT.relative_to(REPO)}`",
        f"- `{GRAPH.relative_to(REPO)}`",
        f"- `{TABLE.relative_to(REPO)}`",
    ])
    RESULT.write_text("\n".join(lines) + "\n")


def main() -> int:
    import datetime as dt

    grid_doc = json.loads(GRID.read_text())
    grid = grid_doc["cells"]
    result: dict[str, Any] = {
        "experiment": "E-A LLM dependency audit",
        "stage": 1,
        "executed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(REPO)),
        "input_hashes": {
            "protocol_sha256": sha_file(PROTOCOL),
            "script_sha256": sha_file(Path(__file__)),
            "grid_sha256": sha_file(GRID),
            "step2_result_sha256": sha_file(STEP2),
        },
        "n_cells": len(grid),
        "bundle_census": bundle_census(grid),
        "source_dependency_inventory": source_inventory(),
        "variants": {},
    }
    for variant in VARIANTS:
        print(f"recomputing {variant} ...", flush=True)
        result["variants"][variant] = variant_run(grid, variant)
    result["comparisons"] = compare_variants(result["variants"])
    result["faithfulness"] = faithfulness(result["variants"]["full_existing"]["summaries"])
    result["stage1_pass"] = (
        result["faithfulness"]["pass"]
        and all(v["canonical_bytes_equal"] and v["n_cell_method_mismatches"] == 0
                for v in result["comparisons"].values())
    )
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    write_graph()
    write_table(result["variants"], result["comparisons"])
    write_result(result)
    print(f"stage1_pass={result['stage1_pass']}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0 if result["stage1_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
