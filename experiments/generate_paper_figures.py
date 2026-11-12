"""Generate frozen paper figures from existing experiment artifacts.

The script is intentionally read-only with respect to experiment results: it
renders figures from already materialized JSON summaries and never recomputes or
fills missing metrics.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "paper" / "figures"
MPL_CACHE = REPO_ROOT / ".matplotlib_cache"
MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

EXP1_SUMMARY = REPO_ROOT / "experiments" / "exp01_overall_reliability" / "results" / "summary.json"
EXP5_MANUAL = REPO_ROOT / "experiments" / "exp05_cve_attribution" / "results" / "dry_run_strict" / "manual_stats.json"
EXP6_SYNTH = REPO_ROOT / "experiments" / "exp03_dynamic_service" / "results" / "exp6a_rerun" / "result.json"
EXP6_REAL = REPO_ROOT / "experiments" / "exp05_long_term_evolution" / "results" / "run1" / "result.json"


METHOD_ORDER = [
    "Ours",
    "B1_reconftw",
    "B2_pentestgpt",
    "B3_static_dag",
    "B4_single_react",
]
METHOD_LABELS = {
    "Ours": "Ours",
    "B1_reconftw": "B1",
    "B2_pentestgpt": "B2",
    "B3_static_dag": "B3",
    "B4_single_react": "B4",
}
METHOD_COLORS = {
    "Ours": "#2f6f9f",
    "B1_reconftw": "#8c8c8c",
    "B2_pentestgpt": "#c26d5a",
    "B3_static_dag": "#6aa56a",
    "B4_single_react": "#b08ac2",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_exp01_main_results() -> None:
    summary = _load_json(EXP1_SUMMARY)["summary"]
    metrics = [
        ("asset_f1", "Asset F1"),
        ("service_acc", "Service F1"),
        ("cve_recall_at_5", "CVE R@5"),
    ]
    methods = [m for m in METHOD_ORDER if m in summary]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), sharey=True)
    for ax, (metric, label) in zip(axes, metrics):
        xs = list(range(len(methods)))
        vals = [summary[m][metric] for m in methods]
        lows = [summary[m].get(f"{metric}_ci_low", summary[m][metric]) for m in methods]
        highs = [summary[m].get(f"{metric}_ci_high", summary[m][metric]) for m in methods]
        yerr = [
            [max(v - lo, 0.0) for v, lo in zip(vals, lows)],
            [max(hi - v, 0.0) for v, hi in zip(vals, highs)],
        ]
        colors = [METHOD_COLORS[m] for m in methods]
        ax.bar(xs, vals, yerr=yerr, capsize=3, color=colors, edgecolor="#222222", linewidth=0.5)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=8)
        ax.set_ylim(0, 0.82)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
        if ax is axes[0]:
            ax.set_ylabel("Score with bootstrap 95% CI")
    handles = [Patch(facecolor=METHOD_COLORS[m], edgecolor="#222222", label=summary[m]["display"]) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle("Exp 1 main result on the 15-org broad replay corpus", fontsize=11)
    fig.tight_layout(rect=(0, 0.14, 1, 0.92))
    _save(fig, "exp01_main_results_ci")


def render_exp05_audit_distribution() -> None:
    stats = _load_json(EXP5_MANUAL)["methods"]
    methods = ["CPE-only", "CVSS + EPSS + KEV priority only", "LLM-only rerank", "RAG-only", "RAG + KG", "Ours", "Ours-strict"]
    labels = ["strong", "reasonable", "weak", "unsupported", "wrong"]
    colors = {
        "strong": "#2f6f9f",
        "reasonable": "#7fb37b",
        "weak": "#d8b365",
        "unsupported": "#d95f5f",
        "wrong": "#7f3b3b",
    }
    display = {
        "CVSS + EPSS + KEV priority only": "CVSS/EPSS/KEV",
        "LLM-only rerank": "LLM-only",
        "RAG + KG": "RAG+KG",
    }

    fig, ax = plt.subplots(figsize=(8.9, 4.25))
    y = list(range(len(methods)))
    left = [0] * len(methods)
    for label in labels:
        vals = [stats[m]["label_counts"].get(label, 0) for m in methods]
        ax.barh(y, vals, left=left, color=colors[label], edgecolor="white", linewidth=0.6, label=label)
        left = [l + v for l, v in zip(left, vals)]

    for idx, m in enumerate(methods):
        pass_rate = stats[m]["pass_rate"]
        ax.text(51.2, idx, f"pass={pass_rate:.2f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([display.get(m, m) for m in methods], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 60)
    ax.set_xlabel("Number of audited asset-CVE pairs (n=50 per method)")
    ax.set_title("Exp 5 manual audit label distribution")
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.45)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "exp05_audit_label_distribution")


def render_exp06_dynamic_anchor() -> None:
    synth = _load_json(EXP6_SYNTH)["comparison"]
    real = _load_json(EXP6_REAL)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8))

    strategies = ["full_rescan", "selective_rescan"]
    strategy_labels = ["Full re-scan", "Selective re-scan"]
    metrics = [
        ("recall", "Recall"),
        ("lifecycle", "Lifecycle acc."),
        ("saving", "Probe saving"),
    ]
    values = {
        "full_rescan": [
            synth["full_rescan"]["recall_mean"],
            synth["full_rescan"]["lifecycle_classification_accuracy_mean"],
            synth["full_rescan"]["probe_saving_rate"],
        ],
        "selective_rescan": [
            synth["selective_rescan"]["recall_mean"],
            synth["selective_rescan"]["lifecycle_classification_accuracy_mean"],
            synth["selective_rescan"]["probe_saving_rate_mean"],
        ],
    }
    width = 0.34
    xs = list(range(len(metrics)))
    for i, strategy in enumerate(strategies):
        off = [x + (i - 0.5) * width for x in xs]
        color = "#8c8c8c" if strategy == "full_rescan" else "#2f6f9f"
        ax1.bar(off, values[strategy], width=width, color=color, edgecolor="#222222", linewidth=0.5, label=strategy_labels[i])
    ax1.set_xticks(xs)
    ax1.set_xticklabels([label for _, label in metrics], fontsize=8)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Rate")
    ax1.set_title("Synthetic dynamic trials")
    ax1.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.5)
    ax1.legend(frameon=False, fontsize=8, loc="lower left")
    ax1.text(
        0.02,
        0.94,
        f"Selective latency: {synth['selective_rescan']['latency_mean_weeks']:.3f} wk",
        transform=ax1.transAxes,
        fontsize=8,
        va="top",
    )

    cases = real["cases"]
    orgs = [case["organization"] for case in cases]
    drift = [case["n_ip_changed"] for case in cases]
    stable = [case["n_hosts"] - case["n_ip_changed"] for case in cases]
    y = list(range(len(orgs)))
    ax2.barh(y, stable, color="#d7e6ef", edgecolor="white", label="No IP drift")
    ax2.barh(y, drift, left=stable, color="#d95f5f", edgecolor="white", label="IP drift")
    for idx, case in enumerate(cases):
        ax2.text(case["n_hosts"] + 0.6, idx, f"{case['n_ip_changed']}/{case['n_hosts']}", va="center", fontsize=8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(orgs, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(case["n_hosts"] for case in cases) + 6)
    ax2.set_xlabel("Hosts in small live sample")
    ax2.set_title("Real public drift anchor")
    ax2.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.45)
    ax2.text(
        0.02,
        0.95,
        "Light: no IP drift; red: IP drift\n"
        f"DNS/HTTP observed: {real['n_dns_observed']}/{real['n_hosts']} and {real['n_http_observed']}/{real['n_hosts']}",
        transform=ax2.transAxes,
        fontsize=8,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.86, "pad": 3},
    )

    fig.suptitle("Exp 6 dynamic detection with a real longitudinal anchor", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    _save(fig, "exp06_dynamic_real_anchor")


def main() -> None:
    render_exp01_main_results()
    render_exp05_audit_distribution()
    render_exp06_dynamic_anchor()
    print("Generated:")
    for name in [
        "exp01_main_results_ci",
        "exp05_audit_label_distribution",
        "exp06_dynamic_real_anchor",
    ]:
        print(f"  paper/figures/{name}.pdf")
        print(f"  paper/figures/{name}.png")


if __name__ == "__main__":
    main()
