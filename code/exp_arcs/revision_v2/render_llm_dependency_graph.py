"""Render the audited E-A dependency paths as a paper-ready PDF figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_name("llm_dependency_graph.pdf")


def box(ax, xy, width, height, title, lines, face, edge):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                                boxstyle="round,pad=0.015,rounding_size=0.025",
                                facecolor=face, edgecolor=edge, linewidth=1.3))
    ax.text(x + 0.025, y + height - 0.035, title, fontsize=9.5,
            fontweight="bold", va="top", color="#0f172a")
    for i, line in enumerate(lines):
        ax.text(x + 0.025, y + height - 0.085 - i * 0.040, line,
                fontsize=7.7, va="top", color="#334155")


def arrow(ax, start, end, *, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.25, color="#e11d48" if dashed else "#475569",
                                 linestyle="--" if dashed else "-",
                                 connectionstyle="arc3,rad=0.0"))


fig, ax = plt.subplots(figsize=(12.5, 7.2))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.text(0.025, 0.96, "E-A: field-level dependency of evaluated metrics",
        fontsize=15, fontweight="bold", va="top", color="#0f172a")

box(ax, (0.03, 0.72), 0.25, 0.155, "Tool and evidence records",
    ["evidence: target, source_family", "source artifact: id, raw_ref"],
    "#f8fafc", "#334155")
box(ax, (0.03, 0.51), 0.25, 0.155, "Formal policy projection",
    ["claim + decision fields", "report_item links"], "#f8fafc", "#334155")
box(ax, (0.03, 0.32), 0.25, 0.13, "Discovery manifest",
    ["resolved_hosts"], "#f8fafc", "#334155")
box(ax, (0.03, 0.09), 0.25, 0.16, "Reporting LLM text",
    ["state.report.summary", "not persisted in E3 bundle"], "#fff1f2", "#e11d48")

box(ax, (0.385, 0.72), 0.245, 0.155, "Candidate universe",
    ["evidence / claim subjects", "+ resolved-host manifest"], "#ecfeff", "#0891b2")
box(ax, (0.385, 0.49), 0.245, 0.175, "Host features and scores",
    ["combined, Isolation Forest", "robust z-score", "no rendered-report field"],
    "#ecfeff", "#0891b2")
box(ax, (0.385, 0.255), 0.245, 0.17, "Contamination and DAG",
    ["formal record differences", "repairable leaves"], "#ecfeff", "#0891b2")

box(ax, (0.73, 0.62), 0.235, 0.17, "Localization metrics",
    ["Detect@1 / Detect@3", "flags and candidate coverage"],
    "#eef2ff", "#4f46e5")
box(ax, (0.73, 0.29), 0.235, 0.19, "Offline repair metrics",
    ["success; specification cost / C*", "formal report_item affects truth"],
    "#eef2ff", "#4f46e5")

for a, b in [((0.28, 0.80), (0.385, 0.80)),
             ((0.28, 0.59), (0.385, 0.58)),
             ((0.28, 0.385), (0.385, 0.79)),
             ((0.28, 0.565), (0.385, 0.34)),
             ((0.63, 0.79), (0.73, 0.71)),
             ((0.63, 0.58), (0.73, 0.69)),
             ((0.63, 0.57), (0.73, 0.40)),
             ((0.63, 0.34), (0.73, 0.38))]:
    arrow(ax, a, b)
arrow(ax, (0.28, 0.17), (0.73, 0.31), dashed=True)
ax.text(0.39, 0.115, "tested absent report-text path", fontsize=8,
        color="#be123c", fontweight="bold")
ax.text(0.03, 0.025,
        "Solid: audited formal-field consumption. Dashed: tested absent dependency. Source audit, not a whole-program proof.",
        fontsize=7.5, color="#475569")
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.12)
plt.close(fig)
print(OUT)
