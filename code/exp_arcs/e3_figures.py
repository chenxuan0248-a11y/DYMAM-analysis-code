"""experiments/exp_arcs/e3_figures — E3 main figure (Cost–Success Pareto).

Reads e3_pareto_v1.json (budget sweep) → a log-x Success-vs-budget frontier for
ARCS / single-point / global. The paper's core figure. Pure plotting, no new data.

    poetry run python -m experiments.exp_arcs.e3_figures
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main() -> int:
    src = Path("experiments/exp_arcs/e3_pareto_v1.json")
    out = Path("experiments/exp_arcs/e3_pareto_frontier.pdf")
    curve = json.loads(src.read_text())["curve"]
    b = [c["budget"] for c in curve]
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(b, [c["arcs"] for c in curve], "-o", ms=3, lw=2, label="ARCS (adaptive)", color="#1b6ca8")
    ax.plot(b, [c["single_point"] for c in curve], "--s", ms=3, lw=1.6, label="Single-point", color="#c1440e")
    ax.plot(b, [c["global"] for c in curve], ":^", ms=3, lw=1.6, label="Global re-collection", color="#5a5a5a")
    ax.set_xscale("symlog")
    ax.set_xlabel("Repair budget B (cost units, symlog)")
    ax.set_ylabel("Success@B  (fraction of 180 cells)")
    ax.set_ylim(-0.03, 1.03)
    ax.axvline(3, color="#999", lw=0.8, ls=":"); ax.text(3, 0.02, " B₀=3", fontsize=8, color="#666")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out)
    print(f"wrote {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
