"""Revision v2 · step 2 — Repair@B step curves (POST-HOC presentation of existing results).

Repair@B = fraction of cells (org-level mean) whose FIXED host-grain repair scope succeeds AND
has spec cost <= B. The scope is never re-selected under the budget, so this is not
"best repair achievable within B". Facets: target alpha 0.106 (combined t=0.5) and 0.020
(combined t=0.9). "+ base domains" is a post-hoc rule (base domains = the pipeline's own
task-scope input, core/graph.py::base_domains_for).

    .venv/bin/python experiments/exp_arcs/revision_v2/step2_repair_curve.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
HERE = REPO / "experiments/exp_arcs/revision_v2"
sys.argv = sys.argv[:1]
import experiments.exp_arcs.revision_v2.step2_baselines as S  # noqa: E402

# fixed categorical order (dataviz reference palette, light; validated 5-slot)
COL = {"combined": "#2a78d6", "iforest": "#eb6834", "robust_z": "#1baf7a", "lof": "#eda100",
       "combined+bd": "#e87ba4"}
LABEL = {"combined": "combined", "iforest": "Isolation Forest", "robust_z": "robust-z", "lof": "LOF",
         "combined+bd": "combined + base domains (post-hoc)"}
SHORT = {"combined": "comb.", "iforest": "IF", "robust_z": "rz", "lof": "LOF", "combined+bd": "comb.+bd"}
FACETS = {"primary": ("target α = 0.106 (combined t = 0.5)", 0.5),
          "secondary": ("target α = 0.020 (combined t = 0.9)", 0.9)}


def org_curve(rows: list[dict], Bs: list[float]) -> list[float]:
    by = defaultdict(list)
    for x in rows:
        by[x["org"]].append(x)
    return [round(sum(sum(1 for x in v if x["success"] and x["cost"] <= B) / len(v) for v in by.values()) / len(by), 4)
            for B in Bs]


def main() -> int:
    r = json.loads((HERE / "step2_baselines_v1.json").read_text())
    ph2 = json.loads((HERE / "step2_posthoc2_v1.json").read_text())["E_repair_at_B"]
    grid = json.loads((S.D / "e3_grid_v1.json").read_text())["cells"]
    from core.graph import base_domains_for       # the pipeline's own task-scope input
    bd = {c["org"]: set(base_domains_for(c["org"])) for c in grid}
    B0 = r["inputs"]["B0"]

    series: dict = {}
    for tag, (_, t) in FACETS.items():
        comb, combbd = [], []
        for c in grid:
            fd = Path(c["faulted_dir"])
            fl = frozenset(S.detector_localize(fd)["flagged_hosts"] | S.peer_anomaly_hosts(fd, min_base_rate=t)
                           | S.manifest_dropped_hosts(fd))
            comb.append({"org": c["org"], "cell": c["cell"], **S.repair(c, fl)})
            combbd.append({"org": c["org"], "cell": c["cell"], **S.repair(c, frozenset(fl | bd[c["org"]]))})
        series[tag] = {"combined": comb, "combined+bd": combbd}
        for det in ("iforest", "robust_z", "lof"):
            series[tag][det] = [{"org": x["org"], "cell": x["cell"], "success": x["success"], "cost": x["cost"]}
                                for x in r["baselines"][f"{det}|manifest_aware"]["by_alpha"][tag]["cells"]]

    # consistency with step2_posthoc2_v1.json grid points
    name2 = {"primary": {"combined": "combined_t0.5", "combined+bd": "combined_t0.5+base_domains",
                         "iforest": "iforest_alpha0.106", "robust_z": "robust_z_alpha0.106", "lof": "lof_alpha0.106"},
             "secondary": {"combined": "combined_t0.9", "combined+bd": "combined_t0.9+base_domains",
                           "iforest": "iforest_alpha0.02", "robust_z": "robust_z_alpha0.02", "lof": "lof_alpha0.02"}}
    for tag in FACETS:
        for m, rows in series[tag].items():
            want = ph2[name2[tag][m]]["repair_at_B"]
            got = dict(zip(want, org_curve(rows, [B0 if b == "B0" else float(b) for b in want])))
            assert got == want, (tag, m, got, want)

    costs = sorted({x["cost"] for tag in series for rows in series[tag].values() for x in rows if x["success"]})
    Bs = sorted({0.5, *costs, 500.0})
    curves = {tag: {m: org_curve(rows, Bs) for m, rows in series[tag].items()} for tag in series}
    data = {"note": "POST-HOC presentation; fixed-scope Repair@B, org-level mean; '+ base domains' is a post-hoc rule",
            "B": Bs, "B0": B0, "curves": curves,
            "inputs": {"step2_baselines_v1_sha256": hashlib.sha256((HERE / "step2_baselines_v1.json").read_bytes()).hexdigest(),
                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    (HERE / "step2_repair_curve_v1.json").write_text(json.dumps(data, indent=1, sort_keys=True))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#2b2b29",
                         "xtick.color": "#5c5c58", "ytick.color": "#5c5c58", "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), sharey=True)
    order = ("combined", "iforest", "robust_z", "lof", "combined+bd")
    for ax, (tag, (title, _)) in zip(axes, FACETS.items()):
        for m in order:
            y = curves[tag][m]
            ax.step(Bs, y, where="post", color=COL[m], lw=2, ls="--" if m == "combined+bd" else "-",
                    label=LABEL[m], solid_capstyle="round")
        ax.axvline(B0, color="#b8b8b2", lw=1, ls=":")
        ax.text(B0 * 1.06, 0.975, "B0", color="#5c5c58", fontsize=8, va="top")
        ax.set_xscale("log")
        ax.set_xlim(1, 500)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=9.5, color="#2b2b29", loc="left")
        ax.set_xlabel("repair budget B (spec cost units, log scale)")
        ax.grid(True, which="major", color="#ececea", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        # direct end labels, nudged apart to avoid collisions
        ends = sorted(((curves[tag][m][-1], m) for m in order), reverse=True)
        placed: list[float] = []
        for v, m in ends:
            yv = v
            while any(abs(yv - p) < 0.045 for p in placed):
                yv -= 0.045
            placed.append(yv)
            ax.text(520, yv, f"{v:.2f} {SHORT[m]}", color="#2b2b29", fontsize=8, va="center")
    axes[0].set_ylabel("Repair@B (fixed scope, success and cost ≤ B)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 0.95, 1))
    out = HERE / "step2_repair_curve.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out} and step2_repair_curve_v1.json; grid-point consistency with posthoc2: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
