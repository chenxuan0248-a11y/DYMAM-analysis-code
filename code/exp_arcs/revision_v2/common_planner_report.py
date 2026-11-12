"""Render the descriptive common-planner table and recovery-cost figure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "common_planner_eval_20261020.json"
TABLE = HERE / "common_planner_table_20261020.tex"
FIGURE = HERE / "common_planner_pareto_20261020.pdf"


def main() -> None:
    data = json.loads(SOURCE.read_text())
    assert data["n_cells"] == 180 and data["n_orgs"] == 20
    summary = data["summary"]
    names = {"combined": "Combined", "iforest": "Isolation Forest",
             "robust_z": "Robust-$z$", "contract_b": "Contract B"}
    policies = {"selective": "Selective", "fixed_four_family": "Four-family"}
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Locator & Action policy & Recovery & $C_{\mathrm{plan}}$ & Hosts & Detect@1 & Truth alert & Clean alert\\",
             r"\midrule"]
    points = []
    for method in ("combined", "iforest", "robust_z", "contract_b"):
        item = summary[method]
        for policy in ("selective", "fixed_four_family"):
            result = item["policies"][policy]
            recovery = result["recovery_cells"] / 180
            cost = result["planned_cost_org_mean"]
            points.append((method, policy, cost, recovery))
            lines.append(f"{names[method]} & {policies[policy]} & "
                         f"{result['recovery_cells']}/180 & {cost:.2f} & "
                         f"{result['alerted_hosts_org_mean']:.2f} & "
                         f"{item['detect1_org_mean']:.3f} & "
                         f"{item['truth_host_alert_recall_org_mean']:.4f} & "
                         f"{item['clean_alert_macro_rate']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TABLE.write_text("\n".join(lines) + "\n")

    colors = {"combined": "#1f77b4", "iforest": "#d95f02",
              "robust_z": "#7570b3", "contract_b": "#7f7f7f"}
    markers = {"selective": "o", "fixed_four_family": "s"}
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    line_styles = {"combined": "-", "iforest": "-.", "robust_z": ":", "contract_b": "--"}
    for method in colors:
        method_points = [(x, y) for m, _, x, y in points if m == method]
        ax.plot([p[0] for p in method_points], [p[1] for p in method_points],
                color=colors[method], ls=line_styles[method], alpha=0.75, lw=1.4)
    for method, policy, cost, recovery in points:
        ax.scatter(cost, recovery, color=colors[method], marker=markers[policy],
                   s=72)
    nondominated = sorted([(x, y) for _, _, x, y in points
                           if not any(x2 <= x and y2 >= y and (x2 < x or y2 > y)
                                      for _, _, x2, y2 in points)])
    ax.plot([x for x, _ in nondominated], [y for _, y in nondominated],
            color="black", ls="--", lw=1, alpha=0.7, label="Observed nondominated points")
    ax.set_xlabel("Mean planned tool-action cost (specification units)")
    ax.set_ylabel("Offline counterfactual recovery")
    ax.set_xlim(28, 130)
    ax.set_ylim(0.60, 0.98)
    ax.grid(alpha=0.2)
    handles = [Line2D([0], [0], color=colors[m], ls=line_styles[m], lw=1.5,
                      label={"combined": "Combined", "iforest": "IF",
                             "robust_z": "Robust-z", "contract_b": "Contract B"}[m])
               for m in colors]
    handles += [Line2D([0], [0], color="black", marker="o", ls="None", label="Selective"),
                Line2D([0], [0], color="black", marker="s", ls="None", label="Four-family"),
                Line2D([0], [0], color="black", ls="--", label="Nondominated")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=8.5, ncol=4, frameon=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.31)
    fig.savefig(FIGURE)
    plt.close(fig)
    print(json.dumps({"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      "table": str(TABLE), "figure": str(FIGURE),
                      "nondominated": nondominated}, indent=2))


if __name__ == "__main__":
    main()
