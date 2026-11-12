"""Reviewer item F — joint (localization x repair) end-to-end table.

Consolidates the two frozen artifacts into ONE end-to-end view:
  - e3_oracle_stats_v1.json : repair-method comparison under Oracle localization
  - e3_detector_v1.json     : ARCS under reference-free detector localization

Only (localization, repair) pairs that were ACTUALLY computed are listed. global
replay is localization-independent (repairs everything); single-point is reported
under Oracle localization (its main-table baseline). No pair is invented.

    poetry run python -m experiments.exp_arcs.e3_joint_endtoend_table
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

OS_ = Path("experiments/exp_arcs/e3_oracle_stats_v1.json")
DET = Path("experiments/exp_arcs/e3_detector_v1.json")
OUT = Path("experiments/exp_arcs/e3_joint_endtoend_table.tex")


def main() -> int:
    s = json.loads(OS_.read_text())
    cells = s["cells"]
    n = len(cells)

    def rate(f):
        return sum(1 for c in cells if f(c)) / n

    sp_succ = rate(lambda c: c["sp_success"])
    arcs_succ = rate(lambda c: c["arcs_success"])
    arcs_ccs = st.mean(c["arcs_cost"] / c["C_star"] for c in cells
                       if c["arcs_success"] and c["C_star"])
    sp_ccs = st.mean(c["sp_cost"] / c["C_star"] for c in cells
                     if c["sp_success"] and c["C_star"])
    gl_cost = st.mean(c["global_cost"] for c in cells)
    gl_ccs = st.mean(c["global_cost"] / c["C_star"] for c in cells if c["C_star"])
    arcs_cost = st.mean(c["arcs_cost"] for c in cells)

    e = json.loads(DET.read_text())["summary"]
    det_h = e["detector_only_success"]["combined"]
    det_h_c = e["cost_recovery"]["combined"]["mean_cost_over_Cstar"]
    det_f = e["detector_only_success"]["combined_family_grain"]
    det_f_c = e["cost_recovery"]["combined_family_grain"]["mean_cost_over_Cstar"]

    lines = [
        r"\begin{table}[t]\centering\small",
        (r"\caption{Offline localization and scope outcomes on the 180-cell grid. "
         r"Oracle rows use clean-reference localization; detector rows use "
         r"faulted-run localization but retain the historical clean-DAG action "
         r"map. Cost/$C^*$ is averaged over successful cells within each row. "
         r"Global re-collection is the all-leaf-action baseline; no actions ran.}"),
        r"\label{tab:joint-e2e}",
        r"\begin{tabular}{llrr}\toprule",
        r"localization & repair & Success & cost/$C^*$\\\midrule",
        f"Oracle & single-point & {sp_succ:.3f} & {sp_ccs:.2f}\\\\",
        f"Oracle & \\textbf{{ARCS}} & \\textbf{{{arcs_succ:.3f}}} & \\textbf{{{arcs_ccs:.2f}}}\\\\",
        f"(any) & global re-collection & 1.000 & {gl_ccs:.1f}\\\\",
        r"\midrule",
        f"detector, host-grain & ARCS & {det_h:.3f} & {det_h_c:.2f}\\\\",
        f"detector, family-grain & ARCS & {det_f:.3f} & {det_f_c:.2f}\\\\",
        r"\midrule",
        (r"\multicolumn{4}{p{.9\linewidth}}{\footnotesize Oracle single-point / ARCS / "
         r"global use the frozen oracle evaluation; detector rows use the "
         r"combined signal under the historical action map. cost/$C^*$ averaged over "
         r"that row's success cells. global cost/$C^*$ shown relative to the same $C^*$ "
         r"base (mean global cost $" + f"{gl_cost:.1f}" + r"$).}\\"),
        r"\bottomrule\end{tabular}\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
