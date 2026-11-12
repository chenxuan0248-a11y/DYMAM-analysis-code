"""Build the E8 end-to-end asset-F1 table from e8_delta_f1_v1.json (no hand-typed
numbers). Host-grain F1 per fault type + overall; marks the silent (Delta F1 = 0)
faults that motivate reference-free localization.

    poetry run python -m experiments.exp_arcs.e8_table
"""
from __future__ import annotations
import json
from pathlib import Path

J = Path("experiments/exp_arcs/e8_delta_f1_v1.json")
OUT = Path("experiments/exp_arcs/e8_delta_f1_table.tex")
ORDER = ["F4@dns", "F4@active_observation", "F4@certificate", "F4@registration",
         "F4-host", "F4-2host-dns", "F10-local", "F10-2host", "F10-3host"]


def main() -> int:
    d = json.loads(J.read_text())
    bt = d["by_fault_type"]
    ov = d["overall"]

    def row(name, r):
        f, rp, c, dl = r["faulted_host"], r["repaired_host"], r["clean_host"], r["delta_host"]
        silent = "$\\star$" if abs(dl) < 1e-9 else ""
        nm = name.replace("_", r"\_")
        return f"{nm} & {f:.4f} & {rp:.4f} & {c:.4f} & {dl:+.4f}{silent}\\\\"

    lines = [
        r"\begin{table}[t]\centering\small",
        (r"\caption{End-to-end asset mapping $F_1$ (host grain) vs.\ ground truth, per "
         r"injected fault type ($180$ cells): faulted report, ARCS-repaired (B1 "
         r"restore-to-clean), and clean ceiling. $\Delta F_1 = $ repaired $-$ faulted. "
         r"Rows marked $\star$ have $\Delta F_1 = 0$: the faulted asset projection is "
         r"byte-identical to clean, so the fault is \emph{output-invisible} --- direct "
         r"motivation for reference-free localization. Repair recovers the "
         r"output-visible damage (F10, F4-host) to the clean value.}"),
        r"\label{tab:e8}",
        r"\begin{tabular}{lrrrr}\toprule",
        r"fault type & faulted & repaired & clean & $\Delta F_1$\\\midrule",
    ]
    for name in ORDER:
        if name in bt:
            lines.append(row(name, bt[name]))
    lines.append(r"\midrule")
    lines.append(
        f"\\textbf{{overall}} & {ov['faulted_host']:.4f} & {ov['repaired_host']:.4f} & "
        f"{ov['clean_host']:.4f} & {ov['delta_host']:+.4f}\\\\")
    lines += [
        r"\midrule",
        (r"\multicolumn{5}{p{.9\linewidth}}{\footnotesize $\Delta F_1$ is small by "
         r"construction: each fault is single-locus ($1$--$3$ hosts) in a $\sim$100-asset "
         r"corpus, and only injected hosts present in the (partial) ground truth can move "
         r"$F_1$. The point is the $\star$ rows' exact zero, not the magnitude.}\\"),
        r"\bottomrule\end{tabular}\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
