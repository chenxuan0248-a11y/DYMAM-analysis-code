"""Build the FULL (all 4 asset-grain metrics) E8 end-to-end F1 table from
e8_delta_f1_v1.json (no hand-typed numbers) -- the appendix companion to the
host-grain-only table in the main body (e8_table.py / Table tab:e8).

    poetry run python -m experiments.exp_arcs.e8_delta_f1_full_table
"""
from __future__ import annotations
import json
from pathlib import Path

J = Path("experiments/exp_arcs/e8_delta_f1_v1.json")
OUT = Path("experiments/exp_arcs/e8_delta_f1_full_table.tex")
ORDER = ["F4@dns", "F4@active_observation", "F4@certificate", "F4@registration",
         "F4-host", "F4-2host-dns", "F10-local", "F10-2host", "F10-3host"]
METRICS = ["subdomain", "host", "port", "cpe"]


def cell(row: dict, metric: str) -> str:
    dl = row[f"delta_{metric}"]
    silent = abs(dl) < 1e-9
    if silent:
        dl = 0.0  # normalize -0.0 -> 0.0 for display; exactness is what $\star$ asserts
    return f"{dl:+.4f}{'$\\star$' if silent else ''}"


def main() -> int:
    d = json.loads(J.read_text())
    bt = d["by_fault_type"]
    ov = d["overall"]

    # Sanity check the claim this table's footnote makes: repaired == clean on
    # every (fault type, metric) cell. If a future data refresh breaks this,
    # fail loudly rather than silently mis-describing the table.
    for name, row in {**bt, "overall": ov}.items():
        for m in METRICS:
            assert row[f"repaired_{m}"] == row[f"clean_{m}"], (name, m)

    rows = []
    for name in ORDER:
        row = bt[name]
        nm = name.replace("_", r"\_")
        cells = " & ".join(cell(row, m) for m in METRICS)
        rows.append(f"{nm} & {cells}\\\\")

    ov_cells = " & ".join(cell(ov, m) for m in METRICS)

    lines = [
        r"\begin{table}[t]\centering\small",
        (r"\caption{Full end-to-end asset-$F_1$ $\Delta F_1=$~repaired$-$faulted "
         r"across all four asset grains ($180$ cells), the companion to the "
         r"host-grain-only Table~\ref{tab:e8} in \S\ref{sec:rq1}. Repaired $\equiv$ "
         r"clean exactly on every one of these $36$ (fault type $\times$ metric) "
         r"cells plus the overall row (verified programmatically, not by "
         r"inspection), so only $\Delta F_1$ is shown; the faulted/repaired/clean "
         r"absolute values are retained in the frozen result artifact. Rows marked "
         r"$\star$ have $\Delta F_1=0$ on that grain (output-invisible on that "
         r"metric). cpe can be slightly negative (e.g.\ "
         r"F4@active\_observation, F10-local): repaired$=$clean, and the clean "
         r"run itself carries a few cpe false positives vs.\ GT, so a fault that "
         r"happens to drop one of those FPs looks marginally ``better'' before "
         r"repair --- a faithful artifact of partial ground truth, not a "
         r"regression (\S\ref{sec:threats}).}"),
        r"\label{tab:e8-full}",
        r"\begin{tabular}{lrrrr}\toprule",
        r"fault type & $\Delta$subdomain & $\Delta$host & $\Delta$port & $\Delta$cpe\\\midrule",
        *rows,
        r"\midrule",
        f"\\textbf{{overall}} & {ov_cells}\\\\",
        r"\bottomrule\end{tabular}\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
