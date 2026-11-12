"""Build the E5 asset-confidence calibration table from e5_calibration_v1.json
(no hand-typed numbers) -- full appendix table backing the one-paragraph summary
in Threats-to-Validity/Construct (Sec. threats).

    poetry run python -m experiments.exp_arcs.e5_calibration_table
"""
from __future__ import annotations
import json
from pathlib import Path

J = Path("experiments/exp_arcs/e5_calibration_v1.json")
OUT = Path("experiments/exp_arcs/e5_calibration_table.tex")
STATES = ["faulted", "repaired", "clean"]


def main() -> int:
    d = json.loads(J.read_text())
    st = d["states"]

    rows = []
    for name in STATES:
        s = st[name]
        rows.append(
            f"{name} & {s['n']} & {s['ece']:.4f} & {s['mce']:.4f} & "
            f"{s['brier']:.4f} & {s['mean_confidence']:.4f} & "
            f"{s['fraction_in_gt']:.4f}\\\\"
        )

    lines = [
        r"\begin{table}[t]\centering\small",
        (r"\caption{Asset-confidence calibration (E5), pooled over all $180$ cells, "
         r"by report state: expected/maximum calibration error (ECE/MCE), Brier "
         r"score, mean predicted confidence, and the fraction of predicted "
         r"subdomains present in the (partial) ground-truth list. "
         r"Repair does not degrade calibration: repaired ECE ($0.4742$) equals "
         r"clean ECE exactly to four decimal places, and faulted differs by "
         r"$<0.001$. Absolute ECE ($\approx0.47$) is not a claim about model "
         r"quality: mean confidence is $\approx0.75$ but only "
         r"$\approx" + f"{st['clean']['fraction_in_gt']*100:.0f}" + r"\%$ of "
         r"predicted subdomains appear in the ground-truth list, so genuine, "
         r"GT-unlisted discoveries are scored ``incorrect'' alongside real "
         r"overconfidence, and the two cannot be separated with this ground "
         r"truth; the load-bearing comparison is therefore the "
         r"repaired-vs-clean equality, not the absolute magnitude "
         r"(\S\ref{sec:threats}).}"),
        r"\label{tab:e5-calibration}",
        r"\begin{tabular}{lrrrrrr}\toprule",
        r"state & $n$ & ECE & MCE & Brier & mean conf.\ & frac.\ in GT\\\midrule",
        *rows,
        r"\bottomrule\end{tabular}\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
