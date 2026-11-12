"""Build the localization signal-ablation LaTeX table from
e7_localization_ablation_v1.json (no hand-typed numbers).

    poetry run python -m experiments.exp_arcs.e7_localization_ablation_table
"""
from __future__ import annotations
import json
from pathlib import Path

J = Path("experiments/exp_arcs/e7_localization_ablation_v1.json")
OUT = Path("experiments/exp_arcs/e7_localization_ablation_table_v2.tex")

SUBTYPE_ORDER = ["F4@dns", "F4@active_observation", "F4@certificate",
                 "F4@registration", "F4-host", "F4-2host-dns",
                 "F10-local", "F10-2host", "F10-3host"]


def main() -> int:
    d = json.loads(J.read_text())
    o = d["overall"]
    loo = d["leave_one_out_success_drop"]
    bs = d["by_subtype"]

    def f(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "--"

    lines = [
        r"\begin{table}[t]\centering\small",
        (r"\caption{Reference-free localization signal ablation on the 180-cell E3 "
         r"grid (host-grain, offline counterfactual recovery with the historical "
         r"clean-DAG action map). The cost row averages over successful cells only. "
         r"\textbf{Leave-one-in} (single columns) and \textbf{leave-one-out} (the drop "
         r"row) isolate each signal. The validator-flag signal (the deterministic support-validation markers, also read by the first-flag/provenance-backtrace probes) "
         r"contributes only $" + f(loo['drop_self']) + r"$ to end-to-end success; the "
         r"entire lift is carried by the two proposed reference-free signals: "
         r"peer-anomaly ($" + f(loo['drop_peer']) + r"$) and discovery-manifest ($" +
         f(loo['drop_manifest']) + r"$).}"),
        r"\label{tab:loc-ablation}",
        r"\begin{tabular}{lccccccc}\toprule",
        (r"fault subtype & flag & peer & manif. & flag$\cup$peer & flag$\cup$manif. "
         r"& peer$\cup$manif. & \textbf{combined}\\"),
        (r" & & & & \emph{(--manif.)} & \emph{(--peer)} & \emph{(--flag)} & \\\midrule"),
    ]
    for st in SUBTYPE_ORDER:
        r = bs[st]
        lines.append(
            f"{st.replace('_',chr(92)+'_')} & {f(r['self'])} & {f(r['peer'])} & "
            f"{f(r['manifest'])} & {f(r['self+peer'])} & {f(r['self+manifest'])} & "
            f"{f(r['peer+manifest'])} & \\textbf{{{f(r['combined'])}}}\\\\")
    lines.append(r"\midrule")
    lines.append(
        f"\\textbf{{overall}} & {f(o['self']['success'])} & {f(o['peer']['success'])} & "
        f"{f(o['manifest']['success'])} & {f(o['self+peer']['success'])} & "
        f"{f(o['self+manifest']['success'])} & {f(o['peer+manifest']['success'])} & "
        f"\\textbf{{{f(o['combined']['success'])}}}\\\\")
    lines.append(
        f"\\emph{{cost/$C^*$ (successes)}} & {f(o['self']['cost_over_Cstar'])} & "
        f"{f(o['peer']['cost_over_Cstar'])} & {f(o['manifest']['cost_over_Cstar'])} & "
        f"{f(o['self+peer']['cost_over_Cstar'])} & {f(o['self+manifest']['cost_over_Cstar'])} & "
        f"{f(o['peer+manifest']['cost_over_Cstar'])} & {f(o['combined']['cost_over_Cstar'])}\\\\")
    lines += [
        r"\midrule",
        r"\multicolumn{8}{p{.97\linewidth}}{\footnotesize Leave-one-out success drop: "
        f"remove validator flag $-{loo['drop_self']:.3f}$, remove manifest $-{loo['drop_manifest']:.3f}$, "
        f"remove peer $-{loo['drop_peer']:.3f}$. "
        r"Manifest recovers the F10 host-omission cells (invisible to the other signals); "
        r"peer recovers common-family F4 drops (dns/ao); validator flagging is near-redundant. "
        r"F4@certificate / F4@registration remain difficult for this peer-frequency "
        r"signal; this is not a general identifiability claim.}\\",
        r"\bottomrule\end{tabular}\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
