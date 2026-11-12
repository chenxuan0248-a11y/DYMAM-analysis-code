"""E-D variant x signal repair-success table from ed_variant_signal_matrix_v1.json.

Success is the unbudgeted A1 counterfactual success of the scope each signal flags
(host-grain unless stated). The bottom rows give the organization-level mean and 95%
CI over all 180 cells, and the mean spec cost over all cells (failed cells included).
Read-only: renders the frozen E-D artifact, recomputes nothing.
"""
from __future__ import annotations
import json
from pathlib import Path

D = Path("experiments/exp_arcs")
SRC = D / "ed_variant_signal_matrix_v1.json"
OUT = D / "ed_matrix_table.tex"

COLS = [("validator_flag", "valid."), ("peer_anomaly", "peer"), ("manifest", "manif."),
        ("combined_host", "comb./host"), ("combined_family", "comb./fam."),
        ("oracle_single_point", "SP$^\\dagger$"), ("oracle_arcs", "ARCS$^\\dagger$")]
ROWS = ["F4@dns", "F4@active_observation", "F4@certificate", "F4@registration",
        "F4-host", "F4-2host-dns", "F10-local", "F10-2host", "F10-3host"]


def _tt(v: str) -> str:
    return v.replace("_", r"\_")


def main() -> int:
    d = json.loads(SRC.read_text())
    m, ci = d["matrix"], d["matrix_org_ci_ALL"]
    head = " & ".join(c for _, c in COLS)
    lines = [
        r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        (r"\caption{Repair success by fault variant and localization signal ($20$ "
         r"organizations per variant). Each signal's flagged region is handed to the "
         r"same host-grain repair mapper (family grain for comb./fam.); Success is the "
         r"unbudgeted counterfactual criterion. valid.\ = validator flags, peer = "
         r"peer-anomaly ($t=0.5$), manif.\ = discovery manifest, comb.\ = their union. "
         r"$^\dagger$Oracle localization (upper references, not deployable): SP = "
         r"single-point repair of the injected site, ARCS = constrained scope. Bottom "
         r"rows: organization-level mean with $95\%$ bootstrap CI, and mean spec cost "
         r"$c/C^\ast$ over all $180$ cells including failed ones.}"),
        r"\label{tab:ed-matrix}",
        r"\begin{tabular}{l" + "r" * len(COLS) + r"}\toprule",
        f"variant & {head}\\\\\\midrule",
    ]
    for v in ROWS:
        cells = " & ".join(f"{m[v][k]['success_rate']:.2f}" for k, _ in COLS)
        lines.append(f"{_tt(v)} & {cells}\\\\")
    lines.append(r"\midrule")
    succ = " & ".join(f"{ci[k]['success_rate']['org_mean']:.3f}" for k, _ in COLS)
    lo_hi = " & ".join(
        "[{:.2f}, {:.2f}]".format(*ci[k]["success_rate"]["ci95"]) for k, _ in COLS)
    cost = " & ".join(
        f"{ci[k]['mean_cost_over_Cstar_all_cells']['org_mean']:.2f}" for k, _ in COLS)
    lines += [f"all: Success & {succ}\\\\",
              f"\\quad 95\\% CI & {lo_hi}\\\\",
              f"all: $c/C^\\ast$ & {cost}\\\\",
              r"\bottomrule\end{tabular}\end{table}"]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
