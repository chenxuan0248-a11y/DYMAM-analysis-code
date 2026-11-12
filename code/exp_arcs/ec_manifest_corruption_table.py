"""E-C manifest-record corruption table from ec_manifest_corruption_v1.json.

Rows: p (independent per-host deletion probability) for drop_true, plus the absent
manifest. drop_irrelevant and random_censorship are not tabulated separately: every
manifest-flagged host in the faulted runs is an injected host (Mf_composition:
114 true, 0 irrelevant), so drop_irrelevant leaves all metrics unchanged and
random_censorship equals drop_true at every p. The script asserts both facts.
Values are exact expectations (no Monte Carlo); CIs are organization-level bootstrap.
"""
from __future__ import annotations
import json
from pathlib import Path

D = Path("experiments/exp_arcs")
SRC = D / "ec_manifest_corruption_v1.json"
OUT = D / "ec_manifest_corruption_table.tex"
P = ["0.0", "0.25", "0.5", "0.75", "1.0"]
MET = ["success", "detect1", "cost_over_Cstar"]


def main() -> int:
    d = json.loads(SRC.read_text())
    s = d["summary"]
    allv, fam = s["all"], s["by_family"]
    comp = s["Mf_composition"]
    assert comp["total_Mf_irrelevant"] == 0, comp
    for p in d["p_grid"]:
        for k in MET:
            a = allv[f"random_censorship|p={p}"][k]["org_mean"]
            b = allv[f"drop_true|p={p}"][k]["org_mean"]
            c = allv[f"drop_irrelevant|p={p}"][k]["org_mean"]
            z = allv["drop_true|p=0.0"][k]["org_mean"]
            assert a == b and c == z, (p, k, a, b, c, z)
    assert all(s["selfcheck"].values()), s["selfcheck"]

    def row(label, key):
        a, f10, f4 = allv[key], fam["F10"][key], fam["F4"][key]
        lo, hi = a["success"]["ci95"]
        return (f"{label} & {a['success']['org_mean']:.3f} [{lo:.2f}, {hi:.2f}] & "
                f"{a['detect1']['org_mean']:.3f} & {a['cost_over_Cstar']['org_mean']:.2f} & "
                f"{f10['success']['org_mean']:.3f} & {f4['success']['org_mean']:.3f}\\\\")

    lines = [
        r"\begin{table}[t]\centering\footnotesize",
        (r"\caption{Sensitivity of the combined detector to the discovery manifest "
         r"(record-level corruption of the frozen manifests; the pipeline is not re-run). "
         r"Each manifest-flagged host is deleted independently with probability $p$; "
         r"$p=1$ equals an absent manifest. All manifest-flagged hosts in the faulted "
         r"runs are injected hosts ($" + str(comp["total_Mf_true"]) + r"$ entries in "
         r"$" + str(comp["cells_with_Mf"]) + r"$ cells, none irrelevant), so deleting "
         r"other manifest entries changes nothing and random deletion coincides with "
         r"deleting true hosts. Exact expectations; Success with organization-level "
         r"$95\%$ CI; cost over all $180$ cells. Clean-run alerts are unchanged "
         r"($" + f"{s['clean_false_alarm']['p=0.0']:.3f}" + r"$) because the manifest "
         r"raises none.}"),
        r"\label{tab:ec-manifest}",
        r"\begin{tabular}{lrrrrr}\toprule",
        r"manifest & Success [95\% CI] & Detect@1 & $c/C^\ast$ & F10 Succ. & F4 Succ.\\\midrule",
    ]
    for p in P:
        label = r"absent ($p=1$)" if p == "1.0" else f"$p={float(p):g}$"
        lines.append(row(label, f"drop_true|p={p}"))
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
