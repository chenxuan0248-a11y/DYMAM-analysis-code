"""Build the full D1 (H2 evidence-redundancy pre-registration) per-claim-type
table from s2_d1_frozen_v1.json (no hand-typed numbers) -- the appendix
companion to the pooled-only summary in RQ1 (Sec. rq1: "0.000 ... pooled 0.285").

    poetry run python -m experiments.exp_arcs.d1_pre_registration_table
"""
from __future__ import annotations
import json
from pathlib import Path

J = Path("experiments/exp_arcs/s2_d1_frozen_v1.json")
OUT = Path("experiments/exp_arcs/d1_pre_registration_table.tex")

# (json key, display name, bold=substantive type)
CONTEXT_TYPES = [
    ("asset_identity", "asset\\_identity", False),
    ("organization_association", "organization\\_association", False),
    ("service_observation", "\\textbf{service\\_observation}", True),
    ("technology_identification", "\\textbf{technology\\_identification}", True),
    ("vulnerability_applicability", "\\textbf{vulnerability\\_applicability}", True),
]


def row(name: str, r: dict, bold_cov: bool = False) -> str:
    cov = f"{r['coverage_2plus']:.3f}"
    if bold_cov:
        cov = f"\\textbf{{{cov}}}"
    return (f"{name} & {r['n']} & {r['P(R=1)']:.4f} & {r['P(R=2)']:.4f} & "
            f"{r['P(R>=3)']:.4f} & {cov}\\\\")


def main() -> int:
    d = json.loads(J.read_text())
    pct = d["per_claim_type_indep"]
    sub = d["d1_substantive_indep"]
    ov = d["overall_indep"]

    rows = [row(name, pct[key], bold) for key, name, bold in CONTEXT_TYPES]

    lines = [
        r"\begin{table}[t]\centering\small",
        (r"\caption{Full D1 pre-registered evidence-redundancy survey "
         r"(\S\ref{sec:design}, \S\ref{sec:rq1}; "
         f"$n={d['n_claims']}$ claims, ${len(d['orgs'])}$ orgs), the per-claim-type "
         r"breakdown behind the pooled numbers reported in \S\ref{sec:rq1}. $R(c)$ is "
         r"the count of independent source families "
         + ", ".join(f"\\texttt{{{f.replace('_', chr(92) + '_')}}}" for f in d["indep_families"])
         + r" supporting claim $c$; Coverage$_{2+}=P(R\geq2)$. The three "
         r"\textbf{bold} rows are the substantive claim types --- the ones ARCS "
         r"actually repairs --- and are the D1 primary metric (pooled "
         f"$n={sub['n']}$, Coverage$_{{2+}}={sub['coverage_2plus']:.3f}$, frozen "
         f"threshold ${d['d1_threshold']}$, verdict "
         + d["d1_verdict"].replace("_", " ") +
         r"); \texttt{asset\_identity}/\texttt{organization\_association} and the "
         r"pooled-overall row are context only, not the primary metric, since "
         r"redundancy in asset existence/org attribution does not make a service, "
         r"technology, or CVE finding substitutable.}"),
        r"\label{tab:d1-full}",
        r"\begin{tabular}{lrrrrr}\toprule",
        r"claim type & $n$ & $P(R=1)$ & $P(R=2)$ & $P(R\geq3)$ & Coverage$_{2+}$\\\midrule",
        *rows,
        r"\midrule",
        (f"\\textbf{{substantive pooled (D1 primary)}} & {sub['n']} & "
         f"{sub['P(R=1)']:.4f} & {sub['P(R=2)']:.4f} & {sub['P(R>=3)']:.4f} & "
         f"\\textbf{{{sub['coverage_2plus']:.3f}}}\\\\"),
        (f"overall (context) & {ov['n']} & {ov['P(R=1)']:.4f} & {ov['P(R=2)']:.4f} & "
         f"{ov['P(R>=3)']:.4f} & {ov['coverage_2plus']:.3f}\\\\"),
        r"\bottomrule\end{tabular}\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
