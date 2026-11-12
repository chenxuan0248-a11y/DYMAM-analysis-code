"""Build the frozen cost/harm/reversibility spec table from cost_harm_spec.yaml
(no hand-typed numbers) -- the concrete numeric spec behind every cost/harm/rho_C
number in the paper (Sec. Approach/RQ3/RQ4), previously never shown to the reader.

    poetry run python -m experiments.exp_arcs.cost_harm_spec_table
"""
from __future__ import annotations
from pathlib import Path

from experiments.exp_arcs.cost_harm import load_spec, spec_version_hash

OUT = Path("experiments/exp_arcs/cost_harm_spec_table.tex")

REV_LABEL = {
    "idempotent": "idempotent",
    "reversible": "reversible",
    "compensatable": "compensatable",
    "irreversible": "irreversible",
}


def main() -> int:
    spec = load_spec()
    actions = spec["actions"]
    fam_to_action = spec["family_to_action"]

    action_to_families: dict[str, list[str]] = {a: [] for a in actions}
    for fam, act in fam_to_action.items():
        action_to_families.setdefault(act, []).append(fam)

    rows = []
    for name, row in actions.items():
        fams = action_to_families.get(name, [])
        fam_str = r", \allowbreak ".join(f.replace("_", r"\_") for f in fams) if fams else r"\emph{unused}"
        rows.append(
            f"\\texttt{{{name.replace('_', chr(92)+'_')}}} & {row['cost']} & "
            f"{REV_LABEL[row['reversibility']]} & {row['harm']} & {fam_str}\\\\"
        )

    unused = [a for a, fams in action_to_families.items() if not fams]
    unused_str = ", ".join(f"\\texttt{{{a.replace('_', chr(92)+'_')}}}" for a in unused)
    h = spec_version_hash()

    lines = [
        r"\begin{table}[t]\centering\small",
        (r"\caption{Frozen repair-action cost/reversibility/harm spec "
         r"(\S\ref{sec:approach}; \texttt{cost\_harm\_spec.yaml}, "
         f"spec\\_version={spec['spec_version']}, sha256={h[:12]}\\ldots), "
         r"the concrete numbers behind every $c(v)$, $h(v)$, $\rho_C$, cost-saving, "
         r"and harm-saving figure in \S\ref{sec:rq3}--\S\ref{sec:rq4}. "
         r"``source family'' is the evidence family (from the attributed DAG) that "
         r"this action regenerates when re-run. "
         f"{unused_str} are defined in the frozen spec but never invoked by any "
         r"family on this passive-recon corpus --- the concrete reason the harm "
         r"budget and irreversibility ban are inert here (\S\ref{sec:disc},"
         r"\S\ref{sec:threats}), not an omission from this table. "
         r"\texttt{nmap\_scan} is a generic active port-scan action; the pipeline's "
         r"passive InternetDB lookup tool (code name \texttt{nmap}) belongs to "
         r"active\_observation and is costed as \texttt{http\_probe}.}"),
        r"\label{tab:cost-harm-spec}",
        r"\begin{tabular}{lrlrl}\toprule",
        r"action & cost $c(v)$ & reversibility & harm $h(v)$ & source family\\\midrule",
        *rows,
        r"\bottomrule\end{tabular}\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
