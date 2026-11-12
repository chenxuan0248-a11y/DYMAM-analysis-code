"""Compact injected-fault representativeness table from
natural_fault_prevalence_v1.json (no hand-typed numbers)."""
from __future__ import annotations
import json
from pathlib import Path

D = Path("experiments/exp_arcs")
OUT = D / "natural_fault_prevalence_table.tex"
FAMS = ["dns", "active_observation", "certificate", "registration"]


def main() -> int:
    d = json.loads((D / "natural_fault_prevalence_v1.json").read_text())
    f4 = d["F4_family_omission_analog"]
    bf = d["F4_by_injected_family"]
    f10 = d["F10_host_omission_analog"]

    def rowfam(fam):
        b = bf[fam]
        nm = b["mean_natural_missing_rate_when_expected"]
        nm_s = f"{nm:.4f}" if nm is not None else "-- (rare)"
        return (f"{fam.replace('_', chr(92)+'_')} & {b['mean_base_rate']:.3f} & "
                f"{b['expected_in_n_orgs']}/20 & {nm_s}\\\\")

    lines = [
        r"\begin{table}[t]\centering\small",
        (r"\caption{Representativeness of the injected faults, measured on the $20$ "
         r"\emph{clean} (un-injected) runs at the same $\geq50\%$ peer base-rate as the "
         r"detector. The F4 \emph{structural form} --- a host lacking a peer-common "
         r"evidence family --- is present naturally on $" +
         f"{f4['mean_rate_hosts_missing_expected_family']*100:.1f}" + r"\%$ of hosts "
         r"(range $[" + f"{f4['range'][0]*100:.0f},{f4['range'][1]*100:.0f}" +
         r"]\%$); we claim only that this form is not an artifact we introduced, not "
         r"that these hosts are confirmed faults. Per injected family: "
         r"active\_observation omission is common naturally, dns omission rare-but-real, "
         r"and certificate/registration are never peer-common (base rate $\approx0.01$), "
         r"so injecting those probes the natural rare-family detection floor. F10 "
         r"(host omission) has \textbf{no} natural analog ($" +
         f"{f10['total_manifest_dropped_hosts_on_clean_runs']}" + r"$ resolved-but-unused "
         r"hosts on clean runs, threshold-independent) and is justified on threat-model "
         r"grounds rather than empirical frequency: discovery completes normally and its "
         r"manifest is captured before the drop (\S\ref{sec:approach}), but the drop still "
         r"measurably shifts the output (\S\ref{sec:rq1}) and, absent an independent host "
         r"manifest, cannot be judged from a single deployed run --- this is a distinct "
         r"rationale from F4's, not the same output-invisibility argument.}"),
        r"\label{tab:representativeness}",
        r"\begin{tabular}{lrrr}\toprule",
        r"injected family (F4) & mean base rate & peer-common in & natural miss rate\\",
        r" & (clean) & \#orgs & (when peer-common)\\\midrule",
        *[rowfam(f) for f in FAMS],
        r"\midrule",
        (r"\multicolumn{4}{p{.9\linewidth}}{\footnotesize These clean-run family-sparse "
         r"hosts are exactly the hosts that constitute the detector's clean-run alert "
         r"rate (\S\ref{sec:rq2}) --- one phenomenon in two roles, not double-counted. "
         r"Absent per-host ground truth we cannot tell a genuine natural omission from a "
         r"host that legitimately lacks the family, so we claim structural "
         r"representativeness only, never that a given natural instance is a fault.}\\"),
        r"\bottomrule\end{tabular}\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
