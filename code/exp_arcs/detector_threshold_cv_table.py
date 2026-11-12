"""Compact LOOO threshold-CV table from detector_threshold_cv_v1.json.

Reports what the CV actually shows: the selected threshold is stable across folds,
plus the per-org held-out F1 spread. It does NOT claim threshold optimality.

Correction 2026-10-08: earlier versions reported the train-minus-heldout F1 gap
(mean -0.000) as evidence of "no systematic optimism". Train F1 is the plain mean
of the 19 training orgs' per-org F1, and all folds select the same t*, so the
fold-averaged gap is 0 by construction and the per-fold gap equals
20/19 * (mean - heldout_i). Both gap rows were removed.
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

D = Path("experiments/exp_arcs")
OUT = D / "detector_threshold_cv_table.tex"


def main() -> int:
    d = json.loads((D / "detector_threshold_cv_v1.json").read_text())
    folds = d["folds"]
    sel = d["selected_t"]
    ho = [f["heldout_f1"] for f in folds]
    ho_lo, ho_hi, ho_sd = min(ho), max(ho), st.pstdev(ho)

    lines = [
        r"\begin{table}[t]\centering\small",
        (r"\caption{Leave-one-organization-out cross-validation of the peer-anomaly "
         r"threshold (detection, host grain, $20$ folds). For each fold, $t^\ast$ is "
         r"selected by $\arg\max$ mean detection-$F_1$ on the $19$ training orgs and "
         r"evaluated on the held-out org. All folds select $t^\ast\!=\!"
         + f"{sel['mode']}" + r"$, so the selection is stable when any one "
         r"organization is left out, and each held-out value is that organization's "
         r"value at $t^\ast$. Because every fold selects the same threshold, the "
         r"fold-averaged train$-$held-out $F_1$ gap is zero by construction and the "
         r"per-fold gap is a rescaling of the held-out deviation; neither is evidence "
         r"about selection optimism, so neither is reported.}"),
        r"\label{tab:threshold-cv}",
        r"\begin{tabular}{lr}\toprule",
        r"quantity & value\\\midrule",
        f"selected $t^\\ast$ (unique across 20 folds) & {sel['unique']}\\\\",
        f"held-out $F_1$, mean & {st.mean(ho):.3f}\\\\",
        f"\\quad held-out $F_1$ range (per org) & $[{ho_lo:.3f},\\,{ho_hi:.3f}]$\\\\",
        f"\\quad held-out $F_1$ std across folds & {ho_sd:.3f}\\\\",
        f"held-out mean precision / recall & {d['heldout_mean']['precision']:.3f} / {d['heldout_mean']['recall']:.3f}\\\\",
        f"held-out mean clean-run alert rate & {d['heldout_mean']['clean_fa']:.3f}\\\\",
        r"\midrule",
        (r"\multicolumn{2}{p{.82\linewidth}}{\footnotesize The per-org $F_1$ spread "
         r"reflects organization heterogeneity. The CV says little beyond selection "
         r"stability: any two training sets share $18$ of $19$ organizations, and "
         r"detection-$F_1$ is near-monotone in $t$ (see threats to validity). "
         r"$t^\ast$ is a detection-level optimum, not an end-to-end repair "
         r"optimum.}\\"),
        r"\bottomrule\end{tabular}\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
