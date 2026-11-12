"""Reviewer item B — PR/ROC figure from detector_pr_roc_v1.json (pure plotting).

Two panels: (left) pooled PR curve with AUC-PR + the 1.3% random baseline;
(right) ROC with AUC-ROC. The t=0.5 (frozen) and t=0.9 operating points are
marked on the knob curve to show precision/false-alarm are a threshold choice.

    poetry run python -m experiments.exp_arcs.detector_pr_roc_figure
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("experiments/exp_arcs")


def main() -> int:
    d = json.loads((D / "detector_pr_roc_v1.json").read_text())
    pr = d["pr_curve"]
    roc = d["roc_curve"]
    knob = d["knob_curve"]
    base = d["baseline_pr"]

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.35))

    # ---- PR ----
    a1.plot([p[0] for p in pr], [p[1] for p in pr], "-", lw=2, color="#1b6ca8",
            label=f"Detector (AUC-PR {d['auc_pr']})")
    a1.axhline(base, color="#777", ls=":", lw=1, label=f"Random ({base})")
    # knob operating points
    for k, mk, col in [(0.5, "o", "#c1440e"), (0.9, "s", "#2a7a2a")]:
        kp = next(x for x in knob if x["t"] == k)
        a1.plot(kp["recall"], kp["precision"], mk, ms=9, color=col,
                label=f"t={k}")
    a1.set_xlabel("Recall"); a1.set_ylabel("Precision")
    a1.set_xlim(-0.02, 1.02); a1.set_ylim(-0.02, 1.02)
    a1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    a1.grid(alpha=0.25)

    # ---- ROC ----
    a2.plot([r[0] for r in roc], [r[1] for r in roc], "-", lw=2, color="#1b6ca8",
            label=f"Detector (AUC-ROC {d['auc_roc']})")
    a2.plot([0, 1], [0, 1], ":", color="#999", lw=1, label="chance")
    a2.set_xlabel("False-positive rate"); a2.set_ylabel("True-positive rate (recall)")
    a2.set_xlim(-0.02, 1.02); a2.set_ylim(-0.02, 1.02)
    a2.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    a2.grid(alpha=0.25)

    fig.tight_layout(pad=1.0)
    out = D / "detector_pr_roc.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
