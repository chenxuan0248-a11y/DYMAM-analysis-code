"""evaluation/calibration — ECE, MCE, Brier, reliability bins.

Two calibration views are reported in the paper:

* **Asset confidence calibration.** ``Asset.confidence`` is set by EvidenceFusion
  (and revised by Reflection); the paired truth label is GT membership
  (the host appears in ``data/gt/<org>_gt.json::subdomains[]``). This measures
  whether the fusion layer is honest about uncertain hosts.
* **CVE retrieval calibration.** ``CVEFinding.retrieval_score`` (cosine score
  from the cve_rag corpus) paired with ``cves_weak`` membership measures whether
  the RAG layer's similarity is a usable probability.

The two views share one bin-based ECE/MCE implementation; reliability bin
tables are returned alongside the scalar so callers (figure_gen) can render the
reliability diagram without redoing the binning.

Conventions
-----------
* **Equal-width binning** with ``n_bins=15``: matches Guo et al. (ICML'17) and
  is what most calibration papers report. Equal-mass binning is also exposed
  (``scheme="quantile"``) for the Nixon et al. (CVPRW'19) view.
* Scalars are weighted by bin mass; empty bins contribute 0.
* Brier score is returned uncalibrated (not decomposed) — the bin table is
  sufficient for the reliability-vs-resolution decomposition downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    n: int
    ece: float          # Expected Calibration Error
    mce: float          # Maximum Calibration Error
    brier: float        # Mean Brier score
    bins: list[dict]    # one row per bin: {lo, hi, n, conf_mean, acc, gap}

    def to_dict(self) -> dict:
        return {"n": self.n, "ece": self.ece, "mce": self.mce, "brier": self.brier,
                "bins": self.bins}


def _bin_edges(scheme: str, n_bins: int, conf: np.ndarray) -> np.ndarray:
    if scheme == "quantile":
        # equal-mass bins; clip to [0,1] and de-dup to avoid zero-width edges
        edges = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    return np.linspace(0.0, 1.0, n_bins + 1)


def calibration(
    confidences: Iterable[float],
    correct: Iterable[bool],
    *,
    n_bins: int = 15,
    scheme: Literal["uniform", "quantile"] = "uniform",
) -> CalibrationResult:
    """Reliability-bin ECE / MCE / Brier for a (confidence, is_correct) sample.

    Empty inputs are tolerated and yield ``n=0`` with zeroed scalars so the
    caller doesn't have to special-case orgs that produced no rows of a given
    kind (e.g. a small GT with no fingerprinted CPEs)."""
    c = np.asarray(list(confidences), dtype=float)
    y = np.asarray(list(correct), dtype=bool).astype(float)
    if c.size == 0:
        return CalibrationResult(0, 0.0, 0.0, 0.0, [])

    c = np.clip(c, 0.0, 1.0)
    edges = _bin_edges(scheme, n_bins, c)
    # right-edge inclusive only for the last bin
    idx = np.clip(np.digitize(c, edges[1:-1], right=False), 0, len(edges) - 2)

    bins: list[dict] = []
    ece = mce = 0.0
    total = c.size
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]),
                         "n": 0, "conf_mean": 0.0, "acc": 0.0, "gap": 0.0})
            continue
        conf_mean = float(c[mask].mean())
        acc = float(y[mask].mean())
        gap = abs(conf_mean - acc)
        ece += (n / total) * gap
        mce = max(mce, gap)
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "n": n, "conf_mean": conf_mean, "acc": acc, "gap": gap})

    brier = float(np.mean((c - y) ** 2))
    return CalibrationResult(int(total), float(ece), float(mce), brier, bins)


# ─────────────────────────────── adapters ───────────────────────────────────
def asset_confidence_pairs(predicted: list[dict], gt: dict) -> tuple[list[float], list[bool]]:
    """(Asset.confidence, in_GT) pairs across every predicted asset."""
    truth = {(s.get("name") or "").strip().lower().rstrip(".")
             for s in gt.get("subdomains", [])}
    conf, correct = [], []
    for a in predicted:
        name = (a.get("name") or "").strip().lower().rstrip(".")
        conf.append(float(a.get("confidence", 0.0)))
        correct.append(name in truth)
    return conf, correct


def cve_retrieval_pairs(predicted: list[dict], gt: dict) -> tuple[list[float], list[bool]]:
    """(CVEFinding.retrieval_score, host has this CVE in cves_weak) pairs.

    CVEs without a ``retrieval_score`` (e.g. NVD-lookup CVEs, not RAG-sourced)
    are dropped — calibration of a non-probabilistic ranker is undefined."""
    truth_map: dict[str, set[str]] = {}
    for s in gt.get("subdomains", []):
        h = (s.get("name") or "").strip().lower().rstrip(".")
        truth_map[h] = {c.strip().upper() for c in s.get("cves_weak", [])}
    conf, correct = [], []
    for a in predicted:
        h = (a.get("name") or "").strip().lower().rstrip(".")
        for c in a.get("cves", []) or []:
            if not isinstance(c, dict):
                continue
            score = c.get("retrieval_score")
            if score is None:
                continue
            cid = (c.get("cve_id") or "").strip().upper()
            conf.append(float(score))
            correct.append(cid in truth_map.get(h, set()))
    return conf, correct
