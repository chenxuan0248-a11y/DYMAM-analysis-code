"""evaluation/ranking — top-K ranking quality on the CVE list.

The paper reports two complementary ranking views over the per-host CVE list:

* **Coverage at K** — Recall@K, Precision@K. "Did the K most-urgent CVEs cover
  the ones that actually matter?"
* **Order quality** — MAP, MRR, nDCG@K. Independent of K and rewards putting
  relevant CVEs near the top.

Score function
--------------
``score_cve_finding`` is the paper's deterministic urgency score and is shared
with the dashboard:

    score = (cvss / 10) · w_cvss + epss · w_epss + 𝟙[is_kev] · w_kev
          + (retrieval_score or 0) · w_rag

Defaults match the dashboard ranker; pass ``weights=`` to ablate. Missing
fields are coalesced to 0 — pre-2015 CVEs that only have CVSS v2 fall back via
``coalesce(cvss_v3, cvss_v2, 0)`` so they remain rankable instead of dropping
to the bottom.

Relevance label is ``cves_weak`` (cf. :mod:`evaluation.metrics` on why); the
ranking metrics are computed per host then **macro-averaged** across hosts so
small-CVE hosts don't get drowned by giant-CVE hosts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

DEFAULT_WEIGHTS = {"cvss": 0.5, "epss": 0.3, "kev": 0.15, "rag": 0.05}


def score_cve_finding(c: dict, weights: dict | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    cvss = c.get("cvss_v3_score")
    if cvss is None:
        cvss = c.get("cvss_v2_score") or 0.0
    cvss = float(cvss) / 10.0
    epss = float(c.get("epss_score") or 0.0)
    kev = 1.0 if c.get("is_kev") else 0.0
    rag = float(c.get("retrieval_score") or 0.0)
    return cvss * w["cvss"] + epss * w["epss"] + kev * w["kev"] + rag * w["rag"]


@dataclass(frozen=True)
class RankingResult:
    name: str            # e.g. "recall@10"
    value: float
    n_hosts: int         # hosts that contributed (had ≥1 GT CVE)

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "n_hosts": self.n_hosts}


# ─────────────────────────────── per-host primitives ────────────────────────
def _ranked_ids(cves: Iterable[dict], weights: dict | None) -> list[str]:
    scored = [((c.get("cve_id") or "").strip().upper(), score_cve_finding(c, weights))
              for c in cves if isinstance(c, dict)]
    scored = [s for s in scored if s[0]]
    scored.sort(key=lambda x: (-x[1], x[0]))   # stable tiebreak by id
    return [cid for cid, _ in scored]


def _recall_at_k(ranked: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    return len(set(ranked[:k]) & truth) / len(truth)


def _precision_at_k(ranked: list[str], truth: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(ranked[:k]) & truth) / k


def _average_precision(ranked: list[str], truth: set[str]) -> float:
    if not truth:
        return 0.0
    hits, ap = 0, 0.0
    for i, cid in enumerate(ranked, 1):
        if cid in truth:
            hits += 1
            ap += hits / i
    return ap / len(truth)


def _reciprocal_rank(ranked: list[str], truth: set[str]) -> float:
    for i, cid in enumerate(ranked, 1):
        if cid in truth:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(ranked: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    gains = np.array([1.0 if cid in truth else 0.0 for cid in ranked[:k]])
    if gains.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    ideal_hits = min(len(truth), k)
    ideal = float((np.ones(ideal_hits) * (1.0 / np.log2(np.arange(2, ideal_hits + 2)))).sum())
    return dcg / ideal if ideal else 0.0


# ─────────────────────────────────── public ─────────────────────────────────
def rank_cves(
    predicted: list[dict],
    gt: dict,
    *,
    ks: tuple[int, ...] = (5, 10, 20),
    weights: dict | None = None,
) -> list[RankingResult]:
    """Macro-averaged ranking metrics across hosts that have ≥1 weak-GT CVE."""
    truth_map: dict[str, set[str]] = {}
    for s in gt.get("subdomains", []):
        h = (s.get("name") or "").strip().lower().rstrip(".")
        truth_map[h] = {c.strip().upper() for c in s.get("cves_weak", [])}

    per_host: dict[str, list[float]] = {f"recall@{k}": [] for k in ks}
    per_host.update({f"precision@{k}": [] for k in ks})
    per_host.update({f"ndcg@{k}": [] for k in ks})
    per_host.update({"map": [], "mrr": []})

    for a in predicted:
        h = (a.get("name") or "").strip().lower().rstrip(".")
        truth = truth_map.get(h, set())
        if not truth:
            continue
        ranked = _ranked_ids(a.get("cves") or [], weights)
        for k in ks:
            per_host[f"recall@{k}"].append(_recall_at_k(ranked, truth, k))
            per_host[f"precision@{k}"].append(_precision_at_k(ranked, truth, k))
            per_host[f"ndcg@{k}"].append(_ndcg_at_k(ranked, truth, k))
        per_host["map"].append(_average_precision(ranked, truth))
        per_host["mrr"].append(_reciprocal_rank(ranked, truth))

    out: list[RankingResult] = []
    for name, vals in per_host.items():
        v = float(np.mean(vals)) if vals else 0.0
        out.append(RankingResult(name=name, value=v, n_hosts=len(vals)))
    return out
