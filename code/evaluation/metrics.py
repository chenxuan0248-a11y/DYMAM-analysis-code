"""evaluation/metrics — set-based correctness against data/gt/<org>_gt.json.

Two layers of granularity are reported because the paper's threat model treats
them differently:

* **Asset-level** (subdomain / host / port / technology) measures recon
  coverage — the question "did we find what's actually there?". GT comes from
  ``data/gt/<org>_gt.json::subdomains[]``.
* **CVE-level** uses the weak ground-truth (``cves_weak``: every CVE whose CPE
  prefix matches the host's fingerprinted CPE). It is intentionally a *recall
  ceiling*, not a precision oracle — the paper reports Recall, Recall@K and
  CVE-level F1 only against ``cves_weak`` and is explicit about its
  over-inclusiveness. For ranking quality on top of Recall, see
  :mod:`evaluation.ranking`.

All functions return :class:`MetricResult` (a frozen dataclass) so the raw
``tp/fp/fn`` are preserved for downstream bootstrap CIs and significance tests.
``score_assets`` and ``score_cves`` are the two entrypoints :mod:`_shared.metrics`
composes into the paper's result row.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class MetricResult:
    name: str
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(precision=self.precision, recall=self.recall, f1=self.f1)
        return d


def _norm_host(name: str) -> str:
    return (name or "").strip().lower().rstrip(".")


def _norm_cve(cid: str) -> str:
    return (cid or "").strip().upper()


def _norm_cpe(cpe: str) -> str:
    # Drop version + edition fragments for a recall-friendly vendor:product key.
    # Matches the GT generator's "cpe-vendor-product-prefix" convention.
    if not cpe:
        return ""
    parts = cpe.lower().replace("cpe:2.3:", "cpe:/").split(":")
    return ":".join(parts[:5])


def _confusion(pred: Iterable[str], truth: Iterable[str]) -> tuple[int, int, int]:
    p, t = set(pred), set(truth)
    p.discard("")
    t.discard("")
    tp = len(p & t)
    fp = len(p - t)
    fn = len(t - p)
    return tp, fp, fn


# ───────────────────────── asset-level (subdomain / host / port / tech) ─────
def score_subdomains(predicted: Iterable[dict], gt_subdomains: Iterable[dict]) -> MetricResult:
    pred = {_norm_host(p.get("name", "")) for p in predicted}
    truth = {_norm_host(t.get("name", "")) for t in gt_subdomains}
    tp, fp, fn = _confusion(pred, truth)
    return MetricResult("subdomain", tp, fp, fn)


def score_hosts(predicted: Iterable[dict], gt_subdomains: Iterable[dict]) -> MetricResult:
    """Resolved-IP coverage. A host counts as predicted only if it carries ≥1 IP."""
    pred = {_norm_host(p["name"]) for p in predicted if p.get("ips")}
    truth = {_norm_host(t["name"]) for t in gt_subdomains if t.get("ips")}
    tp, fp, fn = _confusion(pred, truth)
    return MetricResult("host", tp, fp, fn)


def score_ports(predicted: Iterable[dict], gt_subdomains: Iterable[dict]) -> MetricResult:
    """(host, port) pair coverage — a port is only credited under the right host."""
    pred = {f"{_norm_host(p['name'])}:{int(port)}"
            for p in predicted for port in p.get("ports", [])}
    truth = {f"{_norm_host(t['name'])}:{int(port)}"
             for t in gt_subdomains for port in t.get("ports", [])}
    tp, fp, fn = _confusion(pred, truth)
    return MetricResult("port", tp, fp, fn)


def score_cpes(predicted: Iterable[dict], gt_subdomains: Iterable[dict]) -> MetricResult:
    """Fingerprint coverage at vendor:product granularity (see ``_norm_cpe``)."""
    pred = {f"{_norm_host(p['name'])}|{_norm_cpe(c)}"
            for p in predicted for c in p.get("cpes", [])}
    truth = {f"{_norm_host(t['name'])}|{_norm_cpe(c)}"
             for t in gt_subdomains for c in t.get("cpes", [])}
    tp, fp, fn = _confusion(pred, truth)
    return MetricResult("cpe", tp, fp, fn)


def score_assets(predicted: Iterable[dict], gt: dict) -> dict[str, MetricResult]:
    """All asset-level metrics in one pass. ``gt`` is the loaded ``<org>_gt.json``."""
    pred = list(predicted)
    truth = gt.get("subdomains", [])
    return {
        "subdomain": score_subdomains(pred, truth),
        "host":      score_hosts(pred, truth),
        "port":      score_ports(pred, truth),
        "cpe":       score_cpes(pred, truth),
    }


# ───────────────────────────────────── CVE-level ────────────────────────────
def _gt_cve_map(gt_subdomains: Iterable[dict], weak: bool) -> dict[str, set[str]]:
    key = "cves_weak" if weak else "cves"
    return {_norm_host(s["name"]): {_norm_cve(c) for c in s.get(key, [])}
            for s in gt_subdomains}


def _pred_cve_map(predicted: Iterable[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for a in predicted:
        cves: set[str] = set()
        for c in a.get("cves", []):
            cid = _norm_cve(c.get("cve_id", "") if isinstance(c, dict) else c)
            if cid:
                cves.add(cid)
        out[_norm_host(a["name"])] = cves
    return out


def score_cves(predicted: Iterable[dict], gt: dict, *, weak: bool = True) -> MetricResult:
    """Micro-averaged CVE recall/precision over all (host, CVE) pairs.

    ``weak=True`` is the paper's default — see the module docstring on what
    cves_weak does and doesn't claim. ``weak=False`` is only meaningful once a
    strong CVE oracle is populated (currently empty in the shipped GT)."""
    pred = _pred_cve_map(predicted)
    truth = _gt_cve_map(gt.get("subdomains", []), weak)
    pred_pairs = {(h, c) for h, cs in pred.items() for c in cs}
    truth_pairs = {(h, c) for h, cs in truth.items() for c in cs}
    tp, fp, fn = _confusion(pred_pairs, truth_pairs)
    return MetricResult("cve_weak" if weak else "cve", tp, fp, fn)
