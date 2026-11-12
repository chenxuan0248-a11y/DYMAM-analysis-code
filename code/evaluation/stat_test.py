"""evaluation/stat_test — significance + confidence intervals for paper tables.

We avoid any test that assumes normality across the 5-org × n-seed sample (too
small, often skewed). The shipped tests are:

* :func:`bootstrap_ci` — percentile bootstrap CI for one sample.
* :func:`paired_bootstrap` — two-sided paired test on the mean-difference, the
  default for "system A vs B over matched (org, seed) cells". Effect size is
  reported as the mean of paired deltas (no standardization — units are the
  metric's own, which is what reviewers expect to see).
* :func:`wilcoxon` — paired, distribution-free fallback. Useful when n is tiny
  (≤10 matched pairs) where the bootstrap collapses to a few unique resamples.
* :func:`benjamini_hochberg` — BH-FDR over the family of metric p-values, so
  the paper can report q-values instead of vanilla p-values when many metrics
  are compared simultaneously.

Determinism: the bootstrap uses ``numpy.random.default_rng(seed)`` so any test
is reproducible from the seed alone. Default ``B=10_000`` is the size most
NLP/IR venues expect; bump it for marginal p-values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    test: str
    effect: float        # mean(a) − mean(b)  (or mean(a) for one-sample)
    p_value: float
    ci_low: float
    ci_high: float
    n: int

    def to_dict(self) -> dict:
        return {"test": self.test, "effect": self.effect, "p_value": self.p_value,
                "ci_low": self.ci_low, "ci_high": self.ci_high, "n": self.n}


# ────────────────────────────── one-sample CI ───────────────────────────────
def bootstrap_ci(
    x: Sequence[float],
    *,
    B: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean. Returns (mean, ci_low, ci_high)."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(B, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


# ───────────────────────── two-sample paired tests ──────────────────────────
def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    B: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> TestResult:
    """Two-sided paired bootstrap on mean(a) − mean(b).

    p-value is the standard "achieved significance level": the fraction of
    bootstrap mean-deltas under the null (centred at zero) at least as extreme
    as the observed delta. ``a`` and ``b`` must be matched cell-by-cell."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError("paired_bootstrap: a and b must have equal length")
    n = arr_a.size
    if n == 0:
        return TestResult("paired_bootstrap", 0.0, 1.0, 0.0, 0.0, 0)

    delta = arr_a - arr_b
    observed = float(delta.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    resampled = delta[idx].mean(axis=1)

    # CI: percentile bootstrap of the delta
    lo, hi = np.quantile(resampled, [alpha / 2, 1 - alpha / 2])

    # p-value: centre resample under H0 (mean=0), count |t*| ≥ |t_obs|
    centred = resampled - resampled.mean()
    p = float((np.abs(centred) >= abs(observed)).mean())
    # avoid reporting p=0 when no resample exceeded (Davison & Hinkley §4.2.3)
    p = max(p, 1.0 / B)

    return TestResult("paired_bootstrap", observed, p, float(lo), float(hi), n)


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Wilcoxon signed-rank, two-sided. Zero-difference pairs are dropped
    (``zero_method='wilcox'``)."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError("wilcoxon: a and b must have equal length")
    delta = arr_a - arr_b
    nz = delta[delta != 0]
    if nz.size == 0:
        return TestResult("wilcoxon", 0.0, 1.0, 0.0, 0.0, 0)
    try:
        res = stats.wilcoxon(arr_a, arr_b, zero_method="wilcox", alternative="two-sided")
        p = float(res.pvalue)
    except ValueError:
        # n too small after zero-drop
        p = 1.0
    effect = float(delta.mean())
    # CI from the bootstrap of the same delta — wilcoxon doesn't produce one
    _, lo, hi = bootstrap_ci(delta.tolist())
    return TestResult("wilcoxon", effect, p, lo, hi, int(arr_a.size))


# ────────────────────────────── effect size ─────────────────────────────────
def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> dict:
    """Cliff's delta, a non-parametric effect size = P(a>b) − P(a<b) over all
    cross-pairs, in [-1, 1]. Distribution-free (no normality / no equal-variance
    assumption), which is why it pairs with wilcoxon/bootstrap here. Magnitude
    labels follow the common Romano et al. thresholds:
    |δ|<0.147 negligible, <0.33 small, <0.474 medium, else large.
    δ=+1 means every a exceeds every b (complete separation)."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    n, m = arr_a.size, arr_b.size
    if n == 0 or m == 0:
        return {"delta": 0.0, "magnitude": "undefined", "n_a": int(n), "n_b": int(m)}
    diff = arr_a[:, None] - arr_b[None, :]
    delta = float((np.sum(diff > 0) - np.sum(diff < 0)) / (n * m))
    ad = abs(delta)
    mag = ("negligible" if ad < 0.147 else "small" if ad < 0.33
           else "medium" if ad < 0.474 else "large")
    return {"delta": round(delta, 4), "magnitude": mag, "n_a": int(n), "n_b": int(m)}


# ────────────────────────── multiple-comparison correction ──────────────────
def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> list[dict]:
    """BH-FDR. Returns one row per input p in input order with q-value and
    reject flag at level ``alpha``."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    # raw BH q: p_(k) * n / k
    raw_q = ranked * n / np.arange(1, n + 1)
    # enforce monotonicity from the back so q is non-decreasing in p_(k)
    q_sorted = np.minimum.accumulate(raw_q[::-1])[::-1].clip(max=1.0)
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    reject = q <= alpha
    return [{"p_value": float(p[i]), "q_value": float(q[i]), "reject": bool(reject[i])}
            for i in range(n)]
