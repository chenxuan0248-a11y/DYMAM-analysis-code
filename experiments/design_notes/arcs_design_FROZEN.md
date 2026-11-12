# ARCS design — pre-registered (FROZEN 2026-09-17, before ARCS is built or run)

Phases 0–2 built and determinism-verified the **oracle** side: leaf grid
{F2,F4,F10}×5 orgs (`ilp_oracle_pilot_v1.json`, S*/C* + GATE) and the non-leaf
gate-logic oracle (`non_leaf_oracle_v1.json`, F8/F7). Nothing has measured **ARCS
the system** yet — no `Success@B₀`, `ρC`, or scope P/R exists, and there is no ARCS
selector code in the repo (verified 2026-09-17).

This file freezes ARCS's mechanism, budget, blindness, metrics, and baseline ladder
**before any ARCS run**, mirroring `D1_FROZEN.md` / `non_leaf_oracle_FROZEN.md`, so
ARCS cannot be tuned to its own oracle after the fact.

ARCS repairs **two fault classes**: evidence-layer faults (F2/F4/F10) via leaf
re-collection, and gate-logic faults (F8/F7) via non-leaf re_evaluate/re_render —
the latter detected by decision-consistency features f6/f7 (see `non_leaf_oracle_
FROZEN.md` **NL-1′**, which corrected the earlier NL-1 "boundary" claim on data).

**Pre-registration statement.** As of this freeze: ARCS is not implemented; no ARCS
repair, Success, ρC, or scope P/R number exists; the oracle S*/C* and D(f) labels
are frozen inputs, not ARCS outputs. Every threshold, weight, and feature named
below is fixed here, before the first ARCS run.

Rigor target: CCF-A. Guiding judgement (frozen): since **D1 = H2 Opportunistic**
(no substitution) the oracle already showed S* = forced minimal cover, so the
selection algorithm has little room to be "clever" — the real variance is in
**detection quality** and **budget pressure**, not optimization. The experiment is
therefore designed to **decompose the ρC gap** into detection-error vs
heuristic-loss vs budget, not to crown one selector.

---

## A-ARCS — detection model (blind)

ARCS must decide *what to repair* from the **faulted bundle only** (blindness, F-1),
never from clean / D(f) / oracle. Frozen model:

- **Primary signal = re-collection discrepancy at the evidence/leaf layer.** ARCS
  re-collects a leaf (deterministic replay ⇒ its clean cached value) and flags it
  contaminated iff the re-collected value ≠ the faulted bundle's value. The
  re-collected value is simultaneously the repair (the leaf B1 primitive). Cost is
  charged per `(target, source_family)` invocation (`dag.scope_cost`) — so ARCS
  cannot re-collect everything for free; that is the `repair-all` baseline.
**Frozen feature set `D_θ` (all computable from the faulted bundle alone; every
referenced field verified present 2026-09-17).** Two tiers:

*Tier-1 — evidence-layer (targets the leaf faults F2/F4/F10):*
- **f1 — provenance integrity (hard boolean):** `evidence.raw_hash ≠
  source_artifact[raw_ref].sha256` → tampered content. Catches **F4**. (Link
  verified resolvable, 200/200 consistent on a clean bundle.)
- **f2 — observation status (hard boolean):** `observation_status ≠ "success"` →
  failed/degraded collection. Catches **F2** (wrong-tool discovery failure).
- **f3 — structural integrity (hard boolean):** a decision/report references an
  `evidence_event_id` absent from the bundle (dangling ref), or a source_artifact
  has no derived evidence (dangling upstream). Catches **F10** (dropped records).
- **f4 — host truncation (relative, frozen quantile):** a host's evidence count
  `< median_host − 2·MAD`. Secondary **F10** signal.
- **f5 — weak-support completeness (relative, frozen quantile):** a claim's support
  count `< 25th percentile within its claim_type`. Secondary.

*Tier-2 — decision-consistency (targets the gate-logic faults F8/F7; data-verified):*
- **f6 — decision↔reason consistency (hard boolean):** `decision = report ∧
  reason_codes ⊅ corroborated_observation` (i.e. reason_codes ∈ {llm_only_evidence,
  cve_context_without_applicability, ranking_context_only}) → over-admission.
  **Catches F8 with perfect P/R** on mozilla (flags exactly the 61 over-admitted
  decisions; verified).
- **f7 — decision↔support consistency (hard boolean):** `decision ∈ {abstain,
  reject} ∧ support_state = supported` → a well-supported claim wrongly withheld.
  **Catches F7** (verified on the fixture: F7 sets decision=abstain but leaves
  support_state=supported).

**Composition (frozen):** Ĉ = { nodes flagged by ANY hard signal (f1/f2/f3/f6/f7) }
∪ { nodes flagged by ≥2 soft signals (f4/f5) }. Hard signals are threshold-free
boolean facts (unattackable); soft signals use pre-registered quantiles (2·MAD,
25th pct) and are gated ≥2 to control false positives. A Tier-1 flag maps to leaf
re-collection; a Tier-2 flag maps to non-leaf re_evaluate/re_render (B-ARCS).

**Detection is measured, not assumed.** Report `D_θ` precision/recall vs the true
D(f) (`dfs_pilot_v1.json`) as a first-class result, split Tier-1 (leaf) vs Tier-2
(gate-logic), so the paper separates detection quality from selection quality.

**Self-verifying (favour recall).** ARCS re-collects / re-evaluates a flagged node;
if the recomputed value equals the faulted value it was a false positive — wasted
budget, never a wrong repair. So false positives only cost budget (bounded by B₀)
while false negatives cause failure ⇒ `D_θ` is tuned toward **recall**.

**Consequence for gate-logic (NL-1′, corrected).** f6/f7 fire precisely on F8/F7
(verified), so ARCS detects them and repairs via re_evaluate/re_render → F8/F7 are
**expected successes**, not a boundary. (This corrects the original NL-1; see
`non_leaf_oracle_FROZEN.md` NL-1′.) The residual boundary is only a
fully-consistent-forgery fault (decision+support+reason all coherent), absent from
the current set — noted as future work.

> **Frozen default — features grounded in the data, not fault-specific hacks.** f1–f7
> are generic integrity/completeness/consistency checks any recon-repair system
> would run; that they happen to catch F2/F4/F10/F8/F7 is a property of those faults
> (each leaves a real inconsistency), not reverse-engineering — and `D_θ` is blind
> to which fault is present. The set is frozen here; any change is edited in BEFORE
> the first ARCS run.

---

## B-ARCS — selection algorithm

Given Ĉ (suspected set) and the attributed DAG under budget B₀. The **action /
candidate set** spans both classes: leaf re-collection invocations
`(target, source_family)` for Tier-1 flags, and non-leaf `re_evaluate` /
`re_render` actions (frozen block in `non_leaf_oracle.py`, cost=1, harm=0) for
Tier-2 flags. The selector treats them uniformly as costed actions covering
suspected outputs.

- **Primary = cost-aware greedy over the coverage structure.** Iteratively select
  the action (leaf invocation OR non-leaf re_evaluate/re_render) maximizing marginal
  `(# suspected outputs newly repaired) / cost`, until either all suspected outputs
  are covered or the budget B₀ is exhausted. Full-cover regime = weighted set-cover
  (greedy Hₙ ln-approx); budget-limited regime = budgeted max-coverage (greedy
  (1−1/e)-approx). Deterministic tie-break by invocation id (reproducibility).
- **Decomposition partner = blind ILP.** The SAME solver as `ilp_oracle.py`, fed Ĉ
  (detection output) instead of ground-truth D(f), same frozen `ilp_config.yaml`
  (α=1, β=1, E_min=1.0, H_max, forbid_irreversible). Reported alongside greedy so
  the gap splits cleanly:
  `blind-ILP vs oracle` = pure detection-error cost;
  `greedy vs blind-ILP` = heuristic-loss cost.
  (Because covers are largely forced under Opportunistic, greedy ≈ blind-ILP is the
  *expected* finding, itself evidence that optimization is not the bottleneck.)
- **Harm-aware variant (H4).** Greedy with priority
  `(# suspected outputs) / (α·cost + β·harm)`, same α=β=1 as the oracle objective,
  hard-forbidding `reversibility = irreversible` invocations. Run beside cost-only
  greedy to show whether cheaper/safer repair holds harm within budget.

All three consume only Ĉ + DAG + frozen cost model — never clean/oracle (F-1).

---

## Baseline ladder (lower/upper bounds — mandatory, not "algorithms")

- **repair-all-suspected** — re-collect every repairable leaf ARCS could
  (≈ global re-collect). Upper cost bound; the ρ denominator per org
  (cloudflare 231 / github 310 / mozilla 360 / redhat 291 / wikimedia 385).
- **random-within-budget** — pick invocations at random until B₀ (fixed seed).
  Lower bound showing selection matters.
- **single-point** — repair only the single highest-signal invocation. Also the
  unit defining B₀ (below).

---

## B₀ — reference budget (frozen definition + value)

- **Definition (frozen):** `B₀ = median over the leaf grid of C*(oracle)` — the
  cost of a *typical full single-fault repair*. `C_single-point` = the oracle's C*
  for one injected fault (its minimal full cover); B₀ = the median across the 15
  frozen leaf cells.
- **Value (computed from `ilp_oracle_pilot_v1.json`, recorded not defined):**
  leaf C* = [52,136,144,162,162,180,184,**187**,219,224,251,272,288,322,346] →
  **B₀ = 187**.
- **Per-org sensitivity variant (frozen):** per-org median C*
  (cloudflare 162 / github 224 / mozilla 251 / redhat 180 / wikimedia 272), reported
  as a robustness check; the global B₀ = 187 is the primary.

---

## Metrics (frozen; computed against frozen assets)

- **Success@B₀ (primary).** Per (org, fault): ARCS selects S with cost(S) ≤ B₀,
  applies it to the faulted bundle; success ⟺ the frozen **A1** predicate holds
  (no result-relevant record differs from clean;
  `contamination_closure(clean, repair(S)_faulted)` on report_item empty). Report
  the success rate over the leaf grid. Clean is used ONLY to score, never by ARCS.
- **ρC = C(S_ARCS)/C\*.** Per cell, ARCS's spent cost over the oracle optimal. The
  reference is **per repair class**: leaf cells vs `ilp_oracle_pilot_v1.json`;
  gate-logic cells (F8/F7) vs `non_leaf_oracle_v1.json`. ρC ≥ 1 by construction.
  Reported over cells where ARCS succeeds; failures reported separately (not folded
  into a favourable mean).
- **scope P/R.** Precision/recall of ARCS's repaired set vs the ground-truth
  repairable D(f) — leaf D(f) (`dfs_pilot_v1.json`) for Tier-1, the contaminated
  non-leaf set for F8/F7. Measures whether ARCS repaired the right records,
  independent of cost.

---

## F-1 — blindness constraint (hard, enforced)

ARCS reads only: the faulted bundle, its own re-collection results, the frozen cost
model. It never reads: the clean bundle, D(f)/`dfs_pilot_v1.json`, the oracle
S*/C*/`coverage_map`, or `non_leaf_oracle_v1.json`. Enforced at the interface (the
ARCS entry takes a faulted-bundle handle + a re-collect callback, not clean paths)
and GATE-checked: the ARCS code path asserts no read of clean/oracle artifacts. A1
scoring and ρC/scope-P·R comparison happen OUTSIDE the ARCS boundary, by the
harness, after ARCS returns S.

---

## Grid and the G3 gate

- **Leaf class (H3/H4):** ARCS × {F2,F4,F10} × 5 pilot orgs (`pilot_orgs_FROZEN`),
  leaf re-collection, oracle ref = `ilp_oracle_pilot_v1.json`.
- **Gate-logic class (H3/H4):** ARCS × {F8 (live, 4 orgs + cloudflare no-op), F7
  (record fixture)}, non-leaf re_evaluate/re_render, oracle ref =
  `non_leaf_oracle_v1.json`. Under NL-1′ these are **expected successes** (f6/f7
  fire), reported as a second repair class — not a boundary.
- **G3 (pilot GO/NO-GO):** the 5-org pilot is scored first; scaling to the 20-org
  E3 happens only after G3 = GO, per `pilot_orgs_FROZEN.md`. G3 criteria to be
  fixed in the Phase-5 freeze; this file does not pre-judge them.

---

## Pre-registered predictions (frozen before ARCS runs)

- **H3.** On the leaf grid, ARCS reaches high `Success@B₀` at `ρC ≈ 1` (ARCS ≈
  oracle) — because Opportunistic ⇒ forced covers ⇒ little heuristic gap. Falsified
  if greedy ≪ blind-ILP (heuristic-loss large) or Success@B₀ is low at ρC≈1.
- **H4.** Harm-aware ARCS keeps `H(S)` within budget with no `Success@B₀` loss vs
  cost-only. Falsified if harm-aware sacrifices success or exceeds H_max.
- **NL-1′ (gate-logic repairable; supersedes NL-1).** ARCS detects F8/F7 via f6/f7
  and repairs them (re_evaluate/re_render), reaching high `Success@B₀` on the
  gate-logic class. Falsified if ARCS **fails** on F8/F7 despite the f6/f7 signal →
  revisit the detector / non-leaf action model. (`non_leaf_oracle_FROZEN.md` NL-1′.)
- **Decomposition (frozen reporting rule).** Always report the ρC gap split as
  detection-error (`blind-ILP − oracle`) + heuristic-loss (`greedy − blind-ILP`), so
  the source of any cost gap is attributable, not asserted. Report `D_θ` P/R split
  Tier-1 vs Tier-2 alongside.

---

## Isolation & reproducibility

- ARCS artifacts (`arcs_pilot_v1.json`, …) are separate; they never modify the
  frozen oracle artifacts or the 15 leaf cells (I-1 discipline).
- The run_id-invariant fingerprint + prior-artifact determinism self-check from
  `ilp_oracle_pilot.py` is reused for the ARCS driver (exit 0 ⇔ metrics computed AND
  repro match).
- Deterministic replay (G1 R_replay = 1.0) makes ARCS runs reproducible; re-collect
  callbacks hit the same cache the oracle used.

Frozen 2026-09-17. Implementation proceeds only after this freeze (and after
sign-off on the A-ARCS detector features). No number here may be revised to match an
observed ARCS result.
