# D1 — pre-registered decision (FROZEN 2026-09-15, before E3)

Phase 0.4 (S2 evidence-redundancy survey) → decision **D1** for hypothesis **H2**
(independent source families provide substitutable evidence). Frozen here before
any E3 result is seen, to prevent post-hoc rule fitting.

## Frozen rule

- **R(c)** = number of *independent source families* whose evidence supports
  claim `c`. Independent families (frozen set):
  `{dns, certificate, active_observation, registration}` — genuine independent
  external observations. Excluded as non-independent / derived:
  `llm_inference`, `temporal_memory`, `vulnerability_context`, `other`.
- **Coverage₂₊** = P(R ≥ 2).
- **Primary metric** = Coverage₂₊ over the **substantive** claim types only —
  `service_observation`, `technology_identification`,
  `vulnerability_applicability` — the claim types ARCS actually repairs.
  Redundancy in `asset_identity` / `organization_association` does not make a
  service / technology / CVE finding substitutable, so the pooled-over-all-types
  metric would overstate H2; it is retained as **context only**.
- **Thresholds**: substantive Coverage₂₊ ≥ 0.30 → **H2 Core**; < 0.30 →
  **H2 Opportunistic**; pooled < 0.30 but some substantive type ≥ 0.30 →
  **H2 Conditional**.

Source: `experiments/exp_arcs/s2_evidence_redundancy.py`. No ground truth is read
(avoids selection bias). Evidence→claim support comes from the decision ledger's
`supporting_evidence_event_ids` in the recorded infra_v1 bundles.

## Frozen result (18 orgs, 12,170 claims; code @ e407b5f)

Full JSON: `experiments/exp_arcs/s2_d1_frozen_v1.json`.

| claim type | n | Coverage₂₊ |
|---|---|---|
| asset_identity | 1740 | 0.996 |
| organization_association | 1740 | 0.996 |
| **service_observation** | 3547 | **0.000** |
| **technology_identification** | 1497 | **0.000** |
| **vulnerability_applicability** | 3646 | **0.000** |
| **substantive pooled (D1 primary)** | **8690** | **0.000** |
| overall (context) | 12170 | 0.285 |

## Verdict: **H2 = Opportunistic**

The substantive claim types — every one ARCS repairs — have **zero** independent
second-family redundancy. The apparent overall redundancy is confined to
`asset_identity` / `organization_association` (asset existence / org attribution,
corroborated by dns + active_observation + certificate). Robustness: at 18 orgs
even the *overall* metric is 0.285 < 0.30, so the verdict is H2 Opportunistic
under either the substantive-only or the literal-overall reading.

**Implications for the plan:** H2 (evidence substitution, Step 4.2) is an
opportunistic mechanism — usable only where a substantive claim happens to have
an independent second family, which the survey shows is essentially never in the
current corpus. ARCS's core value must therefore rest on H3 (cost) and H4
(safety), not on H2 substitution. E7-B (−substitution ablation) is expected to
show little effect. This is consistent with the F9 pseudo-redundancy guardrail.
