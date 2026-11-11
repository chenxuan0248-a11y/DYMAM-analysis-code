# Unified clean-run alert-budget comparison protocol

**Frozen:** 2026-11-05 (local, retrospective; post hoc/exploratory). This is not preregistration or independent validation. No network, live target, or external service access is permitted.

## Question and budget
Compare localization and alerting for Combined, Isolation Forest, and Robust-z under the same clean-run candidate-host alert budget. The single primary budget is **0.1055** (macro-average fraction of candidate hosts alerted per clean run), fixed from the previously reported Combined clean-run rate before rerunning fault outcomes. Contract B remains descriptive because it has no calibrated continuous score under the existing rule definition.

## Candidate contract and folds
Use the frozen main candidate contract `E ∪ R` (faulted evidence targets plus same-run manifest hosts) for every detector. For each held-out organization, score each of the other 19 organizations' clean bundles using the detector's existing frozen feature and scoring implementation. Choose the smallest score threshold whose mean per-run clean alert fraction across those 19 organizations is at most 0.1055. Apply that threshold unchanged to the held-out organization's clean bundle and all nine faulted units. Do not use held-out or faulted labels/data to select thresholds.

Combined uses its existing continuous score: 1 for self-diagnosis or manifest flags; otherwise the maximum within-run base rate among evidence families absent from that host. IF and Robust-z use the existing features and scoring functions. A deterministic SHA-256 host-key breaks exact score ties. The fold threshold is selected from the training clean bundles to minimize absolute distance to 0.1055 (ties choose the lower alert rate); this uses no fault labels. Report the realized held-out budget, which may still differ from the target because organization-level distributions differ.

## Outcomes and uncertainty
Report per-detector realized clean macro and micro alert rates, mean alerted hosts per clean/faulted run, target-host alert recall, tie-aware Detect@1, and the selective/fixed-four offline planning outcomes (OPR, mean planned action-specification cost, action count). Aggregate within organization, then across the 20 organizations. For detector contrasts, calculate paired organization-level differences and percentile bootstrap 95% intervals (10,000 resamples; seed 42; organization is the resampling unit).

## Integrity and interpretation
Keep existing operating-point results as a separately labeled descriptive comparison. The new comparison is post hoc and setting-specific; it does not retroactively make the original thresholds equivalent. Hash the protocol, source grid, bundles' relevant input files, cost specification, scripts, plans, and outputs. If any required input or method cannot be reconstructed, report the analysis as incomplete rather than inferring thresholds or scores.

## Protocol amendment after initial run
The first frozen implementation used only a scalar threshold and the existing `score >= threshold` tie rule. Its held-out Combined macro clean alert rate was 0.0872, materially below 0.1055 because the Combined score is discrete. Before accepting any faulted-outcome comparison, the analysis was amended to use a deterministic host-hash tie-break and to select the attainable training-fold alert rate closest to the same 0.1055 target. This amendment is post hoc and is recorded transparently; retain the initial scalar-threshold output as an audit trail, and interpret the amended comparison as exploratory.
