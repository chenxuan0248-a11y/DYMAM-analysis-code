# Broader candidate-set downstream sensitivity protocol

**Frozen:** 2026-11-05 (local, retrospective; post hoc/exploratory). This is not preregistration or independent validation. No network, live target, or external service access is permitted.

## Question
How do localization, alerting, and offline planning results change when the candidate universe includes surviving claim/decision identities in addition to faulted evidence targets and the same-run discovery manifest?

## Candidate rules
For every clean and faulted bundle, reconstruct from that bundle alone:
- `E`: hosts named by `evidence_event.target`;
- `R`: hosts in `discovery_manifest.resolved_hosts`;
- `C_surv`: hosts named as subjects by surviving claims, plus subjects resolved through surviving decisions' claim references.
- Main contract: `E ∪ R`; broader contract: `E ∪ R ∪ C_surv`.

Apply the same rule independently to clean and faulted bundles. Do not add ground-truth hosts to candidate sets. Ground truth is used only for evaluation.

## Scoring and calibration
Recompute per-run feature vectors and detector scores for every candidate under the relevant contract. Keep the four existing localizers (Combined, Isolation Forest, Robust-z, Contract B) and the existing feature definitions, model parameters, five IF seeds, tie-aware Detect@1, and four-family planner. For the exploratory broader-contract analysis, calibrate Combined, IF, and Robust-z using leave-one-organization-out clean-only calibration at a 0.1055 macro clean-candidate alert budget; Contract B remains an uncalibrated rule reference. Thresholds are estimated separately for each candidate contract using the other 19 organizations' clean runs. No faulted labels enter calibration.

## Outcomes
Report candidate target coverage; tie-aware Detect@1; organization-macro target-host alert recall; organization-macro clean-run host alert rate; mean alerts per faulted run; and, for both selective and fixed-four-family policies, OPR, organization-macro planned action-specification cost, and action count. Score plans against the paired clean DAG only after plans are serialized. Compare broader versus main contract using organization-paired differences and percentile bootstrap (10,000 resamples; seed 42; organization is the resampling unit).

## Interpretation
This is a post hoc sensitivity analysis on one frozen corpus. Results estimate the effect of changing the candidate contract within this study and do not establish cross-system performance. Report missing/corrupt inputs as non-computable; do not impute.
