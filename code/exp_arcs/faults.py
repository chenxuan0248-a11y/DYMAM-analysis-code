"""experiments/exp_arcs/faults — the frozen ARCS fault registry (F1–F10).

The plan references ``error_taxonomy.zh.md`` for F1–F10, but that file was never
created; this module is the authoritative machine registry (the doc
``error_taxonomy.zh.md``, generated alongside, is the human copy). Two layers:

* ``record`` faults tamper a *recorded* infra_v1 bundle — implemented in
  ``experiments/mechanism_v1/fault_suite.py`` for validator testing. They
  deliberately leave content-addressed id fields intact, so they are invisible
  to id-level ``replay_equal`` and are NOT usable as the Phase 0.3 fault
  dimension.

* ``live`` faults are injected into the running recon pipeline
  (``core.graph.run_pipeline``) via ``experiments/exp_arcs/interceptors.py``;
  they change what the pipeline produces, so the recorded bundle's ids differ
  from a clean run — this is the fault dimension Phase 0.3 needs.

``inject_node`` is grounded in the *actual* run_pipeline code, which differs
from the plan's stale hints (the plan pointed F4 at ``infra_v1/adapters.py``,
a method-adapter path run_pipeline never executes, and F8 at graph_writer, while
the real evidence gate is ``core.project_and_validate.ProjectAndValidate._evaluate``).
Deviations are recorded per fault.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fault:
    fault_id: str
    name: str
    layer: str            # "record" | "live"
    inject_node: str      # pipeline node / step the fault perturbs
    mechanism: str
    status: str           # "implemented" | "pending"
    impl_ref: str         # where it lives
    repair_class: str     # "leaf" | "gate_logic" | "record" — ARCS repair scope
    # leaf: repairable by re-collecting source_artifact/evidence_event leaves
    #   (n_repairable>0 all orgs; the ARCS repair grid). gate_logic: contamination
    #   is entirely on decision/report (n_repairable=0 all orgs) — needs a
    #   non-leaf repair action, not covered by leaf re-run. record: id-invariant
    #   validator fault, invisible to replay, not in the ARCS repair grid.
    #   Grounded in dfs_pilot_v1.json; see error_taxonomy.zh.md.
    display_name: str = ""     # name used in new tables / the paper; `name` stays the
    #   legacy identifier already serialized in historical artifacts (e.g. dfs_v2.json)
    mechanism_class: str = ""  # short machine-readable mechanism tag (optional)

    @property
    def label(self) -> str:
        return self.display_name or self.name


FAULT_REGISTRY: dict[str, Fault] = {
    # ── record-layer (mechanism_v1.fault_suite; id-invariant validator faults) ──
    "F1": Fault("F1", "delete_source_artifact_payload", "record", "source_artifact",
                "delete a source blob referenced by a source_artifact record",
                "implemented", "mechanism_v1.fault_suite:_delete_source", "record"),
    "F3": Fault("F3", "tamper_source_artifact_payload", "record", "source_artifact",
                "append bytes to a source blob (hash mismatch)",
                "implemented", "mechanism_v1.fault_suite:_tamper_source", "record"),
    "F5": Fault("F5", "swap_snapshot", "record", "evidence_event",
                "rewrite an evidence event's snapshot_id",
                "implemented", "mechanism_v1.fault_suite:_swap_snapshot", "record"),
    "F6": Fault("F6", "swap_scope", "record", "evidence_event",
                "rewrite an evidence event's scope_id",
                "implemented", "mechanism_v1.fault_suite:_swap_scope", "record"),
    "F7": Fault("F7", "tamper_decision", "record", "decision",
                "flip a decision's action to abstain",
                "implemented", "mechanism_v1.fault_suite:_tamper_decision", "record"),
    "F9": Fault("F9", "forge_parent", "record", "evidence_event",
                "forge a non-existent parent_evidence_event_id "
                "(redundancy-hallucination mutator; stress-tests H2)",
                "implemented", "mechanism_v1.fault_suite:_fake_parent", "record"),
    # ── live (exp_arcs.interceptors; perturb the running recon pipeline) ──
    "F2": Fault("F2", "wrong_tool", "live", "discovery",
                "the active-observation tool (httpx) returns a wrong fingerprint "
                "signal (server/powered_by/title overwritten) → wrong technology",
                "implemented", "exp_arcs.interceptors:F2", "leaf"),
    "F8": Fault("F8", "evidence_gate_bypass", "live", "project_and_validate",
                "force every claim's decision to 'report' (bypass the "
                "support/freshness/qualification gate) → over-admission. "
                "PLAN DEVIATION: gate is in project_and_validate, not graph_writer",
                "implemented", "exp_arcs.interceptors:F8", "gate_logic"),
    "F10": Fault("F10", "premature_termination", "live", "discovery",
                 "discovery completes normally, then its returned asset map is "
                 "post-hoc truncated (~half the hosts, deterministic) to MODEL a "
                 "run terminating before enumeration completes — the underlying "
                 "recorder still captures a pre-truncation resolved-hosts "
                 "manifest, which is what makes discovery-manifest-diff "
                 "detection possible; a literal early-abort with no such record "
                 "would not be. Every downstream ledger shrinks from the "
                 "truncation. PLAN DEVIATION: injected at discovery, not "
                 "orchestrator — the static graph is fixed so the orchestrator "
                 "plan is descriptive-only, and a later cut (e.g. llm_reasoning "
                 "findings) is invisible to the typed ledgers the recorder "
                 "captures",
                 "implemented", "exp_arcs.interceptors:F10", "leaf",
                 display_name="host_handoff_drop",
                 mechanism_class="post_hoc_discovery_output_truncation"),
    "F4": Fault("F4", "normalization_bug", "live", "project_and_validate",
                "the evidence-normalization step "
                "(ProjectAndValidate._normalize_evidence) drops a third of the "
                "evidence (by sorted id) → lost support flips decisions / drops "
                "report_items. A pure no-op here is healed by the recorder's "
                "re-canonicalization (reorder only, D(f)=0), so the content is "
                "corrupted instead. PLAN DEVIATION: plan pointed F4 at "
                "infra_v1/adapters.py::normalize (unused by run_pipeline); this "
                "is run_pipeline's closest real re-canonicalization hook",
                "implemented", "exp_arcs.interceptors:F4", "leaf"),
}

LIVE_FAULTS = [f for f, v in FAULT_REGISTRY.items()
               if v.layer == "live" and v.status == "implemented"]
