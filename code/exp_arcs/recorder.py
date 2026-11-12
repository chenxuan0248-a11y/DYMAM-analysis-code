"""experiments/exp_arcs/recorder — live DyMAMState → infra_v1 run bundle.

The live recon pipeline (``core.graph.run_pipeline``) returns an in-memory
``DyMAMState``; the replay + equality machinery (``experiments/infra_v1``)
operates on recorded run bundles (``source_artifact.jsonl`` and friends). This
module is the bridge the ARCS plan's Phase 0.1 requires and no existing code
provided (every prior ``RunStore`` user consumes method-level fixtures, not a
live trajectory).

All five record slices are populated from the P1 typed ledgers on the state
(``evidence_ledger`` / ``claim_ledger`` / ``decision_ledger`` / ``report_items``):

    source_artifact  ← one per deduped evidence observation
    evidence_event   ← evidence_ledger  (raw_ref/raw_hash point at its artifact)
    claim            ← claim_ledger
    decision         ← decision_ledger
    report_item      ← report_items

The core ledgers are NOT infra_v1 records: they carry a single ``qualifiers``
field, lack ``semantic_key`` / ``state_slot_key``, use their own id scheme
(``cl_`` / ``ev_`` / ``dc_`` / ``ri_``), and set ``policy_version`` to
``project-and-validate-v1``. So the recorder derives the infra id fields with
``canonical.py`` and rebuilds every cross-reference through core-id → infra-id
maps. Two invariants keep records replay-stable across two runs of the same
inputs:

  * all timestamps that feed a content id (``created_at`` / ``collected_at`` /
    ``evaluated_at``) use a fixed ``context.evaluated_at``, never wall clock;
  * every derived id is a pure content hash (``canonical.*_id``), so identical
    content yields an identical id regardless of run_id — exactly what
    ``replay_equal`` checks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.infra_v1.bundle import ArtifactBundle
from experiments.infra_v1.canonical import (
    artifact_id,
    canonical_hash,
    canonical_json,
    claim_id as make_claim_id,
    decision_id as make_decision_id,
    event_id as make_event_id,
    normalize_target,
    report_item_id as make_report_item_id,
    semantic_key as make_semantic_key,
    sha256_bytes,
    state_slot_key as make_state_slot_key,
    utc_z,
)
from experiments.infra_v1.canonical import content_id
from experiments.infra_v1.store import RunContext, RunStore

# Fields that identify a raw source observation independent of *when* it was
# seen or which run recorded it. Deliberately excludes ``observed_at``,
# ``evidence_id`` and mode so the payload bytes — and therefore the
# content-addressed ``artifact_id`` — are stable across replays.
_SOURCE_IDENTITY_FIELDS = (
    "source",
    "source_family",
    "target",
    "asset",
    "claim",
    "tool_name",
    "tool_version",
    "raw_hash",
)

METHOD_ID = "arcs_recon"
INPUT_DATA_VERSION = "arcs-recon-inputs-v1"
INFRA_POLICY_VERSION = "policy-v1"  # infra schema const; core uses its own


def _as_dict(item: Any) -> dict[str, Any]:
    """DyMAMState list members may be pydantic models or plain dicts."""
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        return item
    raise TypeError(f"cannot coerce {type(item)!r} to dict")


def _rows(state: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [_as_dict(r) for r in (state.get(key) or [])]


# ─────────────────────────── source_artifact + evidence_event ───────────────

def _source_layer(
    state: dict[str, Any], context: RunContext,
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, tuple[str, str]]]:
    """Map ``evidence_ledger`` to deduplicated SourceArtifact records.

    Returns ``(records, payloads, ev_to_art)`` where ``ev_to_art`` maps each core
    ``evidence_id`` to ``(artifact_id, sha256)`` so evidence events can point
    ``raw_ref`` / ``raw_hash`` at their source blob (the schema's ``raw_hash`` is
    a strict Sha256, which the core 40-char sha1 would fail).
    """
    records: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    ev_to_art: dict[str, tuple[str, str]] = {}
    seen: set[str] = set()

    for ev in _rows(state, "evidence_ledger"):
        envelope = {k: ev.get(k, "") for k in _SOURCE_IDENTITY_FIELDS}
        payload = (canonical_json(envelope) + "\n").encode("utf-8")
        record = {
            "schema_version": "schema-v1",
            "artifact_id": "",
            "run_id": context.run_id,
            "snapshot_id": context.snapshot_id,
            "scope_id": context.scope_id,
            "media_type": "application/json",
            "byte_length": len(payload),
            "sha256": sha256_bytes(payload),
            "storage_ref": "",
            "created_at": utc_z(context.evaluated_at),
            "source_uri_redacted": None,
            "parent_artifact_ids": [],
        }
        aid = artifact_id(record)
        record["artifact_id"] = aid
        ev_to_art[ev.get("evidence_id", "")] = (aid, record["sha256"])
        if aid in seen:
            continue  # identical source content already captured
        seen.add(aid)
        record["storage_ref"] = f"sources/{aid}.json"
        records.append(record)
        payloads[aid] = payload

    return records, payloads, ev_to_art


def _evidence_claim_links(
    state: dict[str, Any], claim_map: dict[str, dict[str, str]],
) -> dict[str, str]:
    """``core evidence_id → claim semantic_key`` from the decision ledger.

    The schema requires a ``success`` EvidenceEvent to name the claim it supports
    (``claim_semantic_key``, a ``sem_...`` id). The core Evidence has no direct
    claim link; the decision ledger carries it (claim_id + supporting/
    contradicting evidence ids). When an evidence supports several claims, pick
    the lexicographically smallest semantic_key so the choice is deterministic.
    """
    link: dict[str, str] = {}
    for dec in _rows(state, "decision_ledger"):
        keys = claim_map.get(dec.get("claim_id", ""))
        if keys is None:
            continue
        sem = keys["semantic_key"]
        for e in (dec.get("supporting_evidence_ids", [])
                  + dec.get("contradicting_evidence_ids", [])):
            if e not in link or sem < link[e]:
                link[e] = sem
    return link


def _evidence_event_layer(
    state: dict[str, Any], context: RunContext,
    ev_to_art: dict[str, tuple[str, str]], ev_to_semantic: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """evidence_ledger → EvidenceEvent records + ``core evidence_id → event_id``.

    Only evidence linked to a claim (via a decision) is recorded: the schema
    forbids a null ``claim_semantic_key`` on a ``success`` event, and evidence
    that no decision references is not part of the reported DAG anyway (so
    dropping it never breaks a decision's evidence remap).
    """
    records: list[dict[str, Any]] = []
    event_map: dict[str, str] = {}
    seen: set[str] = set()

    for ev in _rows(state, "evidence_ledger"):
        core_id = ev.get("evidence_id", "")
        art = ev_to_art.get(core_id)
        semantic = ev_to_semantic.get(core_id)
        if art is None or semantic is None:
            continue
        art_id, art_sha = art
        event = {
            "schema_version": "schema-v1",
            "evidence_event_id": "",
            "run_id": context.run_id,
            "snapshot_id": context.snapshot_id,
            "scope_id": context.scope_id,
            "target": ev.get("target", ""),
            "source": ev.get("source", ""),
            "source_family": ev.get("source_family", "other"),
            "tool_name": ev.get("tool_name", "unknown"),
            "tool_version": ev.get("tool_version", "unknown"),
            "collection_mode": ev.get("collection_mode", context.collection_mode),
            "observation_status": "success",
            "collected_at": utc_z(context.evaluated_at),
            "source_event_time": None,
            "valid_to": None,
            "raw_ref": f"artifact:{art_id}",
            "raw_hash": art_sha,
            "parent_evidence_event_ids": [],
            "relation": ev.get("relation", "supports"),
            "claim_semantic_key": semantic,
            "payload": {
                "asset": ev.get("asset", ""),
                "claim": ev.get("claim", ""),
                "confidence": ev.get("confidence"),
                "relation": ev.get("relation", "supports"),
            },
        }
        eid = make_event_id(event)
        event["evidence_event_id"] = eid
        event_map[core_id] = eid
        if eid in seen:
            continue
        seen.add(eid)
        records.append(event)

    return records, event_map


# ─────────────────────────────────── claim ─────────────────────────────────

def _identity_safe(value: Any) -> Any:
    """Make a value safe for canonical *identity* hashing (content_id).

    ``canonical_identity_json`` forbids non-integral floats (a claim's object /
    qualifiers may carry confidence scores). Integral floats fold to int;
    non-integral floats become their deterministic ``repr`` string. Only the id
    derivation needs this — the stored record keeps the real object (written via
    ``canonical_json``, which allows floats).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else repr(value)
    if isinstance(value, dict):
        return {k: _identity_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_identity_safe(v) for v in value]
    return value


def _derive_claim_keys(
    context: RunContext, claim: dict[str, Any],
) -> tuple[str, str, str]:
    """(semantic_key, state_slot_key, claim_id) for a core claim.

    ``state_slot_key`` branches on the object's internal shape (port / product /
    cve_id); the recon ledger stores a vulnerability object as a bare CVE string,
    so normalize it, and fall back to a generic content-hash slot if any
    claim-type's object lacks the field ``state_slot_key`` expects.
    """
    claim_type = claim["claim_type"]
    subject = claim["subject"]
    predicate = claim["predicate"]
    obj = _identity_safe(claim.get("object"))
    identity = _identity_safe(claim.get("qualifiers") or {})

    semantic = make_semantic_key(
        claim_type=claim_type, scope_id=context.scope_id, subject=subject,
        predicate=predicate, object_value=obj, identity_qualifiers=identity,
    )
    slot_obj = obj
    if claim_type == "vulnerability_applicability" and isinstance(obj, str):
        slot_obj = {"cve_id": obj}
    try:
        slot = make_state_slot_key(
            claim_type=claim_type, scope_id=context.scope_id, subject=subject,
            predicate=predicate, object_value=slot_obj, identity_qualifiers=identity,
        )
    except (KeyError, TypeError, IndexError):
        slot = content_id("slot", {
            "schema_version": "schema-v1", "claim_type": claim_type,
            "scope_id": context.scope_id, "subject": normalize_target(subject),
            "predicate": predicate, "object": obj,
        })
    cid = make_claim_id(
        semantic=semantic, slot=slot, snapshot_id=context.snapshot_id,
    )
    return semantic, slot, cid


def _claim_layer(
    state: dict[str, Any], context: RunContext,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """claim_ledger → Claim records + ``core claim_id → {claim_id, keys}``."""
    records: list[dict[str, Any]] = []
    claim_map: dict[str, dict[str, str]] = {}
    seen: set[str] = set()

    for claim in _rows(state, "claim_ledger"):
        semantic, slot, cid = _derive_claim_keys(context, claim)
        claim_map[claim["claim_id"]] = {
            "claim_id": cid, "semantic_key": semantic, "state_slot_key": slot,
        }
        if cid in seen:
            continue
        seen.add(cid)
        records.append({
            "schema_version": "schema-v1",
            "claim_id": cid,
            "semantic_key": semantic,
            "state_slot_key": slot,
            "claim_type": claim["claim_type"],
            "subject": normalize_target(claim["subject"]),
            "predicate": claim["predicate"],
            "object": claim.get("object"),
            "identity_qualifiers": claim.get("qualifiers") or {},
            "display_qualifiers": {},
            "issuer": claim.get("issuer", "project_and_validate"),
            "created_at": utc_z(context.evaluated_at),
            "snapshot_id": context.snapshot_id,
            "scope_id": context.scope_id,
        })

    return records, claim_map


# ────────────────────────────────── decision ───────────────────────────────

def _remap_events(ids: list[str], event_map: dict[str, str]) -> list[str]:
    return sorted({event_map[e] for e in ids if e in event_map})


def _decision_layer(
    state: dict[str, Any], context: RunContext,
    claim_map: dict[str, dict[str, str]], event_map: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """decision_ledger → Decision records + ``core decision_id → decision_id``."""
    records: list[dict[str, Any]] = []
    decision_map: dict[str, str] = {}
    seen: set[str] = set()

    for dec in _rows(state, "decision_ledger"):
        keys = claim_map.get(dec.get("claim_id", ""))
        if keys is None:
            continue  # decision references a claim outside this run's ledger
        record = {
            "schema_version": "schema-v1",
            "decision_id": "",
            "run_id": context.run_id,
            "claim_id": keys["claim_id"],
            "semantic_key": keys["semantic_key"],
            "state_slot_key": keys["state_slot_key"],
            "support_state": dec["support_state"],
            "temporal_state": dec["temporal_state"],
            "decision": dec["decision"],
            "reason_codes": sorted(dec.get("reason_codes", [])),
            "evaluated_at": utc_z(context.evaluated_at),
            "policy_version": INFRA_POLICY_VERSION,
            "supporting_evidence_event_ids": _remap_events(
                dec.get("supporting_evidence_ids", []), event_map),
            "contradicting_evidence_event_ids": _remap_events(
                dec.get("contradicting_evidence_ids", []), event_map),
            "context_evidence_event_ids": [],
            "previous_decision_id": None,
        }
        did = make_decision_id(record)
        record["decision_id"] = did
        decision_map[dec["decision_id"]] = did
        if did in seen:
            continue
        seen.add(did)
        records.append(record)

    return records, decision_map


# ───────────────────────────────── report_item ─────────────────────────────

def _report_item_layer(
    state: dict[str, Any], context: RunContext,
    claim_map: dict[str, dict[str, str]], decision_map: dict[str, str],
    event_map: dict[str, str],
) -> list[dict[str, Any]]:
    """report_items → ReportItem records (cross-refs remapped to infra ids)."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in _rows(state, "report_items"):
        keys = claim_map.get(item.get("claim_id", ""))
        did = decision_map.get(item.get("decision_id", ""))
        if keys is None or did is None:
            continue
        record = {
            "schema_version": "schema-v1",
            "report_item_id": "",
            "run_id": context.run_id,
            "snapshot_id": context.snapshot_id,
            "scope_id": context.scope_id,
            "claim_id": keys["claim_id"],
            "semantic_key": keys["semantic_key"],
            "state_slot_key": keys["state_slot_key"],
            "decision_id": did,
            "claim_type": item["claim_type"],
            "subject": normalize_target(item["subject"]),
            "predicate": item["predicate"],
            "object": item.get("object"),
            "display_qualifiers": item.get("qualifiers") or {},
            "evidence_event_ids": _remap_events(
                item.get("evidence_ids", []), event_map),
            "policy_version": INFRA_POLICY_VERSION,
        }
        rid = make_report_item_id(record)
        record["report_item_id"] = rid
        if rid in seen:
            continue
        seen.add(rid)
        records.append(record)

    return records


# ─────────────────────────────── run context ───────────────────────────────

def _context_for(state: dict[str, Any], *, evaluated_at: str) -> RunContext:
    config = state["config"]
    org = state.get("organization", getattr(config, "organization", "unknown"))
    base_domains = sorted(state.get("base_domains", []))
    snapshot_id = getattr(config, "snapshot_id", None) or str(org)
    input_data_hash = canonical_hash(
        {"organization": org, "base_domains": base_domains}
    )
    scope_manifest_hash = canonical_hash(
        {"scope_id": config.scope_id, "base_domains": base_domains}
    )
    return RunContext.create(
        method_id=METHOD_ID,
        snapshot_id=snapshot_id,
        scope_id=config.scope_id,
        scope_manifest_hash=scope_manifest_hash,
        evaluated_at=evaluated_at,
        collection_mode=getattr(config, "mode", "replay"),
        seed=config.seed,
        input_data_version=INPUT_DATA_VERSION,
        input_data_hash=input_data_hash,
    )


def _minimal_report(context: RunContext) -> dict[str, Any]:
    # RenderedReport is required by the bundle schema; the ARCS harness compares
    # jsonl record ids, not the report shell, so emit the empty-but-valid shape
    # RunStore.fail() uses.
    return {
        "schema_version": "schema-v1",
        "run_id": context.run_id,
        "snapshot_id": context.snapshot_id,
        "scope_id": context.scope_id,
        "policy_version": "policy-v1",
        "report_item_ids": [],
        "sections": [],
        "rendered_at": utc_z(context.evaluated_at),
    }


def record_run(
    state: dict[str, Any],
    *,
    raw_root: str | Path,
    evaluated_at: str = "2026-01-01T00:00:00Z",
) -> Path:
    """Record a live DyMAMState as an infra_v1 run bundle under ``raw_root``.

    ``evaluated_at`` is fixed (not wall-clock) so a re-record of the same state
    differs only in the intentionally-volatile ``run_id`` — keeping everything
    ``replay_equal`` inspects stable. Returns the finalized run directory.
    """
    context = _context_for(state, evaluated_at=evaluated_at)

    claims, claim_map = _claim_layer(state, context)
    ev_to_semantic = _evidence_claim_links(state, claim_map)
    source_artifacts, payloads, ev_to_art = _source_layer(state, context)
    evidence_events, event_map = _evidence_event_layer(
        state, context, ev_to_art, ev_to_semantic)
    decisions, decision_map = _decision_layer(state, context, claim_map, event_map)
    report_items = _report_item_layer(
        state, context, claim_map, decision_map, event_map)

    bundle = ArtifactBundle(
        source_artifacts=source_artifacts,
        source_payloads=payloads,
        evidence_events=evidence_events,
        claims=claims,
        decisions=decisions,
        report_items=report_items,
        rendered_report=_minimal_report(context),
    )
    store = RunStore(raw_root)
    run_dir = store.finalize(context=context, bundle=bundle)

    # Discovery resolved-host manifest (reference-free omission signal for the
    # detector-only localizer). Written as a side-file so it does not affect the
    # fingerprinted record files (bundle determinism / oracle labels unchanged).
    report = state.get("report") or {}
    (run_dir / "discovery_manifest.json").write_text(json.dumps(
        {"resolved_hosts": report.get("resolved_hosts", []),
         "resolved": report.get("resolved")}, sort_keys=True))
    return run_dir
