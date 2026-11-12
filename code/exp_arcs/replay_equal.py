"""experiments/exp_arcs/replay_equal — bundle-level replay equality (G1).

Two runs are *replay-equal* when, ignoring the intentionally-volatile
``run_id`` and wall-clock fields, they produce the same records in the same
order. The ARCS plan defines this over {event ids, event order, normalized
inputs, decision order}.

infra_v1 record ids are content-addressed (e.g. ``artifact_id`` hashes
{scope_id, snapshot_id, media_type, byte_length, sha256} — no run_id, no
timestamp), so comparing ids already covers normalized inputs at the identity
layer. Per record type this module compares: the id set (membership), the id
sequence (order), and — when ids match — the full normalized record content
(every field except the volatile ``run_id``), making "normalized inputs" an
explicit criterion that also catches divergence in any non-identity field the id
hash does not cover. It reports the first divergence per type.

All five recorder slices are populated (source_artifact / evidence_event /
claim / decision / report_item), so this is the G1 comparator for the full
bundle; unpopulated types compare as empty-equal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from experiments.infra_v1.canonical import canonical_json

# Fields that legitimately vary between two replays of the same inputs and must
# be excluded from the normalized-content comparison. ``run_id`` is the only
# volatile field the recorder emits (timestamps are pinned to a fixed
# ``context.evaluated_at``); it is intentionally not part of any content id.
_VOLATILE_FIELDS = frozenset({"run_id"})

# (filename, id field) per infra_v1 record type, in evidence-DAG order.
_RECORD_SPECS: tuple[tuple[str, str], ...] = (
    ("source_artifact.jsonl", "artifact_id"),
    ("execution_event.jsonl", "event_id"),
    ("evidence_event.jsonl", "evidence_event_id"),
    ("claim.jsonl", "claim_id"),
    ("decision.jsonl", "decision_id"),
    ("report_item.jsonl", "report_item_id"),
)


@dataclass
class ReplayDiff:
    record_type: str
    kind: str  # "order" | "membership"
    detail: str


def _load_rows(run_dir: Path, filename: str) -> list[dict]:
    path = run_dir / filename
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def _load_ids(run_dir: Path, filename: str, id_field: str) -> list[str]:
    ids: list[str] = []
    for row in _load_rows(run_dir, filename):
        if id_field not in row:
            raise KeyError(f"{filename}: record missing id field {id_field!r}")
        ids.append(row[id_field])
    return ids


def _load_normalized(run_dir: Path, filename: str) -> list[str]:
    """Canonical serialization of each record with volatile fields stripped."""
    return [canonical_json({k: v for k, v in row.items() if k not in _VOLATILE_FIELDS})
            for row in _load_rows(run_dir, filename)]


def replay_equal(
    dir_a: str | Path, dir_b: str | Path,
) -> tuple[bool, list[ReplayDiff]]:
    """Return ``(equal, diffs)`` comparing two finalized infra_v1 run dirs.

    ``equal`` is True iff every record type matches on both order and
    membership. ``diffs`` lists each divergence (membership reported before
    order for the same type, since a set difference explains an order one).
    """
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    diffs: list[ReplayDiff] = []

    for filename, id_field in _RECORD_SPECS:
        ids_a = _load_ids(dir_a, filename, id_field)
        ids_b = _load_ids(dir_b, filename, id_field)
        record_type = filename.removesuffix(".jsonl")

        set_a, set_b = set(ids_a), set(ids_b)
        if set_a != set_b:
            only_a = sorted(set_a - set_b)
            only_b = sorted(set_b - set_a)
            diffs.append(ReplayDiff(
                record_type, "membership",
                f"{len(only_a)} only in A, {len(only_b)} only in B "
                f"(A={len(ids_a)}, B={len(ids_b)}); "
                f"e.g. +A={only_a[:2]} +B={only_b[:2]}",
            ))
            continue  # membership diff subsumes any order diff
        if ids_a != ids_b:
            first = next(
                i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y
            )
            diffs.append(ReplayDiff(
                record_type, "order",
                f"same set, first order divergence at index {first}: "
                f"A={ids_a[first]} B={ids_b[first]}",
            ))
            continue

        # Normalized-inputs check (plan criterion): ids match, so verify the
        # full record content matches too, after stripping volatile fields.
        # Ids are content-addressed, so this rarely fires, but it catches any
        # divergence in a non-identity field the id hash does not cover.
        norm_a = _load_normalized(dir_a, filename)
        norm_b = _load_normalized(dir_b, filename)
        if norm_a != norm_b:
            first = next(
                (i for i, (x, y) in enumerate(zip(norm_a, norm_b)) if x != y),
                min(len(norm_a), len(norm_b)),
            )
            diffs.append(ReplayDiff(
                record_type, "normalized_inputs",
                f"ids match but record content differs at index {first}",
            ))

    return (not diffs, diffs)
