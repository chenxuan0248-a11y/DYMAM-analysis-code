"""experiments/exp_arcs/cost_harm — load the frozen cost/harm/reversibility spec.

Step 1.3 output: a versioned c(v) / r(v) / h(v) model, keyed by repair action,
plus a ``source_family → action`` map so the attributed DAG (Step 1.2) can hang
cost / reversibility / harm on every node. ``spec_version_hash`` is the sha256 of
the canonicalized spec, recorded on any run that uses it (reproducibility).
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

from experiments.infra_v1.canonical import canonical_hash

_SPEC_PATH = Path(__file__).with_name("cost_harm_spec.yaml")


@functools.lru_cache(maxsize=1)
def load_spec() -> dict[str, Any]:
    return yaml.safe_load(_SPEC_PATH.read_text(encoding="utf-8"))


def spec_version_hash() -> str:
    """sha256 of the canonicalized spec — freeze/provenance id."""
    return canonical_hash(load_spec())


def action_for_family(source_family: str) -> str:
    spec = load_spec()
    return spec["family_to_action"].get(source_family, spec["family_to_action"]["other"])


def cost_harm_reversibility(source_family: str) -> dict[str, Any]:
    """(c, r, h, reversibility_rank, action) for a node of this source family."""
    spec = load_spec()
    action = action_for_family(source_family)
    row = spec["actions"][action]
    return {
        "action": action,
        "cost": row["cost"],
        "harm": row["harm"],
        "reversibility": row["reversibility"],
        "reversibility_rank": spec["reversibility_rank"][row["reversibility"]],
    }
