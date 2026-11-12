"""experiments/exp_arcs/interceptors_local — LOCAL (single-target) live faults.

The global interceptors (`interceptors.py`, frozen) inject at a pipeline STAGE
(httpx / _evaluate / _run / _normalize_evidence) so EVERY host is perturbed →
inject_node = a stage (not a DAG node), I(f) = Descendants(stage) is undefined and
D(f) is global. The plan's ARCS needs LOCAL faults: inject_node = a real DAG node,
I(f) = a small subtree.

These variants gate the SAME perturbation on one fixed target host, so only that
host's records are contaminated. inject_node is then well-defined as that host's
affected leaves; I(f) = Descendants(those leaves) ∩ Relevant(G).

    from experiments.exp_arcs.interceptors_local import inject_local
    with inject_local("F2", target="accounts.firefox.com"):
        state = run_pipeline(organization="mozilla")

Frozen `interceptors.py` is untouched (the oracle artifacts depend on it).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import agents.discovery as _discovery
from agents.discovery import DiscoveryAgent
from core.project_and_validate import ProjectAndValidate

_TARGET: dict[str, Any] = {"hosts": set(), "family": None}


def _f2_local(orig: Any) -> Any:
    """F2 wrong_tool, only for the target host's httpx fingerprint.

    NB: the httpx tool object carries a ``scoped_arg`` attribute the agent
    framework reads (``agents/_base.py``: ``tool.scoped_arg``). A bare wrapper
    loses it → the tool call crashes for EVERY host → discovery fails globally
    (this is the latent bug that made the stage-level F2 pervasive). Preserve all
    of the original's attributes so only the target's fingerprint is perturbed."""
    import functools

    @functools.wraps(orig)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        res = orig(*args, **kwargs)
        url = kwargs.get("url")
        if url is None and len(args) >= 2:
            url = args[1]
        if url in _TARGET["hosts"]:
            try:
                res.data["server"] = "arcs-F2-wrong-tool"
                res.data["powered_by"] = None
                res.data["title"] = "arcs-F2"
            except Exception:
                pass
        return res
    for attr in ("scoped_arg", "tool_name", "tool_version"):
        if hasattr(orig, attr):
            setattr(wrapped, attr, getattr(orig, attr))
    return wrapped


def _f4_local(orig: Any) -> Any:
    """F4 normalization_bug, drop the target host's evidence.

    NOTE: this is a DIFFERENT mechanism from the frozen global ``interceptors.
    py::_f4_wrap`` (which sorts ALL evidence by id and drops a fixed fraction,
    with no host/family targeting — used for the 10-class taxonomy-validation
    pilot, not the E3 grid). This local variant instead removes exactly the
    evidence matching ``_TARGET["hosts"]``/``_TARGET["family"]``; it is the one
    the paper's E3 main grid (F4@dns, F4@certificate, F4-host, ...) actually
    uses. Keep this distinction in mind when tracing paper claims back to code.

    If ``_TARGET["family"]`` is set, drop ONLY those hosts' evidence of that one
    source_family (a family-scoped local fault: it tests whether the remaining
    families can still support the host's claims — the H2 substitutability axis).
    If family is None, drop all of the target hosts' evidence (the whole-host
    variant — multi-invocation when a host spans ≥2 families).
    ``_TARGET["hosts"]`` may hold >1 host (a multi-point fault: the leaf-repairable
    impact then spans ≥2 (host,family) invocations, which single-point cannot
    cover)."""
    def wrapped(self: Any, rows: Any, *args: Any, **kwargs: Any) -> Any:
        normalized = orig(self, rows, *args, **kwargs)
        hosts, family = _TARGET["hosts"], _TARGET["family"]
        kept = []
        for e in normalized:
            if getattr(e, "target", None) in hosts and (
                family is None or getattr(e, "source_family", None) == family):
                continue
            kept.append(e)
        return kept
    return wrapped


def _f10_local(orig: Any) -> Any:
    """F10 premature_termination, drop the target host(s) from discovery.

    ``orig()`` runs to completion first (the target hosts ARE resolved, and the
    recorder captures a pre-drop resolved-hosts manifest for them) — only the
    RETURNED asset map is then filtered to model a run that never enumerated
    them. This is why discovery-manifest-diff detection (`detector_eval.
    manifest_dropped_hosts`) can recover these hosts: it reads that pre-drop
    manifest. A literal early-abort that also skipped the manifest write would
    leave no such record and would not be reference-free recoverable this way;
    see the paper's external-validity discussion for this modeling choice.
    ``_TARGET["hosts"]`` may hold >1 host (multi-point premature termination)."""
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        res = orig(self, *args, **kwargs)
        if isinstance(res, dict) and isinstance(res.get("assets"), dict):
            hosts = _TARGET["hosts"]
            res["assets"] = {k: v for k, v in res["assets"].items() if k not in hosts}
        return res
    return wrapped


_WRAP = {"F2": ("_discovery.httpx", _f2_local),
         "F4": ("ProjectAndValidate._normalize_evidence", _f4_local),
         "F10": ("DiscoveryAgent._run", _f10_local)}


@contextmanager
def inject_local(fault_id: str, target: str | None = None,
                 family: str | None = None,
                 targets: "list[str] | tuple[str, ...] | set[str] | None" = None,
                 ) -> Iterator[dict[str, Any]]:
    """Patch one LOCAL live fault gated on ``target`` (single host) or ``targets``
    (a set of hosts — a MULTI-POINT fault whose leaf-repairable impact spans ≥2
    (host,family) invocations); and, for F4, optionally one ``family``. Restore on
    exit. Yields ``{fault, targets, family, inject_node}`` where inject_node is
    ('target', host[, family]) for one host or ('targets', [hosts...][, family])
    for several."""
    if fault_id not in _WRAP:
        raise ValueError(f"{fault_id!r} has no local variant (have {sorted(_WRAP)})")
    if family is not None and fault_id != "F4":
        raise ValueError(f"family scoping only applies to F4, not {fault_id!r}")
    hosts = set(targets) if targets is not None else set()
    if target is not None:
        hosts.add(target)
    if not hosts:
        raise ValueError("inject_local needs target or targets")
    _TARGET["hosts"] = hosts
    _TARGET["family"] = family
    ordered = sorted(hosts)
    if len(ordered) == 1:
        inode = ("target", ordered[0]) if family is None else ("target", ordered[0], family)
    else:
        inode = ("targets", ordered) if family is None else ("targets", ordered, family)
    meta = {"fault": fault_id, "targets": ordered, "family": family,
            "inject_node": inode}
    if fault_id == "F2":
        orig = _discovery.httpx
        _discovery.httpx = _f2_local(orig)
        try:
            yield meta
        finally:
            _discovery.httpx = orig
    elif fault_id == "F4":
        orig = ProjectAndValidate._normalize_evidence
        ProjectAndValidate._normalize_evidence = _f4_local(orig)
        try:
            yield meta
        finally:
            ProjectAndValidate._normalize_evidence = orig
    elif fault_id == "F10":
        orig = DiscoveryAgent._run
        DiscoveryAgent._run = _f10_local(orig)
        try:
            yield meta
        finally:
            DiscoveryAgent._run = orig
    _TARGET["hosts"] = set()
    _TARGET["family"] = None
