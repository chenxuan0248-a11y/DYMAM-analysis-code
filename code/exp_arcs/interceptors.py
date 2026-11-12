"""experiments/exp_arcs/interceptors — live fault injection into run_pipeline.

``inject(fault_id)`` is a context manager that monkeypatches one grounded point
of the running recon pipeline for the duration of a ``run_pipeline`` call, then
restores it. Each fault is DETERMINISTIC (a fixed perturbation) so a faulted run
still replays exactly — which is what the Phase 0.3 fault dimension measures: not
"does the fault fire" but "does a *faulted* trajectory reproduce across replays".

    from experiments.exp_arcs.interceptors import inject
    with inject("F8"):
        state = run_pipeline(organization="mozilla")

Patches wrap the real logic and alter its output, so they cannot crash the run
on an unexpected shape (worst case a fault is a no-op, which the verifier flags).
See ``experiments/exp_arcs/faults.py`` for the registry and plan deviations.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import agents.discovery as _discovery
from agents.discovery import DiscoveryAgent
from core.project_and_validate import ProjectAndValidate
from experiments.exp_arcs.faults import FAULT_REGISTRY


class _ToolProxy:
    """Transparent proxy around a ReplayableTool: delegates EVERY attribute to the
    wrapped tool (``name`` / ``scoped_arg`` / ``schema_version`` / ``local`` /
    ``_cache_key`` / ``source`` … — whatever ``agents/_base.py::tool`` reads, now or
    later) and only intercepts the call to mutate the result. A bare-function
    wrapper drops these attributes and crashes the framework's tool dispatch (the
    latent bug ``interceptors_local._f2_local`` documents); this proxy is robust to
    future attribute additions because it forwards by ``__getattr__``."""

    def __init__(self, orig: Any, mutate) -> None:
        object.__setattr__(self, "_orig", orig)
        object.__setattr__(self, "_mutate", mutate)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_orig"), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        res = object.__getattribute__(self, "_orig")(*args, **kwargs)
        object.__getattribute__(self, "_mutate")(res, args, kwargs)
        return res


def _f2_wrap(orig: Any) -> Any:
    """F2 wrong_tool: httpx returns a wrong fingerprint signal (globally)."""
    def _mutate(res: Any, args: tuple, kwargs: dict) -> None:
        try:
            res.data["server"] = "arcs-F2-wrong-tool"
            res.data["powered_by"] = None
            res.data["title"] = "arcs-F2"
        except Exception:  # unexpected shape → leave untouched
            pass
    return _ToolProxy(orig, _mutate)


def _f8_wrap(orig: Any) -> Any:
    """F8 evidence_gate_bypass: force every decision to admit ('report')."""
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        decision = orig(self, *args, **kwargs)
        try:
            decision.support_state = "supported"
            decision.decision = "report"
        except Exception:
            pass
        return decision
    return wrapped


def _f10_wrap(orig: Any) -> Any:
    """F10 premature_termination: discovery stops after ~half the hosts.

    Truncating discovery's asset map (deterministically, first half by sorted
    name) models the run terminating before enumeration completes — every
    downstream ledger (claims/decisions/report_items) shrinks, so it is visible
    in the recorded bundle. (Injecting later — e.g. dropping llm_reasoning's
    findings — is invisible: findings are report-only, not a typed ledger.)
    """
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        res = orig(self, *args, **kwargs)
        if isinstance(res, dict) and isinstance(res.get("assets"), dict):
            assets = res["assets"]
            keep = sorted(assets)[: max(1, len(assets) // 2)]
            res["assets"] = {k: assets[k] for k in keep}
        return res
    return wrapped


def _f4_wrap(orig: Any) -> Any:
    """F4 normalization_bug: the evidence-normalization step drops evidence.

    ``ProjectAndValidate._normalize_evidence`` is the run's real
    re-canonicalization step (run_pipeline has no single host-canonicalization
    hook; this is the closest one it executes). A pure no-op here is *healed*
    downstream — the recorder re-canonicalizes ids, so it only reorders records
    (exactly the "record-layer mutation gets healed" the plan warns about). To
    bug it in a way that is NOT healed, corrupt the *content*: run the real
    normalization, then deterministically discard a third of the evidence (by
    sorted id). Lost support flips decisions / drops report_items → real
    contamination (D(f) > 0).
    """
    def wrapped(self: Any, rows: Any, *args: Any, **kwargs: Any) -> Any:
        normalized = orig(self, rows, *args, **kwargs)
        ordered = sorted(normalized, key=lambda e: e.evidence_id)
        keep = len(ordered) - len(ordered) // 3
        return ordered[:keep]
    return wrapped


@contextmanager
def inject(fault_id: str | None) -> Iterator[dict[str, Any]]:
    """Patch the pipeline for one live fault; restore on exit.

    ``None`` / ``"clean"`` is a no-op. Yields ``{fault, inject_node}``.
    """
    if fault_id in (None, "clean"):
        yield {"fault": "clean", "inject_node": None}
        return

    spec = FAULT_REGISTRY.get(fault_id)
    if spec is None or spec.layer != "live":
        raise ValueError(f"{fault_id!r} is not a live fault "
                         f"(live: see faults.LIVE_FAULTS)")
    if spec.status != "implemented":
        raise NotImplementedError(
            f"{fault_id} ({spec.name}) injection point is pending: {spec.mechanism}")

    meta = {"fault": fault_id, "inject_node": spec.inject_node}

    if fault_id == "F2":
        orig = _discovery.httpx
        _discovery.httpx = _f2_wrap(orig)
        try:
            yield meta
        finally:
            _discovery.httpx = orig
    elif fault_id == "F8":
        orig = ProjectAndValidate._evaluate
        ProjectAndValidate._evaluate = _f8_wrap(orig)
        try:
            yield meta
        finally:
            ProjectAndValidate._evaluate = orig
    elif fault_id == "F10":
        orig = DiscoveryAgent._run
        DiscoveryAgent._run = _f10_wrap(orig)
        try:
            yield meta
        finally:
            DiscoveryAgent._run = orig
    elif fault_id == "F4":
        orig = ProjectAndValidate._normalize_evidence
        ProjectAndValidate._normalize_evidence = _f4_wrap(orig)
        try:
            yield meta
        finally:
            ProjectAndValidate._normalize_evidence = orig
    else:  # pragma: no cover — registry says implemented but no handler
        raise NotImplementedError(f"no interceptor wired for {fault_id}")
