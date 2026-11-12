"""experiments/exp_arcs — ARCS (Constrained Adaptive Repair Scope Selection).

Phase 0.1 first: bridge a live ``core.graph.run_pipeline`` DyMAMState into an
infra_v1 run bundle so the existing replay/equality machinery
(``experiments/infra_v1``) can operate on live recon trajectories, and provide
``replay_equal`` for the G1 gate.

This package currently implements the SOURCE-ARTIFACT slice of the recorder
end-to-end; evidence/claim/decision/report slices are follow-ups.
"""
from experiments.exp_arcs.recorder import record_run
from experiments.exp_arcs.replay_equal import replay_equal

__all__ = ["record_run", "replay_equal"]
