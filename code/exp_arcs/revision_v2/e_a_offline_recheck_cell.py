"""Stage-traced E-A cell with an additional pre-call no-live-tool guard."""
from __future__ import annotations

from experiments.exp_arcs.revision_v2 import llm_dependency_stage_trace_cell as traced
from core.settings import get_settings
from tools._base import ReplayableTool

_traced_call = ReplayableTool.__call__


def _offline_call(self, *, mode=None, **kwargs):
    if not self.local and (mode not in (None, "replay") or
                           (mode is None and not get_settings().replay_mode)):
        raise RuntimeError(f"E-A offline recheck forbids live tool: {self.name}")
    return _traced_call(self, mode=mode, **kwargs)


ReplayableTool.__call__ = _offline_call

if __name__ == "__main__":
    raise SystemExit(traced.cell.main())
