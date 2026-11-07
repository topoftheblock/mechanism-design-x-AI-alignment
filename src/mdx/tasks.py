"""Decision tasks shared by every backend.

A task is a fully specified one-shot decision in the paper's environment. Backends
(the rational benchmark agent, an LLM) receive the same task object, so any gap between
them is attributable to the agent rather than to the harness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class ActionTask:
    """Deployment stage: the agent observes theta and picks a <= limit.

    c   : capability (highest executable action), None = unconstrained
    cap : designer's permitted maximum, None = no cap
    """

    theta: float
    b: float
    c: Optional[float] = None
    cap: Optional[float] = None
    framing: str = "abstract"
    explicit_objective: bool = True
    kind: str = field(default="action", init=False)

    @property
    def limit(self) -> float:
        lims = [x for x in (self.c, self.cap) if x is not None]
        return min(lims) if lims else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReportTask:
    """Evaluation stage: the agent certifies a level c_hat <= c from `grid`, which
    selects the cap schedule[c_hat] for deployment (Example I)."""

    c: float
    b: float
    grid: Sequence[float]
    schedule: Sequence[float]
    framing: str = "abstract"
    explicit_objective: bool = True
    kind: str = field(default="report", init=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["grid"], d["schedule"] = list(self.grid), list(self.schedule)
        return d


Task = ActionTask | ReportTask


@dataclass
class Decision:
    """A backend's answer. `value` is the raw choice; `feasible_value` is the choice
    projected onto what the agent can physically do (infeasible choices are recorded,
    not silently dropped)."""

    value: Optional[float]
    feasible_value: Optional[float]
    infeasible: bool = False
    rationale: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def project(task: Task, value: Optional[float]) -> tuple[Optional[float], bool]:
    """Map a raw choice onto the feasible set. Returns (feasible_value, was_infeasible)."""
    if value is None:
        return None, False
    if isinstance(task, ActionTask):
        if value > task.limit + 1e-9:
            return task.limit, True
        return value, False
    # ReportTask: must be a grid level not above true capability (no counterfeiting)
    grid = [g for g in task.grid if g <= task.c + 1e-9]
    nearest = min(task.grid, key=lambda g: abs(g - value))
    if abs(nearest - value) > 1e-6 or nearest > task.c + 1e-9:
        return max(grid), True
    return nearest, False
