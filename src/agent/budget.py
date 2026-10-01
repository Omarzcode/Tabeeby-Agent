"""Loop budget for the Tabeeby agent.

"A loop without a budget is a bug." The reasoning loop is capped on four
independent axes so a runaway conversation can never burn unbounded time,
tool calls or money:

- steps   : how many times the LLM is called
- seconds : wall-clock time for the whole request
- actions : how many tool calls are actually executed (one LLM turn can ask
            for several tool calls at once, so steps alone do not cap the bill)
- cost    : accumulated USD spent on tokens during this request
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Budget:
    """A cooperative spending limit for one agent run."""

    max_steps: int = 8
    max_seconds: float = 90.0
    max_actions: Optional[int] = 12
    max_cost_usd: Optional[float] = None
    # Stop offering tools once this share of the time budget is used, so the
    # model has time left to turn its findings into an answer.
    answer_time_reserve: float = 0.25

    steps: int = 0
    actions: int = 0
    cost_usd: float = 0.0
    _start: float = field(default_factory=time.monotonic, repr=False)

    def tick(self) -> None:
        """Record one LLM call."""
        self.steps += 1

    def spend(self, usd: float) -> None:
        """Record token spend for this request."""
        self.cost_usd += max(0.0, float(usd or 0.0))

    def allow(self, requested: int) -> int:
        """Grant up to `requested` tool calls and count them against the budget.

        Returns how many calls may run. Calls beyond the grant are dropped,
        never executed.
        """
        if self.max_actions is None:
            granted = requested
        else:
            granted = max(0, min(requested, self.max_actions - self.actions))
        self.actions += granted
        return granted

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def actions_left(self) -> Optional[int]:
        """Remaining tool calls, or None when actions are unlimited."""
        if self.max_actions is None:
            return None
        return max(0, self.max_actions - self.actions)

    def can_use_tools(self) -> bool:
        """Whether tool calls requested in this step may still run.

        Calls are refused on the last allowed step, once the action budget is
        spent, and in the reserved final share of the time budget — so the
        model is told to write its answer instead of being cut off mid-search.
        """
        left = self.actions_left()
        time_ok = self.elapsed < self.max_seconds * (1.0 - self.answer_time_reserve)
        return (left is None or left > 0) and self.steps < self.max_steps - 1 and time_ok

    def exhausted(self) -> bool:
        """True once a hard limit (steps, time, cost) has been reached.

        The action limit is soft: when it is hit the agent may still call the
        LLM — without tools — to turn what it already found into an answer.
        """
        return self.reason() is not None

    def reason(self) -> Optional[str]:
        """Describe which hard limit stopped the loop, or None if none has."""
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            return f"cost budget hit (${self.cost_usd:.4f}/${self.max_cost_usd:.4f})"
        if self.steps >= self.max_steps:
            return f"step budget hit ({self.steps}/{self.max_steps})"
        if self.elapsed >= self.max_seconds:
            return f"time budget hit ({self.elapsed:.1f}s/{self.max_seconds:.0f}s)"
        return None

    def usage(self) -> dict:
        """Snapshot of what this run consumed (returned to API clients)."""
        return {
            "steps": self.steps,
            "max_steps": self.max_steps,
            "actions": self.actions,
            "max_actions": self.max_actions,
            "seconds": round(self.elapsed, 2),
            "max_seconds": self.max_seconds,
            "cost_usd": round(self.cost_usd, 6),
            "max_cost_usd": self.max_cost_usd,
        }
