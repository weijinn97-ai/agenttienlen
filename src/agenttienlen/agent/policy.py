"""Policy interface and Action types.

Every agent strategy implements :class:`Policy`. The orchestrator only ever
sees an :class:`Action` (``Play | Pass``); concrete policies are pluggable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agenttienlen.core.combo import Combo
from agenttienlen.memory.game_state import GameState


@dataclass(frozen=True, slots=True)
class Play:
    combo: Combo

    def __str__(self) -> str:
        return f"PLAY {self.combo}"


@dataclass(frozen=True, slots=True)
class Pass:
    reason: str = ""

    def __str__(self) -> str:
        return f"PASS ({self.reason})" if self.reason else "PASS"


Action = Play | Pass


class Policy(Protocol):
    """Strategy interface. Implementations are pure functions of GameState."""

    def decide(self, state: GameState) -> Action: ...
