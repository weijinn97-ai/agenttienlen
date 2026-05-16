"""Temporal voting: confirm an observation is steady across N frames.

In a real-time game the per-frame readers flicker — cards animate in,
buttons blink, opponents' back stacks momentarily double-count. Without
temporal voting these spurious readings would slam ``GameState`` and
mis-trigger the agent.

:class:`FrameStabilizer` is the spec's confirmation gate (§9): an
observation is published as ``stable`` only after the same key has been
seen for ``required_frames`` consecutive ticks. The stabilizer is
generic on the observation type and on a user-supplied ``key_fn`` that
defines what "same" means (e.g. ``frozenset(cards)`` for hand cards,
``count`` for back-card counts).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class StableResult(Generic[T]):
    """Output of one :meth:`FrameStabilizer.update` call.

    ``is_stable`` is ``True`` iff the latest observation matched all
    previous observations in the rolling window. ``value`` is the latest
    observation (returned even when unstable so callers can inspect
    flicker). ``streak`` is the number of consecutive observations with
    the same key (clamped to ``required_frames``).
    """

    is_stable: bool
    value: T | None
    streak: int


@dataclass(slots=True)
class FrameStabilizer(Generic[T]):
    """Generic N-of-N consecutive-match confirmation gate.

    Typical usage::

        stabilizer = FrameStabilizer[list[Card]](
            key_fn=lambda cards: frozenset(cards),
            required_frames=3,
        )
        for frame in stream:
            result = stabilizer.update(read_hand(frame))
            if result.is_stable and result.value is not None:
                game_state.my_hand = result.value
    """

    key_fn: Callable[[T], Hashable]
    required_frames: int = 3
    _history: deque[Hashable] = field(init=False, repr=False)
    _last_value: T | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.required_frames < 1:
            raise ValueError(f"required_frames must be >= 1: {self.required_frames}")
        self._history = deque(maxlen=self.required_frames)

    def update(self, observation: T) -> StableResult[T]:
        """Push a new observation and return whether the window is stable."""
        key = self.key_fn(observation)
        self._history.append(key)
        self._last_value = observation
        streak = self._current_streak()
        is_stable = len(self._history) >= self.required_frames and streak >= self.required_frames
        return StableResult(is_stable=is_stable, value=observation, streak=streak)

    def reset(self) -> None:
        """Drop the history (e.g. when entering a new hand or recovering)."""
        self._history.clear()
        self._last_value = None

    @property
    def streak(self) -> int:
        return self._current_streak()

    @property
    def last_value(self) -> T | None:
        return self._last_value

    def _current_streak(self) -> int:
        if not self._history:
            return 0
        latest = self._history[-1]
        count = 0
        # deque is iterable from oldest to newest; walk backwards.
        for key in reversed(self._history):
            if key == latest:
                count += 1
            else:
                break
        return count
