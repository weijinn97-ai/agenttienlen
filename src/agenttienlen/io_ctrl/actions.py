"""High-level game actions: click cards in hand, click Play, click Pass.

The orchestrator translates an :class:`agenttienlen.agent.policy.Action` into
a sequence of taps via :class:`GameActions`.

Tap coordinates are calibrated against 1280x720 captures (matching
``vision/layout.py``). For other resolutions, override the layout/button
fields when constructing :class:`GameActions`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

Tapper = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class HandLayout:
    """Geometry of the bot's hand row at the bottom of the screen.

    The bot's cards are laid out evenly between ``x_start`` and ``x_end`` at
    vertical centre ``y_center``. Tapping the card at index ``i`` (0-based,
    left → right) clicks position ``x_start + i * stride``.
    """

    x_start: int = 280
    x_end: int = 1060
    y_center: int = 620


# Default button locations matching the screenshots in this repo.
DEFAULT_PLAY_BUTTON: tuple[int, int] = (1195, 630)
DEFAULT_PASS_BUTTON: tuple[int, int] = (920, 75)


class GameActions:
    """Click cards and buttons via a user-supplied tap callable."""

    def __init__(
        self,
        tap: Tapper,
        *,
        hand_layout: HandLayout | None = None,
        play_button: tuple[int, int] = DEFAULT_PLAY_BUTTON,
        pass_button: tuple[int, int] = DEFAULT_PASS_BUTTON,
    ) -> None:
        self._tap = tap
        self.hand_layout = hand_layout or HandLayout()
        self.play_button = play_button
        self.pass_button = pass_button

    def click_card_at(self, index: int, total: int) -> None:
        """Click the card at hand position ``index`` (0-based) given ``total`` cards visible."""
        if total <= 0 or index < 0 or index >= total:
            raise IndexError(f"card index {index} out of range for {total} cards")
        lo = self.hand_layout
        if total == 1:
            x = (lo.x_start + lo.x_end) // 2
        else:
            stride = (lo.x_end - lo.x_start) / (total - 1)
            x = int(lo.x_start + index * stride)
        self._tap(x, lo.y_center)

    def click_cards(self, indices: Sequence[int], total: int) -> None:
        for i in indices:
            self.click_card_at(i, total)

    def click_play(self) -> None:
        self._tap(*self.play_button)

    def click_pass(self) -> None:
        self._tap(*self.pass_button)
