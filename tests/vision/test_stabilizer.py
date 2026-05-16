"""Tests for :class:`FrameStabilizer`."""

from __future__ import annotations

from collections.abc import Hashable

import pytest

from agenttienlen.vision.stabilizer import FrameStabilizer


def _identity(x: Hashable) -> Hashable:
    return x


class TestUpdate:
    def test_first_observation_is_unstable(self) -> None:
        stab: FrameStabilizer[int] = FrameStabilizer(key_fn=_identity, required_frames=3)
        result = stab.update(7)
        assert result.is_stable is False
        assert result.value == 7
        assert result.streak == 1

    def test_consecutive_matches_become_stable(self) -> None:
        stab: FrameStabilizer[int] = FrameStabilizer(key_fn=_identity, required_frames=3)
        assert stab.update(7).is_stable is False
        assert stab.update(7).is_stable is False
        result = stab.update(7)
        assert result.is_stable is True
        assert result.streak == 3
        assert result.value == 7

    def test_break_resets_streak(self) -> None:
        stab: FrameStabilizer[int] = FrameStabilizer(key_fn=_identity, required_frames=3)
        stab.update(7)
        stab.update(7)
        result = stab.update(9)
        assert result.is_stable is False
        assert result.streak == 1
        assert result.value == 9

    def test_required_frames_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="required_frames"):
            FrameStabilizer(key_fn=_identity, required_frames=0)


class TestKeyFn:
    def test_custom_key_fn_groups_observations(self) -> None:
        # Treat any positive number as the same group, any non-positive as another.
        stab: FrameStabilizer[int] = FrameStabilizer(
            key_fn=lambda n: n > 0,
            required_frames=2,
        )
        stab.update(1)
        result = stab.update(99)  # different value, same key (both > 0)
        assert result.is_stable is True
        assert result.value == 99

    def test_frozenset_key_for_hand_cards(self) -> None:
        stab: FrameStabilizer[list[str]] = FrameStabilizer(
            key_fn=lambda cards: frozenset(cards),
            required_frames=2,
        )
        stab.update(["3c", "5d", "AH"])
        # Same multiset, different order → still stable.
        result = stab.update(["AH", "3c", "5d"])
        assert result.is_stable is True


class TestReset:
    def test_reset_clears_state(self) -> None:
        stab: FrameStabilizer[int] = FrameStabilizer(key_fn=_identity, required_frames=2)
        stab.update(7)
        stab.update(7)
        assert stab.streak == 2
        stab.reset()
        assert stab.streak == 0
        assert stab.last_value is None
        # After reset, a single observation is not stable on its own.
        assert stab.update(7).is_stable is False


class TestProperties:
    def test_last_value_tracks_latest_observation(self) -> None:
        stab: FrameStabilizer[str] = FrameStabilizer(key_fn=_identity, required_frames=2)
        stab.update("a")
        stab.update("b")
        assert stab.last_value == "b"
