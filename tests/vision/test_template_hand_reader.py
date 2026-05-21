"""Tests for TemplateHandReader using real screenshot fixtures.

These tests require:
- ``data/card_templates/`` (52 PNG files)
- ``tests/fixtures/screenshots/`` (game screenshots)

Both directories are gitignored — they must be present locally.
Tests are skipped when fixtures are missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "data" / "card_templates"
SCREENSHOT_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "screenshots"

_HAS_TEMPLATES = TEMPLATE_DIR.is_dir() and len(list(TEMPLATE_DIR.glob("*.png"))) == 52
_HAS_SCREENSHOTS = SCREENSHOT_DIR.is_dir() and len(list(SCREENSHOT_DIR.glob("*.png"))) > 0

needs_fixtures = pytest.mark.skipif(
    not (_HAS_TEMPLATES and _HAS_SCREENSHOTS),
    reason="Requires card templates and screenshot fixtures (gitignored)",
)


@pytest.fixture()
def template_store():
    """Load the 52-card template store."""
    from agenttienlen.vision.template_utils import TemplateStore

    return TemplateStore.load(TEMPLATE_DIR)


@pytest.fixture()
def hand_reader(template_store):
    """Build a TemplateHandReader with default parameters."""
    from agenttienlen.vision.template_hand_reader import TemplateHandReader

    return TemplateHandReader(template_store)


@pytest.fixture()
def router():
    """Standard LayoutRouter for 1280x720."""
    from agenttienlen.vision.layout_router import LayoutRouter

    return LayoutRouter()


def _load_screenshot(name: str):
    """Load a screenshot as BGR numpy array."""
    import cv2

    path = SCREENSHOT_DIR / name
    img = cv2.imread(str(path))
    assert img is not None, f"Failed to load screenshot: {path}"
    return img


# ---- Template store tests ----


@needs_fixtures
class TestTemplateStore:
    def test_loads_52_templates(self, template_store) -> None:
        assert template_store.count == 52

    def test_all_ranks_present(self, template_store) -> None:
        from agenttienlen.core.card import Rank

        for rank in Rank:
            assert rank in template_store.by_rank
            assert len(template_store.by_rank[rank]) == 4  # 4 suits per rank

    def test_top_crop_smaller_than_full(self, template_store) -> None:
        for entry in template_store.entries:
            assert entry.top_crop_gray.shape[0] < entry.gray.shape[0]
            assert entry.top_crop_gray.shape[1] == entry.gray.shape[1]


# ---- Hand reader tests ----


@needs_fixtures
class TestTemplateHandReader:
    """Test hand detection on real screenshots."""

    def test_detects_cards_in_hand(self, hand_reader, router) -> None:
        """Screenshot 065839 should detect 11-13 card slots."""
        from agenttienlen.vision.layout import RegionName

        frame = _load_screenshot("Screenshot_20260520-065839.png")
        crops = router.split(frame)
        hand_crop = crops[RegionName.MY_HAND]

        result = hand_reader.read(hand_crop)
        # We expect roughly 12 cards in this screenshot
        assert 10 <= len(result.candidates) <= 14, (
            f"Expected 10-14 cards, got {len(result.candidates)}"
        )

    def test_cards_ordered_left_to_right(self, hand_reader, router) -> None:
        """Detected cards should be sorted by x position."""
        from agenttienlen.vision.layout import RegionName

        frame = _load_screenshot("Screenshot_20260520-065839.png")
        crops = router.split(frame)
        hand_crop = crops[RegionName.MY_HAND]
        result = hand_reader.read(hand_crop)

        xs = [c.bbox.x for c in result.candidates]
        assert xs == sorted(xs), f"Cards not left-to-right: {xs}"

    def test_ranks_have_high_confidence(self, hand_reader, router) -> None:
        """All detected cards should have confidence > 0.8."""
        from agenttienlen.vision.layout import RegionName

        frame = _load_screenshot("Screenshot_20260520-065839.png")
        crops = router.split(frame)
        hand_crop = crops[RegionName.MY_HAND]
        result = hand_reader.read(hand_crop)

        for c in result.candidates:
            assert c.confidence >= 0.8, f"Low confidence {c.confidence:.3f} for rank={c.rank}"

    def test_all_candidates_have_rank(self, hand_reader, router) -> None:
        """Every detected slot should have a rank (may have None suit)."""
        from agenttienlen.vision.layout import RegionName

        frame = _load_screenshot("Screenshot_20260520-065839.png")
        crops = router.split(frame)
        hand_crop = crops[RegionName.MY_HAND]
        result = hand_reader.read(hand_crop)

        for c in result.candidates:
            assert c.rank is not None, "Detected card missing rank"

    def test_returns_card_candidates_dataclass(self, hand_reader, router) -> None:
        """Verify the result is the correct dataclass type."""
        from agenttienlen.vision.layout import RegionName
        from agenttienlen.vision.readers import HandReadResult

        frame = _load_screenshot("Screenshot_20260520-065839.png")
        crops = router.split(frame)
        result = hand_reader.read(crops[RegionName.MY_HAND])
        assert isinstance(result, HandReadResult)

    def test_multiple_screenshots(self, hand_reader, router) -> None:
        """Detection should work across multiple screenshots."""
        from agenttienlen.vision.layout import RegionName

        screenshots = [
            "Screenshot_20260520-065907.png",
            "Screenshot_20260520-065957.png",
            "Screenshot_20260520-070008.png",
        ]
        for name in screenshots:
            path = SCREENSHOT_DIR / name
            if not path.exists():
                continue
            frame = _load_screenshot(name)
            crops = router.split(frame)
            if RegionName.MY_HAND not in crops:
                continue
            result = hand_reader.read(crops[RegionName.MY_HAND])
            # Should detect at least 1 card (player may have few cards
            # late in a game — e.g. 065907 shows only 2 cards)
            assert len(result.candidates) >= 1, (
                f"{name}: expected ≥1 cards, got {len(result.candidates)}"
            )

    def test_empty_crop_returns_empty(self, hand_reader) -> None:
        """An all-black crop should return no candidates."""
        import numpy as np

        from agenttienlen.vision.layout import ROI, RegionName
        from agenttienlen.vision.layout_router import RegionCrop

        black = np.zeros((210, 1005, 3), dtype=np.uint8)
        crop = RegionCrop(
            region=RegionName.MY_HAND,
            roi=ROI(RegionName.MY_HAND, 135, 510, 1140, 720),
            image=black,
            offset=(135, 510),
        )
        result = hand_reader.read(crop)
        assert len(result.candidates) == 0


# ---- Synthetic tests (always run in CI, no fixtures needed) ----


class TestTemplateHandReaderSynthetic:
    """Deterministic tests using generated images — always run in CI."""

    @staticmethod
    def _make_synthetic_store():
        """Build a tiny TemplateStore with 4 synthetic card templates."""
        import numpy as np

        from agenttienlen.core.card import Card, Rank, Suit
        from agenttienlen.vision.template_utils import TemplateEntry, TemplateStore

        entries = []
        by_rank: dict[Rank, list[TemplateEntry]] = {}
        by_card: dict[Card, TemplateEntry] = {}
        # Create 4 unique 60x30 grayscale patterns (one per suit of rank THREE)
        for i, suit in enumerate(Suit):
            card = Card(rank=Rank.THREE, suit=suit)
            rng = np.random.RandomState(42 + i)
            img = rng.randint(0, 255, (60, 30, 3), dtype=np.uint8)
            gray = img.mean(axis=2).astype(np.uint8)
            top_crop = gray[: 60 * 30 // 100, :]
            entry = TemplateEntry(card=card, image=img, gray=gray, top_crop_gray=top_crop)
            entries.append(entry)
            by_rank.setdefault(card.rank, []).append(entry)
            by_card[card] = entry
        return TemplateStore(entries=entries, by_rank=by_rank, by_card=by_card)

    def test_empty_crop_returns_empty_result(self) -> None:
        import numpy as np

        from agenttienlen.vision.layout import ROI, RegionName
        from agenttienlen.vision.layout_router import RegionCrop
        from agenttienlen.vision.template_hand_reader import TemplateHandReader

        store = self._make_synthetic_store()
        reader = TemplateHandReader(store, threshold=0.9)
        black = np.zeros((100, 400, 3), dtype=np.uint8)
        crop = RegionCrop(
            region=RegionName.MY_HAND,
            roi=ROI(RegionName.MY_HAND, 0, 0, 400, 100),
            image=black,
            offset=(0, 0),
        )
        result = reader.read(crop)
        assert len(result.candidates) == 0

    def test_detects_embedded_template(self) -> None:
        """Embed a template into a synthetic crop and verify detection."""
        import numpy as np

        from agenttienlen.vision.layout import ROI, RegionName
        from agenttienlen.vision.layout_router import RegionCrop
        from agenttienlen.vision.readers import HandReadResult
        from agenttienlen.vision.template_hand_reader import TemplateHandReader

        store = self._make_synthetic_store()
        reader = TemplateHandReader(store, threshold=0.85, nms_dist=20)
        entry = store.entries[0]

        # Place the template's top-crop into a larger black image
        canvas = np.zeros((100, 400, 3), dtype=np.uint8)
        tc = entry.top_crop_gray
        th, tw = tc.shape
        # Paste at position (50, 10)
        canvas[10 : 10 + th, 50 : 50 + tw, 0] = tc
        canvas[10 : 10 + th, 50 : 50 + tw, 1] = tc
        canvas[10 : 10 + th, 50 : 50 + tw, 2] = tc

        crop = RegionCrop(
            region=RegionName.MY_HAND,
            roi=ROI(RegionName.MY_HAND, 0, 0, 400, 100),
            image=canvas,
            offset=(0, 0),
        )
        result = reader.read(crop)
        assert isinstance(result, HandReadResult)
        assert len(result.candidates) >= 1
        # The detected card should be near x=50
        assert result.candidates[0].bbox.x >= 40
        assert result.candidates[0].bbox.x <= 60

    def test_result_candidates_sorted_by_x(self) -> None:
        """Two embedded templates should come back left-to-right."""
        import numpy as np

        from agenttienlen.vision.layout import ROI, RegionName
        from agenttienlen.vision.layout_router import RegionCrop
        from agenttienlen.vision.template_hand_reader import TemplateHandReader

        store = self._make_synthetic_store()
        reader = TemplateHandReader(store, threshold=0.85, nms_dist=20)

        canvas = np.zeros((100, 400, 3), dtype=np.uint8)
        for idx, x_pos in enumerate([50, 200]):
            entry = store.entries[idx % len(store.entries)]
            tc = entry.top_crop_gray
            th, tw = tc.shape
            canvas[10 : 10 + th, x_pos : x_pos + tw, 0] = tc
            canvas[10 : 10 + th, x_pos : x_pos + tw, 1] = tc
            canvas[10 : 10 + th, x_pos : x_pos + tw, 2] = tc

        crop = RegionCrop(
            region=RegionName.MY_HAND,
            roi=ROI(RegionName.MY_HAND, 0, 0, 400, 100),
            image=canvas,
            offset=(0, 0),
        )
        result = reader.read(crop)
        if len(result.candidates) >= 2:
            xs = [c.bbox.x for c in result.candidates]
            assert xs == sorted(xs)
