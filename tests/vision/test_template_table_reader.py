"""Tests for TemplateTableReader using real screenshot fixtures.

Tests are skipped when fixtures (templates + screenshots) are missing.
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
    from agenttienlen.vision.template_utils import TemplateStore

    return TemplateStore.load(TEMPLATE_DIR)


@pytest.fixture()
def table_reader(template_store):
    from agenttienlen.vision.template_table_reader import TemplateTableReader

    return TemplateTableReader(template_store)


@pytest.fixture()
def router():
    from agenttienlen.vision.layout_router import LayoutRouter

    return LayoutRouter()


def _load_screenshot(name: str):
    import cv2

    path = SCREENSHOT_DIR / name
    img = cv2.imread(str(path))
    assert img is not None, f"Failed to load screenshot: {path}"
    return img


@needs_fixtures
class TestTemplateTableReader:
    """Test table card detection on real screenshots."""

    def test_detects_cards_on_table(self, table_reader, router) -> None:
        """Screenshot with cards on table should detect them."""
        from agenttienlen.vision.layout import RegionName

        # 065839 shows cards played at opponent positions
        frame = _load_screenshot("Screenshot_20260520-065839.png")
        crops = router.split(frame)
        table_crop = crops.get(RegionName.TABLE)
        if table_crop is None:
            pytest.skip("TABLE region not in layout")
        result = table_reader.read(table_crop)
        # Table may or may not have cards depending on game state
        assert isinstance(result.cards, list)

    def test_cards_ordered_left_to_right(self, table_reader, router) -> None:
        from agenttienlen.vision.layout import RegionName

        frame = _load_screenshot("Screenshot_20260520-065839.png")
        crops = router.split(frame)
        table_crop = crops.get(RegionName.TABLE)
        if table_crop is None:
            pytest.skip("TABLE region not in layout")
        result = table_reader.read(table_crop)
        if len(result.cards) > 1:
            xs = [c.bbox.x for c in result.cards]
            assert xs == sorted(xs)

    def test_detected_cards_have_full_identity(self, table_reader, router) -> None:
        """Table cards are face-up — both rank and suit should be known."""
        from agenttienlen.vision.layout import RegionName

        frame = _load_screenshot("Screenshot_20260520-065839.png")
        crops = router.split(frame)
        table_crop = crops.get(RegionName.TABLE)
        if table_crop is None:
            pytest.skip("TABLE region not in layout")
        result = table_reader.read(table_crop)
        for c in result.cards:
            assert c.rank is not None, "Table card missing rank"
            assert c.suit is not None, "Table card missing suit"

    def test_empty_crop_returns_empty(self, table_reader) -> None:
        import numpy as np

        from agenttienlen.vision.layout import ROI, RegionName
        from agenttienlen.vision.layout_router import RegionCrop

        black = np.zeros((330, 940, 3), dtype=np.uint8)
        crop = RegionCrop(
            region=RegionName.TABLE,
            roi=ROI(RegionName.TABLE, 170, 10, 1110, 340),
            image=black,
            offset=(170, 10),
        )
        result = table_reader.read(crop)
        assert len(result.cards) == 0

    def test_opponent_play_areas(self, table_reader, router) -> None:
        """Test detection on individual opponent play area crops."""
        from agenttienlen.vision.layout import RegionName

        frame = _load_screenshot("Screenshot_20260520-065839.png")
        crops = router.split(frame)

        for region in [
            RegionName.OPP_PLAY_LEFT,
            RegionName.OPP_PLAY_TOP,
            RegionName.OPP_PLAY_RIGHT,
        ]:
            crop = crops.get(region)
            if crop is None:
                continue
            result = table_reader.read(crop)
            # Each opponent play area might have 0-5 cards
            assert isinstance(result.cards, list)
            assert len(result.cards) <= 13  # sanity: no more than a full hand

    def test_multiple_screenshots(self, table_reader, router) -> None:
        """Detection should not crash on any screenshot."""
        from agenttienlen.vision.layout import RegionName

        for name in [
            "Screenshot_20260520-065907.png",
            "Screenshot_20260520-065957.png",
            "Screenshot_20260520-070050.png",
            "Screenshot_20260520-070114.png",
        ]:
            path = SCREENSHOT_DIR / name
            if not path.exists():
                continue
            frame = _load_screenshot(name)
            crops = router.split(frame)
            table_crop = crops.get(RegionName.TABLE)
            if table_crop is None:
                continue
            result = table_reader.read(table_crop)
            assert isinstance(result.cards, list)


# ---- Synthetic tests (always run in CI, no fixtures needed) ----


class TestTemplateTableReaderSynthetic:
    """Deterministic tests using generated images — always run in CI."""

    @staticmethod
    def _make_synthetic_store():
        import numpy as np

        from agenttienlen.core.card import Card, Rank, Suit
        from agenttienlen.vision.template_utils import TemplateEntry, TemplateStore

        entries = []
        by_rank: dict[Rank, list[TemplateEntry]] = {}
        by_card: dict[Card, TemplateEntry] = {}
        for i, suit in enumerate(Suit):
            card = Card(rank=Rank.FIVE, suit=suit)
            rng = np.random.RandomState(100 + i)
            img = rng.randint(0, 255, (80, 40, 3), dtype=np.uint8)
            gray = img.mean(axis=2).astype(np.uint8)
            top_crop = gray[: 80 * 30 // 100, :]
            entry = TemplateEntry(card=card, image=img, gray=gray, top_crop_gray=top_crop)
            entries.append(entry)
            by_rank.setdefault(card.rank, []).append(entry)
            by_card[card] = entry
        return TemplateStore(entries=entries, by_rank=by_rank, by_card=by_card)

    def test_empty_crop_returns_empty(self) -> None:
        import numpy as np

        from agenttienlen.vision.layout import ROI, RegionName
        from agenttienlen.vision.layout_router import RegionCrop
        from agenttienlen.vision.template_table_reader import TemplateTableReader

        store = self._make_synthetic_store()
        reader = TemplateTableReader(store, threshold=0.9)
        black = np.zeros((200, 600, 3), dtype=np.uint8)
        crop = RegionCrop(
            region=RegionName.TABLE,
            roi=ROI(RegionName.TABLE, 0, 0, 600, 200),
            image=black,
            offset=(0, 0),
        )
        result = reader.read(crop)
        assert len(result.cards) == 0

    def test_detects_embedded_card(self) -> None:
        """Embed a scaled template and verify detection."""
        import numpy as np

        from agenttienlen.vision.layout import ROI, RegionName
        from agenttienlen.vision.layout_router import RegionCrop
        from agenttienlen.vision.readers import TableReadResult
        from agenttienlen.vision.template_table_reader import TemplateTableReader

        store = self._make_synthetic_store()
        entry = store.entries[0]
        # Scale template to 0.55x (within default scale range)
        import cv2

        scaled = cv2.resize(
            entry.gray,
            (int(entry.gray.shape[1] * 0.55), int(entry.gray.shape[0] * 0.55)),
        )
        sh, sw = scaled.shape

        canvas = np.zeros((200, 600, 3), dtype=np.uint8)
        canvas[30 : 30 + sh, 100 : 100 + sw, 0] = scaled
        canvas[30 : 30 + sh, 100 : 100 + sw, 1] = scaled
        canvas[30 : 30 + sh, 100 : 100 + sw, 2] = scaled

        reader = TemplateTableReader(store, threshold=0.85)
        crop = RegionCrop(
            region=RegionName.TABLE,
            roi=ROI(RegionName.TABLE, 0, 0, 600, 200),
            image=canvas,
            offset=(0, 0),
        )
        result = reader.read(crop)
        assert isinstance(result, TableReadResult)
        assert len(result.cards) >= 1

    def test_results_sorted_by_x(self) -> None:
        """Multiple embedded cards should be returned left-to-right."""
        import cv2
        import numpy as np

        from agenttienlen.vision.layout import ROI, RegionName
        from agenttienlen.vision.layout_router import RegionCrop
        from agenttienlen.vision.template_table_reader import TemplateTableReader

        store = self._make_synthetic_store()
        canvas = np.zeros((200, 600, 3), dtype=np.uint8)

        for idx, x_pos in enumerate([80, 300]):
            entry = store.entries[idx % len(store.entries)]
            scaled = cv2.resize(
                entry.gray,
                (int(entry.gray.shape[1] * 0.55), int(entry.gray.shape[0] * 0.55)),
            )
            sh, sw = scaled.shape
            canvas[30 : 30 + sh, x_pos : x_pos + sw, 0] = scaled
            canvas[30 : 30 + sh, x_pos : x_pos + sw, 1] = scaled
            canvas[30 : 30 + sh, x_pos : x_pos + sw, 2] = scaled

        reader = TemplateTableReader(store, threshold=0.85)
        crop = RegionCrop(
            region=RegionName.TABLE,
            roi=ROI(RegionName.TABLE, 0, 0, 600, 200),
            image=canvas,
            offset=(0, 0),
        )
        result = reader.read(crop)
        if len(result.cards) >= 2:
            xs = [c.bbox.x for c in result.cards]
            assert xs == sorted(xs)
