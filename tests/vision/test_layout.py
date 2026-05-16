"""Tests for the ROI layout helpers."""

from __future__ import annotations

import pytest

from agenttienlen.vision.layout import (
    ROI,
    RegionName,
    assign_region,
    default_layout_1280x720,
    full_layout_1280x720,
    scale_layout,
)


class TestRoi:
    def test_width_and_height(self) -> None:
        r = ROI(RegionName.MY_HAND, 10, 20, 50, 80)
        assert r.width == 40
        assert r.height == 60

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ROI(RegionName.MY_HAND, 10, 20, 5, 30)
        with pytest.raises(ValueError, match="non-negative"):
            ROI(RegionName.MY_HAND, 10, 20, 30, 5)


class TestFullLayout:
    def test_covers_all_region_names(self) -> None:
        layout = full_layout_1280x720()
        names = {roi.name for roi in layout}
        assert names == set(RegionName)

    def test_within_screen_bounds(self) -> None:
        for roi in full_layout_1280x720():
            assert 0 <= roi.x1 < roi.x2 <= 1280, roi
            assert 0 <= roi.y1 < roi.y2 <= 720, roi

    def test_default_layout_is_legacy_subset(self) -> None:
        legacy_names = {roi.name for roi in default_layout_1280x720()}
        assert legacy_names == {
            RegionName.MY_HAND,
            RegionName.TABLE,
            RegionName.OPP_LEFT,
            RegionName.OPP_TOP,
            RegionName.OPP_RIGHT,
        }


class TestScaleLayout:
    def test_identity_scale(self) -> None:
        layout = full_layout_1280x720()
        scaled = scale_layout(layout, 1280, 720)
        assert [(r.name, r.x1, r.y1, r.x2, r.y2) for r in scaled] == [
            (r.name, r.x1, r.y1, r.x2, r.y2) for r in layout
        ]

    def test_double_scale(self) -> None:
        original = [ROI(RegionName.MY_HAND, 100, 200, 300, 400)]
        scaled = scale_layout(original, 2560, 1440)
        assert scaled[0].x1 == 200
        assert scaled[0].y1 == 400
        assert scaled[0].x2 == 600
        assert scaled[0].y2 == 800


class TestAssignRegion:
    def test_point_inside_my_hand(self) -> None:
        layout = full_layout_1280x720()
        assert assign_region(600, 600, layout) == RegionName.MY_HAND

    def test_point_inside_avatar_me(self) -> None:
        layout = full_layout_1280x720()
        assert assign_region(75, 600, layout) == RegionName.AVATAR_ME

    def test_point_off_screen_returns_none(self) -> None:
        layout = full_layout_1280x720()
        assert assign_region(2000, 2000, layout) is None
