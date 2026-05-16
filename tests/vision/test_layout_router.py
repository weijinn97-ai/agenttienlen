"""Tests for :class:`LayoutRouter`."""

from __future__ import annotations

import numpy as np
import pytest

from agenttienlen.vision.layout import ROI, RegionName
from agenttienlen.vision.layout_router import LayoutRouter


def _frame(height: int = 720, width: int = 1280) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestSplit:
    def test_default_layout_returns_all_regions(self) -> None:
        router = LayoutRouter()
        crops = router.split(_frame())
        assert set(crops) == set(RegionName)
        for crop in crops.values():
            assert crop.image.shape[0] == crop.roi.height
            assert crop.image.shape[1] == crop.roi.width
            assert crop.offset == (crop.roi.x1, crop.roi.y1)

    def test_clips_out_of_bounds_rois(self) -> None:
        layout = [ROI(RegionName.MY_HAND, 100, 100, 2000, 1000)]
        router = LayoutRouter(layout)
        crops = router.split(_frame(720, 1280))
        crop = crops[RegionName.MY_HAND]
        assert crop.image.shape[:2] == (720 - 100, 1280 - 100)
        assert crop.offset == (100, 100)

    def test_skips_fully_outside_rois(self) -> None:
        layout = [ROI(RegionName.TABLE, 2000, 2000, 2100, 2100)]
        router = LayoutRouter(layout)
        crops = router.split(_frame())
        assert crops == {}

    def test_image_is_a_view(self) -> None:
        layout = [ROI(RegionName.TABLE, 10, 10, 20, 20)]
        router = LayoutRouter(layout)
        frame = _frame()
        crop = router.split(frame)[RegionName.TABLE]
        crop.image[0, 0] = (1, 2, 3)
        assert tuple(frame[10, 10]) == (1, 2, 3)

    def test_rejects_1d_frame(self) -> None:
        router = LayoutRouter()
        with pytest.raises(ValueError, match="frame must be 2D"):
            router.split(np.zeros((100,), dtype=np.uint8))


class TestCrop:
    def test_returns_named_region(self) -> None:
        router = LayoutRouter()
        crop = router.crop(_frame(), RegionName.MY_HAND)
        assert crop is not None
        assert crop.region is RegionName.MY_HAND

    def test_returns_none_when_region_not_in_layout(self) -> None:
        router = LayoutRouter([ROI(RegionName.MY_HAND, 0, 0, 10, 10)])
        assert router.crop(_frame(), RegionName.AVATAR_TOP) is None
