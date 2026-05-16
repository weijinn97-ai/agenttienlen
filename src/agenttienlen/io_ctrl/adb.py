"""ADB controller for emulators (MEmu / BlueStacks / LDPlayer) or real Android.

`adbutils` is imported lazily so unit tests of pure-Python modules do not
require it.

Typical use::

    ctl = AdbController(serial="127.0.0.1:21503")  # default MEmu port
    frame = ctl.screencap()  # numpy BGR
    ctl.tap(640, 360)
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class AdbController:
    """Thin adbutils wrapper. Constructed cheap; connects on first call."""

    def __init__(
        self, serial: str | None = None, *, host: str = "127.0.0.1", port: int = 5037
    ) -> None:
        self.serial = serial
        self.host = host
        self.port = port
        self._device: object | None = None

    def _ensure_device(self) -> object:
        if self._device is not None:
            return self._device
        import adbutils  # heavy import deferred

        adb = adbutils.AdbClient(host=self.host, port=self.port)
        if self.serial:
            self._device = adb.device(self.serial)
        else:
            devices = adb.device_list()
            if not devices:
                raise RuntimeError("No ADB devices found. Start an emulator or connect a device.")
            self._device = devices[0]
        return self._device

    def screencap(self) -> np.ndarray:
        """Capture the current screen as a BGR numpy array."""
        import cv2
        import numpy as np

        device = self._ensure_device()
        png_bytes: bytes = device.screencap()  # type: ignore[attr-defined]
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to decode screencap PNG")
        return frame

    def tap(self, x: int, y: int) -> None:
        device = self._ensure_device()
        device.shell(f"input tap {x} {y}")  # type: ignore[attr-defined]

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> None:
        device = self._ensure_device()
        device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")  # type: ignore[attr-defined]

    # ``io`` is imported only for typed BytesIO hints downstream.
    _IO_FOR_TYPE_HINT = io  # noqa: RUF100
