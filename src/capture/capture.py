"""Synchronous webcam capture. Images stay local and are never saved here."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from types import TracebackType
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Frame:
    """BGR uint8 image, one-based sequence, and application read-completion time.

    timestamp_ns uses a high-resolution monotonic clock, not camera exposure time.
    Clock ties advance by 1 ns to keep timestamps strictly increasing.
    Consumers should treat image as read-only.
    """

    image: NDArray[np.uint8]
    sequence: int
    timestamp_ns: int


class CaptureError(RuntimeError):
    """The camera could not be opened or a frame could not be read."""


class CameraCapture:
    """Own one camera; use with a context manager for reliable cleanup."""

    def __init__(self, camera_index: int = 0) -> None:
        if camera_index < 0:
            raise ValueError("Camera index must be non-negative.")
        self.camera_index = camera_index
        self._device: cv2.VideoCapture | None = None
        self._sequence = 0
        self._last_timestamp_ns: int | None = None

    def open(self) -> CameraCapture:
        if self._device is not None:
            raise RuntimeError("Camera is already open.")
        device = cv2.VideoCapture()
        try:
            if not device.open(self.camera_index) or not device.isOpened():
                raise CaptureError(
                    f"Cannot open camera {self.camera_index}. Check the camera index, "
                    "Windows camera permissions, and whether another app is using it."
                )
        except BaseException as error:
            device.release()
            if isinstance(error, cv2.error):
                raise CaptureError(f"Cannot open camera {self.camera_index}: {error}") from error
            raise
        self._device = device
        self._sequence = 0
        self._last_timestamp_ns = None
        return self

    def read(self) -> Frame:
        if self._device is None:
            raise RuntimeError("Open the camera before reading frames.")
        try:
            success, image = self._device.read()
        except cv2.error as error:
            raise CaptureError(f"Cannot read from camera {self.camera_index}: {error}") from error
        if not success or image is None or image.size == 0:
            raise CaptureError(f"Camera {self.camera_index} failed to deliver a frame.")
        # On Windows/Python 3.12, monotonic_ns uses a coarse tick clock that
        # can repeat between camera reads. perf_counter_ns is high resolution.
        timestamp_ns = perf_counter_ns()
        if self._last_timestamp_ns is not None:
            timestamp_ns = max(timestamp_ns, self._last_timestamp_ns + 1)
        self._last_timestamp_ns = timestamp_ns
        self._sequence += 1
        # OpenCV's stub allows other dtypes; webcam BGR frames use uint8.
        return Frame(
            image=cast(NDArray[np.uint8], image),
            sequence=self._sequence,
            timestamp_ns=timestamp_ns,
        )

    def close(self) -> None:
        if self._device is not None:
            device = self._device
            self._device = None
            device.release()

    def __enter__(self) -> CameraCapture:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
