"""Scalar smoothers using elapsed seconds, independent of pose and GUI code."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

SmoothingMethod = Literal["ema", "moving_average", "none"]


@dataclass(frozen=True)
class SmoothingSettings:
    method: SmoothingMethod = "ema"
    seconds: float = 0.35
    max_gap_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.method not in ("ema", "moving_average", "none"):
            raise ValueError("Smoothing method must be ema, moving_average, or none")
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("Smoothing seconds must be finite and non-negative")
        if not math.isfinite(self.max_gap_seconds) or self.max_gap_seconds <= 0:
            raise ValueError("Maximum tracking gap must be finite and positive")


class SignalSmoother:
    """EMA time constant or sample-weighted trailing moving-average window.

    EMA: alpha = 1-exp(-dt/tau); y = y_previous + alpha*(x-y_previous).
    Moving average: mean of observed samples in [t-window, t], inclusive.
    Both seed from the first valid value; neither fills missing observations.
    Missing/nonfinite values and excessive gaps reset the filter.
    """

    def __init__(self, settings: SmoothingSettings):
        self.settings = settings
        self._time: int | None = None
        self._last_seen: int | None = None
        self._value: float | None = None
        self._window: deque[tuple[int, float]] = deque()

    def reset(self) -> None:
        self._time = None
        self._value = None
        self._window.clear()

    def update(self, timestamp_ns: int, value: float | None) -> float | None:
        if self._last_seen is not None and timestamp_ns <= self._last_seen:
            raise ValueError("Smoothing timestamps must strictly increase")
        self._last_seen = timestamp_ns
        if value is None or not math.isfinite(value):
            self.reset()
            return None
        dt = (timestamp_ns - self._time) / 1e9 if self._time is not None else None
        if dt is not None and dt > self.settings.max_gap_seconds:
            self.reset()
            dt = None
        method, seconds = self.settings.method, self.settings.seconds
        if method == "none" or seconds == 0 or self._value is None:
            smoothed = value
        elif method == "ema":
            assert dt is not None
            alpha = -math.expm1(-dt / seconds)
            smoothed = self._value + alpha * (value - self._value)
        else:
            smoothed = value  # Replaced by the window mean below.
        if method == "moving_average" and seconds > 0:
            self._window.append((timestamp_ns, value))
            cutoff = timestamp_ns - round(seconds * 1e9)
            while self._window[0][0] < cutoff:
                self._window.popleft()
            smoothed = math.fsum(point[1] for point in self._window) / len(self._window)
        self._time, self._value = timestamp_ns, smoothed
        return smoothed
