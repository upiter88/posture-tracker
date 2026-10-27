"""Periodic diagnostics for the capture -> pose pipeline."""

import logging
from time import monotonic

from capture import Frame
from pose import PoseObservation


class PoseDiagnostics:
    """Accumulate one interval at a time, without retaining images or observations."""

    def __init__(self) -> None:
        self._last_report = monotonic()
        self._frames = 0
        self._detected = 0
        self._inference_seconds = 0.0

    def report(self, frame: Frame, pose: PoseObservation, inference_seconds: float) -> None:
        self._frames += 1
        self._detected += int(pose.detected)
        self._inference_seconds += inference_seconds
        now = monotonic()
        elapsed = now - self._last_report
        if elapsed < 1.0:
            return
        confidence = (
            ", ".join(f"{name}={point.visibility:.2f}/{point.presence:.2f}" for name, point in pose.landmarks.items())
            or "no pose"
        )
        logging.info(
            "Frames=%d | %dx%d | pipeline=%.1f FPS | pose=%d/%d | "
            "inference=%.1f ms | latest visibility/presence: %s | timestamp_ns=%d",
            frame.sequence,
            pose.image_width,
            pose.image_height,
            self._frames / elapsed,
            self._detected,
            self._frames,
            1000 * self._inference_seconds / self._frames,
            confidence,
            pose.timestamp_ns,
        )
        self._last_report = now
        self._frames = 0
        self._detected = 0
        self._inference_seconds = 0.0
