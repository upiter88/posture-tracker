"""OpenCV dashboard: camera, status, and bounded raw/smoothed signal traces."""

from __future__ import annotations

from collections import deque
from types import TracebackType

import cv2
import numpy as np

from calibration_profile import CalibrationProfile
from capture import Frame
from evaluation import PostureResult
from pose import PoseObservation
from preview import draw_pose

BACKGROUND = (27, 24, 21)
PANEL = (40, 36, 32)
TEXT = (237, 235, 229)
MUTED = (156, 151, 142)
RAW = (160, 153, 143)
SMOOTHED = (83, 211, 244)
ENTRY = (118, 117, 245)
CLEAR = (170, 203, 99)
STATUS_COLORS = {"WITHIN_REFERENCE": CLEAR, "DEVIATION": ENTRY, "UNKNOWN": SMOOTHED}


def _text(image, text: str, position, color=TEXT, scale=0.5):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _coordinates(bounds, now, history_seconds, low, high):
    """Return pixel transforms bound to one chart's time/value limits."""
    left, right, bottom, chart_top = bounds

    def x_pixel(timestamp):
        return round(right + (timestamp - now) / 1e9 / history_seconds * (right - left))

    def y_pixel(value):
        return round(bottom - (value - low) / (high - low) * (bottom - chart_top))

    return x_pixel, y_pixel


def raw_thresholds(profile: CalibrationProfile, feature: str) -> list[tuple[float, str]]:
    """Convert score-space thresholds back to the displayed feature's native units."""
    lines = []
    for rule in profile.rules:
        if rule.feature != feature:
            continue
        signs = (-1,) if rule.direction == "decrease" else (1,) if rule.direction == "increase" else (-1, 1)
        for sign in signs:
            lines.extend(
                (
                    (rule.reference.centre + sign * rule.enter_threshold * rule.reference.scale, "enter"),
                    (rule.reference.centre + sign * rule.clear_threshold * rule.reference.scale, "clear"),
                )
            )
    return lines


def _trace(image, history, feature, attribute, coordinates, color, max_gap_seconds):
    x_pixel, y_pixel = coordinates
    run = []
    previous_time = None
    for item in history:
        signal = item.signals.get(feature)
        value = getattr(signal, attribute) if signal else None
        gap = previous_time is not None and (item.timestamp_ns - previous_time) / 1e9 > max_gap_seconds
        if value is None or gap:
            if len(run) >= 2:
                cv2.polylines(image, [np.array(run, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
            run = []
        if value is not None:
            run.append((x_pixel(item.timestamp_ns), y_pixel(value)))
        previous_time = item.timestamp_ns
    if len(run) >= 2:
        cv2.polylines(
            image, [np.array(run, dtype=np.int32)], False, color, 2 if attribute == "smoothed" else 1, cv2.LINE_AA
        )


def render_dashboard(
    frame: Frame,
    pose: PoseObservation,
    result: PostureResult,
    profile: CalibrationProfile,
    history: deque[PostureResult],
    features: tuple[str, ...],
    method: str,
    history_seconds: float,
    max_gap_seconds: float,
):
    """Pure renderer for tests and screenshots; input frames stay unchanged."""
    height = max(620, 95 + 200 * len(features))
    canvas = np.full((height, 1400, 3), BACKGROUND, dtype=np.uint8)
    color = STATUS_COLORS[result.status]
    _text(canvas, "POSTURE TRACKER", (24, 31), MUTED, 0.48)
    _text(canvas, result.status.replace("_", " "), (24, 67), color, 0.85)
    _text(canvas, result.reason[:84], (24, 91), MUTED, 0.43)
    camera = draw_pose(frame, pose)
    factor = min(640 / camera.shape[1], 480 / camera.shape[0])
    camera = cv2.resize(camera, (round(camera.shape[1] * factor), round(camera.shape[0] * factor)))
    x, y = 24 + (640 - camera.shape[1]) // 2, 110 + (480 - camera.shape[0]) // 2
    canvas[y : y + camera.shape[0], x : x + camera.shape[1]] = camera
    _text(canvas, "Q / Esc quit    1 raw    2 EMA    3 moving average", (24, height - 14), MUTED, 0.44)
    _text(canvas, f"SIGNALS   |   {method}   |   {profile.evaluation.smoothing_seconds:g}s", (700, 32), TEXT, 0.54)
    _text(canvas, "RAW", (700, 56), RAW, 0.42)
    _text(canvas, "SMOOTHED", (760, 56), SMOOTHED, 0.42)
    _text(canvas, "ENTER", (880, 56), ENTRY, 0.42)
    _text(canvas, "CLEAR", (960, 56), CLEAR, 0.42)
    for index, feature in enumerate(features):
        top = 78 + index * 200
        cv2.rectangle(canvas, (695, top), (1380, top + 185), PANEL, -1)
        signal = result.signals.get(feature)
        raw = f"{signal.raw:.4f}" if signal and signal.raw is not None else "--"
        smooth = f"{signal.smoothed:.4f}" if signal and signal.smoothed is not None else "--"
        _text(canvas, feature, (712, top + 23), TEXT, 0.53)
        _text(canvas, f"raw {raw}   smooth {smooth}", (1030, top + 23), MUTED, 0.41)
        threshold_lines = raw_thresholds(profile, feature)
        data = [
            value
            for item in history
            if feature in item.signals
            for value in (item.signals[feature].raw, item.signals[feature].smoothed)
            if value is not None
        ]
        data.extend(value for value, _ in threshold_lines)
        if not data:
            _text(canvas, "Waiting for usable measurements", (765, top + 100), MUTED)
            continue
        low, high = min(data), max(data)
        padding = max((high - low) * 0.12, abs(high) * 0.005, 0.001)
        low, high = low - padding, high + padding
        left, right, bottom, chart_top = 770, 1360, top + 155, top + 40
        now = result.timestamp_ns

        x_pixel, y_pixel = _coordinates((left, right, bottom, chart_top), now, history_seconds, low, high)

        for fraction in (0, 0.5, 1):
            value = low + fraction * (high - low)
            yy = y_pixel(value)
            cv2.line(canvas, (left, yy), (right, yy), (63, 57, 51), 1)
            _text(canvas, f"{value:.3f}", (705, yy + 4), MUTED, 0.38)
        for value, label in threshold_lines:
            yy = y_pixel(value)
            for xx in range(left, right, 12):
                cv2.line(canvas, (xx, yy), (xx + 6, yy), ENTRY if label == "enter" else CLEAR, 1)
        for attribute, trace_color in (("raw", RAW), ("smoothed", SMOOTHED)):
            _trace(canvas, history, feature, attribute, (x_pixel, y_pixel), trace_color, max_gap_seconds)
        _text(canvas, f"-{history_seconds:g}s", (left, top + 176), MUTED, 0.38)
        _text(canvas, "now", (right - 24, top + 176), MUTED, 0.38)
    pending = [rule for rule in result.rules if rule.pending_target is not None]
    if pending:
        rule = pending[0]
        target = "deviation" if rule.pending_target else "reference"
        _text(
            canvas,
            f"Confirming {target}: {rule.pending_seconds:.1f}/{rule.required_seconds:.1f}s",
            (700, height - 20),
            color,
            0.47,
        )
    return canvas


class LivePreview:
    WINDOW_NAME = "Posture Tracker - Live evaluation"

    def __init__(self, profile, features, method="ema", history_seconds=15.0, max_gap_seconds=0.5):
        self.profile = profile
        self.features = features
        self.method = method
        self.history_seconds = history_seconds
        self.max_gap_seconds = max_gap_seconds
        self.history: deque[PostureResult] = deque()
        self.last_key = -1

    def __enter__(self):
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1400, max(620, 95 + 200 * len(self.features)))
        return self

    def show(self, frame, pose, result) -> bool:
        self.history.append(result)
        cutoff = result.timestamp_ns - round(self.history_seconds * 1e9)
        while self.history and self.history[0].timestamp_ns < cutoff:
            self.history.popleft()
        image = render_dashboard(
            frame,
            pose,
            result,
            self.profile,
            self.history,
            self.features,
            self.method,
            self.history_seconds,
            self.max_gap_seconds,
        )
        cv2.imshow(self.WINDOW_NAME, image)
        self.last_key = cv2.waitKey(1) & 0xFF
        return (
            self.last_key not in (ord("q"), ord("Q"), 27)
            and cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
        )

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ):
        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except cv2.error:
            pass
