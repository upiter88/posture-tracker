"""Posture features with each formal definition beside its calculation.

Notation: P=(x*image_width, y*image_height) is a landmark in pixels.
Image x increases rightward; y increases downward. Left/right are anatomical.
S is the shoulder midpoint; W is Euclidean shoulder width in pixels.
The decorators build the registries directly: no separate list of names.
"""

from __future__ import annotations

import math
import tomllib
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from inspect import cleandoc
from pathlib import Path
from typing import Protocol

from pose import PoseObservation

FEATURE_VERSION = 1
Point = tuple[float, float]
History = Sequence[tuple[int, float]]


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    units: str
    required_landmarks: tuple[str, ...]
    documentation: str
    calculate: Callable[[_Geometry], float]


FEATURES: dict[str, FeatureDefinition] = {}


class FeatureCalculation(Protocol):
    __name__: str
    __doc__: str | None

    def __call__(self, geometry: _Geometry, /) -> float: ...


def feature(*, units: str, landmarks: tuple[str, ...]):
    """Register a calculation using its function name and explanatory docstring."""

    def register(calculate: FeatureCalculation):
        FEATURES[calculate.__name__] = FeatureDefinition(
            calculate.__name__, units, landmarks, calculate.__doc__ or "", calculate
        )
        return calculate

    return register


class MeasurementUnavailable(ValueError):
    """A feature lacks reliable landmarks or usable geometry."""


class _Geometry:
    """Shared coordinate conversion and quality checks; no feature dispatch."""

    def __init__(self, pose: PoseObservation, config: FeatureConfig):
        self.pose = pose
        self.config = config

    def point(self, name: str) -> Point:
        point = self.pose.landmarks.get(name)
        if point is None:
            raise MeasurementUnavailable(f"missing:{name}")
        if not all(math.isfinite(v) for v in (point.x, point.y, point.visibility, point.presence)):
            raise MeasurementUnavailable(f"nonfinite:{name}")
        if point.visibility < self.config.min_visibility or point.presence < self.config.min_presence:
            raise MeasurementUnavailable(f"low_confidence:{name}")
        if not (0 <= point.x < 1 and 0 <= point.y < 1):
            raise MeasurementUnavailable(f"out_of_frame:{name}")
        if self.pose.image_width <= 0 or self.pose.image_height <= 0:
            raise MeasurementUnavailable("invalid_image_dimensions")
        return point.x * self.pose.image_width, point.y * self.pose.image_height

    def midpoint(self, first: str, second: str) -> Point:
        a, b = self.point(first), self.point(second)
        return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2

    def shoulder_midpoint(self) -> Point:
        return self.midpoint("left_shoulder", "right_shoulder")

    def shoulder_width_px(self) -> float:
        width = math.dist(self.point("left_shoulder"), self.point("right_shoulder"))
        if width < self.config.min_shoulder_width_px:
            raise MeasurementUnavailable("shoulder_width_too_small")
        return width

    def head(self) -> Point:
        if self.config.head_reference == "nose":
            return self.point("nose")
        return self.midpoint("left_eye", "right_eye")


@feature(units="degrees", landmarks=("left_shoulder", "right_shoulder"))
def shoulder_angle_deg(geometry: _Geometry) -> float:
    """Shoulder-line angle relative to image horizontal.

    Definition: degrees(atan2(B_y - A_y, B_x - A_x)), where A is the
    image-left shoulder and B the image-right shoulder (sort by x then y).
    Range: [-90, 90]. Positive means downward slope to image right; level is 0.
    This is image tilt, not anatomical roll. Requires a usable shoulder width.
    """
    geometry.shoulder_width_px()
    image_left, image_right = sorted((geometry.point("left_shoulder"), geometry.point("right_shoulder")))
    return math.degrees(math.atan2(image_right[1] - image_left[1], image_right[0] - image_left[0]))


@feature(units="shoulder widths", landmarks=("left_shoulder", "right_shoulder", "head_reference"))
def head_offset_x(geometry: _Geometry) -> float:
    """Horizontal head position relative to the shoulder midpoint.

    Definition: (H_x - S_x) / W.
    H is the eye midpoint by default, or the nose via head_reference config.
    Positive means image right of S; zero means centered. Normalization by W
    reduces sensitivity to camera distance.
    """
    head = geometry.head()
    shoulders = geometry.shoulder_midpoint()
    return (head[0] - shoulders[0]) / geometry.shoulder_width_px()


@feature(units="shoulder widths", landmarks=("left_shoulder", "right_shoulder", "head_reference"))
def head_offset_y(geometry: _Geometry) -> float:
    """Vertical head position above the shoulder midpoint.

    Definition: (S_y - H_y) / W.
    H is the eye midpoint by default, or the nose via head_reference config.
    Positive means above S. A decrease means less vertical separation;
    it does not by itself establish forward slouching.
    """
    head = geometry.head()
    shoulders = geometry.shoulder_midpoint()
    return (shoulders[1] - head[1]) / geometry.shoulder_width_px()


@feature(units="shoulder widths", landmarks=("mouth_left", "mouth_right", "left_shoulder", "right_shoulder"))
def mouth_shoulder_vertical(geometry: _Geometry) -> float:
    """Vertical separation of mouth and shoulder lines at their midpoints.

    Definition: (S_y - M_y) / W, where M=(mouth_left + mouth_right)/2.
    Positive means the mouth midpoint is above the shoulder midpoint.
    Distance between lines means signed midpoint y separation here,
    not the shortest distance between infinite lines.
    """
    mouth = geometry.midpoint("mouth_left", "mouth_right")
    shoulders = geometry.shoulder_midpoint()
    return (shoulders[1] - mouth[1]) / geometry.shoulder_width_px()


@feature(
    units="shoulder widths",
    landmarks=("left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder"),
)
def eye_ear_vertical(geometry: _Geometry) -> float:
    """Vertical separation of eye and ear lines at their midpoints.

    Definition: (Ears_y - Eyes_y) / W, with each pair averaged in pixels.
    Positive means the eye midpoint is above the ear midpoint.
    Candidate head-orientation signal: anatomy, viewpoint, and tracking noise
    also affect it. Compare with a personal reference.
    """
    eyes = geometry.midpoint("left_eye", "right_eye")
    ears = geometry.midpoint("left_ear", "right_ear")
    return (ears[1] - eyes[1]) / geometry.shoulder_width_px()


@feature(units="image widths", landmarks=("left_shoulder", "right_shoulder"))
def shoulder_width(geometry: _Geometry) -> float:
    """Apparent Euclidean shoulder width relative to image width.

    Definition: W / image_width.
    Larger values usually mean greater apparent scale; body rotation also
    changes it. This is not meters. W includes both x and y displacement;
    tilted shoulders retain their Euclidean width. May exceed 1 in tall images.
    """
    return geometry.shoulder_width_px() / geometry.pose.image_width


@feature(units="dimensionless ratio", landmarks=("nose", "left_ear", "right_ear"))
def facial_asymmetry(geometry: _Geometry) -> float:
    """Imbalance of anatomical left/right nose-to-ear distances.

    Definition: (d_left - d_right) / (d_left + d_right), where
    d_left=||nose-left_ear||_2 and d_right=||nose-right_ear||_2 in pixels.
    Range: [-1, 1]. Positive means the anatomical left distance is larger;
    zero means equal projected distances. Candidate head-turn signal,
    not a calibrated yaw angle. Does not require shoulders.
    """
    nose = geometry.point("nose")
    left_distance = math.dist(nose, geometry.point("left_ear"))
    right_distance = math.dist(nose, geometry.point("right_ear"))
    if left_distance + right_distance < 1e-6:
        raise MeasurementUnavailable("facial_scale_too_small")
    return (left_distance - right_distance) / (left_distance + right_distance)


@dataclass(frozen=True)
class TemporalFeatureDefinition:
    suffix: str
    units: str
    documentation: str
    calculate: Callable[[History], float]


TEMPORAL_FEATURES: dict[str, TemporalFeatureDefinition] = {}


def temporal_feature(*, suffix: str, units: str):
    """Register an operation applied independently to each enabled base feature."""

    def register(calculate: Callable[[History], float]):
        TEMPORAL_FEATURES[suffix] = TemporalFeatureDefinition(suffix, units, calculate.__doc__ or "", calculate)
        return calculate

    return register


@temporal_feature(suffix="std", units="same as the base feature")
def short_window_variation(history: History) -> float:
    """Population standard deviation of recent valid observations.

    Definition: sqrt(sum((f_i - mean(f))**2) / n) for samples in the inclusive
    interval [t-window_seconds, t]. Requires at least two valid samples.
    Samples are equally weighted, not weighted by elapsed time.
    Larger means greater recent variation, not necessarily bad posture.
    """
    mean = math.fsum(value for _, value in history) / len(history)
    return math.sqrt(math.fsum((value - mean) ** 2 for _, value in history) / len(history))


@temporal_feature(suffix="velocity", units="base feature units per second")
def instantaneous_velocity(history: History) -> float:
    """Signed backward finite-difference velocity using the latest two samples.

    Definition: (f(t) - f(previous)) / ((t_ns - previous_ns) / 1e9).
    Positive means the base feature increased. Instantaneous, not a fitted
    trend across the window; sensitive to frame noise. Both samples must be
    inside the window. No assumed FPS or interpolation across tracking gaps.
    """
    previous_time, previous_value = history[-2]
    timestamp, value = history[-1]
    return (value - previous_value) / ((timestamp - previous_time) / 1e9)


def feature_reference_markdown() -> str:
    """Render the reference from the same definitions used for calculation."""
    lines = [
        "# Feature reference",
        "",
        "Generated from the definitions beside each calculation in `src/features.py`.",
        "",
        "Coordinates use pixels: P=(x*image_width, y*image_height). Image x increases rightward; "
        "y increases downward. Left/right landmark names are anatomical. S is the shoulder midpoint; "
        "W is Euclidean shoulder width in pixels. Model-relative z is saved but not used by these features.",
        "",
        "Edit `features.toml` to enable features, choose the head reference, change confidence limits, "
        "or adjust the motion window. Calibration mode loads this file automatically. "
        "Each required point must be finite, in-frame, and meet both confidence thresholds. "
        "Distances divided by W require W >= min_shoulder_width_px. "
        "An unavailable feature is null with a reason; raw landmarks are preserved. "
        "measurement_valid means all enabled base features are usable; it does not judge posture "
        "or require temporal warmup to finish.",
        "",
    ]
    for definition in FEATURES.values():
        lines.extend(
            [
                f"## {definition.name}",
                "",
                f"Units: {definition.units}.",
                "",
                f"Required landmarks: {', '.join(definition.required_landmarks)}. "
                "`head_reference` means the configured nose or both eyes.",
                "",
                cleandoc(definition.documentation),
                "",
            ]
        )
    lines.extend(
        [
            "## Movement and stability",
            "",
            "Each enabled base feature gets the temporal columns below. History resets independently "
            "when its base feature becomes unavailable or the gap exceeds max_gap_seconds. "
            "Fewer than two samples in the window yields null with temporal_warmup. "
            "A stationary stream yields zero after warmup. The window may be partially filled. "
            "Labels do not reset history: use unlabeled transition segments when changing posture.",
            "",
        ]
    )
    for definition in TEMPORAL_FEATURES.values():
        lines.extend(
            [
                f"### <base_feature>__{definition.suffix}",
                "",
                f"Units: {definition.units}.",
                "",
                cleandoc(definition.documentation),
                "",
            ]
        )
    lines.extend(
        [
            "## Adding a feature",
            "",
            "Add a named function decorated with @feature next to its docstring containing the formal "
            "definition, description, signs, units, and limitations. Declare required landmarks in the "
            "decorator and use geometry helpers for validation. The registry, output schema, and session "
            "metadata discover it automatically. Add its name to features.toml if enabled is explicitly "
            "listed. Increment FEATURE_VERSION when definitions change, and test known geometry. "
            "Regenerate this document with the command in README.md.",
            "",
        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class FeatureConfig:
    """TOML options; default names come from the definitions above."""

    enabled: tuple[str, ...] = tuple(FEATURES)
    min_visibility: float = 0.5
    min_presence: float = 0.5
    min_shoulder_width_px: float = 5.0
    head_reference: str = "eyes_midpoint"
    window_seconds: float = 1.0
    max_gap_seconds: float = 0.5
    temporal_enabled: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.enabled, tuple)
            or not self.enabled
            or not all(isinstance(name, str) for name in self.enabled)
            or len(set(self.enabled)) != len(self.enabled)
            or set(self.enabled) - FEATURES.keys()
        ):
            raise ValueError(f"enabled must contain unique names from {tuple(FEATURES)}")
        for name in ("min_visibility", "min_presence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and between 0 and 1")
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("min_shoulder_width_px", "window_seconds", "max_gap_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.head_reference not in ("nose", "eyes_midpoint"):
            raise ValueError("head_reference must be nose or eyes_midpoint")
        if not isinstance(self.temporal_enabled, bool):
            raise ValueError("temporal_enabled must be a boolean")

    @classmethod
    def load(cls, path: Path) -> FeatureConfig:
        with path.open("rb") as source:
            document = tomllib.load(source)
        if set(document) != {"features"} or not isinstance(document["features"], dict):
            raise ValueError("Configuration must contain only a [features] table")
        settings = document["features"]
        if "enabled" in settings:
            names = settings["enabled"]
            if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                raise ValueError("enabled must be an array of feature names")
            settings["enabled"] = tuple(names)
        try:
            return cls(**settings)
        except TypeError as error:
            raise ValueError(f"Invalid feature configuration: {error}") from error

    @property
    def columns(self) -> tuple[str, ...]:
        columns = list(self.enabled)
        if self.temporal_enabled:
            for name in self.enabled:
                columns.extend(f"{name}__{suffix}" for suffix in TEMPORAL_FEATURES)
        return tuple(columns)


@dataclass(frozen=True)
class PostureFeatures:
    timestamp_ns: int
    values: dict[str, float | None]
    reasons: dict[str, str]

    @property
    def measurement_valid(self) -> bool:
        """All enabled base measurements usable; temporal warmup is separate."""
        return all(value is not None for name, value in self.values.items() if "__" not in name)


@dataclass
class FeatureExtractor:
    config: FeatureConfig = field(default_factory=FeatureConfig)
    _history: dict[str, deque[tuple[int, float]]] = field(default_factory=dict, init=False)
    _last_timestamp_ns: int | None = field(default=None, init=False)

    def extract(self, pose: PoseObservation) -> PostureFeatures:
        if self._last_timestamp_ns is not None and pose.timestamp_ns <= self._last_timestamp_ns:
            raise ValueError("Feature timestamps must strictly increase")
        self._last_timestamp_ns = pose.timestamp_ns
        geometry = _Geometry(pose, self.config)
        values: dict[str, float | None] = {}
        reasons: dict[str, str] = {}
        for name in self.config.enabled:
            try:
                values[name] = FEATURES[name].calculate(geometry)
            except MeasurementUnavailable as error:
                values[name] = None
                reasons[name] = str(error)
            if self.config.temporal_enabled:
                self._motion(name, pose.timestamp_ns, values, reasons)
        return PostureFeatures(pose.timestamp_ns, values, reasons)

    def _motion(self, name: str, timestamp: int, values: dict[str, float | None], reasons: dict[str, str]) -> None:
        history = self._history.setdefault(name, deque())
        value = values[name]
        reason = None
        if value is None:
            history.clear()
            reason = "base_measurement_unavailable"
        else:
            if history and (timestamp - history[-1][0]) / 1e9 > self.config.max_gap_seconds:
                history.clear()
            cutoff = timestamp - self.config.window_seconds * 1e9
            while history and history[0][0] < cutoff:
                history.popleft()
            history.append((timestamp, value))
            if len(history) < 2:
                reason = "temporal_warmup"
        for suffix, definition in TEMPORAL_FEATURES.items():
            column = f"{name}__{suffix}"
            if reason is not None:
                values[column] = None
                reasons[column] = reason
            else:
                values[column] = definition.calculate(list(history))
