"""Synchronous MediaPipe adapter; downstream stages receive ordinary dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from capture import Frame


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"

# MediaPipe indices stay inside this adapter. Names use the person's anatomical sides.
_LANDMARK_INDICES = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
}


@dataclass(frozen=True)
class PoseLandmark:
    """Image coordinates: x rightward / width, y downward / height.

    z is model-relative depth, scaled approximately like x, with smaller values
    closer to the camera. It is not metric distance. Coordinates may lie outside
    the image; preserve them rather than clipping them. Confidence is in [0, 1].
    """

    x: float
    y: float
    z: float
    visibility: float
    presence: float


@dataclass(frozen=True)
class PoseObservation:
    """Landmarks from one frame, retaining its original capture timestamp.

    detected means the model returned a pose, not that all landmarks are usable.
    The feature stage must check confidence and geometry for each measurement.
    Multiply x by image_width and y by image_height before calculating angles.
    Missing detections have an empty landmark mapping; previous poses are not reused.
    """

    timestamp_ns: int
    image_width: int
    image_height: int
    landmarks: Mapping[str, PoseLandmark]

    @property
    def detected(self) -> bool:
        return bool(self.landmarks)


class PoseError(RuntimeError):
    """The model could not be loaded or pose inference failed."""


class PoseEstimator:
    """Own a single-person landmarker, tracking sequential frames in VIDEO mode."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._landmarker = None
        self._origin_ns = None
        self._last_timestamp_ns = None
        self._last_timestamp_ms = -1

    def open(self) -> PoseEstimator:
        if self._landmarker is not None:
            raise RuntimeError("Pose estimator is already open.")
        if not self.model_path.is_file():
            raise PoseError(
                f"Pose model not found: {self.model_path}. "
                "Download the Lite model using the instructions in models/README.md."
            )
        try:
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        except (OSError, RuntimeError, ValueError) as error:
            raise PoseError(f"Cannot load pose model {self.model_path}: {error}") from error
        self._origin_ns = None
        self._last_timestamp_ns = None
        self._last_timestamp_ms = -1
        return self

    def estimate(self, frame: Frame) -> PoseObservation:
        if self._landmarker is None:
            raise RuntimeError("Open the pose estimator before estimating poses.")
        if self._last_timestamp_ns is not None and frame.timestamp_ns <= self._last_timestamp_ns:
            raise ValueError("Pose frames must have strictly increasing capture timestamps.")
        if self._origin_ns is None:
            self._origin_ns = frame.timestamp_ns
        # Relative time avoids huge clock values. Rounding collisions advance the
        # model clock by 1 ms; observations always retain the real capture time.
        timestamp_ms = max(
            (frame.timestamp_ns - self._origin_ns) // 1_000_000,
            self._last_timestamp_ms + 1,
        )
        try:
            rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(image, timestamp_ms)
            landmarks = {}
            if result.pose_landmarks:
                points = result.pose_landmarks[0]
                for name, index in _LANDMARK_INDICES.items():
                    point = points[index]
                    values = (point.x, point.y, point.z, point.visibility, point.presence)
                    if not all(value is not None and math.isfinite(value) for value in values):
                        raise ValueError(f"Invalid coordinates or confidence for {name}.")
                    if not all(0 <= value <= 1 for value in values[3:]):
                        raise ValueError(f"Invalid confidence for {name}.")
                    landmarks[name] = PoseLandmark(*values)
        except (cv2.error, OSError, RuntimeError, ValueError, TypeError, IndexError) as error:
            raise PoseError(f"Pose inference failed for frame {frame.sequence}: {error}") from error
        self._last_timestamp_ns = frame.timestamp_ns
        self._last_timestamp_ms = timestamp_ms
        height, width = frame.image.shape[:2]
        return PoseObservation(frame.timestamp_ns, width, height, MappingProxyType(landmarks))

    def close(self) -> None:
        if self._landmarker is not None:
            landmarker = self._landmarker
            self._landmarker = None
            try:
                landmarker.close()
            except (RuntimeError, ValueError) as error:
                raise PoseError(f"Cannot close pose model: {error}") from error

    def __enter__(self) -> PoseEstimator:
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
