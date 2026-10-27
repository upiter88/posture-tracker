"""Bounded, atomic Parquet parts for manually labeled calibration sessions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from types import TracebackType
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from capture import Frame
from features import FEATURE_VERSION, FEATURES, TEMPORAL_FEATURES, FeatureConfig, PostureFeatures
from pose import LANDMARK_NAMES, PoseObservation

SCHEMA_VERSION = 1
SAMPLES_FILENAME = "samples.parquet"
LABEL_EVENTS_FILENAME = "label_events.parquet"
LABEL_KEYS = {ord("2"): "valid", ord("1"): "invalid", ord("0"): "unlabeled"}


class RecordingError(RuntimeError):
    """A recording could not be created or safely written."""


class CalibrationRecorder:
    """No images retained. Each completed part survives later interruption."""

    def __init__(
        self,
        output: Path,
        config: FeatureConfig,
        camera_index: int,
        model_path: Path,
        batch_size: int = 300,
        flush_seconds: float = 5.0,
    ) -> None:
        if batch_size <= 0 or flush_seconds <= 0:
            raise ValueError("Recording batch size and flush interval must be positive")
        started = datetime.now(UTC)
        self.session_id = f"{started:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
        self.path = output / self.session_id
        self.config = config
        self.model_path = model_path
        self.batch_size = batch_size
        self.flush_seconds = flush_seconds
        self.label = "unlabeled"
        self.segment_id = 0
        self.frame_count = 0
        self.saved_count = 0
        self._origin: int | None = None
        self._last_timestamp: int | None = None
        self._last_flush: int | None = None
        self._part = 0
        self._rows: list[dict] = []
        self._events: list[dict] = []
        self._opened = False
        self._closed = False
        self._schema = self._sample_schema()
        self._event_schema = pa.schema(
            [
                ("session_id", pa.string()),
                ("label_segment_id", pa.int64()),
                ("posture_label", pa.string()),
                ("event_timestamp_ns", pa.int64()),
                ("first_frame_sequence", pa.int64()),
            ]
        )
        self._metadata = {
            "schema_version": SCHEMA_VERSION,
            "feature_version": FEATURE_VERSION,
            "session_id": self.session_id,
            "started_utc": started.isoformat(),
            "camera_index": camera_index,
            "model_path": str(model_path.resolve()),
            "landmark_names": LANDMARK_NAMES,
            "feature_config": asdict(config),
            "feature_definitions": {
                name: {
                    "units": FEATURES[name].units,
                    "required_landmarks": FEATURES[name].required_landmarks,
                    "documentation": FEATURES[name].documentation,
                }
                for name in config.enabled
            },
            "temporal_definitions": {
                name: {"units": definition.units, "documentation": definition.documentation}
                for name, definition in TEMPORAL_FEATURES.items()
                if config.temporal_enabled
            },
            "complete": False,
        }

    def _sample_schema(self) -> pa.Schema:
        fields = [
            ("session_id", pa.string()),
            ("frame_sequence", pa.int64()),
            ("capture_timestamp_ns", pa.int64()),
            ("elapsed_ns", pa.int64()),
            ("label_segment_id", pa.int64()),
            ("posture_label", pa.string()),
            ("pose_detected", pa.bool_()),
            ("image_width", pa.int64()),
            ("image_height", pa.int64()),
            ("measurement_valid", pa.bool_()),
        ]
        for name in LANDMARK_NAMES:
            fields.extend((f"{name}_{axis}", pa.float64()) for axis in ("x", "y", "z", "visibility", "presence"))
        for name in self.config.columns:
            fields.extend(((name, pa.float64()), (f"{name}__reason", pa.string())))
        return pa.schema(fields, metadata={b"schema_version": str(SCHEMA_VERSION).encode()})

    def __enter__(self) -> CalibrationRecorder:
        if self._opened or self._closed:
            raise RuntimeError("Recorder cannot be reopened")
        try:
            # The model hash identifies the exact binary used for this session.
            self._metadata["model_sha256"] = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
            self.path.mkdir(parents=True, exist_ok=False)
            (self.path / SAMPLES_FILENAME).mkdir()
            (self.path / LABEL_EVENTS_FILENAME).mkdir()
            self._write_metadata()
        except OSError as error:
            raise RecordingError(f"Cannot create recording at {self.path}: {error}") from error
        self._opened = True
        return self

    def set_label(self, key: int, timestamp_ns: int | None = None) -> None:
        """Called after recording the displayed frame; affects future frames only."""
        self._require_open()
        label = LABEL_KEYS.get(key)
        if label is None or label == self.label:
            return
        self.label = label
        self.segment_id += 1
        self._events.append(
            {
                "session_id": self.session_id,
                "label_segment_id": self.segment_id,
                "posture_label": label,
                # Use the same clock as CameraCapture so events align with frames.
                "event_timestamp_ns": perf_counter_ns() if timestamp_ns is None else timestamp_ns,
                "first_frame_sequence": None,
            }
        )

    def record(self, frame: Frame, pose: PoseObservation, features: PostureFeatures) -> None:
        self._require_open()
        if frame.timestamp_ns != pose.timestamp_ns or pose.timestamp_ns != features.timestamp_ns:
            raise ValueError("Recording inputs must refer to the same capture timestamp")
        if self._last_timestamp is not None and frame.timestamp_ns <= self._last_timestamp:
            raise ValueError("Recording timestamps must strictly increase")
        self._last_timestamp = frame.timestamp_ns
        if self._origin is None:
            self._origin = self._last_flush = frame.timestamp_ns
        for event in self._events:
            if event["label_segment_id"] == self.segment_id and event["first_frame_sequence"] is None:
                event["first_frame_sequence"] = frame.sequence
        row = {
            "session_id": self.session_id,
            "frame_sequence": frame.sequence,
            "capture_timestamp_ns": frame.timestamp_ns,
            "elapsed_ns": frame.timestamp_ns - self._origin,
            "label_segment_id": self.segment_id,
            "posture_label": self.label,
            "pose_detected": pose.detected,
            "image_width": pose.image_width,
            "image_height": pose.image_height,
            "measurement_valid": features.measurement_valid,
        }
        for name in LANDMARK_NAMES:
            point = pose.landmarks.get(name)
            for axis in ("x", "y", "z", "visibility", "presence"):
                row[f"{name}_{axis}"] = getattr(point, axis) if point is not None else None
        for name in self.config.columns:
            row[name] = features.values[name]
            row[f"{name}__reason"] = features.reasons.get(name)
        self._rows.append(row)
        self.frame_count += 1
        if len(self._rows) >= self.batch_size or (
            self._last_flush is not None and (frame.timestamp_ns - self._last_flush) / 1e9 >= self.flush_seconds
        ):
            self.flush()
            self._last_flush = frame.timestamp_ns

    def _require_open(self) -> None:
        if not self._opened or self._closed:
            raise RuntimeError("Recorder must be open")

    def _write_part(self, directory: str, rows: list[dict], schema: pa.Schema) -> None:
        if not rows:
            return
        path = self.path / directory / f"part-{self._part:06d}.parquet"
        temporary = path.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
        temporary.replace(path)

    def flush(self, *, include_pending_events: bool = False) -> None:
        self._require_open()
        try:
            self._write_part(SAMPLES_FILENAME, self._rows, self._schema)
            effective = [
                event for event in self._events if include_pending_events or event["first_frame_sequence"] is not None
            ]
            self._write_part(LABEL_EVENTS_FILENAME, effective, self._event_schema)
            self._events = [event for event in self._events if event not in effective]
        except (OSError, pa.ArrowException) as error:
            raise RecordingError(f"Cannot flush recording at {self.path}: {error}") from error
        self.saved_count += len(self._rows)
        self._rows.clear()
        self._part += 1

    def _write_metadata(self) -> None:
        temporary = self.path / "session.json.tmp"
        temporary.write_text(json.dumps(self._metadata, indent=2), encoding="utf-8")
        temporary.replace(self.path / "session.json")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.flush(include_pending_events=True)
            self._metadata.update(
                complete=exc_type is None or exc_type is KeyboardInterrupt,
                ended_utc=datetime.now(UTC).isoformat(),
                frame_count=self.frame_count,
                saved_count=self.saved_count,
            )
            self._write_metadata()
        except (OSError, pa.ArrowException) as error:
            raise RecordingError(f"Cannot finish recording at {self.path}: {error}") from error
        finally:
            self._closed = True
