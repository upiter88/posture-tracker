"""Read recorded features for fitting/replay without opening camera or model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from calibration_profile import CaptureIdentity
from features import FEATURE_VERSION, FeatureConfig, PostureFeatures


@dataclass(frozen=True)
class RecordedSample:
    sequence: int
    segment_id: int
    label: str
    features: PostureFeatures


@dataclass(frozen=True)
class CalibrationRecording:
    session_id: str
    extraction: FeatureConfig
    capture: CaptureIdentity
    samples: tuple[RecordedSample, ...]


def latest_recording(root: Path) -> Path:
    manifests = list(root.glob("*/session.json"))
    if not manifests:
        raise ValueError(f"No recordings in {root}; first run app.py --calibrate")
    return max(manifests, key=lambda path: json.loads(path.read_text())["started_utc"]).parent


def load_recording(path: Path) -> CalibrationRecording:
    metadata = json.loads((path / "session.json").read_text(encoding="utf-8"))
    if metadata.get("complete") is not True:
        raise ValueError(f"Recording {path.name} did not finish cleanly; inspect it before fitting")
    if metadata["feature_version"] != FEATURE_VERSION or metadata["schema_version"] != 1:
        raise ValueError(f"Recording {path.name} uses incompatible schema or feature definitions")
    config = dict(metadata["feature_config"])
    config["enabled"] = tuple(config["enabled"])
    extraction = FeatureConfig(**config)
    dataset = path / "samples.parquet"
    if not dataset.exists():
        dataset = path / "samples"
    files = sorted(dataset.glob("*.parquet")) if dataset.is_dir() else [dataset]
    if not files:
        raise ValueError(f"Recording {path.name} has no samples")
    rows = pq.read_table(files).to_pylist()
    if not rows or len(rows) != metadata["saved_count"] or len(rows) != metadata["frame_count"]:
        raise ValueError(f"Recording {path.name} has inconsistent frame counts")
    first = rows[0]
    capture = CaptureIdentity(
        metadata["camera_index"], first["image_width"], first["image_height"], metadata["model_sha256"]
    )
    samples = []
    previous_time, previous_sequence = -1, 0
    previous_segment = None
    previous_label = None
    seen_segments = set()
    for row in rows:
        timestamp, sequence, segment = row["capture_timestamp_ns"], row["frame_sequence"], row["label_segment_id"]
        if timestamp <= previous_time or sequence != previous_sequence + 1:
            raise ValueError(f"Recording {path.name} has unordered timestamps or missing/duplicate frames")
        if row["session_id"] != metadata["session_id"] or (row["image_width"], row["image_height"]) != (
            capture.image_width,
            capture.image_height,
        ):
            raise ValueError(f"Recording {path.name} changes session identity or image size")
        label = row["posture_label"]
        if label not in ("valid", "invalid", "unlabeled"):
            raise ValueError(f"Recording {path.name} has an unknown label: {label}")
        if segment != previous_segment:
            if segment in seen_segments:
                raise ValueError("Label segments must be contiguous")
            seen_segments.add(segment)
        elif label != previous_label:
            raise ValueError("A label segment cannot contain different posture labels")
        features = PostureFeatures(
            timestamp,
            {name: row[name] for name in extraction.columns},
            {name: row[f"{name}__reason"] for name in extraction.columns if row[f"{name}__reason"]},
        )
        samples.append(RecordedSample(sequence, segment, label, features))
        previous_time, previous_sequence, previous_segment, previous_label = timestamp, sequence, segment, label
    return CalibrationRecording(metadata["session_id"], extraction, capture, tuple(samples))
