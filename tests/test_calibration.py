"""Fit/validate/replay synthetic labeled holds, independent of MediaPipe."""

import json
from dataclasses import asdict, replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from calibration import FitSettings, RuleSpec, fit_profile
from calibration_data import CalibrationRecording, RecordedSample, load_recording
from calibration_profile import CaptureIdentity, EvaluationSettings
from features import FEATURE_VERSION, FeatureConfig, PostureFeatures
from replay import replay_recording


def recording(session="test"):
    samples = []
    sequence = 0
    for segment, (label, value) in enumerate([("valid", 0.7), ("invalid", 0.54), ("valid", 0.71), ("invalid", 0.55)]):
        for index in range(81):
            sequence += 1
            values = {"head_offset_y": value + 0.002 * np.sin(index)}
            samples.append(
                RecordedSample(sequence, segment, label, PostureFeatures(sequence * 100_000_000, values, {}))
            )
    return CalibrationRecording(
        session,
        FeatureConfig(enabled=("head_offset_y",), temporal_enabled=False),
        CaptureIdentity(0, 640, 480, "a" * 64),
        tuple(samples),
    )


def fit(data):
    return fit_profile(data, (RuleSpec("head_offset_y", "head"),), FitSettings(), EvaluationSettings(0.2, 0.5, 0.3))


def test_fitting_and_replay_are_separate_and_use_real_references():
    source = recording()
    profile, report = fit([source])
    assert report["rules"][0]["balanced_accuracy"] == 1
    assert report["rules"][0]["validation"] == "leave_hold_out"
    rule = profile.rules[0]
    assert rule.reference.centre == pytest.approx(0.705, abs=0.002)
    assert rule.reference.scale >= 0.005
    raw_boundary = rule.reference.centre - rule.enter_threshold * rule.reference.scale
    assert 0.56 < raw_boundary < 0.69
    results, replay = replay_recording(source, profile)
    assert results[0].status == "UNKNOWN"
    assert replay.missed_invalid_holds == 0
    assert len(replay.detection_delays_seconds) == 2
    # Synthetic source has no unlabeled gaps: temporal latency is visible.
    assert replay.false_warning_frames > 0
    assert all(delay >= 0.5 for delay in replay.detection_delays_seconds)


def test_session_holdout_and_incompatible_sources():
    source = recording()
    _, report = fit([source, recording("another")])
    assert report["rules"][0]["validation"] == "leave_session_out"
    with pytest.raises(ValueError, match="unique"):
        fit([source, source])
    with pytest.raises(ValueError, match="share capture"):
        fit([source, replace(recording("another"), capture=replace(source.capture, image_width=1280))])


def test_insufficient_missing_and_uninformative_labels_are_rejected():
    source = recording()
    with pytest.raises(ValueError, match="usable holds"):
        fit([replace(source, samples=source.samples[:81])])
    identical = tuple(
        replace(sample, features=replace(sample.features, values={"head_offset_y": 0.7})) for sample in source.samples
    )
    with pytest.raises(ValueError, match="below"):
        fit([replace(source, samples=identical)])
    absent = tuple(replace(sample, features=replace(sample.features, values={})) for sample in source.samples)
    with pytest.raises(ValueError, match="usable holds"):
        fit([replace(source, samples=absent)])
    sparse = tuple(
        replace(sample, features=replace(sample.features, timestamp_ns=sample.features.timestamp_ns * 10))
        for sample in source.samples
    )
    with pytest.raises(ValueError, match="usable holds"):
        fit([replace(source, samples=sparse)])


def test_recording_loader_validates_counts_versions_and_order(tmp_path):
    source = recording()
    metadata = {
        "schema_version": 1,
        "feature_version": FEATURE_VERSION,
        "session_id": source.session_id,
        "complete": True,
        "saved_count": len(source.samples),
        "frame_count": len(source.samples),
        "feature_config": asdict(source.extraction),
        "camera_index": 0,
        "model_sha256": "a" * 64,
    }
    path = tmp_path / "session.json"
    path.write_text(json.dumps(metadata))
    dataset = tmp_path / "samples.parquet"
    dataset.mkdir()
    rows = [
        {
            "session_id": source.session_id,
            "frame_sequence": sample.sequence,
            "capture_timestamp_ns": sample.features.timestamp_ns,
            "label_segment_id": sample.segment_id,
            "posture_label": sample.label,
            "image_width": 640,
            "image_height": 480,
            "head_offset_y": sample.features.values["head_offset_y"],
            "head_offset_y__reason": None,
        }
        for sample in source.samples
    ]
    pq.write_table(pa.Table.from_pylist(rows), dataset / "part-000000.parquet")
    loaded = load_recording(tmp_path)
    assert loaded == source
    metadata["saved_count"] -= 1
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="frame counts"):
        load_recording(tmp_path)
    metadata["saved_count"] += 1
    metadata["complete"] = False
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="finish cleanly"):
        load_recording(tmp_path)
