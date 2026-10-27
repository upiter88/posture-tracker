"""Parquet round trips and recording lifecycle without camera hardware."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from capture import Frame
from features import FeatureConfig, FeatureExtractor
from pose import PoseLandmark, PoseObservation
from recording import CalibrationRecorder, RecordingError


@pytest.fixture
def recorder(tmp_path):
    model = tmp_path / "model.task"
    model.write_bytes(b"model identity")
    return CalibrationRecorder(tmp_path / "sessions", FeatureConfig(), 0, model, batch_size=2)


def record_frame(recorder, extractor, sequence):
    timestamp = sequence * 100_000_000
    frame = Frame(np.zeros((80, 100, 3), dtype=np.uint8), sequence, timestamp)
    # A partial pose is kept, even though most features are unavailable.
    landmarks = {"nose": PoseLandmark(0.5, 0.4, -0.1, 0.2, 0.8)} if sequence == 1 else {}
    pose = PoseObservation(timestamp, 100, 80, landmarks)
    recorder.record(frame, pose, extractor.extract(pose))


def test_labels_apply_to_next_frame_and_raw_failures_are_preserved(recorder):
    extractor = FeatureExtractor()
    with recorder:
        record_frame(recorder, extractor, 1)
        recorder.set_label(ord("2"), 150_000_000)
        recorder.set_label(ord("2"), 160_000_000)  # no duplicate segment
        record_frame(recorder, extractor, 2)
        assert recorder.saved_count == 2
        assert len(list((recorder.path / "samples.parquet").glob("*.parquet"))) == 1
        recorder.set_label(ord("1"), 250_000_000)
        record_frame(recorder, extractor, 3)
        recorder.set_label(ord("0"), 350_000_000)  # no following frame
    rows = pq.read_table(recorder.path / "samples.parquet").to_pylist()
    assert [row["posture_label"] for row in rows] == ["unlabeled", "valid", "invalid"]
    assert [row["label_segment_id"] for row in rows] == [0, 1, 2]
    assert [row["elapsed_ns"] for row in rows] == [0, 100_000_000, 200_000_000]
    assert rows[0]["nose_z"] == -0.1
    assert rows[0]["nose_visibility"] == 0.2
    assert rows[1]["nose_x"] is None
    assert not rows[1]["pose_detected"]
    assert rows[0]["shoulder_width"] is None
    assert not any(row["measurement_valid"] for row in rows)
    events = pq.read_table(recorder.path / "label_events.parquet").to_pylist()
    assert [row["first_frame_sequence"] for row in events] == [2, 3, None]
    metadata = json.loads((recorder.path / "session.json").read_text())
    assert metadata["complete"]
    assert metadata["saved_count"] == 3
    assert len(metadata["model_sha256"]) == 64
    assert "Definition:" in metadata["feature_definitions"]["shoulder_width"]["documentation"]
    assert not list(recorder.path.rglob("*.tmp"))


def test_exception_flushes_tail_and_marks_session_incomplete(recorder):
    with pytest.raises(RuntimeError, match="inference failure"), recorder:
        record_frame(recorder, FeatureExtractor(), 1)
        raise RuntimeError("inference failure")
    assert pq.read_table(recorder.path / "samples.parquet").num_rows == 1
    assert not json.loads((recorder.path / "session.json").read_text())["complete"]


def test_write_failure_is_reported_and_previous_part_survives(recorder, monkeypatch):
    extractor = FeatureExtractor()
    with pytest.raises(RecordingError, match="Cannot flush"), recorder:
        record_frame(recorder, extractor, 1)
        record_frame(recorder, extractor, 2)

        def fail_write(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("recording.pq.write_table", fail_write)
        record_frame(recorder, extractor, 3)
    assert pq.read_table(recorder.path / "samples.parquet").num_rows == 2
    assert not json.loads((recorder.path / "session.json").read_text())["complete"]


def test_keyboard_interrupt_flushes_and_marks_completed(recorder):
    with pytest.raises(KeyboardInterrupt), recorder:
        record_frame(recorder, FeatureExtractor(), 1)
        raise KeyboardInterrupt
    assert recorder.saved_count == 1
    assert json.loads((recorder.path / "session.json").read_text())["complete"]
