"""Pose contract and lifecycle checks without a model file or physical camera."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from capture import Frame
from pose import PoseError, PoseLandmarksTracker


def make_frame(timestamp_ns=1_000_000_000):
    return Frame(np.full((20, 30, 3), (10, 20, 30), dtype=np.uint8), 1, timestamp_ns)


def result_with_pose():
    return SimpleNamespace(
        pose_landmarks=[
            [SimpleNamespace(x=index / 100, y=0.4, z=-0.1, visibility=0.8, presence=0.9) for index in range(33)]
        ]
    )


@pytest.fixture
def backend(monkeypatch, tmp_path):
    model = tmp_path / "model.task"
    model.touch()
    landmarker = Mock()
    landmarker.detect_for_video.return_value = result_with_pose()
    factory = Mock(return_value=landmarker)
    monkeypatch.setattr("pose.PoseLandmarker.create_from_options", factory)
    return PoseLandmarksTracker(model), landmarker, factory


def test_maps_landmarks_and_converts_bgr_without_mutating_frame(backend):
    estimator, landmarker, factory = backend
    frame = make_frame()
    original = frame.image.copy()
    with estimator:
        observation = estimator.estimate(frame)
    image, timestamp_ms = landmarker.detect_for_video.call_args.args
    np.testing.assert_array_equal(image.numpy_view()[0, 0], [30, 20, 10])
    np.testing.assert_array_equal(frame.image, original)
    assert timestamp_ms == 0
    assert observation.timestamp_ns == frame.timestamp_ns
    assert (observation.image_width, observation.image_height) == (30, 20)
    assert observation.detected
    assert set(observation.landmarks) == {
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
    }
    assert observation.landmarks["left_shoulder"].x == 0.11
    assert observation.landmarks["mouth_left"].x == 0.09
    assert observation.landmarks["mouth_right"].x == 0.10
    assert observation.landmarks["right_shoulder"].x == 0.12
    assert observation.landmarks["nose"].visibility == 0.8
    assert observation.landmarks["nose"].presence == 0.9
    assert factory.call_args.args[0].num_poses == 1
    landmarker.close.assert_called_once()


def test_missing_detection_does_not_reuse_previous_pose(backend):
    estimator, landmarker, _ = backend
    with estimator:
        assert estimator.estimate(make_frame()).detected
        landmarker.detect_for_video.return_value = SimpleNamespace(pose_landmarks=[])
        observation = estimator.estimate(make_frame(1_100_000_000))
    assert not observation.detected
    assert not observation.landmarks
    assert observation.timestamp_ns == 1_100_000_000


def test_timestamp_rounding_and_reopen(backend):
    estimator, landmarker, _ = backend
    with estimator:
        for timestamp_ns in [1_000_000_000, 1_000_000_001, 1_050_000_000]:
            assert estimator.estimate(make_frame(timestamp_ns)).timestamp_ns == timestamp_ns
    with estimator:
        estimator.estimate(make_frame())
    assert [call.args[1] for call in landmarker.detect_for_video.call_args_list] == [0, 1, 50, 0]
    estimator.close()
    assert landmarker.close.call_count == 2


def test_capture_clock_ties_survive_calibration_pipeline(backend, monkeypatch, tmp_path):
    import pyarrow.parquet as pq

    from capture import CameraCapture
    from features import FeatureConfig, FeatureExtractor
    from recording import CalibrationRecorder

    estimator, landmarker, _ = backend
    device = Mock()
    device.read.return_value = (True, make_frame().image)
    monkeypatch.setattr("capture.cv2.VideoCapture", lambda: device)
    times = iter([1_000_000_000, 1_000_000_000, 1_050_000_000])
    monkeypatch.setattr("capture.perf_counter_ns", lambda: next(times))
    monkeypatch.setattr("recording.perf_counter_ns", lambda: 1_025_000_000)
    config = FeatureConfig()
    extractor = FeatureExtractor(config)
    recorder = CalibrationRecorder(tmp_path / "sessions", config, 0, estimator.model_path)
    with estimator, CameraCapture() as camera, recorder:
        for index in range(3):
            frame = camera.read()
            pose = estimator.estimate(frame)
            recorder.record(frame, pose, extractor.extract(pose))
            if index == 1:
                recorder.set_label(ord("2"))

    rows = pq.read_table(recorder.path / "samples.parquet").to_pylist()
    assert [row["capture_timestamp_ns"] for row in rows] == [1_000_000_000, 1_000_000_001, 1_050_000_000]
    assert [row["posture_label"] for row in rows] == ["unlabeled", "unlabeled", "valid"]
    assert [call.args[1] for call in landmarker.detect_for_video.call_args_list] == [0, 1, 50]
    events = pq.read_table(recorder.path / "label_events.parquet").to_pylist()
    assert events[0]["event_timestamp_ns"] == 1_025_000_000
    assert events[0]["first_frame_sequence"] == 3


@pytest.mark.parametrize("timestamp_ns", [1_000_000_000, 999_999_999])
def test_rejects_duplicate_or_backwards_capture_time(backend, timestamp_ns):
    estimator, landmarker, _ = backend
    with estimator:
        estimator.estimate(make_frame())
        with pytest.raises(ValueError, match="strictly increasing"):
            estimator.estimate(make_frame(timestamp_ns))
    assert landmarker.detect_for_video.call_count == 1


def test_low_confidence_is_preserved_for_feature_stage(backend):
    estimator, landmarker, _ = backend
    landmarker.detect_for_video.return_value.pose_landmarks[0][11].visibility = 0.01
    with estimator:
        observation = estimator.estimate(make_frame())
    assert observation.detected
    assert observation.landmarks["left_shoulder"].visibility == 0.01


@pytest.mark.parametrize("failure", [RuntimeError("backend failed"), ValueError("bad image")])
def test_inference_error_is_not_a_missing_detection_and_closes_model(backend, failure):
    estimator, landmarker, _ = backend
    landmarker.detect_for_video.side_effect = failure
    with pytest.raises(PoseError, match="frame 1"):
        with estimator:
            estimator.estimate(make_frame())
    landmarker.close.assert_called_once()


def test_rejects_nonfinite_model_output(backend):
    estimator, landmarker, _ = backend
    landmarker.detect_for_video.return_value.pose_landmarks[0][0].x = float("nan")
    with pytest.raises(PoseError, match="Invalid coordinates"):
        with estimator:
            estimator.estimate(make_frame())


def test_missing_model_has_setup_instructions(tmp_path):
    with pytest.raises(PoseError, match="models/README.md"):
        PoseLandmarksTracker(tmp_path / "missing.task").open()


def test_initialization_error_is_actionable(backend):
    estimator, _, factory = backend
    factory.side_effect = RuntimeError("invalid model")
    with pytest.raises(PoseError, match="Cannot load pose model"):
        estimator.open()


def test_interrupt_closes_model(backend):
    estimator, landmarker, _ = backend
    with pytest.raises(KeyboardInterrupt):
        with estimator:
            raise KeyboardInterrupt
    landmarker.close.assert_called_once()


def test_lifecycle_guards(backend):
    estimator, _, _ = backend
    with pytest.raises(RuntimeError, match="Open the pose estimator"):
        estimator.estimate(make_frame())
    with estimator:
        with pytest.raises(RuntimeError, match="already open"):
            estimator.open()
