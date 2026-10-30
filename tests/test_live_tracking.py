"""Live rendering and orchestration using fake camera/window adapters."""

import hashlib
from collections import deque

import numpy as np
import pytest

import app
import live_tracking
from calibration_profile import CalibrationProfile, CaptureIdentity, EvaluationSettings, FeatureReference, FeatureRule
from capture import Frame
from evaluation import PostureEvaluator
from features import FeatureConfig, PostureFeatures
from live_preview import raw_thresholds, render_dashboard
from pose import PoseObservation


def profile(model_hash="a" * 64):
    return CalibrationProfile(
        FeatureConfig(enabled=("head_offset_y", "mouth_shoulder_vertical"), temporal_enabled=False),
        (FeatureRule("head", "head_offset_y", FeatureReference(0.7, 0.03, 100), "decrease", 3, 1.5),),
        EvaluationSettings(0.35, 1.5, 0.75),
        CaptureIdentity(0, 640, 480, model_hash),
        ("test",),
        "test",
    )


def test_dashboard_renders_native_thresholds_and_preserves_camera_image():
    p = profile()
    evaluator = PostureEvaluator(p, display_features=("mouth_shoulder_vertical",))
    history = deque()
    for index in range(151):
        value = 0.7 if index < 90 else 0.56
        sample = PostureFeatures(
            index * 33_333_333, {"head_offset_y": value, "mouth_shoulder_vertical": value - 0.2}, {}
        )
        history.append(evaluator.update(sample))
    frame = Frame(np.zeros((480, 640, 3), dtype=np.uint8), 151, sample.timestamp_ns)
    pose = PoseObservation(sample.timestamp_ns, 640, 480, {})
    image = render_dashboard(
        frame, pose, history[-1], p, history, ("head_offset_y", "mouth_shoulder_vertical"), "ema", 15, 0.5
    )
    assert image.shape == (620, 1400, 3)
    assert image.dtype == np.uint8
    assert not frame.image.any()
    assert raw_thresholds(p, "head_offset_y")[0][0] == pytest.approx(0.61)
    assert raw_thresholds(p, "head_offset_y")[1][0] == pytest.approx(0.655)


def test_live_runner_switches_filters_and_closes_resources(monkeypatch, tmp_path):
    model = tmp_path / "model.task"
    model.write_bytes(b"test model")
    p = profile(hashlib.sha256(model.read_bytes()).hexdigest())
    path = tmp_path / "profile.json"
    p.save(path)
    events = []

    class Tracker:
        model_path = model

        def __enter__(self):
            events.append("tracker open")
            return self

        def estimate(self, frame):
            return PoseObservation(frame.timestamp_ns, 640, 480, {})

        def __exit__(self, *args):
            events.append("tracker close")

    class Camera:
        count = 0

        def __enter__(self):
            events.append("camera open")
            return self

        def read(self):
            self.count += 1
            return Frame(np.zeros((480, 640, 3), dtype=np.uint8), self.count, self.count * 100_000_000)

        def __exit__(self, *args):
            events.append("camera close")

    class Preview:
        history = deque()
        last_key = -1
        keys = iter([ord("1"), ord("3"), ord("q")])

        def __init__(self, *args):
            self.method = args[2]

        def __enter__(self):
            events.append("preview open")
            return self

        def show(self, frame, pose, result):
            assert result.status == "UNKNOWN"
            events.append(self.method)
            self.last_key = next(self.keys)
            return self.last_key != ord("q")

        def __exit__(self, *args):
            events.append("preview close")

    monkeypatch.setattr(live_tracking, "PoseLandmarksTracker", Tracker)
    monkeypatch.setattr(live_tracking, "CameraCapture", lambda index: Camera())
    monkeypatch.setattr(live_tracking, "LivePreview", Preview)
    assert live_tracking.run_live(path) == 0
    assert events == [
        "tracker open",
        "camera open",
        "preview open",
        "ema",
        "none",
        "moving_average",
        "preview close",
        "camera close",
        "tracker close",
    ]


def test_app_track_dispatch_and_mode_conflict(monkeypatch, tmp_path):
    path = tmp_path / "profile.json"
    calls = []
    monkeypatch.setattr(live_tracking, "run_live", lambda *args: calls.append(args) or 0)
    monkeypatch.setattr("sys.argv", ["app.py", "--track", "--profile", str(path), "--duration", "2"])
    assert app.main() == 0
    assert calls == [(path, 0, 2)]
    monkeypatch.setattr("sys.argv", ["app.py", "--track", "--calibrate"])
    with pytest.raises(SystemExit) as error:
        app.main()
    assert error.value.code == 2


def test_missing_profile_fails_without_opening_camera(monkeypatch, tmp_path):
    monkeypatch.setattr(live_tracking, "CameraCapture", lambda index: pytest.fail("camera should not open"))
    assert live_tracking.run_live(tmp_path / "missing.json") == 1
