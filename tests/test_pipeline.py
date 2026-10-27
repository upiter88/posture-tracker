"""Check stage order, reporting, and cleanup without camera or model hardware."""

import logging
from unittest.mock import Mock

import numpy as np
import pytest

import app
from capture import Frame
from pose import PoseError, PoseLandmark, PoseObservation
from preview import draw_pose
from reporting import PoseDiagnostics


@pytest.mark.parametrize("inference_fails", [False, True])
def test_default_pipeline_order_and_cleanup(monkeypatch, inference_fails):
    events = []
    frame = Frame(np.zeros((80, 100, 3), dtype=np.uint8), 1, 10)
    pose = PoseObservation(10, 100, 80, {})

    class Camera:
        def __enter__(self):
            events.append("open camera")
            return self

        def read(self):
            events.append("capture")
            return frame

        def __exit__(self, *args):
            events.append("close camera")

    class Estimator:
        def __enter__(self):
            events.append("open pose")
            return self

        def estimate(self, actual_frame):
            assert actual_frame is frame
            events.append("pose")
            if inference_fails:
                raise PoseError("inference failed")
            return pose

        def __exit__(self, *args):
            events.append("close pose")

    class Preview:
        def __enter__(self):
            return self

        def show(self, actual_frame, actual_pose):
            assert actual_frame is frame and actual_pose is pose
            events.append("preview")
            return False

        def __exit__(self, *args):
            events.append("close preview")

    def report(actual_frame, actual_pose, seconds):
        assert actual_frame is frame and actual_pose is pose
        assert seconds >= 0
        events.append("report")

    monkeypatch.setattr("sys.argv", ["app.py", "--preview"])
    monkeypatch.setattr(app, "CameraCapture", lambda index: Camera())
    monkeypatch.setattr(app, "PoseEstimator", Estimator)
    monkeypatch.setattr(app, "VideoPreview", Preview)
    monkeypatch.setattr(app, "PoseDiagnostics", lambda: Mock(report=report))
    assert app.main() == (1 if inference_fails else 0)
    stages = ["capture", "pose"] if inference_fails else ["capture", "pose", "report", "preview"]
    assert events == [
        "open pose",
        "open camera",
        *stages,
        "close preview",
        "close camera",
        "close pose",
    ]


def test_diagnostics_reports_missing_pose_and_interval_counts(monkeypatch, caplog):
    times = iter([0.0, 0.5, 1.1, 2.2])
    monkeypatch.setattr("reporting.monotonic", lambda: next(times))
    diagnostics = PoseDiagnostics()
    frame = Frame(np.zeros((10, 10, 3), dtype=np.uint8), 1, 10)
    found = PoseObservation(10, 10, 10, {"nose": PoseLandmark(0.5, 0.5, 0, 0.9, 0.8)})
    missing = PoseObservation(10, 10, 10, {})
    with caplog.at_level(logging.INFO):
        diagnostics.report(frame, found, 0.01)
        assert not caplog.records
        diagnostics.report(frame, missing, 0.03)
        assert "pose=1/2" in caplog.messages[-1]
        assert "inference=20.0 ms" in caplog.messages[-1]
        assert "no pose" in caplog.messages[-1]
        diagnostics.report(frame, missing, 0.01)
        assert "pose=0/1" in caplog.messages[-1]


def test_preview_does_not_mutate_capture_and_skips_unusable_landmarks():
    frame = Frame(np.zeros((80, 100, 3), dtype=np.uint8), 1, 10)
    original = frame.image.copy()
    pose = PoseObservation(
        10,
        100,
        80,
        {
            "nose": PoseLandmark(0.5, 0.5, 0, 0.9, 0.9),
            "left_eye": PoseLandmark(float("nan"), 0.4, 0, 0.9, 0.9),
            "left_ear": PoseLandmark(1e30, 0.4, 0, 0.9, 0.9),
            "right_ear": PoseLandmark(0.2, 0.4, 0, 0.1, 0.9),
        },
    )
    rendered = draw_pose(frame, pose)
    np.testing.assert_array_equal(frame.image, original)
    assert rendered.any()
    missing = draw_pose(frame, PoseObservation(10, 100, 80, {}))
    assert missing.any()


def test_calibration_pipeline_records_preview_labels_and_cleans_up(monkeypatch, tmp_path):
    import json

    import pyarrow.parquet as pq

    events = []
    model = tmp_path / "model.task"
    model.write_bytes(b"fake model")

    class Camera:
        sequence = 0

        def __enter__(self):
            return self

        def read(self):
            self.sequence += 1
            return Frame(np.zeros((80, 100, 3), dtype=np.uint8), self.sequence, self.sequence * 100_000_000)

        def __exit__(self, *args):
            events.append("camera closed")

    class Estimator:
        model_path = model

        def __enter__(self):
            return self

        def estimate(self, frame):
            return PoseObservation(frame.timestamp_ns, 100, 80, {})

        def __exit__(self, *args):
            events.append("pose closed")

    class Preview:
        last_key = -1
        status_text = None
        keys = iter([ord("2"), ord("1"), ord("0"), ord("q")])

        def __enter__(self):
            return self

        def show(self, frame, pose):
            assert "unusable" in self.status_text
            self.last_key = next(self.keys)
            return self.last_key != ord("q")

        def __exit__(self, *args):
            events.append("preview closed")

    monkeypatch.setattr("sys.argv", ["app.py", "--calibrate", "--output", str(tmp_path / "sessions")])
    monkeypatch.setattr(app, "CameraCapture", lambda index: Camera())
    monkeypatch.setattr(app, "PoseEstimator", Estimator)
    monkeypatch.setattr(app, "VideoPreview", Preview)
    assert app.main() == 0
    session = next((tmp_path / "sessions").iterdir())
    samples = pq.read_table(session / "samples.parquet").to_pylist()
    assert [row["posture_label"] for row in samples] == ["unlabeled", "valid", "invalid", "unlabeled"]
    assert json.loads((session / "session.json").read_text())["complete"]
    assert events == ["preview closed", "camera closed", "pose closed"]


def test_invalid_calibration_config_fails_before_opening_camera(monkeypatch):
    def invalid_config(path):
        raise ValueError("unknown feature")

    camera = Mock()
    monkeypatch.setattr("sys.argv", ["app.py", "--calibrate"])
    monkeypatch.setattr(app.FeatureConfig, "load", invalid_config)
    monkeypatch.setattr(app, "CameraCapture", camera)
    with pytest.raises(SystemExit) as error:
        app.main()
    assert error.value.code == 2
    camera.assert_not_called()


def test_preview_normalizes_uppercase_label_and_preserves_source(monkeypatch):
    from preview import VideoPreview

    frame = Frame(np.zeros((80, 640, 3), dtype=np.uint8), 1, 10)
    pose = PoseObservation(10, 640, 80, {})
    displayed = []
    monkeypatch.setattr("preview.cv2.imshow", lambda name, image: displayed.append(image))
    monkeypatch.setattr("preview.cv2.waitKey", lambda delay: ord("2"))
    monkeypatch.setattr("preview.cv2.getWindowProperty", lambda *args: 1)
    preview = VideoPreview()
    preview.status_text = "UNLABELED | usable | 1.0s | saved 0/1"
    assert preview.show(frame, pose)
    assert preview.last_key == ord("2")
    assert displayed[0].any()
    assert not frame.image.any()
