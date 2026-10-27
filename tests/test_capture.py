"""Capture lifecycle checks without opening a physical camera."""

from unittest.mock import Mock

import numpy as np
import pytest

from capture import CameraCapture, CaptureError


@pytest.fixture
def device(monkeypatch):
    camera_device = Mock()
    monkeypatch.setattr("capture.cv2.VideoCapture", lambda: camera_device)
    return camera_device


def test_failed_open_releases_device(device):
    device.open.return_value = False
    with pytest.raises(CaptureError):
        with CameraCapture():
            pytest.fail("Unavailable camera must not enter the context")
    device.release.assert_called_once()


def test_read_failure_releases_device(device):
    device.read.return_value = (False, None)
    with pytest.raises(CaptureError):
        with CameraCapture() as camera:
            camera.read()
    device.release.assert_called_once()


def test_interrupt_releases_device(device):
    with pytest.raises(KeyboardInterrupt):
        with CameraCapture():
            raise KeyboardInterrupt
    device.release.assert_called_once()


def test_frame_metadata_and_reopen(device):
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    device.read.return_value = (True, image)
    camera = CameraCapture()
    with camera:
        first = camera.read()
        second = camera.read()
        assert (first.sequence, second.sequence) == (1, 2)
        assert second.timestamp_ns > first.timestamp_ns
        assert first.image is image
    with camera:
        assert camera.read().sequence == 1
    camera.close()
    assert device.release.call_count == 2


def test_clock_ties_advance_by_one_nanosecond_and_reopen_resets(device, monkeypatch):
    device.read.return_value = (True, np.zeros((2, 3, 3), dtype=np.uint8))
    times = iter([1_000_000_000, 1_000_000_000, 1_000_000_100, 1_050_000_000, 100])
    monkeypatch.setattr("capture.perf_counter_ns", lambda: next(times))
    camera = CameraCapture()
    with camera:
        assert [camera.read().timestamp_ns for _ in range(4)] == [
            1_000_000_000,
            1_000_000_001,
            1_000_000_100,
            1_050_000_000,
        ]
    with camera:
        assert camera.read().timestamp_ns == 100
