"""Own live resources and coordinate the existing capture and feature adapters."""

import hashlib
import logging
from dataclasses import replace
from pathlib import Path
from time import monotonic

import cv2

from calibration_profile import CalibrationProfile, CaptureIdentity, ProfileError
from capture import CameraCapture, CaptureError
from evaluation import PostureEvaluator
from features import FeatureExtractor
from live_preview import LivePreview
from pose import PoseError, PoseLandmarksTracker
from sound_alert import PostureSoundAlert
from tracking_config import TrackingSettings

FILTER_KEYS = {ord("1"): "none", ord("2"): "ema", ord("3"): "moving_average"}


def run_live(profile_path: Path, camera_index: int = 0, duration: float | None = None) -> int:
    try:
        settings = TrackingSettings.load()
        if not profile_path.is_file():
            raise ProfileError("No fitted profile. First run: uv run src/fit_profile.py")
        # Reference/thresholds remain fitted; live timing is editable without refitting.
        profile = replace(CalibrationProfile.load(profile_path), evaluation=settings.evaluation)
        displayed = settings.display.features or profile.required_features[:4]
        method = settings.smoothing.method
        evaluator = PostureEvaluator(profile, method, settings.smoothing.max_gap_seconds, displayed)
        extractor = FeatureExtractor(profile.extraction)
        with PoseLandmarksTracker() as tracker:
            model_hash = hashlib.sha256(tracker.model_path.read_bytes()).hexdigest()
            with (
                CameraCapture(camera_index) as camera,
                LivePreview(
                    profile,
                    displayed,
                    method,
                    settings.display.history_seconds,
                    settings.smoothing.max_gap_seconds,
                ) as preview,
                PostureSoundAlert(settings.sound) as sound,
            ):
                started = monotonic()
                checked_capture = False
                previous_status = None
                while duration is None or monotonic() - started < duration:
                    frame = camera.read()
                    if not checked_capture:
                        height, width = frame.image.shape[:2]
                        profile.check_compatibility(CaptureIdentity(camera_index, width, height, model_hash))
                        checked_capture = True
                    pose = tracker.estimate(frame)
                    sample = extractor.extract(pose)
                    result = evaluator.update(sample)
                    sound.update(result)
                    if result.status != previous_status:
                        logging.info("%s: %s", result.status, result.reason)
                        previous_status = result.status
                    if not preview.show(frame, pose, result):
                        break
                    selected = FILTER_KEYS.get(preview.last_key)
                    if selected is not None and selected != method:
                        sound.stop()
                        method = selected
                        evaluator = PostureEvaluator(profile, method, settings.smoothing.max_gap_seconds, displayed)
                        preview.method = method
                        preview.history.clear()
                        logging.info("Filter changed to %s; resetting history and posture state.", method)
        return 0
    except KeyboardInterrupt:
        logging.info("Stopped live tracking.")
        return 0
    except (OSError, ValueError, TypeError, CaptureError, PoseError, cv2.error) as error:
        logging.error("Live tracking: %s", error)
        return 1
