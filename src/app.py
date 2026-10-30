"""Capture -> pose -> diagnostics -> optional preview: python src/app.py."""

import argparse
import logging
import math
from contextlib import ExitStack
from pathlib import Path
from time import monotonic

import cv2

from calibration.calibration_profile import DEFAULT_PROFILE_PATH
from calibration.recording import CalibrationRecorder, RecordingError
from capture.capture import CameraCapture, CaptureError
from capture.pose import PoseError, PoseLandmarksTracker
from posture.features import FeatureConfig, FeatureExtractor
from presentation.preview import VideoPreview
from presentation.reporting import PoseDiagnostics


def camera_index(value: str) -> int:
    index = int(value)
    if index < 0:
        raise argparse.ArgumentTypeError("camera index must be non-negative")
    return index


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("duration must be a finite positive number")
    return seconds


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate webcam poses with an optional live preview.")
    parser.add_argument("--camera", type=camera_index, default=0, help="camera index (default: 0)")
    parser.add_argument("--duration", type=positive_seconds, help="stop after this many seconds of capture")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="show video; quit with Q, Esc, or the window close button",
    )
    parser.add_argument("--calibrate", action="store_true", help="record labeled landmarks and features; opens preview")
    parser.add_argument(
        "--track", action="store_true", help="evaluate posture with camera and raw/smoothed signal plots"
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH, help="saved live calibration profile")
    parser.add_argument(
        "--output", type=Path, default=Path("data/calibration"), help="calibration session parent directory"
    )
    args = parser.parse_args()
    if args.track and args.calibrate:
        parser.error("--track and --calibrate are separate modes")
    if args.track:
        from tracking.live_tracking import run_live

        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        return run_live(args.profile, args.camera, args.duration)
    try:
        feature_config = (
            FeatureConfig.load(Path(__file__).resolve().parent.parent / "config/features.toml")
            if args.calibrate
            else FeatureConfig()
        )
    except (OSError, ValueError) as error:
        parser.error(f"Cannot load feature configuration: {error}")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    count = 0
    started: float | None = None
    try:
        logging.info("Opening camera %d; press Ctrl+C to stop.", args.camera)
        with ExitStack() as stack:
            estimator = stack.enter_context(PoseLandmarksTracker())
            camera = stack.enter_context(CameraCapture(args.camera))
            preview = stack.enter_context(VideoPreview()) if args.preview or args.calibrate else None
            recorder = None
            extractor = None
            if args.calibrate:
                extractor = FeatureExtractor(feature_config)
                recorder = stack.enter_context(
                    CalibrationRecorder(args.output, feature_config, args.camera, estimator.model_path)
                )
                logging.info("Recording to %s; focus preview and use V/I/U to label posture.", recorder.path)
            if preview is not None:
                logging.info("Preview enabled; press Q or Esc in the window to stop.")
            started = monotonic()
            diagnostics = PoseDiagnostics()
            while args.duration is None or monotonic() - started < args.duration:
                # 1. Capture: BGR image and original monotonic timestamp.
                frame = camera.read()
                count = frame.sequence

                # 2. Pose: model inference only; no posture classification yet.
                inference_started = monotonic()
                pose = estimator.estimate(frame)
                inference_seconds = monotonic() - inference_started

                # 3. Report: summarize detection, confidence, and processing rate.
                diagnostics.report(frame, pose, inference_seconds)

                if recorder is not None and extractor is not None:
                    features = extractor.extract(pose)
                    recorder.record(frame, pose, features)
                    if preview is not None:
                        quality = "usable" if features.measurement_valid else "unusable"
                        preview.status_text = (
                            f"{recorder.label.upper()} | {quality} | {monotonic() - started:.1f}s | "
                            f"saved {recorder.saved_count}/{recorder.frame_count}"
                        )

                # 4. Preview: draw on a copy, keeping the captured image read-only.
                if preview is not None:
                    if not preview.show(frame, pose):
                        logging.info("Preview closed by user.")
                        break
                    if recorder is not None:
                        recorder.set_label(preview.last_key)
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    except (CaptureError, PoseError, RecordingError) as error:
        logging.error("%s", error)
        return 1
    except cv2.error as error:
        logging.error("OpenCV failed: %s. Preview requires the GUI-enabled opencv-python package.", error)
        return 1
    finally:
        if started is not None:
            logging.info(
                "Pipeline ended: %d frames in %.2f seconds; resources closed.",
                count,
                monotonic() - started,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
