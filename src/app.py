"""Capture -> pose -> diagnostics -> optional preview: python src/app.py."""

import argparse
import logging
import math
from contextlib import ExitStack
from time import monotonic

import cv2

from capture import CameraCapture, CaptureError
from pose import PoseError, PoseEstimator
from preview import VideoPreview
from reporting import PoseDiagnostics


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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    count = 0
    started: float | None = None
    try:
        logging.info("Opening camera %d; press Ctrl+C to stop.", args.camera)
        with ExitStack() as stack:
            estimator = stack.enter_context(PoseEstimator())
            camera = stack.enter_context(CameraCapture(args.camera))
            preview = stack.enter_context(VideoPreview()) if args.preview else None
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

                # 4. Preview: draw on a copy, keeping the captured image read-only.
                if preview is not None and not preview.show(frame, pose):
                    logging.info("Preview closed by user.")
                    break
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    except (CaptureError, PoseError) as error:
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
