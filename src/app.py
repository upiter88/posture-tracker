"""Console-only camera diagnostics: python app.py [--camera 0] [--duration 10]."""

import argparse
import logging
import math
from time import monotonic

from capture import CameraCapture, CaptureError


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
    parser = argparse.ArgumentParser(description="Capture webcam frames without a preview or recording.")
    parser.add_argument("--camera", type=camera_index, default=0, help="camera index (default: 0)")
    parser.add_argument("--duration", type=positive_seconds, help="stop after this many seconds of capture")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    count = 0
    started = None
    try:
        logging.info("Opening camera %d; press Ctrl+C to stop.", args.camera)
        with CameraCapture(args.camera) as camera:
            started = monotonic()
            last_report = started
            last_count = 0
            while args.duration is None or monotonic() - started < args.duration:
                frame = camera.read()
                count = frame.sequence
                now = monotonic()
                height, width = frame.image.shape[:2]
                if count == 1:
                    logging.info("Receiving %dx%d BGR frames.", width, height)
                if now - last_report >= 1.0:
                    logging.info(
                        "Frames=%d | %dx%d | %.1f FPS | timestamp_ns=%d",
                        count, width, height,
                        (count - last_count) / (now - last_report), frame.timestamp_ns,
                    )
                    last_report, last_count = now, count
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    except CaptureError as error:
        logging.error("%s", error)
        return 1
    finally:
        if started is not None:
            logging.info("Capture ended: %d frames in %.2f seconds; camera released.", count, monotonic() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
