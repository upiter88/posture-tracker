"""Optional OpenCV window for inspecting captured frames."""

import math

import cv2

from capture import Frame
from pose import PoseObservation


def draw_pose(frame: Frame, pose: PoseObservation):
    """Render confident, in-frame landmarks without changing the source image."""
    image = frame.image.copy()
    height, width = image.shape[:2]
    pixels = {}
    for name, point in pose.landmarks.items():
        if (not all(math.isfinite(value) for value in (point.x, point.y, point.visibility, point.presence))
                or min(point.visibility, point.presence) < 0.5
                or not (0 <= point.x < 1 and 0 <= point.y < 1)):
            continue
        pixels[name] = (int(point.x * width), int(point.y * height))
    for start, end in (
        ('left_shoulder', 'right_shoulder'),
        ('left_ear', 'left_eye'), ('left_eye', 'nose'),
        ('right_ear', 'right_eye'), ('right_eye', 'nose'),
    ):
        if start in pixels and end in pixels:
            cv2.line(image, pixels[start], pixels[end], (0, 200, 0), 2)
    for name, pixel in pixels.items():
        cv2.circle(image, pixel, 4, (0, 255, 255), -1)
        cv2.putText(image, name, (pixel[0] + 6, pixel[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    if not pose.detected:
        cv2.putText(image, 'No pose detected', (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return image


class VideoPreview:
    WINDOW_NAME = 'Posture Tracker - Camera'

    def __enter__(self):
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        return self

    def show(self, frame: Frame, pose: PoseObservation) -> bool:
        """Display a frame; return False when the user requests shutdown."""
        cv2.imshow(self.WINDOW_NAME, draw_pose(frame, pose))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            return False
        return cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1

    def __exit__(self, exc_type, exc_value, traceback):
        # The user may already have closed the window using its title-bar button.
        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except cv2.error:
            pass
