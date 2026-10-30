"""Nonblocking Windows posture tones with local PCM volume control."""

import importlib
import logging
import math
import struct
import sys
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation import PostureResult
from tracking_config import SoundSettings


def write_tone(path: Path, volume: float) -> None:
    """Create a short 880 Hz tone with faded edges to avoid clicks."""
    rate = 44_100
    count = int(rate * 0.3)
    fade = int(rate * 0.015)
    samples = (
        round(
            32767
            * volume
            * min(1, index / fade, (count - 1 - index) / fade)
            * math.sin(2 * math.pi * 880 * index / rate)
        )
        for index in range(count)
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(struct.pack(f"<{count}h", *samples))


class PostureSoundAlert:
    def __init__(self, settings: SoundSettings):
        self.settings = settings
        self._directory = None
        self._player = None
        self._path = None
        self._last_alert_ns = None

    def __enter__(self):
        if not self.settings.enabled or self.settings.volume == 0:
            return self
        if sys.platform != "win32":
            logging.warning("Posture sound alerts currently require Windows; visual tracking remains available.")
            return self
        try:
            self._player = importlib.import_module("winsound")
            self._directory = TemporaryDirectory(prefix="posture-sound-")
            self._path = Path(self._directory.name) / "alert.wav"
            write_tone(self._path, self.settings.volume)
        except (ImportError, OSError, RuntimeError) as error:
            self._disable(error)
        return self

    def _disable(self, error: Exception) -> None:
        logging.warning("Posture sound disabled: %s", error)
        self._player = None

    def update(self, result: PostureResult) -> None:
        if result.status != "DEVIATION":
            self.stop()
            return
        if self._player is None:
            return
        if (
            self._last_alert_ns is None
            or (result.timestamp_ns - self._last_alert_ns) / 1e9 >= self.settings.repeat_seconds
        ):
            try:
                self._player.PlaySound(str(self._path), self._player.SND_FILENAME | self._player.SND_ASYNC)
                self._last_alert_ns = result.timestamp_ns
            except (OSError, RuntimeError) as error:
                self._disable(error)

    def stop(self) -> None:
        if self._last_alert_ns is not None and self._player is not None:
            try:
                self._player.PlaySound(None, 0)
            except (OSError, RuntimeError) as error:
                self._disable(error)
        self._last_alert_ns = None

    def __exit__(self, *args):
        self.stop()
        if self._directory is not None:
            self._directory.cleanup()
