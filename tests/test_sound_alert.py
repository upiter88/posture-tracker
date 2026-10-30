"""Alert timing, audio amplitude, configuration, and resource cleanup without speakers."""

import struct
import wave
from types import SimpleNamespace

import pytest

import sound_alert
from evaluation import PostureResult
from sound_alert import PostureSoundAlert, write_tone
from tracking_config import DEFAULT_SETTINGS_PATH, SoundSettings, TrackingSettings


def result(seconds, status="DEVIATION"):
    return PostureResult(int(seconds * 1e9), status, "test", {}, ())


@pytest.fixture
def player(monkeypatch):
    calls = []
    player = SimpleNamespace(SND_FILENAME=1, SND_ASYNC=2, PlaySound=lambda *args: calls.append(args))
    monkeypatch.setattr(sound_alert.sys, "platform", "win32")
    monkeypatch.setattr(sound_alert.importlib, "import_module", lambda name: player)
    return player, calls


def test_alert_repeats_only_for_deviation_and_cleans_up(player):
    _, calls = player
    with PostureSoundAlert(SoundSettings()) as alert:
        path = alert._path
        assert path.is_file()
        alert.update(result(0, "UNKNOWN"))
        alert.update(result(1, "WITHIN_REFERENCE"))
        assert calls == []
        alert.update(result(2))
        alert.update(result(6.99))
        assert len(calls) == 1
        alert.update(result(7))
        assert calls == [(str(path), 3), (str(path), 3)]
        alert.update(result(8, "UNKNOWN"))
        assert calls[-1] == (None, 0)
        alert.update(result(8.1))
        assert calls[-1] == (str(path), 3)
        alert.update(result(8.2, "WITHIN_REFERENCE"))
        assert calls[-1] == (None, 0)
        alert.update(result(9))
        alert.stop()  # Filter changes reset alert timing as well as posture timing.
        alert.update(result(9.1))
        assert calls[-1] == (str(path), 3)
    assert calls[-1] == (None, 0)
    assert not path.exists()


@pytest.mark.parametrize("settings", [SoundSettings(enabled=False), SoundSettings(volume=0)])
def test_muted_alert_does_not_initialize_audio(settings, player):
    with PostureSoundAlert(settings) as alert:
        alert.update(result(0))
        assert alert._directory is None
    assert player[1] == []


def test_audio_failure_does_not_interrupt_tracking(player, caplog):
    def fail(*args):
        raise RuntimeError("no output device")

    player[0].PlaySound = fail
    with PostureSoundAlert(SoundSettings()) as alert:
        path = alert._path
        alert.update(result(0))
        alert.update(result(10))
    assert "Posture sound disabled: no output device" in caplog.text
    assert not path.exists()


def test_tone_volume_scales_pcm_and_has_faded_edges(tmp_path):
    peaks = []
    for volume in (0, 0.25, 1):
        path = tmp_path / "tone.wav"
        write_tone(path, volume)
        with wave.open(str(path), "rb") as source:
            assert source.getnchannels() == 1
            assert source.getsampwidth() == 2
            assert source.getnframes() / source.getframerate() == pytest.approx(0.3)
            pcm = source.readframes(source.getnframes())
        samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
        assert samples[0] == samples[-1] == 0
        peaks.append(max(map(abs, samples)))
    assert peaks[0] == 0
    assert peaks[1] / peaks[2] == pytest.approx(0.25, abs=0.0001)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"volume": -0.1},
        {"volume": 1.1},
        {"volume": float("nan")},
        {"volume": True},
        {"repeat_seconds": 0.1},
        {"repeat_seconds": float("inf")},
        {"enabled": "yes"},
    ],
)
def test_invalid_sound_settings(kwargs):
    with pytest.raises(ValueError):
        SoundSettings(**kwargs)


def test_sound_settings_are_optional_and_loaded(tmp_path):
    original = DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8")
    path = tmp_path / "tracking.toml"
    legacy = original.split("[sound]")[0]
    path.write_text(legacy, encoding="utf-8")
    assert TrackingSettings.load(path).sound == SoundSettings()
    path.write_text(legacy + "[sound]\nenabled = false\nvolume = 0.6\nrepeat_seconds = 10\n", encoding="utf-8")
    assert TrackingSettings.load(path).sound == SoundSettings(False, 0.6, 10)
