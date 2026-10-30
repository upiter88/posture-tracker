"""Small, human-edited settings for fitting, smoothing, and the dashboard."""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from calibration.calibration import FitSettings, RuleSpec
from calibration.calibration_profile import EvaluationSettings
from posture.smoothing import SmoothingSettings

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config/tracking.toml"


@dataclass(frozen=True)
class DisplaySettings:
    features: tuple[str, ...] = ()
    history_seconds: float = 15.0

    def __post_init__(self):
        if len(self.features) > 4 or len(set(self.features)) != len(self.features):
            raise ValueError("Display supports up to four unique feature outputs")
        if not math.isfinite(self.history_seconds) or self.history_seconds <= 0:
            raise ValueError("Display history must be finite and positive")


@dataclass(frozen=True)
class SoundSettings:
    enabled: bool = True
    volume: float = 0.3
    repeat_seconds: float = 5.0

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("Sound enabled must be a boolean")
        if isinstance(self.volume, bool) or not math.isfinite(self.volume) or not 0 <= self.volume <= 1:
            raise ValueError("Sound volume must be finite and between 0 and 1")
        if isinstance(self.repeat_seconds, bool) or not math.isfinite(self.repeat_seconds) or self.repeat_seconds < 0.3:
            raise ValueError("Sound repeat_seconds must be finite and at least 0.3")


@dataclass(frozen=True)
class TrackingSettings:
    rules: tuple[RuleSpec, ...]
    fitting: FitSettings
    smoothing: SmoothingSettings
    evaluation: EvaluationSettings
    display: DisplaySettings
    sound: SoundSettings = SoundSettings()

    @classmethod
    def load(cls, path: Path = DEFAULT_SETTINGS_PATH):
        with path.open("rb") as source:
            data = tomllib.load(source)
        if set(data) - {"sound"} != {"rules", "fitting", "smoothing", "evaluation", "display"}:
            raise ValueError(
                "config/tracking.toml requires rules, fitting, smoothing, evaluation, and display sections"
            )
        smoothing = SmoothingSettings(**data["smoothing"])
        evaluation = EvaluationSettings(smoothing_seconds=smoothing.seconds, **data["evaluation"])
        display = dict(data["display"])
        display["features"] = tuple(display.get("features", ()))
        rules = tuple(RuleSpec(**rule) for rule in data["rules"])
        if not rules or len({rule.name for rule in rules}) != len(rules):
            raise ValueError("Configure at least one rule, with unique names")
        return cls(
            rules,
            FitSettings(**data["fitting"]),
            smoothing,
            evaluation,
            DisplaySettings(**display),
            SoundSettings(**data.get("sound", {})),
        )
