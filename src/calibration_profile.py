"""Versioned personal calibration data, independent of fitting and evaluation.

The profile snapshots extraction settings and references registry outputs by name.
It never reads features.toml implicitly. See CALIBRATION_PROFILES.md for the JSON
contract, feature selection, score units, compatibility, and manipulation examples.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal
from uuid import uuid4

from features import FEATURE_VERSION, FeatureConfig, PostureFeatures

PROFILE_SCHEMA_VERSION = 1
DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent.parent / "data/profiles/current.json"


class ProfileError(ValueError):
    """A profile is invalid, incompatible, or cannot be read/written."""


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{name} must be a nonempty string")


def _number(value: object, name: str, *, minimum: float = 0, inclusive: bool = True) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProfileError(f"{name} must be a finite number")
    if value < minimum or (not inclusive and value == minimum):
        raise ProfileError(f"{name} must be {'>=' if inclusive else '>'} {minimum}")


def _integer(value: object, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ProfileError(f"{name} must be an integer >= {minimum}")


def _object(value: object, schema) -> dict:
    """Check saved dataclass fields without maintaining a second field list."""
    if not isinstance(value, dict):
        raise ProfileError(f"{schema.__name__} must be a JSON object")
    expected = {item.name for item in fields(schema)}
    if set(value) != expected:
        raise ProfileError(
            f"{schema.__name__}: missing fields {sorted(expected - value.keys())}; "
            f"unknown fields {sorted(value.keys() - expected)}"
        )
    return dict(value)


def _array(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise ProfileError(f"{name} must be a JSON array")
    return value


@dataclass(frozen=True)
class FeatureReference:
    """Centre and positive scale in the referenced output's native units.

    Step 2 will determine how these are fitted (including a noise floor). A scale
    of zero cannot define a normalized deviation and is rejected here.
    """

    centre: float
    scale: float
    sample_count: int

    def __post_init__(self) -> None:
        if isinstance(self.centre, bool) or not isinstance(self.centre, (int, float)) or not math.isfinite(self.centre):
            raise ProfileError("reference.centre must be finite")
        _number(self.scale, "reference.scale", inclusive=False)
        _integer(self.sample_count, "reference.sample_count")


@dataclass(frozen=True)
class FeatureRule:
    """One independently named rule on a registered base or temporal output.

    Future evaluator score:
      decrease: (centre - value) / scale
      increase: (value - centre) / scale
      either:   abs(value - centre) / scale

    Thresholds are dimensionless score units, not raw feature values. Enter must
    exceed clear to encode hysteresis. This module stores data; it makes no
    posture decisions and does not infer thresholds from the reference.
    """

    name: str
    feature: str
    reference: FeatureReference
    direction: Literal["decrease", "increase", "either"]
    enter_threshold: float
    clear_threshold: float

    def __post_init__(self) -> None:
        _text(self.name, "rule.name")
        _text(self.feature, "rule.feature")
        if not isinstance(self.reference, FeatureReference):
            raise ProfileError("rule.reference must be a FeatureReference")
        if self.direction not in ("decrease", "increase", "either"):
            raise ProfileError("rule.direction must be decrease, increase, or either")
        _number(self.enter_threshold, "rule.enter_threshold", inclusive=False)
        _number(self.clear_threshold, "rule.clear_threshold")
        if self.clear_threshold >= self.enter_threshold:
            raise ProfileError("rule.clear_threshold must be smaller than enter_threshold")


@dataclass(frozen=True)
class EvaluationSettings:
    """Explicit provisional timing, in seconds; evaluated in a later milestone.

    Zero smoothing disables smoothing. Persistence durations must be positive.
    No timing defaults are silently presented as learned calibration results.
    """

    smoothing_seconds: float
    enter_seconds: float
    clear_seconds: float

    def __post_init__(self) -> None:
        _number(self.smoothing_seconds, "evaluation.smoothing_seconds")
        _number(self.enter_seconds, "evaluation.enter_seconds", inclusive=False)
        _number(self.clear_seconds, "evaluation.clear_seconds", inclusive=False)


@dataclass(frozen=True)
class CaptureIdentity:
    """Compatibility checks for known setup properties, not camera-pose detection."""

    camera_index: int
    image_width: int
    image_height: int
    model_sha256: str

    def __post_init__(self) -> None:
        _integer(self.camera_index, "capture.camera_index", minimum=0)
        _integer(self.image_width, "capture.image_width")
        _integer(self.image_height, "capture.image_height")
        if (
            not isinstance(self.model_sha256, str)
            or len(self.model_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.model_sha256)
        ):
            raise ProfileError("capture.model_sha256 must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class CalibrationProfile:
    extraction: FeatureConfig
    rules: tuple[FeatureRule, ...]
    evaluation: EvaluationSettings
    capture: CaptureIdentity
    source_sessions: tuple[str, ...]
    method: str
    combination: Literal["any", "all"] = "any"
    profile_id: str = field(default_factory=lambda: str(uuid4()))
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: int = PROFILE_SCHEMA_VERSION
    feature_version: int = FEATURE_VERSION

    def __post_init__(self) -> None:
        _integer(self.schema_version, "schema_version")
        _integer(self.feature_version, "feature_version")
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ProfileError(f"Unsupported profile schema version: {self.schema_version}")
        if self.feature_version != FEATURE_VERSION:
            raise ProfileError(f"Incompatible feature version: {self.feature_version}; expected {FEATURE_VERSION}")
        _text(self.profile_id, "profile_id")
        _text(self.method, "method")
        _text(self.created_utc, "created_utc")
        try:
            created = datetime.fromisoformat(self.created_utc)
        except ValueError as error:
            raise ProfileError("created_utc must be an ISO timestamp with UTC offset") from error
        if created.utcoffset() != UTC.utcoffset(created):
            raise ProfileError("created_utc must have a UTC offset of zero")
        if not isinstance(self.extraction, FeatureConfig):
            raise ProfileError("extraction must be a FeatureConfig")
        if not isinstance(self.evaluation, EvaluationSettings) or not isinstance(self.capture, CaptureIdentity):
            raise ProfileError("evaluation and capture must use their profile dataclasses")
        if (
            not isinstance(self.rules, tuple)
            or not self.rules
            or not all(isinstance(r, FeatureRule) for r in self.rules)
        ):
            raise ProfileError("rules must be a nonempty tuple of FeatureRule values")
        if len({rule.name for rule in self.rules}) != len(self.rules):
            raise ProfileError("Rule names must be unique")
        for rule in self.rules:
            if rule.feature not in self.extraction.columns:
                raise ProfileError(f"Rule {rule.name!r}: {rule.feature!r} is not an output of extraction")
        if self.combination not in ("any", "all"):
            raise ProfileError("combination must be any or all")
        if not isinstance(self.source_sessions, tuple) or not self.source_sessions:
            raise ProfileError("source_sessions must be a nonempty tuple of recording IDs")
        for session in self.source_sessions:
            _text(session, "source_sessions item")
        if len(set(self.source_sessions)) != len(self.source_sessions):
            raise ProfileError("source_sessions must be unique")

    @property
    def required_features(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(rule.feature for rule in self.rules))

    @property
    def diagnostic_features(self) -> tuple[str, ...]:
        return tuple(name for name in self.extraction.columns if name not in self.required_features)

    def measurements_available(self, sample: PostureFeatures) -> bool:
        """Inspect active values only; missing diagnostics do not block a rule."""
        for name in self.required_features:
            value = sample.values.get(name)
            if value is None or not math.isfinite(value):
                return False
        return True

    def check_compatibility(self, capture: CaptureIdentity) -> None:
        """Call before using the profile; never silently rescale or change models."""
        if capture != self.capture:
            differences = [name for name, value in asdict(self.capture).items() if value != asdict(capture)[name]]
            raise ProfileError(f"Profile capture mismatch: {', '.join(differences)}; recalibration required")

    def to_dict(self) -> dict:
        """Serialize dataclasses, converting the three tuple fields to JSON arrays."""
        data = asdict(self)
        data["extraction"]["enabled"] = list(self.extraction.enabled)
        data["rules"] = list(data["rules"])
        data["source_sessions"] = list(self.source_sessions)
        return data

    @classmethod
    def from_dict(cls, value: object) -> CalibrationProfile:
        """Rebuild nested dataclasses; their constructors validate the values."""
        try:
            data = _object(value, cls)
            config = _object(data["extraction"], FeatureConfig)
            config["enabled"] = tuple(_array(config["enabled"], "extraction.enabled"))
            data["extraction"] = FeatureConfig(**config)
            rules = []
            for item in _array(data["rules"], "rules"):
                rule = _object(item, FeatureRule)
                rule["reference"] = FeatureReference(**_object(rule["reference"], FeatureReference))
                rules.append(FeatureRule(**rule))
            data["rules"] = tuple(rules)
            data["evaluation"] = EvaluationSettings(**_object(data["evaluation"], EvaluationSettings))
            data["capture"] = CaptureIdentity(**_object(data["capture"], CaptureIdentity))
            data["source_sessions"] = tuple(_array(data["source_sessions"], "source_sessions"))
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise ProfileError(f"Invalid profile: {error}") from error

    def save(self, path: Path = DEFAULT_PROFILE_PATH) -> None:
        """Publish a complete JSON file atomically; failed writes preserve the old file."""
        temporary: Path | None = None
        try:
            payload = json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as output:
                temporary = Path(output.name)
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(path)
        except (OSError, ValueError) as error:
            raise ProfileError(f"Cannot save profile at {path}: {error}") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path = DEFAULT_PROFILE_PATH) -> CalibrationProfile:
        try:
            data = json.loads(
                path.read_text(encoding="utf-8"), object_pairs_hook=_unique_fields, parse_constant=_reject_constant
            )
        except (OSError, ValueError) as error:
            raise ProfileError(f"Cannot load profile at {path}: {error}") from error
        return cls.from_dict(data)


def _unique_fields(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProfileError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProfileError(f"Nonfinite JSON constant: {value}")
