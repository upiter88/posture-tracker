"""Camera-independent posture decisions with smoothing, hysteresis, and timing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from calibration.calibration_profile import CalibrationProfile, FeatureRule
from posture.features import PostureFeatures
from posture.smoothing import SignalSmoother, SmoothingSettings

PostureStatus = Literal["WITHIN_REFERENCE", "DEVIATION", "UNKNOWN"]


@dataclass(frozen=True)
class SignalValue:
    raw: float | None
    smoothed: float | None


@dataclass(frozen=True)
class RuleResult:
    name: str
    feature: str
    score: float | None
    deviating: bool | None
    pending_target: bool | None = None
    pending_seconds: float = 0.0
    required_seconds: float = 0.0


@dataclass(frozen=True)
class PostureResult:
    timestamp_ns: int
    status: PostureStatus
    reason: str
    signals: dict[str, SignalValue]
    rules: tuple[RuleResult, ...]


def deviation_score(value: float, rule: FeatureRule) -> float:
    difference = (value - rule.reference.centre) / rule.reference.scale
    if rule.direction == "decrease":
        return -difference
    return abs(difference) if rule.direction == "either" else difference


@dataclass
class _RuleState:
    active: bool | None = None
    target: bool | None = None
    since_ns: int | None = None

    def update(self, rule: FeatureRule, score: float, timestamp: int, enter_s: float, clear_s: float) -> RuleResult:
        target = None
        if self.active is not True and score >= rule.enter_threshold:
            target = True
        elif self.active is not False and score <= rule.clear_threshold:
            target = False
        if target != self.target:
            self.target, self.since_ns = target, timestamp if target is not None else None
        elapsed = (timestamp - self.since_ns) / 1e9 if self.since_ns is not None else 0.0
        required = enter_s if target else clear_s
        if target is not None and elapsed >= required:
            self.active = target
            self.target, self.since_ns = None, None
            elapsed = 0.0
        return RuleResult(
            rule.name,
            rule.feature,
            score,
            self.active,
            self.target,
            elapsed,
            required if self.target is not None else 0.0,
        )


class PostureEvaluator:
    """Initial state is UNKNOWN until a sustained reference or deviation exists.

    All active inputs must be available, even for an 'any' combination. Missing
    diagnostics do not affect decisions. Tracking loss resets pending timers and
    filter memory; reacquisition must establish a fresh sustained state.
    """

    def __init__(
        self, profile: CalibrationProfile, method="ema", max_gap_seconds=0.5, display_features: tuple[str, ...] = ()
    ):
        self.profile = profile
        self.settings = SmoothingSettings(method, profile.evaluation.smoothing_seconds, max_gap_seconds)
        names = tuple(dict.fromkeys((*profile.required_features, *display_features)))
        if set(names) - set(profile.extraction.columns):
            raise ValueError("Display features must be outputs of the profile extraction settings")
        self._filters = {name: SignalSmoother(self.settings) for name in names}
        self._states = {rule.name: _RuleState() for rule in profile.rules}
        self._last_time: int | None = None

    def _reset(self) -> None:
        for smoother in self._filters.values():
            smoother.reset()
        self._states = {rule.name: _RuleState() for rule in self.profile.rules}

    def update(self, sample: PostureFeatures) -> PostureResult:
        timestamp = sample.timestamp_ns
        if self._last_time is not None:
            if timestamp <= self._last_time:
                raise ValueError("Evaluation timestamps must strictly increase")
            if (timestamp - self._last_time) / 1e9 > self.settings.max_gap_seconds:
                self._reset()
        self._last_time = timestamp
        raw = {name: sample.values.get(name) for name in self._filters}
        raw = {name: value if value is not None and math.isfinite(value) else None for name, value in raw.items()}
        missing = [name for name in self.profile.required_features if raw[name] is None]
        if missing:
            self._reset()
            reason = "; ".join(f"{name}: {sample.reasons.get(name, 'unavailable')}" for name in missing)
            signals = {name: SignalValue(value, None) for name, value in raw.items()}
            return PostureResult(timestamp, "UNKNOWN", reason, signals, ())
        signals = {
            name: SignalValue(value, self._filters[name].update(timestamp, value)) for name, value in raw.items()
        }
        results = []
        for rule in self.profile.rules:
            value = signals[rule.feature].smoothed
            assert value is not None  # Required raw inputs were checked above; filters seed immediately.
            result = self._states[rule.name].update(
                rule,
                deviation_score(value, rule),
                timestamp,
                self.profile.evaluation.enter_seconds,
                self.profile.evaluation.clear_seconds,
            )
            results.append(result)
        if any(result.deviating is None for result in results):
            status: PostureStatus = "UNKNOWN"
            ambiguous = any(result.deviating is None and result.pending_target is None for result in results)
            reason = (
                "Ambiguous posture: return to your calibrated reference"
                if ambiguous
                else "Settling: hold a stable posture"
            )
        else:
            deviating = [result.deviating for result in results]
            combined = any(deviating) if self.profile.combination == "any" else all(deviating)
            status = "DEVIATION" if combined else "WITHIN_REFERENCE"
            reason = "Sustained deviation from reference" if combined else "Within calibrated reference"
        return PostureResult(timestamp, status, reason, signals, tuple(results))
