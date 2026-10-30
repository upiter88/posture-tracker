"""Fit explicit feature rules, with equal influence per labeled posture hold."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from calibration_data import CalibrationRecording
from calibration_profile import CalibrationProfile, EvaluationSettings, FeatureReference, FeatureRule
from evaluation import deviation_score


@dataclass(frozen=True)
class RuleSpec:
    feature: str
    name: str = "posture"
    direction: Literal["decrease", "increase", "either"] = "decrease"
    min_scale: float = 0.005

    def __post_init__(self):
        if not self.name or not self.feature or self.direction not in ("decrease", "increase", "either"):
            raise ValueError("Each rule needs a name, output feature, and valid direction")
        if not math.isfinite(self.min_scale) or self.min_scale <= 0:
            raise ValueError("Rule min_scale must be finite and positive, in native feature units")


@dataclass(frozen=True)
class FitSettings:
    trim_seconds: float = 1.0
    min_hold_seconds: float = 2.0
    min_holds_per_label: int = 2
    min_valid_fraction: float = 0.8
    min_balanced_accuracy: float = 0.8
    clear_ratio: float = 0.5

    def __post_init__(self):
        if not math.isfinite(self.trim_seconds) or self.trim_seconds < 0:
            raise ValueError("trim_seconds must be finite and non-negative")
        if not math.isfinite(self.min_hold_seconds) or self.min_hold_seconds <= 0 or self.min_holds_per_label < 2:
            raise ValueError("Fitting requires positive hold duration and at least two holds per label")
        if not 0 < self.min_valid_fraction <= 1 or not 0.5 <= self.min_balanced_accuracy <= 1:
            raise ValueError("Invalid fitting coverage or balanced accuracy limit")
        if not 0 < self.clear_ratio < 1:
            raise ValueError("clear_ratio must be between zero and one")


@dataclass(frozen=True)
class _Hold:
    session_id: str
    segment_id: int
    label: str
    values: np.ndarray


def _collect_holds(recordings: list[CalibrationRecording], feature: str, settings: FitSettings):
    holds, excluded = [], []
    for recording in recordings:
        for segment in dict.fromkeys(sample.segment_id for sample in recording.samples):
            group = [sample for sample in recording.samples if sample.segment_id == segment]
            label = group[0].label
            if label == "unlabeled":
                continue
            start = group[0].features.timestamp_ns + round(settings.trim_seconds * 1e9)
            end = group[-1].features.timestamp_ns - round(settings.trim_seconds * 1e9)
            interior = [sample for sample in group if start <= sample.features.timestamp_ns <= end]
            usable = [
                sample
                for sample in interior
                if (value := sample.features.values.get(feature)) is not None and math.isfinite(value)
            ]
            duration = (usable[-1].features.timestamp_ns - usable[0].features.timestamp_ns) / 1e9 if usable else 0
            continuous = all(
                (later.features.timestamp_ns - earlier.features.timestamp_ns) / 1e9
                <= recording.extraction.max_gap_seconds
                for earlier, later in zip(usable, usable[1:], strict=False)
            )
            if (
                not interior
                or len(usable) / len(interior) < settings.min_valid_fraction
                or duration < settings.min_hold_seconds
                or not continuous
            ):
                excluded.append(
                    {
                        "session": recording.session_id,
                        "segment": segment,
                        "reason": "insufficient usable duration, coverage, or continuity after trimming",
                    }
                )
                continue
            holds.append(
                _Hold(
                    recording.session_id,
                    segment,
                    label,
                    np.array([sample.features.values[feature] for sample in usable], dtype=float),
                )
            )
    return holds, excluded


def _fit_rule(holds: list[_Hold], spec: RuleSpec, settings: FitSettings) -> FeatureRule:
    valid = [hold for hold in holds if hold.label == "valid"]
    centre = float(np.median([np.median(hold.values) for hold in valid]))
    # Median within-hold absolute deviations about the shared centre. Each hold
    # contributes once; a long hold cannot dominate the reference or its spread.
    spread = 1.4826 * float(np.median([np.median(np.abs(hold.values - centre)) for hold in valid]))
    reference = FeatureReference(centre, max(spread, spec.min_scale), sum(len(hold.values) for hold in valid))
    scores, labels, weights = [], [], []
    for hold in holds:
        difference = (hold.values - centre) / reference.scale
        score = (
            -difference
            if spec.direction == "decrease"
            else np.abs(difference)
            if spec.direction == "either"
            else difference
        )
        scores.extend(score)
        invalid = hold.label == "invalid"
        labels.extend([invalid] * len(score))
        class_holds = sum(other.label == hold.label for other in holds)
        weights.extend([1 / (class_holds * len(score))] * len(score))
    scores, labels, weights = np.array(scores), np.array(labels), np.array(weights)
    unique, inverse = np.unique(scores, return_inverse=True)
    invalid_weight = np.bincount(inverse[labels], weights=weights[labels], minlength=len(unique))
    valid_weight = np.bincount(inverse[~labels], weights=weights[~labels], minlength=len(unique))
    invalid_below = np.r_[0, np.cumsum(invalid_weight)] / invalid_weight.sum()
    valid_below = np.r_[0, np.cumsum(valid_weight)] / valid_weight.sum()
    thresholds = np.r_[
        np.nextafter(unique[0], -np.inf), (unique[:-1] + unique[1:]) / 2, np.nextafter(unique[-1], np.inf)
    ]
    balanced_accuracy = (valid_below + 1 - invalid_below) / 2
    candidates = np.flatnonzero((thresholds > 0) & np.isfinite(thresholds))
    if not len(candidates):
        raise ValueError(f"{spec.name}: no positive deviation threshold separates the labels")
    best = candidates[np.argmax(balanced_accuracy[candidates])]
    entry = float(thresholds[best])
    return FeatureRule(spec.name, spec.feature, reference, spec.direction, entry, entry * settings.clear_ratio)


def fit_profile(
    recordings: list[CalibrationRecording],
    specs: tuple[RuleSpec, ...],
    settings: FitSettings,
    evaluation: EvaluationSettings,
) -> tuple[CalibrationProfile, dict]:
    if not recordings or not specs:
        raise ValueError("Provide at least one recording and one rule")
    first = recordings[0]
    if len({recording.session_id for recording in recordings}) != len(recordings):
        raise ValueError("Each source recording must be unique")
    for recording in recordings:
        if recording.capture != first.capture or recording.extraction != first.extraction:
            raise ValueError("Source recordings must share capture identity and extraction settings")
    rules, reports = [], []
    for spec in specs:
        if spec.feature not in first.extraction.columns:
            raise ValueError(f"{spec.feature!r} was not extracted in the source recording")
        holds, excluded = _collect_holds(recordings, spec.feature, settings)
        counts = {label: sum(hold.label == label for hold in holds) for label in ("valid", "invalid")}
        if min(counts.values()) < settings.min_holds_per_label:
            raise ValueError(f"{spec.name}: need {settings.min_holds_per_label} usable holds per label; got {counts}")
        folds = []
        # With multiple sessions, hold out a whole session. Otherwise hold out
        # complete posture segments. Nothing from the held-out group fits its rule.
        session_validation = len(recordings) > 1
        groups = [hold.session_id if session_validation else hold.segment_id for hold in holds]
        for group in dict.fromkeys(groups):
            train = [hold for hold, key in zip(holds, groups, strict=True) if key != group]
            test = [hold for hold, key in zip(holds, groups, strict=True) if key == group]
            if {hold.label for hold in train} != {"valid", "invalid"}:
                raise ValueError("Each validation fold must leave both labels in training")
            fitted = _fit_rule(train, spec, settings)
            for hold in test:
                predicted = np.array(
                    [deviation_score(float(value), fitted) >= fitted.enter_threshold for value in hold.values]
                )
                folds.append(
                    {
                        "session": hold.session_id,
                        "segment": hold.segment_id,
                        "label": hold.label,
                        "recall": float(np.mean(predicted == (hold.label == "invalid"))),
                    }
                )
        recalls = {
            label: float(np.mean([fold["recall"] for fold in folds if fold["label"] == label]))
            for label in ("valid", "invalid")
        }
        accuracy = (recalls["valid"] + recalls["invalid"]) / 2
        if accuracy < settings.min_balanced_accuracy:
            raise ValueError(
                f"{spec.name}: blocked balanced accuracy {accuracy:.1%} is below {settings.min_balanced_accuracy:.1%}; collect clearer calibration holds"
            )
        rules.append(_fit_rule(holds, spec, settings))
        reports.append(
            {
                "rule": spec.name,
                "feature": spec.feature,
                "holds": counts,
                "excluded": excluded,
                "validation": "leave_session_out" if session_validation else "leave_hold_out",
                "balanced_accuracy": accuracy,
                "recalls": recalls,
                "folds": folds,
            }
        )
    profile = CalibrationProfile(
        first.extraction,
        tuple(rules),
        evaluation,
        first.capture,
        tuple(recording.session_id for recording in recordings),
        "balanced-hold-threshold-v1",
    )
    return profile, {
        "rules": reports,
        "timing_policy": "provisional, configured; not learned",
        "validation_note": "Raw held-out feature thresholds; temporal replay is reported separately.",
    }
