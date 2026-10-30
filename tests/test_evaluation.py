"""Elapsed-time smoothing and decision-state behavior without camera hardware."""

import math
from dataclasses import replace

import pytest

from calibration_profile import CalibrationProfile, CaptureIdentity, EvaluationSettings, FeatureReference, FeatureRule
from evaluation import PostureEvaluator
from features import FeatureConfig, PostureFeatures
from smoothing import SignalSmoother, SmoothingSettings


def profile():
    return CalibrationProfile(
        FeatureConfig(enabled=("head_offset_y", "mouth_shoulder_vertical"), temporal_enabled=False),
        (FeatureRule("head", "head_offset_y", FeatureReference(10, 1, 100), "decrease", 2, 1),),
        EvaluationSettings(0, 1, 0.5),
        CaptureIdentity(0, 640, 480, "a" * 64),
        ("test",),
        "test",
    )


def sample(seconds, value=10, **diagnostics):
    return PostureFeatures(round(seconds * 1e9), {"head_offset_y": value, **diagnostics}, {})


def test_ema_uses_time_constant_and_resets_missing_values():
    smoother = SignalSmoother(SmoothingSettings("ema", 1, 2))
    assert smoother.update(0, 0) == 0
    assert smoother.update(round(math.log(2) * 1e9), 1) == pytest.approx(0.5)
    assert smoother.update(1_000_000_000, None) is None
    assert smoother.update(1_100_000_000, 9) == 9
    assert smoother.update(4_000_000_000, 2) == 2  # Long gap: no interpolation.
    with pytest.raises(ValueError):
        smoother.update(4_000_000_000, 2)


def test_moving_average_window_and_disabled_smoothing():
    smoother = SignalSmoother(SmoothingSettings("moving_average", 1, 2))
    assert smoother.update(0, 0) == 0
    assert smoother.update(500_000_000, 2) == 1
    assert smoother.update(1_000_000_000, 4) == 2  # Inclusive left boundary.
    assert smoother.update(1_500_000_000, 6) == 4
    raw = SignalSmoother(SmoothingSettings("none", 10, 2))
    raw.update(0, 0)
    assert raw.update(1_000_000_000, 7) == 7


def test_hysteresis_requires_continuous_elapsed_persistence():
    evaluator = PostureEvaluator(profile(), method="none", max_gap_seconds=10)
    assert evaluator.update(sample(0)).status == "UNKNOWN"
    assert evaluator.update(sample(0.5)).status == "WITHIN_REFERENCE"
    assert evaluator.update(sample(1, 8)).status == "WITHIN_REFERENCE"
    assert evaluator.update(sample(1.5, 8)).rules[0].pending_seconds == 0.5
    assert evaluator.update(sample(2, 8)).status == "DEVIATION"
    evaluator.update(sample(2.1, 9))  # Start clearing.
    assert evaluator.update(sample(2.4, 8.5)).status == "DEVIATION"  # Deadband cancels pending clear.
    evaluator.update(sample(2.5, 9))
    assert evaluator.update(sample(3, 9)).status == "WITHIN_REFERENCE"


def test_missing_active_input_and_gaps_reset_state_but_diagnostics_do_not():
    evaluator = PostureEvaluator(
        profile(), method="none", max_gap_seconds=0.5, display_features=("mouth_shoulder_vertical",)
    )
    evaluator.update(sample(0))
    assert evaluator.update(sample(0.5)).status == "WITHIN_REFERENCE"
    assert evaluator.update(sample(0.6, None)).status == "UNKNOWN"
    assert evaluator.update(sample(0.7)).status == "UNKNOWN"
    assert evaluator.update(sample(1.2)).status == "WITHIN_REFERENCE"
    assert evaluator.update(sample(3)).status == "UNKNOWN"  # Gap is not sustained reference time.
    with pytest.raises(ValueError):
        evaluator.update(sample(3))


@pytest.mark.parametrize("combination,status", [("any", "DEVIATION"), ("all", "WITHIN_REFERENCE")])
def test_multiple_rules_obey_explicit_combination(combination, status):
    p = profile()
    mouth = replace(p.rules[0], name="mouth", feature="mouth_shoulder_vertical")
    evaluator = PostureEvaluator(
        replace(p, rules=(*p.rules, mouth), combination=combination), method="none", max_gap_seconds=10
    )
    evaluator.update(sample(0, mouth_shoulder_vertical=7))
    assert evaluator.update(sample(1, mouth_shoulder_vertical=7)).status == status
