"""Profile round trips, compatibility, and selection independent of diagnostics."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from calibration_profile import (
    CalibrationProfile,
    CaptureIdentity,
    EvaluationSettings,
    FeatureReference,
    FeatureRule,
    ProfileError,
)
from features import FEATURES, FeatureConfig, FeatureExtractor, PostureFeatures, feature
from pose import PoseLandmark, PoseObservation


@pytest.fixture
def profile():
    return CalibrationProfile(
        extraction=FeatureConfig(),
        rules=(FeatureRule("head_height", "head_offset_y", FeatureReference(0.7, 0.05, 100), "decrease", 2, 1),),
        evaluation=EvaluationSettings(0.2, 2, 1),
        capture=CaptureIdentity(0, 640, 480, "a" * 64),
        source_sessions=("example-session",),
        method="manual-test-fixture",
    )


def test_round_trip_preserves_complete_extraction_and_rule_contract(profile, tmp_path):
    profile = replace(profile, extraction=replace(profile.extraction, min_presence=0.8, head_reference="nose"))
    path = tmp_path / "profiles/current.json"
    profile.save(path)
    loaded = CalibrationProfile.load(path)
    assert loaded == profile
    assert loaded.extraction.min_presence == 0.8
    assert loaded.required_features == ("head_offset_y",)
    assert loaded.rules[0].reference.scale == 0.05
    assert loaded.rules[0].enter_threshold == 2
    assert loaded.combination == "any"
    saved = json.loads(path.read_text())
    assert saved["extraction"]["enabled"] == list(profile.extraction.enabled)


def test_missing_diagnostics_do_not_block_selected_rule(profile):
    pose = PoseObservation(
        1,
        640,
        480,
        {
            "left_shoulder": PoseLandmark(0.3, 0.7, 0, 0.9, 0.9),
            "right_shoulder": PoseLandmark(0.7, 0.7, 0, 0.9, 0.9),
            "left_eye": PoseLandmark(0.45, 0.3, 0, 0.9, 0.9),
            "right_eye": PoseLandmark(0.55, 0.3, 0, 0.9, 0.9),
        },
    )
    sample = FeatureExtractor(profile.extraction).extract(pose)
    assert not sample.measurement_valid  # Mouth and ear diagnostics are missing.
    assert profile.measurements_available(sample)
    assert "facial_asymmetry" in profile.diagnostic_features
    assert not profile.measurements_available(PostureFeatures(2, {}, {}))
    assert not profile.measurements_available(PostureFeatures(2, {"head_offset_y": float("nan")}, {}))


def test_temporal_rule_resolves_registry_and_observes_warmup(profile, tmp_path):
    active = replace(profile.rules[0], feature="head_offset_y__std", direction="increase")
    temporal = replace(profile, rules=(active,))
    assert not temporal.measurements_available(PostureFeatures(1, {active.feature: None}, {}))
    assert temporal.measurements_available(PostureFeatures(2, {active.feature: 0.01}, {}))
    path = tmp_path / "temporal.json"
    temporal.save(path)
    assert CalibrationProfile.load(path) == temporal
    with pytest.raises(ProfileError, match="not an output"):
        replace(temporal, extraction=replace(temporal.extraction, temporal_enabled=False))


def test_select_remove_and_combine_features_without_parallel_feature_lists(profile):
    extraction = replace(profile.extraction, enabled=("head_offset_y", "mouth_shoulder_vertical"))
    mouth_rule = replace(profile.rules[0], name="mouth_height", feature="mouth_shoulder_vertical")
    selected = replace(profile, extraction=extraction, rules=(*profile.rules, mouth_rule), combination="all")
    assert selected.required_features == ("head_offset_y", "mouth_shoulder_vertical")
    with pytest.raises(ProfileError, match="not an output"):
        replace(selected, extraction=replace(extraction, enabled=("head_offset_y",)))
    with pytest.raises(ProfileError, match="unique"):
        replace(selected, rules=(*selected.rules, selected.rules[0]))


def test_new_registered_feature_needs_no_profile_code_change(profile, monkeypatch):
    # Register a real new output temporarily; restore the registry after the test.
    monkeypatch.setattr("features.FEATURES", dict(FEATURES))

    @feature(units="image widths", landmarks=("nose",))
    def test_nose_x(geometry):
        """Definition: nose pixel x / image width. Demonstration feature."""
        return geometry.point("nose")[0] / geometry.pose.image_width

    extraction = replace(profile.extraction, enabled=("test_nose_x",), temporal_enabled=False)
    rule = replace(profile.rules[0], feature="test_nose_x", name="new_feature")
    extended = replace(profile, extraction=extraction, rules=(rule,))
    assert CalibrationProfile.from_dict(extended.to_dict()) == extended
    pose = PoseObservation(1, 640, 480, {"nose": PoseLandmark(0.4, 0.3, 0, 1, 1)})
    assert FeatureExtractor(extraction).extract(pose).values == {"test_nose_x": 0.4}


@pytest.mark.parametrize(
    "changes",
    [
        {"scale": 0},
        {"scale": float("inf")},
        {"centre": float("nan")},
        {"sample_count": True},
    ],
)
def test_invalid_reference_rejected(profile, changes):
    with pytest.raises(ProfileError):
        replace(profile.rules[0].reference, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"enter_threshold": 0},
        {"clear_threshold": 2},
        {"clear_threshold": -1},
        {"enter_threshold": True},
        {"direction": "upwards"},
        {"name": ""},
    ],
)
def test_invalid_rule_rejected(profile, changes):
    with pytest.raises(ProfileError):
        replace(profile.rules[0], **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"rules": ()},
        {"source_sessions": ()},
        {"source_sessions": ("a", "a")},
        {"created_utc": "2026-10-28T12:00:00"},
        {"created_utc": "2026-10-28T12:00:00+01:00"},
        {"feature_version": 999},
        {"schema_version": True},
        {"combination": "majority"},
    ],
)
def test_invalid_profile_rejected(profile, changes):
    with pytest.raises(ProfileError):
        replace(profile, **changes)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_field",
        "missing_field",
        "missing_config",
        "unknown_output",
        "disabled_output",
        "future_schema",
        "future_features",
    ],
)
def test_malformed_or_incompatible_json_rejected(profile, mutation):
    data = profile.to_dict()
    if mutation == "unknown_field":
        data["typo"] = 1
    elif mutation == "missing_field":
        del data["rules"]
    elif mutation == "missing_config":
        del data["extraction"]["min_presence"]
    elif mutation == "unknown_output":
        data["rules"][0]["feature"] = "typo"
    elif mutation == "disabled_output":
        data["extraction"]["enabled"] = ["shoulder_width"]
    elif mutation == "future_schema":
        data["schema_version"] = 999
    elif mutation == "future_features":
        data["feature_version"] = 999
    with pytest.raises(ProfileError):
        CalibrationProfile.from_dict(data)


def test_model_and_resolution_compatibility(profile):
    profile.check_compatibility(profile.capture)
    for capture in (
        replace(profile.capture, image_width=1280),
        replace(profile.capture, model_sha256="b" * 64),
        replace(profile.capture, camera_index=1),
    ):
        with pytest.raises(ProfileError, match="capture mismatch"):
            profile.check_compatibility(capture)


@pytest.mark.parametrize("target", ["calibration_profile.Path.replace", "calibration_profile.os.fsync"])
def test_atomic_failed_publish_preserves_previous_profile(profile, tmp_path, target):
    path = tmp_path / "current.json"
    profile.save(path)
    old_bytes = path.read_bytes()
    updated = replace(profile, method="new-manual-revision")
    with patch(target, side_effect=OSError("disk failure")):
        with pytest.raises(ProfileError, match="Cannot save"):
            updated.save(path)
    assert path.read_bytes() == old_bytes
    assert not list(tmp_path.glob("*.tmp"))
    updated.save(path)
    assert CalibrationProfile.load(path) == updated


@pytest.mark.parametrize("payload", ["{broken", '{"schema_version":1,"schema_version":2}', '{"a": NaN}'])
def test_invalid_json_is_actionable(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload)
    with pytest.raises(ProfileError, match="Cannot load"):
        CalibrationProfile.load(path)


def test_missing_file_and_invalid_temporal_settings(profile, tmp_path):
    with pytest.raises(ProfileError, match="Cannot load"):
        CalibrationProfile.load(tmp_path / "missing.json")
    with pytest.raises(ProfileError):
        replace(profile.evaluation, enter_seconds=0)
    with pytest.raises(ProfileError):
        replace(profile.capture, image_width=True)


def test_documented_synthetic_example_is_current():
    path = Path(__file__).resolve().parents[1] / "examples/calibration_profile.json"
    example = CalibrationProfile.load(path)
    assert example.method == "illustrative-not-fitted"
    assert example.required_features == ("head_offset_y",)
    assert example.diagnostic_features == ("shoulder_width",)
