"""Known geometry, normalization, feature availability, and elapsed-time motion."""

import math
from dataclasses import replace
from pathlib import Path

import pytest

from features import FEATURES, FeatureConfig, FeatureExtractor, feature_reference_markdown
from pose import PoseLandmark, PoseObservation


def observation(timestamp_ns=0, *, points=None, scale=1.0, shift=(0, 0)):
    pixels = {
        "left_shoulder": (700, 350),
        "right_shoulder": (300, 350),
        "left_eye": (550, 150),
        "right_eye": (450, 150),
        "left_ear": (600, 175),
        "right_ear": (400, 175),
        "nose": (500, 175),
        "mouth_left": (525, 225),
        "mouth_right": (475, 225),
    }
    if points:
        pixels.update(points)
    landmarks = {
        name: PoseLandmark((x * scale + shift[0]) / 1000, (y * scale + shift[1]) / 500, 0, 0.9, 0.9)
        for name, (x, y) in pixels.items()
    }
    return PoseObservation(timestamp_ns, 1000, 500, landmarks)


def test_known_geometry_and_units():
    result = FeatureExtractor(FeatureConfig(temporal_enabled=False)).extract(observation())
    assert result.measurement_valid
    assert result.values == pytest.approx(
        {
            "shoulder_angle_deg": 0,
            "head_offset_x": 0,
            "head_offset_y": 0.5,
            "mouth_shoulder_vertical": 0.3125,
            "eye_ear_vertical": 0.0625,
            "shoulder_width": 0.4,
            "facial_asymmetry": 0,
        }
    )


def test_pixel_aspect_ratio_angle_and_signed_midpoint_distance():
    pose = observation(points={"left_shoulder": (700, 450)})
    values = FeatureExtractor().extract(pose).values
    width = math.hypot(400, 100)
    assert values["shoulder_angle_deg"] == pytest.approx(math.degrees(math.atan2(100, 400)))
    assert values["shoulder_width"] == pytest.approx(width / 1000)
    assert values["mouth_shoulder_vertical"] == pytest.approx((400 - 225) / width)


def test_normalized_features_invariant_to_translation_and_scale():
    config = FeatureConfig(temporal_enabled=False)
    original = FeatureExtractor(config).extract(observation()).values
    moved = FeatureExtractor(config).extract(observation(scale=0.7, shift=(70, 30))).values
    for name in FEATURES:
        expected = original[name] * 0.7 if name == "shoulder_width" else original[name]
        assert moved[name] == pytest.approx(expected)


def test_head_reference_and_anatomical_asymmetry():
    pose = observation(points={"nose": (550, 175)})
    config = FeatureConfig(head_reference="nose", temporal_enabled=False)
    values = FeatureExtractor(config).extract(pose).values
    assert values["head_offset_x"] == pytest.approx(0.125)
    assert values["head_offset_y"] == pytest.approx(0.4375)
    # Anatomical left ear is image-right in this front-facing example.
    assert values["facial_asymmetry"] == pytest.approx(-0.5)


def test_quality_is_per_feature_and_tiny_shoulders_are_unusable():
    pose = observation()
    landmarks = dict(pose.landmarks)
    landmarks["left_shoulder"] = replace(landmarks["left_shoulder"], visibility=0.1)
    result = FeatureExtractor().extract(replace(pose, landmarks=landmarks))
    assert not result.measurement_valid
    assert result.values["facial_asymmetry"] == 0
    assert result.values["shoulder_width"] is None
    assert result.reasons["shoulder_width"] == "low_confidence:left_shoulder"
    collapsed = observation(points={"left_shoulder": (300, 350)})
    result = FeatureExtractor().extract(collapsed)
    assert result.reasons["head_offset_x"] == "shoulder_width_too_small"


@pytest.mark.parametrize("coordinate,reason", [(float("nan"), "nonfinite"), (1.2, "out_of_frame")])
def test_unusable_landmarks_preserve_other_features(coordinate, reason):
    pose = observation()
    landmarks = dict(pose.landmarks)
    landmarks["mouth_left"] = replace(landmarks["mouth_left"], x=coordinate)
    result = FeatureExtractor().extract(replace(pose, landmarks=landmarks))
    assert result.reasons["mouth_shoulder_vertical"] == f"{reason}:mouth_left"
    assert result.values["head_offset_y"] == 0.5


def test_motion_uses_real_time_window_and_resets_on_missing_or_gap():
    extractor = FeatureExtractor(FeatureConfig(enabled=("head_offset_x",)))
    first = extractor.extract(observation())
    assert first.measurement_valid
    assert first.reasons["head_offset_x__std"] == "temporal_warmup"
    second = extractor.extract(observation(200_000_000, points={"left_eye": (590, 150), "right_eye": (490, 150)}))
    assert second.values["head_offset_x__velocity"] == pytest.approx(0.5)
    assert second.values["head_offset_x__std"] == pytest.approx(0.05)
    missing = extractor.extract(PoseObservation(300_000_000, 1000, 500, {}))
    assert missing.values["head_offset_x__std"] is None
    recovered = extractor.extract(observation(400_000_000))
    assert recovered.reasons["head_offset_x__velocity"] == "temporal_warmup"
    after_gap = extractor.extract(observation(1_000_000_000))
    assert after_gap.reasons["head_offset_x__velocity"] == "temporal_warmup"
    stationary = extractor.extract(observation(1_100_000_000))
    assert stationary.values["head_offset_x__std"] == 0
    assert stationary.values["head_offset_x__velocity"] == 0
    with pytest.raises(ValueError, match="strictly increase"):
        extractor.extract(observation(1_100_000_000))


def test_motion_discards_samples_outside_window():
    extractor = FeatureExtractor(FeatureConfig(enabled=("head_offset_x",), window_seconds=0.1))
    extractor.extract(observation())
    result = extractor.extract(observation(200_000_000))
    assert result.reasons["head_offset_x__std"] == "temporal_warmup"


@pytest.mark.parametrize(
    "settings",
    [
        {"enabled": ("typo",)},
        {"enabled": ()},
        {"enabled": ("shoulder_width", "shoulder_width")},
        {"window_seconds": 0},
        {"max_gap_seconds": float("inf")},
        {"min_presence": 1.1},
        {"min_visibility": True},
        {"head_reference": "chin"},
        {"temporal_enabled": "true"},
    ],
)
def test_invalid_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        FeatureConfig(**settings)


def test_project_config_and_generated_documentation_are_current():
    root = Path(__file__).resolve().parents[1]
    config = FeatureConfig.load(root / "features.toml")
    assert config.enabled == tuple(FEATURES)
    assert (root / "FEATURES.md").read_text(encoding="utf-8") == feature_reference_markdown()
    for name, definition in FEATURES.items():
        assert definition.calculate.__name__ == name
        assert "Definition:" in definition.documentation
