"""Replay the exact live evaluator over recordings; no camera or GUI required."""

from dataclasses import dataclass

from calibration_data import CalibrationRecording
from calibration_profile import CalibrationProfile
from evaluation import PostureEvaluator, PostureResult


@dataclass(frozen=True)
class ReplayReport:
    session_id: str
    labeled_frames: int
    unknown_labeled_frames: int
    false_warning_episodes: int
    false_warning_frames: int
    missed_invalid_holds: int
    detection_delays_seconds: tuple[float, ...]
    recovery_delays_seconds: tuple[float, ...]


def replay_recording(
    recording: CalibrationRecording, profile: CalibrationProfile, method="ema", max_gap_seconds=0.5
) -> tuple[tuple[PostureResult, ...], ReplayReport]:
    profile.check_compatibility(recording.capture)
    if recording.extraction != profile.extraction:
        raise ValueError("Replay extraction settings differ from the profile")
    evaluator = PostureEvaluator(profile, method, max_gap_seconds)
    results = tuple(evaluator.update(sample.features) for sample in recording.samples)
    labeled = unknown = false_frames = false_episodes = 0
    previous_false = False
    previous_segment = None
    for sample, result in zip(recording.samples, results, strict=True):
        is_labeled = sample.label in ("valid", "invalid")
        labeled += is_labeled
        unknown += is_labeled and result.status == "UNKNOWN"
        false = sample.label == "valid" and result.status == "DEVIATION"
        false_frames += false
        if false and (not previous_false or sample.segment_id != previous_segment):
            false_episodes += 1
        previous_false, previous_segment = false, sample.segment_id
    missed = 0
    detections, recoveries = [], []
    for segment in dict.fromkeys(sample.segment_id for sample in recording.samples):
        group = [
            (sample, result)
            for sample, result in zip(recording.samples, results, strict=True)
            if sample.segment_id == segment
        ]
        label = group[0][0].label
        if label not in ("valid", "invalid"):
            continue
        target = "DEVIATION" if label == "invalid" else "WITHIN_REFERENCE"
        reached = next((sample.features.timestamp_ns for sample, result in group if result.status == target), None)
        if reached is None:
            missed += label == "invalid"
        else:
            delay = (reached - group[0][0].features.timestamp_ns) / 1e9
            (detections if label == "invalid" else recoveries).append(delay)
    return results, ReplayReport(
        recording.session_id,
        labeled,
        unknown,
        false_episodes,
        false_frames,
        missed,
        tuple(detections),
        tuple(recoveries),
    )


def main() -> int:
    """Write an evaluation report and raw/smoothed traces for another recording."""
    import argparse
    import json
    from dataclasses import asdict, replace
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    from calibration_data import load_recording
    from calibration_profile import DEFAULT_PROFILE_PATH
    from tracking_config import TrackingSettings

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--output", type=Path, help="output directory; default: session/analysis")
    args = parser.parse_args()
    try:
        settings = TrackingSettings.load()
        profile = replace(CalibrationProfile.load(args.profile), evaluation=settings.evaluation)
        recording = load_recording(args.session)
        results, report = replay_recording(
            recording, profile, settings.smoothing.method, settings.smoothing.max_gap_seconds
        )
        output = args.output or args.session / "analysis"
        output.mkdir(parents=True, exist_ok=True)
        summary = {
            **asdict(report),
            "profile_id": profile.profile_id,
            "source_used_for_fitting": recording.session_id in profile.source_sessions,
            "smoothing_method": settings.smoothing.method,
            "evaluation_settings": asdict(profile.evaluation),
            "delay_note": "Delays start at labeled holds, not at the unknown physical motion onset.",
        }
        (output / "posture_replay.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        trace = []
        for sample, result in zip(recording.samples, results, strict=True):
            row = {
                "timestamp_ns": result.timestamp_ns,
                "sequence": sample.sequence,
                "label": sample.label,
                "segment_id": sample.segment_id,
                "status": result.status,
                "reason": result.reason,
            }
            for name, signal in result.signals.items():
                row[f"{name}__raw"] = signal.raw
                row[f"{name}__smoothed"] = signal.smoothed
            trace.append(row)
        pq.write_table(pa.Table.from_pylist(trace), output / "posture_replay.parquet", compression="zstd")
        print(json.dumps(summary, indent=2))
        print(f"Saved report and raw/smoothed traces to {output}")
        return 0
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Cannot replay recording: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
