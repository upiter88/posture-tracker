"""Fit and publish a profile from recorded holds; also replay temporal decisions.

Usage: uv run src/fit_profile.py
       uv run src/fit_profile.py --session data/calibration/<id> [--session ...]
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from calibration.calibration import fit_profile
from calibration.calibration_data import latest_recording, load_recording
from calibration.calibration_profile import DEFAULT_PROFILE_PATH
from calibration.replay import replay_recording
from tracking.tracking_config import TrackingSettings

DATA_ROOT = Path(__file__).resolve().parent.parent / "data/calibration"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, action="append", help="source session; repeat for multiple sessions")
    parser.add_argument("--output", type=Path, default=DEFAULT_PROFILE_PATH)
    args = parser.parse_args()
    try:
        settings = TrackingSettings.load()
        paths = args.session or [latest_recording(DATA_ROOT)]
        recordings = [load_recording(path) for path in paths]
        profile, report = fit_profile(recordings, settings.rules, settings.fitting, settings.evaluation)
        report["profile_id"] = profile.profile_id
        report["smoothing_method"] = settings.smoothing.method
        report["fitting_settings"] = asdict(settings.fitting)
        report["temporal_replay"] = [
            asdict(
                replay_recording(recording, profile, settings.smoothing.method, settings.smoothing.max_gap_seconds)[1]
            )
            for recording in recordings
        ]
        report["replay_note"] = "Replay of source calibration recordings, not an independent test."
        # Finish all validation and replay before replacing the published profile.
        profile.save(args.output)
        report_path = args.output.with_suffix(".fit.json")
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved profile: {args.output}\nFit/replay report: {report_path}")
        for rule in report["rules"]:
            print(f"{rule['rule']}: {rule['validation']} balanced accuracy {rule['balanced_accuracy']:.1%}")
        for replay in report["temporal_replay"]:
            print(
                f"{replay['session_id']}: {replay['false_warning_episodes']} false-warning episodes, "
                f"{replay['missed_invalid_holds']} missed invalid holds, "
                f"{replay['unknown_labeled_frames']} unknown labeled frames"
            )
        return 0
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Cannot fit profile: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
