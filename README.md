# Posture Tracker

A local webcam application with an explicit capture → pose → diagnostics → optional
preview pipeline. See [ARCHITECTURE.md](ARCHITECTURE.md) for the module design.

## Development setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from
the project root:

```powershell
uv sync --locked
uv run pre-commit install
```

uv manages `.venv` and the Python version specified in `.python-version`.
`pyproject.toml` contains runtime dependencies, development tools, pytest settings,
Ruff settings, and ty settings. Commit `uv.lock` to keep dependency versions reproducible.
No manual virtual environment activation is needed.

## Run

```powershell
uv run src/app.py
uv run src/app.py --preview --duration 10
```

Pose inference requires the pretrained Lite model at
`models/pose_landmarker_lite.task`. The model binary is excluded from Git and is
not downloaded by uv or the application.

## Calibration recordings

```powershell
uv run src/app.py --calibrate
uv run src/app.py --calibrate --duration 60 --output data/calibration
```

Calibration mode automatically opens the preview and loads the project-level
[`features.toml`](features.toml). Edit that file to select features, change confidence
thresholds, choose nose/eye midpoint as the head reference, or adjust temporal
settings. Invalid settings fail before opening the camera. Ordinary capture and
preview modes do not create recordings.

Focus the preview window to use **V** (valid), **I** (invalid), **U** (unlabeled),
and **Q/Esc** (finish). Labels persist until changed and start as unlabeled.
A label change applies from the next captured frame, never retroactively to the
displayed frame. Repeated presses of the current label do not create new segments.
Use U while changing posture, then V/I once settled. Window close, duration expiry,
and Ctrl+C also finish the recording and flush buffered samples.

The preview shows the label, base-feature measurement quality, elapsed seconds,
and saved/observed frame counts. Measurement quality and your posture judgment
are separate: an unusable measurement can still carry your manual label.

Each session creates a unique directory under the output path:

```text
<UTC-start>-<unique-id>/
  session.json
  samples/part-000000.parquet
  samples/part-000001.parquet
  label_events/part-000001.parquet
```

Sample files contain one row per processed frame: original monotonic capture
timestamp, elapsed nanoseconds, frame sequence, session/label segment IDs, label,
image size, detection/measurement flags, all nine tracked landmarks with raw
normalized x/y/model-relative z and confidence, and configured feature columns.
Missing coordinates and unavailable features are null, not zero. Each feature
has a `<name>__reason` column; a null reason means available. No images are saved.

Raw landmarks and derived features share a row so they remain aligned. Raw data
is not filtered, smoothed, or overwritten by feature calculation, and can be used
to recompute different features offline. `session.json` stores schema/feature
versions, effective settings, feature definitions/units, model path and SHA-256,
UTC start/end times, counts, and completion status. Monotonic timestamps are for
within-session timing; do not compare their absolute values across sessions.

Label-event parts store each change's monotonic input timestamp and first affected
frame sequence. A change with no following frame has a null sequence. Empty
sessions have no sample parts; sessions without label changes have no event parts.

Buffers flush every 5 capture seconds or 300 frames and on orderly shutdown.
Each part is written to a temporary file and renamed when complete. Read only
`*.parquet` files; ignore `.tmp` leftovers after a crash. Previously completed
parts remain readable, but an abrupt process/power failure can lose buffered
samples. Sample and event parts are individually atomic, not one transaction.
Use per-frame labels as the authoritative annotations if a crash interrupts an
event write. A session left with `complete=false` requires review; handled camera
or inference errors flush the remaining samples and keep this flag false.

Read a completed nonempty sample dataset with PyArrow:

```python
from pathlib import Path
import pyarrow.parquet as pq

session = Path("data/calibration/<session-directory>")
samples = pq.read_table(sorted((session / "samples").glob("*.parquet")))
```

See [`FEATURES.md`](FEATURES.md) for formulas, signs, units, limitations, temporal
behavior, and instructions for adding features. Each definition also lives beside
its named calculation in `src/features.py`; the registry derives feature names
from those functions. Regenerate the reference after changing a definition:

```powershell
uv run python -c "import sys; sys.path.insert(0, 'src'); from pathlib import Path; from features import feature_reference_markdown; Path('FEATURES.md').write_text(feature_reference_markdown(), encoding='utf-8')"
```

This stage collects labeled calibration data and features. Reference fitting,
automatic posture classification, and a recording-quality analysis report remain
future work.

## Checks

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv check --locked
uv run pre-commit run --all-files
```

`uv check` runs ty on `src`; tests are excluded from type checking.

The pre-commit hooks run Ruff linting with safe fixes, formatting, and `uv check --locked`.
They use `uv run --locked` so hooks and manual checks use the same locked Ruff
version. If a hook edits a file, review and stage the changes before committing
again. Each developer installs the hooks once after cloning.

Add runtime dependencies with `uv add <package>` and development tools with
`uv add --dev <package>`. Run `uv lock --upgrade` for an intentional dependency
upgrade, then run the checks and commit both `pyproject.toml` and `uv.lock` as
applicable.
