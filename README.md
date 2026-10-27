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
and Ruff settings. Commit `uv.lock` to keep dependency versions reproducible.
No manual virtual environment activation is needed.

## Run

```powershell
uv run src/app.py
uv run src/app.py --preview --duration 10
```

Pose inference requires the pretrained Lite model at
`models/pose_landmarker_lite.task`. The model binary is excluded from Git and is
not downloaded by uv or the application.

## Checks

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pre-commit run --all-files
```

The pre-commit hooks run Ruff linting with safe fixes, followed by formatting.
They use `uv run --locked` so hooks and manual checks use the same locked Ruff
version. If a hook edits a file, review and stage the changes before committing
again. Each developer installs the hooks once after cloning.

Add runtime dependencies with `uv add <package>` and development tools with
`uv add --dev <package>`. Run `uv lock --upgrade` for an intentional dependency
upgrade, then run the checks and commit both `pyproject.toml` and `uv.lock` as
applicable.
