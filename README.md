# Posture Tracker

The first milestone captures webcam frames locally and prints diagnostics. No preview window, image recording, or posture classification is included.

## Setup and run (PowerShell)

Use Python 3.12 and a project virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Stop with `Ctrl+C`, or run a bounded check:

```powershell
.\.venv\Scripts\python.exe app.py --duration 10
.\.venv\Scripts\python.exe app.py --camera 1 --duration 10
```

Camera 0 is the default. Capture uses the device's default resolution and frame rate, with automatic OpenCV backend selection. Logs show actual frame dimensions, observed loop FPS, and monotonic timestamps. The first reporting interval includes camera warm-up. The optional duration starts after opening the camera and is checked between reads; it cannot interrupt a blocked camera driver call.

If the camera cannot open, check its index, Windows camera permissions for desktop apps, and whether another application is using it. Read failures stop capture with exit code 1. Frames are BGR NumPy arrays; timestamps mark read completion rather than exposure time.

The capture module exposes `CameraCapture` as a context manager and returns `Frame` values. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline and future milestones.

Only one OpenCV package variant should be installed in an environment. This milestone uses `opencv-python-headless` because it needs no GUI. Revisit dependency compatibility when adding MediaPipe.

## Verification

Install development dependencies and run the camera-independent pytest checks in `tests/`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -v
```

These checks cover failed opening, read failure, interruption cleanup, frame metadata, and reopening. Hardware delivery and observed FPS still need a successful local camera run. The development sandbox could not open camera 0, so physical-camera behavior has not yet been verified. A duration limit is checked between reads and cannot interrupt a blocked open/read operation.
