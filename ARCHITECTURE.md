# Posture Tracker — Software Architecture

## Purpose and scope

Build a local Python application that uses a webcam to detect sustained deviations from a user's calibrated reference posture. Develop and validate one module at a time, beginning with camera capture.

The initial application has an optional live camera preview, with no configuration framework, cloud service, or custom model training. It runs in a single process with a synchronous processing loop and prints results to the console. A reference posture is a user-selected baseline, not a medical assessment of correct posture.

## Initial assumptions

- One seated user and one fixed, front-facing webcam.
- The user's head and shoulders are visible; hips may be outside the frame.
- Processing initially targets approximately 10–15 frames per second, subject to measurement on the actual hardware.
- Camera position remains stable after calibration; moving it requires recalibration.
- Images are processed locally and are not saved by default.

A front-facing camera is the starting constraint. Forward slouching and neck displacement may be difficult to infer reliably from this view and require separate validation.

## Module overview

| Module | Responsibility | Input | Output |
| --- | --- | --- | --- |
| `capture` | Open the webcam, read frames, attach timestamps, release the device | Camera index and a few optional capture parameters | `Frame` |
| `pose` | Run a pretrained pose model and expose relevant landmarks and confidence | `Frame` | `PoseObservation` |
| `features` | Calculate normalized geometric measurements and determine measurement validity | `PoseObservation` | `PostureFeatures` |
| `calibration` | Collect valid reference samples and calculate a stable baseline | `PostureFeatures` during calibration | `ReferencePosture` |
| `evaluation` | Compare features with the baseline, smooth measurements, and apply persistence rules | Features, baseline, timestamp | `PostureResult` |
| `reporting` | Print status changes and diagnostic information | Calibration progress or `PostureResult` | Console output |
| `app` | Own module lifetimes and coordinate the processing loop | Small set of constants or command-line arguments | Application execution and exit status |
| `preview` | Optionally display video with pose landmarks and handle window shutdown | `Frame`, `PoseObservation` | Continue/stop decision |

These are logical boundaries. Introduce files and types as each stage is implemented; empty modules and a plugin framework are unnecessary.

## Communication and data flow

```text
capture -> Frame -> pose -> PoseObservation -> features -> PostureFeatures
                                                              |
                                  during calibration ----------+-> calibration
                                                              |       |
                                  after calibration -----------+       v
                                                              | ReferencePosture
                                                              v       |
                                                           evaluation <+
                                                              |
                                                        PostureResult
                                                              |
                                                           reporting
```

`app` passes values directly between modules through function or method calls. Modules do not communicate through global state, queues, or an event bus. There are no background workers initially.

Use small dataclasses for shared values and pure functions for geometry. Keep OpenCV and MediaPipe details inside their respective adapters where practical.

### Proposed data contracts

- **`Frame`**: image array, sequence number, and monotonically increasing capture timestamp. Initially use OpenCV's BGR image format; conversion to the pose model's expected format belongs to `pose`. Downstream modules treat the image as read-only. The timestamp represents application capture time, not a guaranteed hardware exposure timestamp.
- **`PoseObservation`**: original frame timestamp in nanoseconds, image width/height, and a read-only mapping of named `PoseLandmark` values (`x`, `y`, `z`, visibility, presence). `detected` indicates a returned pose, not measurement validity; missing detections have an empty mapping. x increases rightward relative to image width, y downward relative to image height; z is model-relative depth, not metric distance. Left/right names describe the person's anatomical sides. Convert normalized image coordinates to aspect-correct coordinates before calculating angles.
- **`PostureFeatures`**: timestamp, named measurements, validity, and reasons for unavailable measurements. Normalize distances by shoulder width where applicable and reject unusable scale estimates.
- **`ReferencePosture`**: baseline values, sample count, and variation statistics from valid calibration samples. Keep this in memory initially.
- **`PostureResult`**: timestamp, status (`WITHIN_REFERENCE`, `DEVIATION`, or `UNKNOWN`), reasons, and relevant measured deviations.

Camera read failures are capture errors, not posture results. Missing or unreliable landmarks produce invalid measurements and ultimately `UNKNOWN`; they must never be interpreted as good posture.

## Module behavior

### Camera capture — first implementation

Use OpenCV's `VideoCapture` API through `opencv-contrib-python`, which also satisfies MediaPipe's OpenCV dependency. Install only this OpenCV distribution, since the OpenCV packages share the `cv2` namespace. The implemented `CameraCapture` component provides explicit open/read/close behavior and a context manager so the device is released on normal exit, exceptions, and keyboard interruption.

Start with the default camera index. Optional width, height, and frame-rate requests can be added only when needed; hardware may not honor them. Keep the interface small and avoid backend selection logic until an observed device issue requires it.

The command-line runner captures frames and estimates poses by default. The optional `--preview` flag displays each frame with selected landmarks through `src/preview.py`; the default remains console-only. `app` passes `Frame` and `PoseObservation` values directly to the preview and stops when it returns False (Q, Esc, or window close). `app` owns the model, camera, and optional preview through an exit stack. Distinguish an unavailable camera from a failure while reading an already opened camera, report a clear error, and exit cleanly. Do not add automatic reconnection initially.

### Pose estimation

`src/pose.py` implements a context-managed `PoseEstimator` using MediaPipe Pose Landmarker with the local Lite float16 version 1 model, one person, VIDEO mode, confidence thresholds of 0.5, and segmentation disabled. Model setup instructions are in `models/README.md`; the model binary is excluded from Git and is never downloaded at application startup.

`estimate(frame)` converts BGR to RGB without modifying the source image and returns the nose, eyes, ears, and shoulders with confidence information. The model receives milliseconds relative to the first frame. Capture timestamps must strictly increase; millisecond rounding collisions advance the model timestamp by 1 ms. Observations preserve the original nanosecond timestamp. Reopening the estimator resets the model clock.

No detection returns an empty observation, without carrying forward the previous pose. Model loading or inference failures raise `PoseError` and stop the pipeline. A returned pose may still contain unreliable landmarks: the feature module will decide measurement validity. The preview draws only in-frame landmarks with visibility and presence of at least 0.5; this display filter is not a posture decision.

### Feature extraction

Begin with shoulder tilt, sideways head displacement, and changes in head position relative to the shoulders. Define each feature's units, required landmarks, and sign convention during implementation. Avoid requiring hips for the initial upper-body prototype.

Validate feature behavior before deciding which measurements should drive posture classification. Apparent head movement may reflect distance changes or rotation rather than slouching.

### Calibration

Collect several seconds of valid measurements while the user holds their chosen reference posture. Reject incomplete observations and detect excessive movement before accepting a baseline. The exact duration and stability limits remain implementation decisions.

Calibration progress is separate from posture status. Evaluation begins only after a valid baseline exists.

### Evaluation and temporal state

Compare current features with the reference, smooth measurement noise, and require deviations to persist before changing status. Use elapsed time rather than frame counts so behavior remains consistent when processing speed varies.

Use hysteresis: separate thresholds for entering and clearing a deviation. Invalid observations produce `UNKNOWN` and reset pending persistence timers so missing frames cannot count toward a sustained deviation. Thresholds, smoothing strength, and durations are provisional until tested on recordings.

### Reporting and orchestration

Use standard-library logging for diagnostics and console status output. Report status changes and periodic diagnostics without printing every frame. `app` owns startup, calibration, the processing loop, error handling, and shutdown.

## Tools and design patterns

| Tool or pattern | Role |
| --- | --- |
| Python | Application runtime |
| `opencv-contrib-python` | Camera access, image conversion, and preview; satisfies MediaPipe's dependency |
| `mediapipe` | Pretrained pose landmark estimation |
| `numpy` | Geometry, calibration statistics, and smoothing |
| `dataclasses`, `logging`, `time` | Data contracts, diagnostics, and monotonic timing |
| `pytest` | Focused tests of geometry, validity, and temporal decisions when those modules exist |
| Processing pipeline | Clear staged transformation of camera frames into posture results |
| Adapter boundaries | Contain camera and pose-library dependencies |
| Pure functions | Make geometric calculations easy to inspect and test |
| Small state machines | Represent calibration and time-dependent posture decisions explicitly |
| Context-managed resources | Ensure the camera and model are released reliably |

Install dependencies incrementally. The pose milestone adds MediaPipe and its separately downloaded local model.

## Incremental development plan

1. **Camera capture:** implement the capture component and a console runner. Verify usable frames, timestamps, observed rate, clean interruption, device release, and a clear error for an unavailable camera.
2. **Pose estimation:** connect captured frames to MediaPipe and inspect landmark availability and inference rate without posture classification.
3. **Feature extraction:** calculate and log a small set of measurements; verify responses to intentional leaning and head movements. Add focused geometry tests.
4. **Calibration:** collect a stable baseline and verify handling of movement or insufficient valid samples.
5. **Temporal classification:** add reference comparisons, smoothing, persistence, hysteresis, and explicit unknown status. Test timing and tracking-loss behavior.
6. **Evaluation:** use explicitly recorded, labeled clips to measure false warnings, missed deviations, and detection delay; revise features and thresholds based on the results.

Complete and review each milestone before expanding the next. Beyond the optional camera preview, keep UI, extensive configuration, persistence, alerts, concurrency, and custom training outside the initial scope.

## Reference

- [MediaPipe Pose Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)

## Current development state

Milestone 2 is implemented with an explicit synchronous loop in `src/app.py`:

```text
camera.read() -> estimator.estimate(frame) -> diagnostics.report(frame, pose, timing)
                                                        -> preview.show(frame, pose) [optional]
```

Each stage is a direct call with a named intermediate value; there is no generic pipeline engine. Capture lives in `src/capture.py`, inference and contracts in `src/pose.py`, interval diagnostics in `src/reporting.py`, and rendering in `src/preview.py`. Diagnostics report overall processing FPS, detected poses per interval, mean inference latency, and the latest landmark visibility/presence. Preview renders on an image copy.

Run from the project root with `python src/app.py`, after dependency installation and model setup described in `models/README.md`. Pose estimation is always enabled; no pose or model CLI arguments were added. Existing `--camera`, `--duration`, and `--preview` options remain. `pytest.ini` adds `src/` to the test import path. Camera-independent tests cover capture, pose conversion and missing detections, timestamp handling, stage order, cleanup, diagnostics, and image preservation. Real model inference, upper-body tracking quality, and hardware processing rate still require verification. Feature extraction is the next implementation milestone after pose validation and review.
