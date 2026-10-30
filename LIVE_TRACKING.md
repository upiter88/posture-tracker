# Live posture tracking

## Quick start

A profile has been fitted from `20261027T213729-54efcdb3` and saved locally to
`data/profiles/current.json`. It is personal calibration data and is Git-ignored.
With the same camera setup, run:

```powershell
uv run src/app.py --track
```

The dashboard shows the camera, current status, raw and smoothed feature traces,
entry/clearing thresholds, and pending-state progress. Raw traces are gray;
smoothed traces are yellow. Entry thresholds are red; clearing thresholds are
green. Traces use the feature's native units, not standardized score units.

- **1:** show/use unsmoothed signals.
- **2:** exponential smoothing (default).
- **3:** trailing moving average.
- **Q / Esc / close window:** stop and release camera/model resources.

Changing the filter clears plot history and restarts decision timing. The same
smoothed values shown in the graphs drive evaluation. Raw values remain available
for comparison; they do not bypass smoothing or trigger separate warnings.

Use `--camera 1`, `--duration 30`, or `--profile path/to/profile.json` as needed.
`--track` and `--calibrate` are separate modes. Existing default capture and
calibration workflows remain available. The existing calibration label controls
are defined by `recording.py`; filter-selection keys apply only in tracking mode.

## Configuration

Edit `tracking.toml`. It has six short sections:

| Section | Purpose | When changes apply |
|---|---|---|
| `[[rules]]` | Active feature, direction, reference scale floor | Refit profile |
| `[fitting]` | Trimming, minimum usable holds/coverage, validation acceptance, hysteresis ratio | Refit profile |
| `[smoothing]` | EMA/moving-average/raw choice, duration, maximum tracking gap | Next live/replay run |
| `[evaluation]` | Entry and clearing persistence in seconds | Next live/replay run |
| `[display]` | Up to four raw/smoothed outputs and history duration | Next live run |
| `[sound]` | Enable posture tones, volume, and repeat interval | Next live run |

On Windows, tracking plays a short tone when the evaluator enters `DEVIATION`,
then repeats every `sound.repeat_seconds` (default 5 seconds) while it persists.
The existing entry delay applies before the first tone. Recovery, UNKNOWN status,
filter changes, and exit stop playback. Playback runs asynchronously so it does
not pause capture or inference. Replay and calibration do not play alerts.

Set `sound.volume` from `0.0` (mute) to `1.0` (maximum tone amplitude); the default
is `0.3`. System and application mixer volume also affect perceived loudness.
Set `sound.enabled = false` to disable sound. The repeat interval must be at least
0.3 seconds (the tone duration). Settings apply on the next tracking run and do
not require refitting. Older configurations without `[sound]` use these defaults.
Sound uses Windows' built-in playback; other platforms or audio playback failures
log a warning and continue visual tracking without sound.

Example: add `mouth_shoulder_vertical` to `display.features` to inspect it without
making it an active decision rule. Adding a `[[rules]]` block makes a feature
active when fitting the next profile. Rule/output names are checked against the
existing extraction configuration; no feature-specific dispatch is required.
All selected outputs must have been recorded. Calibration extraction settings
remain in `features.toml`; the profile preserves those used in its recordings.

**Settings precedence:** the profile supplies extraction settings, reference
statistics, active rules, thresholds, and capture identity. Live/replay entry
points explicitly override its evaluation timing with `tracking.toml` so you can
experiment without refitting references. The profile's timing values preserve the
initial fitting/replay policy. The camera-independent `PostureEvaluator` API uses
the supplied profile's settings directly. Replay reports record effective timing
and smoothing method for reproducibility.

## Fit a profile

```powershell
uv run src/fit_profile.py
uv run src/fit_profile.py --session data/calibration/<session-id>
uv run src/fit_profile.py --session data/calibration/<first> --session data/calibration/<second>
```

Without `--session`, the newest recording by metadata start time is selected.
The fitter rejects incomplete sessions, inconsistent counts/timestamps, unknown
labels, incompatible extraction/model/capture settings, insufficient usable holds,
and insufficient blocked validation accuracy. It does not silently fall back to
an older recording if the newest one fails. `--output` selects another profile
path; the default replaces `data/profiles/current.json` atomically.

Each feature is fitted independently:

1. Remove unlabeled segments and trim both ends of labeled holds.
2. Keep holds with enough finite measurements, coverage, and duration after trimming.
   Consecutive usable samples must not exceed the recording extraction settings'
   `max_gap_seconds`; missing time cannot substitute for a stable measured hold.
3. Reference centre = median of valid-hold medians. Each hold has equal influence.
4. Scale = max(1.4826 times the median of each valid hold's median absolute
   deviation about the shared centre, configured `min_scale`). The floor uses
   native feature units. This is a robust spread estimate, not a physical length
   or a standard error; it includes between-hold reference variation.
5. Fit a score-space threshold maximizing balanced accuracy with equal weight per
   hold within each label. The configured direction determines the score.
6. Validate by holding out whole sessions when multiple sessions are supplied,
   otherwise whole posture holds. Each fold fits its own reference, scale, and
   threshold using training data only. Both labels must remain in training.
7. Reject rules below the acceptance limit; otherwise fit the final rule on all
   retained holds. Clearing threshold is entry threshold times `clear_ratio`.

A `.fit.json` beside the profile records all folds, excluded holds, acceptance
settings, and temporal replay results. Smoothing durations, persistence, and the
clearing ratio are explicit provisional policy, not automatically optimized.
Multiple feature scores are validated independently; the combined rule set still
requires separate validation. Fitting relies on deliberate labels and settling
intervals, rather than an additional automatic motion-labeling rule.

## Evaluation behavior

- `WITHIN_REFERENCE`: reference posture has been established continuously for the
  clearing duration, or a previously established reference remains inside hysteresis.
- `DEVIATION`: the configured deviation has persisted for the entry duration.
- `UNKNOWN`: initial settling, an ambiguous initial score between thresholds,
  unavailable active measurements, or recovery after a tracking gap.

Scores are `(centre-value)/scale` for decrease, `(value-centre)/scale` for increase,
and `abs(value-centre)/scale` for either direction. Entry occurs at or above the
entry threshold; clearing occurs at or below the clearing threshold. Moving into
the hysteresis band cancels a pending change and retains an established state.
At startup the band has no established state, so it stays UNKNOWN until the
signal clearly establishes either reference or deviation. Missing active inputs
produce UNKNOWN immediately and clear filter/state memory. Missing diagnostic
features do not block active rules. Durations use capture timestamps, not FPS.

For multiple rules, all active measurements and states must first be established;
then the profile's explicit `any` or `all` combination applies. The fitter currently
creates `any` profiles. Switching filters or exceeding `max_gap_seconds` restarts
settling; time spent without usable tracking does not count toward persistence.

## Smoothing choices

`ema`: `alpha=1-exp(-dt/tau)` and `y=previous_y+alpha*(x-previous_y)`.
`seconds` is the time constant tau. Actual elapsed time handles variable FPS.

`moving_average`: the sample-weighted mean of observed values in the inclusive
trailing `[t-window, t]` window. `seconds` is window length. It starts with the
available partial window; samples are not interpolated or time-weighted.

`none`, or duration zero: raw values pass through directly. All modes still use
hysteresis and persistence. Missing/nonfinite measurements reset smoothers rather
than extrapolating a potentially stale posture.

EMA and moving average are transparent baselines for comparing lag and noise.
A Kalman filter is intentionally deferred: it would require a defined motion
model and process/measurement-noise tuning. There is no evidence yet that that
extra model improves these signals. The smoothing module is independent so another
filter can be added after comparing replay results.

## Replay another recording

```powershell
uv run src/replay.py --session data/calibration/<session-id>
```

This uses the same evaluator as live tracking and writes
`analysis/posture_replay.json` plus `analysis/posture_replay.parquet` into that
recording. The Parquet file includes frame identity, labels, statuses, and active
raw/smoothed signals. `--profile` and `--output` select alternate inputs/output.

The summary counts UNKNOWN labeled frames, false-warning episodes/frames, missed
invalid holds, and detection/recovery delays. Delays start at labeled hold onset,
not at the actual physical motion onset; transitions labeled U can allow a state
to settle before the next labeled hold, giving a delay of zero. A report explicitly
marks whether the recording was used for fitting. Source replay is not an
independent accuracy estimate. Record new desk-work sessions and camera positions
before treating calibration replay as evidence of everyday warning reliability.

## Modules

| Module | Responsibility |
|---|---|
| `calibration_data.py` | Read and validate saved recordings |
| `calibration.py` | Fit references/thresholds and perform blocked validation |
| `fit_profile.py` | Publish a fitted profile and its report |
| `smoothing.py` | Standalone time-based signal filters |
| `evaluation.py` | Scores, hysteresis, persistence, UNKNOWN handling |
| `replay.py` | Offline evaluation and trace/report export |
| `live_preview.py` | Camera + plots, without inference or posture decisions |
| `live_tracking.py` | Resource ownership and live orchestration |
| `sound_alert.py` | Nonblocking posture tones and repeat timing |
| `tracking_config.py` | Read the small fitting/live configuration |

`features.py`, capture, pose tracking, and the existing preview are reused.
`app.py` only dispatches the new tracking mode. The model hash, resolution, and
camera index must match the profile; physical camera movement still requires
manual recalibration because it cannot be detected from those identifiers alone.
