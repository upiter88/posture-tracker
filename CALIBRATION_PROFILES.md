# Calibration profiles

`src/calibration/calibration_profile.py` stores personal calibration data independently of
feature extraction, fitting, and live evaluation. `features.py` stays unchanged.

## The simple model

A profile contains five ordinary frozen dataclasses:

| Type | Purpose |
|---|---|
| `FeatureReference` | Centre, positive scale, and reference sample count |
| `FeatureRule` | Output name, reference, direction, and entry/clearing thresholds |
| `EvaluationSettings` | Smoothing and entry/clearing durations, in seconds |
| `CaptureIdentity` | Camera index, actual resolution, and model SHA-256 |
| `CalibrationProfile` | All the above, extraction settings, recording IDs, and versions |

**Extraction and active rules are separate.** `extraction.enabled` lists base
features to calculate. `rules` selects the outputs used for posture decisions.
Everything else is diagnostic. Rule names are checked against the existing
`FeatureConfig.columns`, including generated temporal outputs such as
`head_offset_y__std`. There is no separate list of supported feature names.

`measurements_available(sample)` checks only active output values. Missing ears
or mouth landmarks do not block a head-height rule if its own value is available.
An active temporal output remains unavailable while it warms up.

## JSON settings reference

This reference describes the loadable profile JSON, such as
[`examples/calibration_profile.json`](examples/calibration_profile.json) or
`data/profiles/current.json`. The companion `.fit.json` and replay JSON files are
reports, not profiles, and cannot be passed to `CalibrationProfile.load()`.

**Every field below is required in saved JSON**, including fields that have
Python constructor defaults. Unknown fields are rejected at every object level.
Numbers must be finite; JSON comments, trailing commas, duplicate keys, `NaN`,
and `Infinity` are rejected. Defaults below explain object construction, not
permission to omit fields from a file. Nested values must use the shown object
and array structure; for a complete file, start from the saved example.

### Which file controls live behavior?

| Setting | Effective source in `app.py --track` and the `replay.py` CLI |
|---|---|
| Extracted features, quality checks, temporal-feature calculation | Profile `extraction` |
| Active rules, personal references, thresholds, rule combination | Profile `rules` and `combination` |
| Expected camera, resolution, and model | Profile `capture` |
| Smoothing duration | `config/tracking.toml` `[smoothing].seconds`, overriding profile `evaluation.smoothing_seconds` |
| Entry and clearing durations | `config/tracking.toml` `[evaluation].enter_seconds` / `clear_seconds`, overriding the matching profile fields |
| Smoothing method and live gap reset | `config/tracking.toml` `[smoothing].method` / `max_gap_seconds`; live keys 1/2/3 can change method for that run |
| Displayed signals and plot history | `config/tracking.toml` `[display]` |

`CalibrationProfile.load()` itself applies no overrides and reads no TOML.
The live/replay entry points explicitly replace the three evaluation timing values
in memory; the JSON on disk is unchanged. When constructing `PostureEvaluator`
directly in Python, its supplied profile controls those durations; its `method`
and `max_gap_seconds` arguments default to `"ema"` and `0.5`.
Consequently, editing only `evaluation` in JSON will **not** change the timing
used by the standard live/replay commands. Edit `config/tracking.toml` for those runs.

### Top-level fields

| JSON field | Type / allowed values | Meaning and effect |
|---|---|---|
| `extraction` | Object; all fields below required | Snapshot of feature extraction settings. It replaces any need to read `config/features.toml` when using the profile. |
| `rules` | Nonempty array of rule objects | Active posture rules. Extracted outputs absent from these rules remain diagnostics. |
| `evaluation` | Object with three durations | Stored smoothing/persistence policy. Subject to the live/replay overrides above. |
| `capture` | Object with four identity fields | Expected capture/model setup; checked before live use and replay. These are compatibility constraints, not device/model selection commands. |
| `combination` | `"any"` or `"all"`; constructor default `"any"` | Once **all active inputs and rule states are established**, `any` reports deviation if at least one rule deviates; `all` requires every rule to deviate. Missing an active input or an unestablished rule produces `UNKNOWN` under either policy. The fitter currently creates `any` profiles. |
| `source_sessions` | Nonempty array of unique, nonblank strings | Recording IDs used to construct the profile. Provenance only; paths are not resolved and recordings need not exist on the loading machine. |
| `method` | Nonblank string | How the values were produced, e.g. `"balanced-hold-threshold-v1"` or `"illustrative-not-fitted"`. Descriptive provenance; changing it does not select a fitting algorithm or change evaluation. |
| `profile_id` | Nonblank string; constructor generates a UUID | Identity used in reports and for associating results with a calibration. It does not affect decisions. Uniqueness across files is a caller responsibility. |
| `created_utc` | ISO 8601 timestamp with an explicit zero UTC offset | Creation time, e.g. `"2026-10-29T12:00:00+00:00"` or a timestamp ending in `Z`. Constructor uses the current UTC time. A naive timestamp or nonzero offset is rejected. |
| `schema_version` | Integer, currently exactly `1` | Profile storage-format version. Unsupported versions are rejected; do not edit it to force an incompatible file to load. |
| `feature_version` | Integer, currently exactly `1` | Version of the implemented feature mathematics. Must equal `features.FEATURE_VERSION`; recalibrate after incompatible calculation changes. It is separate from the storage schema version. |

### `extraction`

These are the existing `FeatureConfig` fields. Changing them can change which
measurements are available or their numerical meaning. Refit and validate after
changing extraction settings rather than reusing old references blindly.

| JSON field | Type / range | Constructor default | Meaning and practical effect |
|---|---|---|---|
| `extraction.enabled` | Nonempty array of unique registered **base feature** names | All registered base features | Features to calculate. Listing a feature here does not make it an active posture rule. Temporal names such as `head_offset_y__std` belong in `rules[].feature`, not this array. Removing a base used by any active base/temporal rule is rejected. |
| `extraction.min_visibility` | Number in `[0, 1]` | `0.5` | Minimum visibility confidence for every landmark needed by a feature. Values below it make that measurement unavailable. Increasing it can reject more samples. |
| `extraction.min_presence` | Number in `[0, 1]` | `0.5` | Minimum landmark presence confidence. Both this and visibility requirements must pass; changing either affects coverage, not the posture threshold. |
| `extraction.min_shoulder_width_px` | Number strictly greater than `0`; pixels | `5.0` | Minimum usable Euclidean shoulder width in aspect-correct pixel coordinates. Rejects unreliable shoulder scale before dividing distances by it. It is not a minimum acceptable body width. |
| `extraction.head_reference` | `"eyes_midpoint"` or `"nose"` | `"eyes_midpoint"` | Point used by `head_offset_x` and `head_offset_y`: average of both eyes, or nose. Other features keep their own definitions. Switching this requires new references/thresholds for the affected rules. |
| `extraction.window_seconds` | Number strictly greater than `0`; seconds | `1.0` | Trailing history window for derived variation/velocity outputs. Larger windows cover more recent samples for variation; velocity still uses only the latest two samples inside the window. Has no temporal-extraction effect when `temporal_enabled` is false. |
| `extraction.max_gap_seconds` | Number strictly greater than `0`; seconds | `0.5` | Gap above which each derived feature's history resets. Invalid base measurements also reset that feature's history. This is **not** the final signal smoother/evaluator gap limit in `config/tracking.toml`. The fitter also uses each source recording's original extraction gap limit to reject discontinuous holds; editing a fitted profile does not redo that validation. |
| `extraction.temporal_enabled` | JSON boolean | `true` | Generate `__std` and `__velocity` for every enabled base. False removes these outputs and rejects active rules that reference them. A temporal output needs at least two valid samples in its window, otherwise it is unavailable during warmup. |

Currently registered base names are `shoulder_angle_deg`, `head_offset_x`,
`head_offset_y`, `mouth_shoulder_vertical`, `eye_ear_vertical`, `shoulder_width`,
and `facial_asymmetry`. Their formulas, signs, required landmarks, and native units
are in [FEATURES.md](FEATURES.md). Distances are not all in the same units:
shoulder width uses image widths, angle uses degrees, facial asymmetry is a ratio,
and the other geometric distances use shoulder widths. `__std` keeps its base's
units; `__velocity` uses base units per second.

Temporal feature extraction and live smoothing are different stages. For example,
`head_offset_y__std` measures recent variation; the evaluator can then apply EMA
to that variation output if it is an active rule. Neither operation modifies the
raw landmarks or the base feature calculation.

### `rules[]` and `rules[].reference`

Each array element has the following fields. Rules have no constructor defaults:
all values must be supplied. Two rules may use the same feature, but their names
must differ; this does not make correlated rules independent evidence.

| JSON field | Type / range | Meaning and practical effect |
|---|---|---|
| `rules[].name` | Nonblank string, unique within the profile | Rule identity used in evaluation state and reports, e.g. `"head_height"`. It is a label, not a registered feature name. |
| `rules[].feature` | One output name from the configured extraction | Signal the rule evaluates, e.g. `"head_offset_y"` or `"head_offset_y__std"`. Its base must be in `extraction.enabled`; a temporal output additionally requires `temporal_enabled: true`. |
| `rules[].reference` | Object with the three fields below | Personal reference centre, scale, and sample provenance for this exact output. A temporal output needs its own reference in its own units. |
| `rules[].reference.centre` | Any finite number, including negative values | Personal baseline in the selected output's native units. It is not necessarily zero or the same across users, camera positions, or head-reference choices. |
| `rules[].reference.scale` | Finite number strictly greater than `0` | Converts a raw difference from the centre into a dimensionless score. For a fixed raw difference and score threshold, a larger scale reduces sensitivity. The fitter uses a robust spread plus a configured floor; it is not necessarily a standard deviation, so scores are not automatically Gaussian z-scores. |
| `rules[].reference.sample_count` | Integer at least `1` | Number of reference samples represented by the fit. Metadata only; not a live window size, hold count, confidence weight, or persistence setting. |
| `rules[].direction` | `"decrease"`, `"increase"`, or `"either"` | Which departures from the centre are considered deviations. The score definitions are in the threshold section below. Refit thresholds when changing this choice. |
| `rules[].enter_threshold` | Finite number strictly greater than `0` | Dimensionless score at or above which deviation confirmation starts. Increasing it requires a larger departure from the reference. The condition must persist for the effective `enter_seconds`. |
| `rules[].clear_threshold` | Finite number with `0 <= clear_threshold < enter_threshold` | Score at or below which reference/clearing confirmation starts. Decreasing it requires a closer return to reference before clearing. The condition must persist for the effective `clear_seconds`. |

Rules use **smoothed output values** when smoothing is enabled. Values strictly
between entry and clearing thresholds form the hysteresis band: an established
state is retained and a pending change is canceled. At startup no established
state exists, so remaining in the band keeps the result `UNKNOWN`. Missing active
measurements immediately produce `UNKNOWN` and reset filter/state memory;
missing diagnostics do not block the active rules.

### `evaluation`

These three fields have no Python constructor defaults. The durations are stored
as seconds and use actual timestamps, not frame counts. In the standard live and
replay commands they are overridden by `config/tracking.toml`, as described above.

| JSON field | Type / range | Meaning and practical effect |
|---|---|---|
| `evaluation.smoothing_seconds` | Finite number at least `0`; seconds | EMA time constant or moving-average window, depending on the separately selected method. Larger values generally suppress more noise and add lag. Zero disables smoothing. The method itself is not stored in profile JSON. |
| `evaluation.enter_seconds` | Finite number strictly greater than `0`; seconds | Continuous time the score must satisfy the entry condition before a rule becomes deviating. Longer duration suppresses brief excursions but delays detection. |
| `evaluation.clear_seconds` | Finite number strictly greater than `0`; seconds | Continuous time the score must satisfy the clearing condition before reference posture is established or a deviation clears. Applies during initial settling too. |

For EMA, `alpha = 1-exp(-dt/smoothing_seconds)`; the duration is a time constant,
not an alpha coefficient. For moving average, it is a trailing sample-weighted
window that starts partially filled. With method `"none"`, values pass through
regardless of the stored duration. Disabling smoothing does not disable
hysteresis or persistence. Tracking gaps/missing active measurements reset timing;
missing time cannot complete a pending decision.

### `capture`

These four fields have no constructor defaults. They describe the setup used
for calibration and do not request changes to the camera or model. Change the
actual setup and collect/refit matching calibration rather than editing identity
fields simply to bypass a mismatch.

| JSON field | Type / range | Meaning and practical effect |
|---|---|---|
| `capture.camera_index` | Integer at least `0` | Expected camera index. Actual live selection comes from `--camera` (default `0`), so a profile for camera `1` also needs `--camera 1`. An index does not uniquely identify physical hardware across reconnections. |
| `capture.image_width` | Integer at least `1`; pixels | Expected actual captured image width, not a requested capture resolution. Must match the live frame or replay recording. |
| `capture.image_height` | Integer at least `1`; pixels | Expected actual image height. Width and height preserve the calibrated coordinate/aspect setup. |
| `capture.model_sha256` | Exactly 64 lowercase hexadecimal characters | SHA-256 of the pose model binary used in calibration. Must match the actual model. It is neither a model path nor a setting that selects another model. |

Matching these fields cannot detect camera movement or a changed seating setup.
The all-zero digest in the synthetic example is intentionally illustrative and
will not match the actual model for live use.

### What is not a profile field?

Filter method, live gap limit, plot history/display selection, fitting trim and
coverage limits, `min_scale`, and fitting acceptance criteria live in
[`config/tracking.toml`](config/tracking.toml), not profile JSON. `clear_ratio` is a fitting
policy there; the profile stores the resulting explicit `clear_threshold`.
Feature descriptions/formulas remain in [FEATURES.md](FEATURES.md). Adding any of
these as extra JSON keys makes loading fail rather than configuring the app.

To try an edited candidate without replacing the current profile:

```powershell
uv run src/app.py --track --profile data/profiles/candidate.json
```

Changing references, quality settings, features, or thresholds makes the existing
validation report stale. Save a new profile identity/method and evaluate the
candidate with replay before interpreting earlier fit metrics as its performance.

## Create and save

This example uses synthetic values, not fitted calibration. Import these modules
with `src` on Python's import path, as in the application and tests.

```python
from pathlib import Path

from calibration.calibration_profile import (
    CalibrationProfile,
    CaptureIdentity,
    EvaluationSettings,
    FeatureReference,
    FeatureRule,
)
from posture.features import FeatureConfig

profile = CalibrationProfile(
    extraction=FeatureConfig(
        enabled=("head_offset_y", "shoulder_width"),
        temporal_enabled=False,
    ),
    rules=(
        FeatureRule(
            name="head_height",
            feature="head_offset_y",
            reference=FeatureReference(centre=0.70, scale=0.05, sample_count=100),
            direction="decrease",
            enter_threshold=2.0,
            clear_threshold=1.0,
        ),
    ),
    evaluation=EvaluationSettings(0.2, 2.0, 1.0),
    capture=CaptureIdentity(0, 640, 480, "0" * 64),  # Synthetic model hash.
    source_sessions=("synthetic-example",),
    method="illustrative-not-fitted",
)
profile.save(Path("data/profiles/example.json"))
loaded = CalibrationProfile.load(Path("data/profiles/example.json"))
```

See `examples/calibration_profile.json` for the complete saved example. Omitting
`path` uses `data/profiles/current.json`. Saving uses a temporary file and atomic
rename so a failed publication preserves the previous profile. Personal profiles
are Git-ignored. Use `fit_profile.py` to fit and publish a real profile; see `LIVE_TRACKING.md`.

## Change features

Use `dataclasses.replace` to create a changed, revalidated profile:

```python
from dataclasses import replace

# Remove a diagnostic feature. Active rules must still be available.
head_only = replace(
    loaded,
    extraction=replace(loaded.extraction, enabled=("head_offset_y",)),
)

# Change reference values without rebuilding unrelated settings.
new_rule = replace(head_only.rules[0], reference=FeatureReference(0.68, 0.04, 200))
updated = replace(head_only, rules=(new_rule,))

# Select a temporal output using the same rule type.
# The example numbers are illustrative and need their own fit in these units.
movement_rule = FeatureRule(
    name="head_variation",
    feature="head_offset_y__std",
    reference=FeatureReference(0.003, 0.001, 100),
    direction="increase",
    enter_threshold=3.0,
    clear_threshold=1.0,
)
movement_example = replace(
    head_only,
    extraction=replace(head_only.extraction, temporal_enabled=True),
    rules=(movement_rule,),
)
```

Adding a decorated feature to the existing registry makes it available to profiles
through `FeatureConfig.columns`. Profile code needs no per-feature changes. Expose
new landmarks in the pose adapter first if the calculation requires them.

`replace` preserves identity and creation time. Give a newly published revision
new identity/provenance fields, and save candidates separately while reviewing.
Changing settings does not establish that they are fitted or validated.

## Threshold meaning

Reference centre and scale use the selected output's native units. The evaluator uses these dimensionless scores:

```text
decrease: (centre - value) / scale
increase: (value - centre) / scale
either:   abs(value - centre) / scale
```

Entry must exceed clearing threshold; both thresholds and scale must be finite,
with scale and entry strictly positive. With centre 0.70 and scale 0.05, an entry
score of 2 corresponds to a raw decreasing-feature boundary of 0.60. Timing is
explicitly supplied and is not presented as a learned result.

Multiple rules carry an explicit `any`/`all` combination policy for the evaluator. Availability of all active inputs is checked first. Missing required
inputs must become `UNKNOWN`; diagnostic failures do not affect this check.

## Persistence and compatibility

JSON mirrors the dataclass fields. The loader reconstructs the nested dataclasses
and their constructors validate values. Field checks are derived from those same
dataclasses rather than maintained in parallel lists.

The complete extraction settings are saved with the profile. Loading never reads
`config/features.toml`, so editing the TOML does not silently alter an existing profile.
Feature formulas and descriptions stay in `features.py` / `FEATURES.md`; they are
not copied into every JSON file. `feature_version` identifies their semantics:
changes to feature calculations must increment `FEATURE_VERSION` and regenerate
profiles. Documentation-only edits no longer invalidate saved profiles.

The loader rejects incomplete/unknown fields, malformed JSON, duplicate keys,
nonfinite numbers, invalid rules, disabled outputs, and incompatible schema or
feature versions. Errors raise `ProfileError` with context.

Call `check_compatibility(actual_capture_identity)` before using a profile. Camera
index, actual resolution, and model hash must match. These checks cannot detect a
physical camera move. Recording IDs and `method` preserve provenance; this storage
module does not fit values or verify source recordings exist on disk.

Automatic fitting and replay/live evaluation are implemented in separate modules.
See `LIVE_TRACKING.md` for commands and live timing overrides.
