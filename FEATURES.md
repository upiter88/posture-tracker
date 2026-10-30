# Feature reference

Generated from the definitions beside each calculation in `src/posture/features.py`.

Coordinates use pixels: P=(x*image_width, y*image_height). Image x increases rightward; y increases downward. Left/right landmark names are anatomical. S is the shoulder midpoint; W is Euclidean shoulder width in pixels. Model-relative z is saved but not used by these features.

Edit `config/features.toml` to enable features, choose the head reference, change confidence limits, or adjust the motion window. Calibration mode loads this file automatically. Each required point must be finite, in-frame, and meet both confidence thresholds. Distances divided by W require W >= min_shoulder_width_px. An unavailable feature is null with a reason; raw landmarks are preserved. measurement_valid means all enabled base features are usable; it does not judge posture or require temporal warmup to finish.

## shoulder_angle_deg

Units: degrees.

Required landmarks: left_shoulder, right_shoulder. `head_reference` means the configured nose or both eyes.

Shoulder-line angle relative to image horizontal.

Definition: degrees(atan2(B_y - A_y, B_x - A_x)), where A is the
image-left shoulder and B the image-right shoulder (sort by x then y).
Range: [-90, 90]. Positive means downward slope to image right; level is 0.
This is image tilt, not anatomical roll. Requires a usable shoulder width.

## head_offset_x

Units: shoulder widths.

Required landmarks: left_shoulder, right_shoulder, head_reference. `head_reference` means the configured nose or both eyes.

Horizontal head position relative to the shoulder midpoint.

Definition: (H_x - S_x) / W.
H is the eye midpoint by default, or the nose via head_reference config.
Positive means image right of S; zero means centered. Normalization by W
reduces sensitivity to camera distance.

## head_offset_y

Units: shoulder widths.

Required landmarks: left_shoulder, right_shoulder, head_reference. `head_reference` means the configured nose or both eyes.

Vertical head position above the shoulder midpoint.

Definition: (S_y - H_y) / W.
H is the eye midpoint by default, or the nose via head_reference config.
Positive means above S. A decrease means less vertical separation;
it does not by itself establish forward slouching.

## mouth_shoulder_vertical

Units: shoulder widths.

Required landmarks: mouth_left, mouth_right, left_shoulder, right_shoulder. `head_reference` means the configured nose or both eyes.

Vertical separation of mouth and shoulder lines at their midpoints.

Definition: (S_y - M_y) / W, where M=(mouth_left + mouth_right)/2.
Positive means the mouth midpoint is above the shoulder midpoint.
Distance between lines means signed midpoint y separation here,
not the shortest distance between infinite lines.

## eye_ear_vertical

Units: shoulder widths.

Required landmarks: left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder. `head_reference` means the configured nose or both eyes.

Vertical separation of eye and ear lines at their midpoints.

Definition: (Ears_y - Eyes_y) / W, with each pair averaged in pixels.
Positive means the eye midpoint is above the ear midpoint.
Candidate head-orientation signal: anatomy, viewpoint, and tracking noise
also affect it. Compare with a personal reference.

## shoulder_width

Units: image widths.

Required landmarks: left_shoulder, right_shoulder. `head_reference` means the configured nose or both eyes.

Apparent Euclidean shoulder width relative to image width.

Definition: W / image_width.
Larger values usually mean greater apparent scale; body rotation also
changes it. This is not meters. W includes both x and y displacement;
tilted shoulders retain their Euclidean width. May exceed 1 in tall images.

## facial_asymmetry

Units: dimensionless ratio.

Required landmarks: nose, left_ear, right_ear. `head_reference` means the configured nose or both eyes.

Imbalance of anatomical left/right nose-to-ear distances.

Definition: (d_left - d_right) / (d_left + d_right), where
d_left=||nose-left_ear||_2 and d_right=||nose-right_ear||_2 in pixels.
Range: [-1, 1]. Positive means the anatomical left distance is larger;
zero means equal projected distances. Candidate head-turn signal,
not a calibrated yaw angle. Does not require shoulders.

## Movement and stability

Each enabled base feature gets the temporal columns below. History resets independently when its base feature becomes unavailable or the gap exceeds max_gap_seconds. Fewer than two samples in the window yields null with temporal_warmup. A stationary stream yields zero after warmup. The window may be partially filled. Labels do not reset history: use unlabeled transition segments when changing posture.

### <base_feature>__std

Units: same as the base feature.

Population standard deviation of recent valid observations.

Definition: sqrt(sum((f_i - mean(f))**2) / n) for samples in the inclusive
interval [t-window_seconds, t]. Requires at least two valid samples.
Samples are equally weighted, not weighted by elapsed time.
Larger means greater recent variation, not necessarily bad posture.

### <base_feature>__velocity

Units: base feature units per second.

Signed backward finite-difference velocity using the latest two samples.

Definition: (f(t) - f(previous)) / ((t_ns - previous_ns) / 1e9).
Positive means the base feature increased. Instantaneous, not a fitted
trend across the window; sensitive to frame noise. Both samples must be
inside the window. No assumed FPS or interpolation across tracking gaps.

## Adding a feature

Add a named function decorated with @feature next to its docstring containing the formal definition, description, signs, units, and limitations. Declare required landmarks in the decorator and use geometry helpers for validation. The registry, output schema, and session metadata discover it automatically. Add its name to config/features.toml if enabled is explicitly listed. Increment FEATURE_VERSION when definitions change, and test known geometry. Regenerate this document with the command in README.md.
