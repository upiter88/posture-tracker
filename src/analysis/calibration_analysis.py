# %% [markdown]
# # Calibration recording analysis
#
# Run this percent-format notebook cell by cell in an editor supporting `# %%`,
# or execute `uv run python src/analysis/calibration_analysis.py` from the project.
# It reads raw data without changing it. Results go to `<session>/analysis/`.
# Dependencies (available in the locked project environment): NumPy, PyArrow,
# Matplotlib, plus the project's feature definitions. No pandas or sklearn needed.
#
# Workflow: select a session; inspect integrity, labels, and tracking quality;
# recompute features; exclude unlabeled transitions; compare distributions;
# hold out whole posture segments for threshold evaluation; inspect redundancy,
# motion, timing, and boundary sensitivity; export plots, statistics, and a report.
# All thresholds fitted here are exploratory, not production settings.

# %% Imports and reproducible settings
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Pin the recording so rerunning this notebook reproduces this analysis.
# Set SESSION_ID = None to select the newest recording by metadata start time.
SESSION_ID: str | None = "20261027T213729-54efcdb3"
TRIM_SECONDS = 1.0
TRIM_SENSITIVITY_SECONDS = (0.0, 1.0, 2.0, 3.0)
SHOW_PLOTS = True  # Set True for interactive notebook use.


def find_project_root() -> Path:
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "src/posture/features.py").is_file() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Run this notebook from inside the PostureTracker project")


PROJECT_ROOT = find_project_root()
DATA_ROOT = PROJECT_ROOT / "data/calibration"
if SESSION_ID is None:
    manifests = list(DATA_ROOT.glob("*/session.json"))
    if not manifests:
        raise FileNotFoundError(f"No recordings in {DATA_ROOT}")
    manifest = max(manifests, key=lambda p: json.loads(p.read_text(encoding="utf-8"))["started_utc"])
    SESSION = manifest.parent
else:
    SESSION = DATA_ROOT / SESSION_ID
OUTPUT = SESSION / "analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / ".matplotlib"))

import matplotlib  # noqa: E402

if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from capture.pose import PoseLandmark, PoseObservation  # noqa: E402
from posture.features import FEATURE_VERSION, FeatureConfig, FeatureExtractor  # noqa: E402

# %% [markdown]
# ## 1. Load the original recording
#
# Both the original directory names (`samples`, `label_events`) and the newer
# `.parquet` dataset directories are supported. An absent event dataset is reported
# explicitly; it must not silently look like an absence of label changes.

# %%
metadata = json.loads((SESSION / "session.json").read_text(encoding="utf-8"))


def dataset_files(stem: str) -> list[Path]:
    for path in (SESSION / f"{stem}.parquet", SESSION / stem):
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(path.glob("*.parquet"))
    return []


sample_files = dataset_files("samples")
event_files = dataset_files("label_events")
if not sample_files:
    raise ValueError(f"No saved samples in {SESSION}")
rows = pq.read_table(sample_files).to_pylist()
events = pq.read_table(event_files).to_pylist() if event_files else []
if len(rows) < 2:
    raise ValueError("At least two frames are required for timing analysis")
base_features = list(metadata["feature_config"]["enabled"])
settings = dict(metadata["feature_config"])
settings["enabled"] = tuple(settings["enabled"])
feature_config = FeatureConfig(**settings)
feature_names = list(feature_config.columns)
values = {
    name: np.array([np.nan if row[name] is None else row[name] for row in rows], dtype=float) for name in feature_names
}
timestamps = np.array([row["capture_timestamp_ns"] for row in rows], dtype=np.int64)
time_s = (timestamps - timestamps[0]) / 1e9
sequences = np.array([row["frame_sequence"] for row in rows])
labels = np.array([row["posture_label"] for row in rows])
segment_ids = np.array([row["label_segment_id"] for row in rows])
is_invalid = labels == "invalid"
intervals_s = np.diff(timestamps) / 1e9
if not np.all(intervals_s > 0) or not np.all(np.diff(sequences) > 0):
    raise ValueError("Frames must be unique and ordered; inspect the original dataset before proceeding")
# Approximate label durations: hold each label until the next capture; estimate
# the final frame interval using the median. These are not video exposure times.
frame_weights_s = np.r_[intervals_s, np.median(intervals_s)]

print(f"Session: {SESSION.name}; {len(rows):,} frames; {time_s[-1]:.2f} seconds")
print("Labels:", dict(Counter(labels)))
print("Complete:", metadata["complete"])

# %% [markdown]
# ## 2. Integrity, event alignment, and measurement availability
#
# Model confidence and `measurement_valid` indicate availability under configured
# rules; they do not establish anatomical accuracy or validity of the posture.
# Validate every event against its first affected frame. A null first frame means
# the user changed labels but no subsequent sample was captured.

# %%
sequence_lookup = {row["frame_sequence"]: index for index, row in enumerate(rows)}
event_checks = []
for event in events:
    first_sequence = event["first_frame_sequence"]
    check = {"segment_id": event["label_segment_id"], "applied": first_sequence is not None}
    if first_sequence is not None:
        index = sequence_lookup.get(first_sequence)
        check["first_frame_exists"] = index is not None
        if index is not None:
            check.update(
                label_matches=rows[index]["posture_label"] == event["posture_label"],
                segment_matches=rows[index]["label_segment_id"] == event["label_segment_id"],
                first_frame_in_segment=index == 0 or segment_ids[index - 1] != segment_ids[index],
                input_to_capture_ms=float((timestamps[index] - event["event_timestamp_ns"]) / 1e6),
            )
    event_checks.append(check)

quality = {
    "frames": len(rows),
    "metadata_frame_count": metadata.get("frame_count"),
    "metadata_saved_count": metadata.get("saved_count"),
    "counts_match": len(rows) == metadata.get("frame_count") == metadata.get("saved_count"),
    "complete": metadata["complete"],
    "duration_seconds": float(time_s[-1]),
    "fps": float((len(rows) - 1) / time_s[-1]),
    "sequences_contiguous": bool(np.all(np.diff(sequences) == 1)),
    "elapsed_times_match": all(
        row["elapsed_ns"] == int(ts - timestamps[0]) for row, ts in zip(rows, timestamps, strict=False)
    ),
    "session_ids_match": all(row["session_id"] == metadata["session_id"] for row in rows),
    "frame_interval_ms_p50_p95_max": list(map(float, np.quantile(intervals_s * 1000, [0.5, 0.95, 1]))),
    "gaps_over_500ms": int(np.sum(intervals_s > 0.5)),
    "detected_frames": sum(row["pose_detected"] for row in rows),
    "measurement_valid_frames": sum(row["measurement_valid"] for row in rows),
    "label_frames": dict(Counter(map(str, labels))),
    "label_seconds_estimated": {
        str(label): float(frame_weights_s[labels == label].sum()) for label in np.unique(labels)
    },
    "resolution": sorted({(row["image_width"], row["image_height"]) for row in rows}),
    "temporary_files": [str(path.relative_to(SESSION)) for path in SESSION.rglob("*.tmp")],
    "event_files": len(event_files),
    "label_events": len(events),
    "missing_counts": {name: int(np.sum(~np.isfinite(array))) for name, array in values.items()},
    "unavailable_reasons": {
        name: dict(Counter(row[f"{name}__reason"] for row in rows if row[f"{name}__reason"] is not None))
        for name in feature_names
    },
}
confidence = {
    landmark: {
        score: dict(
            zip(
                ("min", "p05", "median"),
                map(
                    float,
                    np.nanquantile(
                        [np.nan if row[f"{landmark}_{score}"] is None else row[f"{landmark}_{score}"] for row in rows],
                        [0, 0.05, 0.5],
                    ),
                ),
                strict=False,
            )
        )
        for score in ("visibility", "presence")
    }
    for landmark in metadata["landmark_names"]
}
print(json.dumps(quality, indent=2))
print("Event checks:", event_checks)

# %% [markdown]
# ## 3. Recompute every feature from raw landmarks
#
# This checks consistency between the dataset and the *current* implementation.
# If feature versions differ, discrepancies may reflect intentional code changes.
# Never overwrite the original feature values during this check.

# %%
extractor = FeatureExtractor(feature_config)
recompute = {
    "recorded_version": metadata["feature_version"],
    "current_version": FEATURE_VERSION,
    "null_mismatches": 0,
    "max_absolute_error": {name: 0.0 for name in feature_names},
}
for row in rows:
    landmarks = {
        name: PoseLandmark(*(row[f"{name}_{axis}"] for axis in ("x", "y", "z", "visibility", "presence")))
        for name in metadata["landmark_names"]
        if row[f"{name}_x"] is not None
    }
    observation = PoseObservation(row["capture_timestamp_ns"], row["image_width"], row["image_height"], landmarks)
    recalculated = extractor.extract(observation)
    for name in feature_names:
        saved, calculated = row[name], recalculated.values[name]
        if (saved is None) != (calculated is None):
            recompute["null_mismatches"] += 1
        elif saved is not None:
            recompute["max_absolute_error"][name] = max(recompute["max_absolute_error"][name], abs(saved - calculated))
print("Recomputation:", recompute)

# %% [markdown]
# ## 4. Define segments and analysis exclusions
#
# Unlabeled frames remain on plots and in quality checks, but are excluded from
# valid/invalid comparisons. The primary analysis additionally excludes one second
# at both ends of each labeled hold. This avoids using the previous posture's
# rolling history and reduces residual settling contamination. The sensitivity
# analysis below repeats the calculation with 0, 1, 2, and 3 seconds excluded.
# No samples are deleted or selected by their agreement with an expected outcome.

# %%
segments: list[dict] = []
for segment_id in np.unique(segment_ids):
    indices = np.flatnonzero(segment_ids == segment_id)
    if len(np.unique(labels[indices])) != 1 or np.any(np.diff(indices) != 1):
        raise ValueError(f"Segment {segment_id} must be contiguous and have one label")
    segments.append(
        {
            "id": int(segment_id),
            "label": str(labels[indices[0]]),
            "start_s": float(time_s[indices[0]]),
            "end_s": float(time_s[indices[-1]]),
            "frames": len(indices),
            "duration_s": float(frame_weights_s[indices].sum()),
        }
    )


def labeled_interior(trim_seconds: float) -> np.ndarray:
    mask = np.zeros(len(rows), dtype=bool)
    for segment in segments:
        if segment["label"] in ("valid", "invalid"):
            mask |= (
                (segment_ids == segment["id"])
                & (time_s >= segment["start_s"] + trim_seconds)
                & (time_s <= segment["end_s"] - trim_seconds)
            )
    return mask


primary_mask = labeled_interior(TRIM_SECONDS)
for segment in segments:
    mask = primary_mask & (segment_ids == segment["id"])
    segment["retained_frames"] = int(mask.sum())
    segment["feature_medians"] = {
        name: float(np.nanmedian(values[name][mask])) if np.any(mask & np.isfinite(values[name])) else None
        for name in base_features
    }
print("Retained labels:", dict(Counter(labels[primary_mask])))
print("Holds:", Counter(segment["label"] for segment in segments))

# %% [markdown]
# ## 5. Descriptive separation and whole-segment validation
#
# Quantiles describe the retained frames. Rank AUC is the probability a randomly
# selected invalid frame has a higher feature value than a valid frame, with ties
# counting 0.5. Direction-free rank separation is max(AUC, 1-AUC); it is descriptive
# and its direction is selected using the whole recording, not a held-out score.
#
# For prediction, fit an exact class-balanced threshold on all but one complete
# labeled segment, then predict the held-out segment. Repeat for every segment.
# Balanced accuracy = (invalid recall + valid recall)/2. This avoids random-frame
# leakage, but holds from the same session are still related. No unseen-session
# performance claim or significance test is justified by this single recording.
#
# Test two feature transforms separately: raw values, and absolute deviation from
# the *training-valid* median. Both direction and threshold are fitted only on
# training data. Report both; choosing the better result is itself model selection.


# %%
def rank_auc(x: np.ndarray, target: np.ndarray) -> float:
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    first_ranks = np.r_[0, np.cumsum(counts)[:-1]] + 1
    ranks = (first_ranks + (counts - 1) / 2)[inverse]
    positives, negatives = int(target.sum()), int((~target).sum())
    return float((ranks[target].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def fit_threshold(x: np.ndarray, target: np.ndarray) -> tuple[float, str]:
    unique, inverse = np.unique(x, return_inverse=True)
    positive_counts = np.bincount(inverse[target], minlength=len(unique))
    negative_counts = np.bincount(inverse[~target], minlength=len(unique))
    positive_below = np.r_[0, np.cumsum(positive_counts)] / positive_counts.sum()
    negative_below = np.r_[0, np.cumsum(negative_counts)] / negative_counts.sum()
    scores = np.stack(((positive_below + 1 - negative_below) / 2, (1 - positive_below + negative_below) / 2))
    direction, split = np.unravel_index(np.argmax(scores), scores.shape)
    threshold = -np.inf if split == 0 else np.inf if split == len(unique) else (unique[split - 1] + unique[split]) / 2
    return float(threshold), "below" if direction == 0 else "above_or_equal"


def classify(x: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    return x < threshold if direction == "below" else x >= threshold


def classification_metrics(prediction: np.ndarray, mask: np.ndarray) -> dict:
    invalid_recall = float(np.mean(prediction[mask & is_invalid]))
    valid_recall = float(np.mean(~prediction[mask & ~is_invalid]))
    return {
        "balanced_accuracy": (invalid_recall + valid_recall) / 2,
        "invalid_recall": invalid_recall,
        "valid_recall": valid_recall,
        "false_invalid_frames": int(np.sum(prediction & mask & ~is_invalid)),
        "missed_invalid_frames": int(np.sum(~prediction & mask & is_invalid)),
    }


def evaluate_threshold(name: str, mask: np.ndarray, transform: str = "raw") -> dict | None:
    x = values[name]
    mask = mask & np.isfinite(x)
    holds_per_class = [len(np.unique(segment_ids[mask & (labels == label)])) for label in ("valid", "invalid")]
    if min(holds_per_class) < 2:
        return None  # Holding out a segment must leave both labels in training.
    predictions = np.zeros(len(rows), dtype=bool)
    folds = []
    for hold in np.unique(segment_ids[mask]):
        train, test = mask & (segment_ids != hold), mask & (segment_ids == hold)
        centre = float(np.median(x[train & ~is_invalid])) if transform == "deviation" else 0.0
        transformed = np.abs(x - centre) if transform == "deviation" else x
        threshold, direction = fit_threshold(transformed[train], is_invalid[train])
        predictions[test] = classify(transformed[test], threshold, direction)
        folds.append(
            {
                "segment": int(hold),
                "label": str(labels[test][0]),
                "frames": int(test.sum()),
                "recall": float(np.mean(predictions[test] == is_invalid[test])),
                "threshold": threshold,
                "direction": direction,
                "training_valid_centre": centre,
            }
        )
    result = classification_metrics(predictions, mask)
    result["folds"] = folds
    # Macro metric gives each hold equal weight within its class, unlike pooling frames.
    result["segment_macro_balanced_accuracy"] = float(
        np.mean(
            [np.mean([fold["recall"] for fold in folds if fold["label"] == label]) for label in ("valid", "invalid")]
        )
    )
    centre = float(np.median(x[mask & ~is_invalid])) if transform == "deviation" else 0.0
    transformed = np.abs(x - centre) if transform == "deviation" else x
    threshold, direction = fit_threshold(transformed[mask], is_invalid[mask])
    result["full_data_fit"] = {"threshold": threshold, "direction": direction, "valid_centre": centre}
    return result


statistics: dict[str, dict] = {}
for name in feature_names:
    x = values[name]
    mask = primary_mask & np.isfinite(x)
    by_label = {}
    for label in ("valid", "invalid"):
        sample = x[mask & (labels == label)]
        by_label[label] = (
            dict(
                zip(
                    ("p05", "p25", "median", "p75", "p95"),
                    map(float, np.quantile(sample, [0.05, 0.25, 0.5, 0.75, 0.95])),
                    strict=False,
                )
            )
            if len(sample)
            else None
        )
    auc = rank_auc(x[mask], is_invalid[mask]) if all(by_label.values()) else None
    statistics[name] = {
        "by_label": by_label,
        # Inspect complete labeled ranges too, independently of boundary trimming.
        "untrimmed_class_range": {
            label: list(map(float, np.quantile(x[(labels == label) & np.isfinite(x)], [0, 1])))
            if np.any((labels == label) & np.isfinite(x))
            else None
            for label in ("valid", "invalid")
        },
        "auc_invalid_high": auc,
        "direction_free_rank_separation": max(auc, 1 - auc) if auc is not None else None,
        "blocked_threshold": {mode: evaluate_threshold(name, primary_mask, mode) for mode in ("raw", "deviation")},
    }
    raw_score = statistics[name]["blocked_threshold"]["raw"]
    print(
        name,
        "medians",
        {label: stats["median"] if stats else None for label, stats in by_label.items()},
        "blocked BA",
        round(raw_score["balanced_accuracy"], 4) if raw_score else None,
    )

# %% [markdown]
# ## 6. Redundancy and motion
#
# Correlations use complete rows for all enabled base features within labeled
# interiors. High correlation can reflect shared geometry, shared normalization,
# or the particular movements recorded; it is not proof of independent usefulness.
# Motion features are evaluated above using the same exclusions and held-out holds.
# Zero velocity can describe either a sustained valid or a sustained invalid pose.

# %%
complete_case_mask = primary_mask & np.all(np.isfinite(np.array([values[name] for name in base_features])), axis=0)
correlation = np.corrcoef(np.array([values[name][complete_case_mask] for name in base_features]))
sensitivity: dict[str, dict] = {}
for trim in TRIM_SENSITIVITY_SECONDS:
    mask = labeled_interior(trim)
    sensitivity[str(trim)] = {
        "retained_frames": int(mask.sum()),
        "label_frames": dict(Counter(map(str, labels[mask]))),
        "features": {name: evaluate_threshold(name, mask) for name in base_features},
    }
print(
    "Boundary sensitivity (head_offset_y):",
    {trim: data["features"].get("head_offset_y", {}).get("balanced_accuracy") for trim, data in sensitivity.items()},
)

# %% [markdown]
# ### Check the normalization denominator
#
# A shoulder-normalized distance can change because its pixel numerator changes,
# because shoulder width changes, or both. Reconstruct pixel distances before
# attributing all separation to head movement: distance_px = feature * width_px.
# Pixel distances are a within-camera diagnostic, not resolution-invariant features.

# %%
normalization: dict[str, dict] = {}
width_px = values["shoulder_width"] * np.array([row["image_width"] for row in rows])
for name in ("head_offset_y", "mouth_shoulder_vertical", "eye_ear_vertical"):
    if name not in values:
        continue
    pixel_distance = values[name] * width_px
    normalization[name] = {}
    for label in ("valid", "invalid"):
        mask = primary_mask & (labels == label) & np.isfinite(pixel_distance)
        normalization[name][label] = {
            "median_pixel_distance": float(np.median(pixel_distance[mask])),
            "median_shoulder_width_px": float(np.median(width_px[mask])),
            "pixel_distance_p05_p95": list(map(float, np.quantile(pixel_distance[mask], [0.05, 0.95]))),
        }
print("Normalization diagnostic:", normalization)

# %% [markdown]
# ## 7. Timing relative to explicit unlabeled transitions
#
# Use a centered 0.5-second median-smoothed head-height trace only as an offline
# alignment diagnostic. Fit its threshold on labeled interiors. Inspect each
# valid/invalid change separated by an unlabeled interval: does a matching-direction
# crossing fall inside that interval? This is not independent ground truth or a
# causal measurement of labeling error. Do not automatically shift saved labels.

# %%
timing: list[dict] = []
head_fit = statistics.get("head_offset_y", {}).get("blocked_threshold", {}).get("raw")
if head_fit is not None:
    fit = head_fit["full_data_fit"]
    head = values["head_offset_y"]
    smooth = np.array(
        [np.nanmedian(head[(time_s >= timestamp - 0.25) & (time_s <= timestamp + 0.25)]) for timestamp in time_s]
    )
    predicted_invalid = classify(smooth, fit["threshold"], fit["direction"])
    crossings = np.flatnonzero(predicted_invalid[1:] != predicted_invalid[:-1]) + 1
    previous_labeled = None
    for segment in segments:
        if segment["label"] == "unlabeled":
            continue
        if previous_labeled is not None and previous_labeled["label"] != segment["label"]:
            candidates = [
                index
                for index in crossings
                if previous_labeled["end_s"] <= time_s[index] <= segment["start_s"]
                and predicted_invalid[index] == (segment["label"] == "invalid")
            ]
            timing.append(
                {
                    "from_segment": previous_labeled["id"],
                    "to_segment": segment["id"],
                    "unlabeled_gap_s": segment["start_s"] - previous_labeled["end_s"],
                    "crossings_inside_gap": [float(time_s[index]) for index in candidates],
                    "matching_crossing_inside_gap": bool(candidates),
                }
            )
        previous_labeled = segment
print("Transition checks:", timing)

# %% [markdown]
# ## 8. Frozen thresholds from the preceding recording
#
# These constants come from the previous analysis in this chat. The earlier raw
# recording is no longer present locally, so its fitting cannot be rerun here.
# Keep these constants frozen: evaluating them on this fresh session is a useful
# transfer check, unlike fitting and testing a threshold on the same samples.
# Same-person/camera setup still limits the breadth of this test.

# %%
previous_thresholds = {
    "head_offset_y": 0.6157579466250872,
    "mouth_shoulder_vertical": 0.4069558352262105,
    "eye_ear_vertical": 0.025909867141036415,
}
transfer: dict[str, dict] = {}
for name, threshold in previous_thresholds.items():
    if name in values:
        mask = primary_mask & np.isfinite(values[name])
        all_labeled = labeled_interior(0) & np.isfinite(values[name])
        transfer[name] = {
            "threshold": threshold,
            **classification_metrics(values[name] < threshold, mask),
            "all_labeled_frames": classification_metrics(values[name] < threshold, all_labeled),
        }
print("Previous threshold transfer:", transfer)

# %% [markdown]
# ## 9. Plots
#
# Shaded intervals show manual labels, not model predictions. Distribution dots
# show individual hold medians, to avoid treating thousands of nearby frames as
# thousands of independent observations. Whiskers show 5th-95th percentiles.

# %%
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
colors = {"valid": "#177e62", "invalid": "#cf4d48", "unlabeled": "#a6adb4"}


def save_plot(figure, filename: str) -> None:
    figure.savefig(OUTPUT / filename, dpi=150, facecolor="white")
    if SHOW_PLOTS:
        plt.show()
    plt.close(figure)


figure, axes = plt.subplots(
    len(base_features), 1, figsize=(14, 2 * len(base_features)), sharex=True, layout="constrained", squeeze=False
)
for axis, name in zip(axes.flat, base_features, strict=False):
    for segment in segments:
        axis.axvspan(segment["start_s"], segment["end_s"], color=colors[segment["label"]], alpha=0.18)
    axis.plot(time_s, values[name], color="#284d70", linewidth=0.7)
    axis.set_ylabel(name.replace("_", "\n"), rotation=0, ha="right", va="center", fontsize=9)
    axis.grid(axis="y", alpha=0.15)
axes.flat[0].legend(
    handles=[Patch(facecolor=color, alpha=0.4, label=label.title()) for label, color in colors.items()], ncol=3
)
axes.flat[-1].set_xlabel("Seconds since first captured frame")
figure.suptitle(f"Calibration timeline | {SESSION.name}", fontsize=15)
save_plot(figure, "feature_timeline.png")

ncols = 4
nrows = (len(base_features) + ncols - 1) // ncols
figure, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), layout="constrained", squeeze=False)
for axis, name in zip(axes.flat, base_features, strict=False):
    arrays = [
        values[name][primary_mask & (labels == label) & np.isfinite(values[name])] for label in ("valid", "invalid")
    ]
    boxes = axis.boxplot(arrays, tick_labels=["Valid", "Invalid"], showfliers=False, patch_artist=True, whis=(5, 95))
    for box, label in zip(boxes["boxes"], ("valid", "invalid"), strict=False):
        box.set_facecolor(colors[label])
        box.set_alpha(0.35)
    for position, label in enumerate(("valid", "invalid"), 1):
        medians = [
            segment["feature_medians"][name]
            for segment in segments
            if segment["label"] == label and segment["feature_medians"][name] is not None
        ]
        axis.scatter(
            np.linspace(position - 0.08, position + 0.08, len(medians)),
            medians,
            color=colors[label],
            edgecolors="white",
            zorder=3,
        )
    axis.set_title(name, fontsize=11)
    axis.grid(axis="y", alpha=0.15)
for axis in list(axes.flat)[len(base_features) :]:
    axis.axis("off")
figure.suptitle(f"Valid vs invalid | {TRIM_SECONDS:g}s excluded at each end; dots = hold medians", fontsize=15)
save_plot(figure, "feature_distributions.png")

figure, axis = plt.subplots(figsize=(9, 7), layout="constrained")
image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
axis.set_xticks(range(len(base_features)), base_features, rotation=40, ha="right")
axis.set_yticks(range(len(base_features)), base_features)
for i in range(len(base_features)):
    for j in range(len(base_features)):
        axis.text(
            j,
            i,
            f"{correlation[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if abs(correlation[i, j]) > 0.65 else "#222",
        )
figure.colorbar(image, ax=axis, label="Pearson correlation, labeled interiors")
axis.set_title("Feature redundancy")
save_plot(figure, "feature_correlation.png")

# %% [markdown]
# ## 10. Export machine-readable results and an evidence-based report
#
# JSON keeps every quantile, fold, fitted threshold, exclusion setting, and timing
# check. Infinite thresholds represent constant classifiers; JSON encodes them as
# strings to preserve their meaning while keeping strict JSON valid.


# %%
def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


results = {
    "session": SESSION.name,
    "primary_trim_seconds": TRIM_SECONDS,
    "quality": quality,
    "event_checks": event_checks,
    "confidence": confidence,
    "recompute": recompute,
    "segments": segments,
    "features": statistics,
    "base_feature_names": base_features,
    "base_feature_correlation": correlation.tolist(),
    "normalization_diagnostic": normalization,
    "trim_sensitivity": sensitivity,
    "transition_alignment": timing,
    "previous_threshold_transfer": transfer,
}
(OUTPUT / "statistics.json").write_text(json.dumps(json_safe(results), indent=2, allow_nan=False), encoding="utf-8")

report = [
    f"# Calibration analysis: {SESSION.name}",
    "",
    "## Coverage and quality",
    "",
    f"- Complete: {quality['complete']}; counts match: {quality['counts_match']}.",
    f"- {len(rows):,} frames over {time_s[-1]:.2f}s at {quality['fps']:.2f} FPS.",
    f"- Labels (frames): {quality['label_frames']}.",
    f"- Estimated label seconds: {quality['label_seconds_estimated']}.",
    f"- Holds: {dict(Counter(s['label'] for s in segments))}.",
    f"- Usable base measurements: {quality['measurement_valid_frames']}/{len(rows)} frames.",
    f"- Feature null mismatches: {recompute['null_mismatches']}; maximum recompute error: "
    f"{max(recompute['max_absolute_error'].values()):.3g}.",
    "",
    f"## Feature comparison ({TRIM_SECONDS:g}s trimmed from each labeled hold boundary)",
    "",
    "| Feature | Valid median | Invalid median | Blocked balanced accuracy (raw / deviation) |",
    "|---|---:|---:|---:|",
]
for name in base_features:
    stats = statistics[name]
    scores = [stats["blocked_threshold"][mode] for mode in ("raw", "deviation")]
    report.append(
        f"| {name} | {stats['by_label']['valid']['median']:.4f} | "
        f"{stats['by_label']['invalid']['median']:.4f} | "
        + " / ".join(f"{100 * score['balanced_accuracy']:.1f}%" if score else "unavailable" for score in scores)
        + " |"
    )
report += [
    "",
    "Balanced accuracy averages valid and invalid recall. Complete holds are excluded from "
    "training one at a time. Threshold direction and deviation centre are fitted on training data only. "
    "These are within-session exploratory estimates, not performance on unseen users/days. "
    "Adjacent frames are correlated, and ranking multiple features adds selection optimism.",
    "",
    "## Boundary sensitivity (raw-threshold blocked balanced accuracy)",
    "",
    "| Trim seconds per end | Retained frames | Head height | Mouth height | Eye-ear height |",
    "|---:|---:|---:|---:|---:|",
]
for trim, result in sensitivity.items():
    scores = [result["features"].get(name) for name in previous_thresholds]
    report.append(
        f"| {trim} | {result['retained_frames']} | "
        + " | ".join(f"{100 * score['balanced_accuracy']:.1f}%" if score else "unavailable" for score in scores)
        + " |"
    )
report += [
    "",
    "Different trim settings evaluate different subsets; higher scores do not demonstrate an improved model.",
    "",
    "## Frozen earlier thresholds on this recording",
    "",
    "| Feature | Earlier threshold (invalid below) | Balanced accuracy | Invalid recall | Valid recall |",
    "|---|---:|---:|---:|---:|",
]
for name, result in transfer.items():
    report.append(
        f"| {name} | {result['threshold']:.5f} | {100 * result['balanced_accuracy']:.1f}% | "
        f"{100 * result['invalid_recall']:.1f}% | {100 * result['valid_recall']:.1f}% |"
    )
report += [
    "",
    "The earlier thresholds are frozen constants from the preceding analysis; the older raw data "
    "is not currently available for reanalysis. New-session threshold fits are not substitutes for "
    "these transfer results.",
    "",
    "## Timing and interpretation",
    "",
    f"Matching head-height crossings inside explicit transition gaps: "
    f"{sum(item['matching_crossing_inside_gap'] for item in timing)}/{len(timing)} label changes. "
    "This diagnostic uses a smoothed feature and a threshold fitted to the recording; it is not independent "
    "ground truth. See statistics.json for exact crossings, intervals, and all fold results.",
    "",
    "## Artifacts and reproduction",
    "",
    "- feature_timeline.png: raw features and all manual labels.",
    "- feature_distributions.png: class distributions and individual hold medians.",
    "- feature_correlation.png: feature redundancy.",
    "- statistics.json: complete calculations and validation folds.",
    "",
    "Every analysis step is in src/analysis/calibration_analysis.py, a percent-format Python notebook. "
    "Run it cell by cell or execute `uv run python src/analysis/calibration_analysis.py`. "
    "Raw samples are read-only. Edit SESSION_ID to inspect a different recording.",
    "",
]
(OUTPUT / "report.md").write_text("\n".join(report), encoding="utf-8")
print(f"Saved report, statistics, and plots to {OUTPUT}")

# %% [markdown]
# ## 11. Conclusions and practical next steps
#
# The generated narrative below accompanies the calculations, rather than relying
# on conclusions that only appeared in a chat. Inspect the plots as well: a perfect
# holdout score can still reflect limited movement diversity. This is one additional
# recording with thirteen labeled holds, not thousands of independent experiments.
#
# The original session is no longer available to reproduce its fitting. The
# frozen-threshold check is therefore deliberately separate from fresh fitting.

# %%
vertical_names = [name for name in previous_thresholds if name in statistics]
conclusions = ["## Conclusions and next steps", ""]
for name in vertical_names:
    score = statistics[name]["blocked_threshold"]["raw"]
    if score is not None:
        conclusions.append(
            f"- **{name}**: {100 * score['balanced_accuracy']:.1f}% whole-hold-out balanced accuracy "
            f"on this recording's labeled interiors. Earlier frozen threshold: "
            f"{100 * transfer[name]['balanced_accuracy']:.1f}% balanced accuracy, "
            f"{100 * transfer[name]['invalid_recall']:.1f}% invalid recall."
        )
if "head_offset_y" in base_features and "mouth_shoulder_vertical" in base_features:
    i, j = base_features.index("head_offset_y"), base_features.index("mouth_shoulder_vertical")
    conclusions.append(
        f"- Head height and mouth height correlate at r={correlation[i, j]:.4f}. "
        "They largely repeat the same signal here. Start with one primary feature; "
        "test whether the others add value on new movement types before combining votes."
    )
conclusions += [
    f"- The transition diagnostic finds a matching head-height crossing inside "
    f"{sum(item['matching_crossing_inside_gap'] for item in timing)}/{len(timing)} "
    "explicit unlabeled transition gaps. The untrimmed and trimmed results are in the "
    "sensitivity table; agreement across trim widths supports cleaner labels but is not "
    "independent proof that every label is correct.",
    "- Shoulder angle, horizontal head position, width, and facial asymmetry are secondary "
    "signals for this recording. Inspect their per-hold folds before deciding to use them. "
    "Near-perfect vertical separation does not establish their usefulness for lateral lean or head tilt.",
    "- Motion should primarily support stability/settling assessment. A held invalid posture "
    "can have zero velocity. The JSON includes every rolling-variation and signed-velocity score.",
]
if "head_offset_y" in normalization:
    valid = normalization["head_offset_y"]["valid"]
    invalid = normalization["head_offset_y"]["invalid"]
    conclusions.append(
        f"- Median head-to-shoulder pixel separation: {valid['median_pixel_distance']:.1f} valid "
        f"vs {invalid['median_pixel_distance']:.1f} invalid. Median shoulder width: "
        f"{valid['median_shoulder_width_px']:.1f}px vs {invalid['median_shoulder_width_px']:.1f}px. "
        "Both numerator and denominator contribute to the normalized feature difference. "
        "Retain apparent width as a setup/scale diagnostic."
    )
conclusions += [
    "- For the next evaluation, keep unlabeled transitions and add different invalid subtypes, "
    "valid head turns, small posture deviations, and ordinary desk activity. Record new days "
    "and camera placements, then hold out complete sessions for final testing.",
    "- Use a personal stable reference rather than promoting these fitted cutoffs to global "
    "rules. Tune persistence/hysteresis on separate recordings. Perfect separation of this "
    "sample does not establish warning delay or false-alarm rates during normal work.",
    "",
]
report_path = OUTPUT / "report.md"
report_path.write_text("\n".join(report + [""] + conclusions), encoding="utf-8")
print("\n".join(conclusions))
