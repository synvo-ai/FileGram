"""Fixed-dimension behavioral fingerprint vector computation.

Each fingerprint is a 17-feature vector covering all 6 L/M/R dimensions.
Used for per-session deviation scoring (episodic channel).
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from .schema import FEATURE_TO_DIMENSION, FINGERPRINT_FEATURES


def compute_fingerprint(features: dict[str, Any]) -> list[float]:
    """Extract a fixed-dimension vector from per-trajectory procedural features.

    Args:
        features: Output of FeatureExtractor.extract_all()

    Returns:
        17-element float vector.
    """
    fp = []
    for attr, key in FINGERPRINT_FEATURES:
        attr_dict = features.get(attr, {})
        val = attr_dict.get(key, 0)
        fp.append(float(val) if isinstance(val, (int, float, bool)) else 0.0)
    return fp


def normalize_fingerprints(
    fingerprints: dict[str, list[float]],
) -> dict[str, list[float]]:
    """Z-score normalize fingerprints per dimension.

    Args:
        fingerprints: {trajectory_id: 17-element vector}

    Returns:
        Same structure, z-score normalized per feature dimension.
    """
    if len(fingerprints) < 2:
        return dict(fingerprints)

    n_dims = len(FINGERPRINT_FEATURES)
    vectors = list(fingerprints.values())
    ids = list(fingerprints.keys())

    normalized = {}
    for tid in ids:
        normalized[tid] = [0.0] * n_dims

    for dim in range(n_dims):
        vals = [v[dim] for v in vectors]
        mu = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        for i, tid in enumerate(ids):
            if std > 0:
                normalized[tid][dim] = (vectors[i][dim] - mu) / std
            else:
                normalized[tid][dim] = 0.0

    return normalized


def compute_deviations(
    fingerprints: dict[str, list[float]],
    threshold: float = 1.5,
) -> tuple[list[float], dict[str, float], dict[str, bool]]:
    """Compute centroid, per-session distances, and deviation flags.

    Args:
        fingerprints: {trajectory_id: fingerprint_vector} (already normalized)
        threshold: Flag as deviation if distance > mean + threshold * std

    Returns:
        (centroid, distances, flags)
    """
    if not fingerprints:
        return [], {}, {}

    n_dims = len(FINGERPRINT_FEATURES)
    ids = list(fingerprints.keys())
    vectors = list(fingerprints.values())

    # Compute centroid
    centroid = [statistics.mean(v[d] for v in vectors) for d in range(n_dims)]

    # Euclidean distance from centroid
    distances: dict[str, float] = {}
    for tid, vec in zip(ids, vectors):
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, centroid)))
        distances[tid] = round(dist, 4)

    # Flag deviations
    if len(distances) < 3:
        flags = {tid: False for tid in ids}
        return centroid, distances, flags

    dist_vals = list(distances.values())
    mu = statistics.mean(dist_vals)
    std = statistics.stdev(dist_vals)
    cutoff = mu + threshold * std

    flags = {tid: dist > cutoff for tid, dist in distances.items()}

    return centroid, distances, flags


def locate_shifted_dimensions(
    fingerprint: list[float],
    centroid: list[float],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """For a deviant engram, locate which dimensions shifted most.

    Args:
        fingerprint: The deviant session's normalized fingerprint.
        centroid: The profile centroid.
        top_k: Return top-k shifted dimensions.

    Returns:
        List of {"feature": "attr.key", "delta": float, "dimension": str}
    """
    diffs = []
    for i, (attr, key) in enumerate(FINGERPRINT_FEATURES):
        delta = abs(fingerprint[i] - centroid[i])
        diffs.append(
            {
                "feature": f"{attr}.{key}",
                "delta": round(delta, 3),
                "dimension": FEATURE_TO_DIMENSION.get(attr, attr),
            }
        )

    diffs.sort(key=lambda x: -x["delta"])
    return diffs[:top_k]


def detect_absences(per_trajectory_features: list[dict[str, Any]]) -> list[str]:
    """Detect consistently absent behaviors across all trajectories.

    Returns human-readable absence descriptions.
    """
    if not per_trajectory_features:
        return []

    absences = []
    n = len(per_trajectory_features)

    # Never created directories
    if all(f.get("directory_style", {}).get("dirs_created", 0) == 0 for f in per_trajectory_features):
        absences.append(f"Never created directories in any of {n} trajectories")

    # Never used search
    if all(f.get("reading_strategy", {}).get("total_searches", 0) == 0 for f in per_trajectory_features):
        absences.append(f"Never used search tools in any of {n} trajectories")

    # Never edited files
    if all(f.get("edit_strategy", {}).get("total_edits", 0) == 0 for f in per_trajectory_features):
        absences.append(f"Never edited files after creation in any of {n} trajectories")

    # Never created backups
    if all(f.get("version_strategy", {}).get("backup_copies", 0) == 0 for f in per_trajectory_features):
        absences.append(f"Never created backup copies in any of {n} trajectories")

    # Never deleted files
    if all(f.get("version_strategy", {}).get("total_deletes", 0) == 0 for f in per_trajectory_features):
        absences.append(f"Never deleted files in any of {n} trajectories")

    # Never browsed directories
    if all(f.get("reading_strategy", {}).get("total_browses", 0) == 0 for f in per_trajectory_features):
        absences.append(f"Never browsed directories in any of {n} trajectories")

    # Never created images or structured data
    if all(
        f.get("cross_modal_behavior", {}).get("image_files_created", 0) == 0
        and f.get("cross_modal_behavior", {}).get("structured_files_created", 0) == 0
        for f in per_trajectory_features
    ):
        absences.append(f"Never created image or structured data files in any of {n} trajectories")

    return absences
