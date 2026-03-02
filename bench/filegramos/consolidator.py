"""Stage 2: Cross-engram consolidation with perturbation detection.

Consolidates multiple Engrams into a MemoryStore with three channels:
  1. Procedural: aggregated statistics + patterns + L/M/R classifications
  2. Semantic: stratified content samples + merged filenames/dirs
  3. Episodic: fingerprint-based deviation scoring + consistency + absences

Classification and pattern detection are performed by LLM (required).
"""

from __future__ import annotations

import json
from typing import Any

from .aggregation import FeatureAggregator
from .engram import Engram, MemoryStore
from .fingerprint import (
    compute_deviations,
    detect_absences,
    locate_shifted_dimensions,
    normalize_fingerprints,
)
from .sampler import StratifiedSampler
from .schema import build_classification_prompt
from .tuning import SEMANTIC_BUDGET


class EngramConsolidator:
    """Consolidate engrams into a three-channel MemoryStore.

    Uses LLM for dimension classification and behavioral pattern detection.
    llm_fn is required — no deterministic fallback.
    """

    def __init__(
        self,
        semantic_budget: int = SEMANTIC_BUDGET,
        deviation_threshold: float = 1.5,
        llm_fn=None,
    ):
        if llm_fn is None:
            raise ValueError("llm_fn is required for EngramConsolidator")
        self._semantic_budget = semantic_budget
        self._deviation_threshold = deviation_threshold
        self._llm_fn = llm_fn
        self._sampler = StratifiedSampler(total_token_budget=semantic_budget)

    def consolidate(
        self,
        engrams: list[Engram],
        profile_id: str = "",
    ) -> MemoryStore:
        """Consolidate a list of engrams into a unified MemoryStore.

        Args:
            engrams: List of Engram objects (one per trajectory).
            profile_id: Profile identifier for the store.

        Returns:
            A fully populated MemoryStore.
        """
        if not engrams:
            return MemoryStore(profile_id=profile_id)

        # --- Episodic channel (compute first, needed for weighting) ---
        raw_fps = {e.trajectory_id: e.fingerprint for e in engrams}
        norm_fps = normalize_fingerprints(raw_fps)
        centroid, distances, deviation_flags = compute_deviations(
            norm_fps,
            threshold=self._deviation_threshold,
        )

        # Locate shifted dimensions for deviant engrams
        deviation_details: dict[str, list[dict]] = {}
        for eng in engrams:
            tid = eng.trajectory_id
            if deviation_flags.get(tid, False) and centroid:
                details = locate_shifted_dimensions(
                    norm_fps[tid],
                    centroid,
                    top_k=3,
                )
                deviation_details[tid] = details

        # --- Channel 1: Procedural ---
        # Weight deviant engrams lower for aggregation
        weighted_features = self._weight_features(engrams, deviation_flags)
        aggregator = FeatureAggregator(weighted_features)
        procedural_aggregate = aggregator.aggregate_all()

        # Classification via LLM
        dimension_classifications, behavioral_patterns = self._llm_classify(
            procedural_aggregate,
        )
        if dimension_classifications is None:
            raise RuntimeError("LLM classification failed — check llm_fn and model availability")

        # --- Channel 2: Semantic ---
        # Prefer non-deviant engrams for representative samples
        non_deviant = [e for e in engrams if not deviation_flags.get(e.trajectory_id, False)]
        sample_pool = non_deviant if non_deviant else engrams
        representative_samples = self._sampler.select_samples(sample_pool)

        # Merge all filenames and dir structure (deduplicated)
        all_filenames = self._merge_filenames(engrams)
        dir_structure_union = self._merge_dir_structure(engrams)

        # Collect LLM behavioral narratives (skip deviant trajectories)
        llm_narratives: dict[str, dict] = {}
        for eng in engrams:
            if eng.semantic.llm_encoding and not deviation_flags.get(eng.trajectory_id, False):
                llm_narratives[eng.trajectory_id] = eng.semantic.llm_encoding

        # --- Channel 3: Episodic (continued) ---
        all_features = [e.procedural for e in engrams]
        consistency_flags = FeatureAggregator.aggregate_episodic(all_features)
        absence_flags = detect_absences(all_features)

        return MemoryStore(
            profile_id=profile_id,
            engrams=engrams,
            # Channel 1
            procedural_aggregate=procedural_aggregate,
            behavioral_patterns=behavioral_patterns,
            dimension_classifications=dimension_classifications,
            # Channel 2
            representative_samples=representative_samples,
            all_filenames=all_filenames,
            dir_structure_union=dir_structure_union,
            llm_narratives=llm_narratives,
            # Channel 3
            centroid=centroid,
            per_session_distances=distances,
            deviation_flags=deviation_flags,
            deviation_details=deviation_details,
            consistency_flags=consistency_flags,
            absence_flags=absence_flags,
        )

    def _weight_features(
        self,
        engrams: list[Engram],
        deviation_flags: dict[str, bool],
    ) -> list[dict[str, Any]]:
        """Return procedural features, duplicating non-deviant entries for weighting.

        Simple approach: include non-deviant engrams twice, deviant once.
        This dilutes the noise from perturbed trajectories in aggregation.
        """
        features = []
        for eng in engrams:
            features.append(eng.procedural)
            if not deviation_flags.get(eng.trajectory_id, False):
                # Non-deviant: add again for extra weight
                features.append(eng.procedural)
        return features

    def _merge_filenames(self, engrams: list[Engram]) -> list[str]:
        """Merge and deduplicate filenames across all engrams."""
        seen: set[str] = set()
        result: list[str] = []
        for eng in engrams:
            for fn in eng.semantic.created_filenames:
                if fn not in seen:
                    seen.add(fn)
                    result.append(fn)
        return result

    def _merge_dir_structure(self, engrams: list[Engram]) -> list[str]:
        """Merge and deduplicate directory structure diffs."""
        seen: set[str] = set()
        result: list[str] = []
        for eng in engrams:
            for d in eng.semantic.dir_structure_diff:
                if d not in seen:
                    seen.add(d)
                    result.append(d)
        return result

    # --- LLM-based classification ---

    def _llm_classify(
        self,
        procedural_aggregate: dict[str, dict[str, Any]],
    ) -> tuple[list[str] | None, list[str] | None]:
        """Use LLM to classify dimensions and detect behavioral patterns.

        Returns (classifications, patterns) or (None, None) on failure.
        """
        stats_text = self._format_stats_for_llm(procedural_aggregate)
        n = next(iter(procedural_aggregate.values()), {}).get("_n_trajectories", "?")

        prompt = build_classification_prompt(
            n_trajectories=n,
            stats_text=stats_text,
        )

        try:
            response = self._llm_fn(prompt)
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                print("  [filegramos] LLM classification: no JSON found, falling back")
                return None, None

            data = json.loads(text[start : end + 1])

            # Convert classifications to standard string format:
            #   "attribute → classification [CONFIDENCE]: evidence"
            classifications = []
            for c in data.get("classifications", []):
                attr = c.get("attribute", "?")
                cls = c.get("classification", "?")
                conf = c.get("confidence", "LOW")
                evidence = c.get("evidence", "")
                classifications.append(f"{attr} → {cls} [{conf}]: {evidence}")

            patterns = data.get("patterns", [])

            if classifications:
                print(f"  [filegramos] LLM classification: {len(classifications)} dims, {len(patterns)} patterns")
                return classifications, patterns

            return None, None

        except Exception as e:
            print(f"  [filegramos] LLM classification failed: {e}, falling back")
            return None, None

    @staticmethod
    def _format_stats_for_llm(
        procedural_aggregate: dict[str, dict[str, Any]],
    ) -> str:
        """Format procedural aggregate statistics as readable text for LLM."""
        lines: list[str] = []
        for attr_name, features in procedural_aggregate.items():
            lines.append(f"\n### {attr_name}")
            n = features.get("_n_trajectories", "?")
            lines.append(f"  (based on {n} trajectories)")
            for key, value in sorted(features.items()):
                if key.startswith("_"):
                    continue
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.3f}")
                elif isinstance(value, dict):
                    lines.append(f"  {key}: {value}")
                else:
                    lines.append(f"  {key}: {value}")
        return "\n".join(lines)
