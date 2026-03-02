"""FileGramOS Simple: three-channel profile extraction adapter.

This is NOT a baseline — it's the method we expect to outperform all baselines.
Registered alongside baselines so the same evaluation runner can compare them.

Three-channel approach:
  Channel 1 (Procedural): Deterministic feature extraction + aggregation (statistics)
  Channel 2 (Semantic): Content previews of user-created files, diffs, filenames
  Channel 3 (Episodic): Cross-trajectory consistency metrics
  Final step: LLM synthesis from all three channels
"""

from __future__ import annotations

from typing import Any

from .base import BaseAdapter
from .registry import register_adapter

try:
    from ..filegramos.aggregation import FeatureAggregator
    from ..filegramos.feature_extraction import FeatureExtractor
    from ..filegramos.normalizer import EventNormalizer
except ImportError:
    from filegramos.aggregation import FeatureAggregator
    from filegramos.feature_extraction import FeatureExtractor
    from filegramos.normalizer import EventNormalizer

SYNTHESIS_PROMPT = """\
You are an expert at inferring user work habit profiles from file-system \
behavioral data. Below is a three-channel analysis deterministically \
extracted from {n_trajectories} task trajectories performed by a single user.

## Channel 1: Procedural Patterns (aggregated statistics)
{procedural_features}

## Channel 2: Semantic Content (user-created output samples)
{semantic_content}

## Channel 3: Episodic Consistency
{episodic_consistency}

## Inference Task

Based on ALL three channels above, infer this user's work habit profile. \
For each attribute, infer the most likely value based on the evidence and \
cite the specific evidence that supports your inference.

IMPORTANT: Use Channel 2 (semantic content) to judge tone, output structure, \
and naming convention — these are best assessed from actual content, not just \
statistics. Use Channel 1 for reading_strategy, edit_strategy, and other \
behavioral patterns. Use Channel 3 for confidence calibration.

Attributes to infer:
{attributes_list}

Respond in JSON format:
{{
  "inferred_profile": {{
    "<attribute_name>": {{
      "value": "<inferred value>",
      "justification": "<which evidence from which channel supports this>"
    }},
    ...
  }}
}}
"""


@register_adapter("filegramos_simple")
class FileGramOSSimpleAdapter(BaseAdapter):
    """Three-channel profile extraction — our proposed method."""

    def __init__(self, config: dict[str, Any] | None = None, llm_fn=None):
        super().__init__(config, llm_fn=llm_fn)
        self.name = "filegramos_simple"
        self._normalizer = EventNormalizer()
        self._per_trajectory_features: list[dict[str, Any]] = []
        self._per_trajectory_semantic: list[dict[str, Any]] = []

    def ingest(self, trajectories: list[dict[str, Any]]) -> None:
        """Extract deterministic features and semantic content from each trajectory."""
        self._per_trajectory_features = []
        self._per_trajectory_semantic = []
        for traj in trajectories:
            normalized = self._normalizer.normalize_all(traj["events"])
            extractor = FeatureExtractor(normalized)
            features = extractor.extract_all()
            semantic = extractor.extract_semantic_channel()
            self._per_trajectory_features.append(features)
            self._per_trajectory_semantic.append(semantic)
            self._on_trajectory_done(traj["task_id"])

    def infer_profile(self, profile_attributes: list[str]) -> dict[str, str]:
        """Build three-channel synthesis prompt."""
        # Channel 1: Procedural (aggregated statistics)
        aggregator = FeatureAggregator(self._per_trajectory_features)
        procedural_text = aggregator.to_summary_text()

        # Channel 2: Semantic (content previews)
        semantic_merged = FeatureAggregator.aggregate_semantic(self._per_trajectory_semantic)
        semantic_text = FeatureAggregator.to_semantic_text(semantic_merged)

        # Channel 3: Episodic (cross-trajectory consistency)
        episodic = FeatureAggregator.aggregate_episodic(self._per_trajectory_features)
        episodic_text = FeatureAggregator.to_episodic_text(episodic)

        # Build attribute list (open-ended, no value options)
        attributes_list = "\n".join(f"- {attr}" for attr in profile_attributes)

        prompt = SYNTHESIS_PROMPT.format(
            n_trajectories=len(self._per_trajectory_features),
            procedural_features=procedural_text,
            semantic_content=semantic_text or "(no content available)",
            episodic_consistency=episodic_text,
            attributes_list=attributes_list,
        )

        return {"_prompt": prompt, "_method": "filegramos_simple"}

    def get_extracted_features(self) -> list[dict[str, Any]]:
        """Return raw per-trajectory features for inspection."""
        return self._per_trajectory_features

    def get_aggregated_features(self) -> dict[str, dict[str, Any]]:
        """Return aggregated features for inspection."""
        aggregator = FeatureAggregator(self._per_trajectory_features)
        return aggregator.aggregate_all()

    def _get_ingest_state(self):
        return {"features": self._per_trajectory_features, "semantic": self._per_trajectory_semantic}

    def _set_ingest_state(self, state):
        self._per_trajectory_features = state["features"]
        self._per_trajectory_semantic = state["semantic"]

    def reset(self) -> None:
        super().reset()
        self._per_trajectory_features = []
        self._per_trajectory_semantic = []
