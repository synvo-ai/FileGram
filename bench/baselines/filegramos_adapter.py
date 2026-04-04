"""FileGramOS: engram-based three-channel memory adapter.

Uses typed Engram memory units, fingerprint-based deviation detection,
stratified content sampling, and query-adaptive retrieval.
"""

from __future__ import annotations

from typing import Any

from .base import BaseAdapter
from .registry import register_adapter

try:
    from ..filegramos.consolidator import EngramConsolidator
    from ..filegramos.encoder import EngramEncoder
    from ..filegramos.engram import MemoryStore
    from ..filegramos.normalizer import EventNormalizer
    from ..filegramos.retriever import QueryAdaptiveRetriever
except ImportError:
    from filegramos.consolidator import EngramConsolidator
    from filegramos.encoder import EngramEncoder
    from filegramos.engram import MemoryStore
    from filegramos.normalizer import EventNormalizer
    from filegramos.retriever import QueryAdaptiveRetriever


SYNTHESIS_PROMPT = """\
You are an expert at inferring user work habit profiles from file-system \
behavioral data. Below is a three-channel analysis from {n_trajectories} task \
trajectories performed by a single user, organized using the FileGramOS \
engram-based memory architecture.

{channel_context}

## Inference Task

Based on ALL three channels above, infer this user's work habit profile. \
For each attribute, cite the specific evidence that supports your inference.

IMPORTANT guidance:
- Channel 1 (Procedural) is most reliable for: working_style, thoroughness, \
error_handling, reading_strategy, directory_style, edit_strategy, version_strategy
- Channel 2 (Semantic) is most reliable for: name, role, language, tone, \
output_structure, documentation
- Channel 3 (Episodic) tells you about behavioral consistency and any anomalies. \
If deviations are flagged, weight those trajectories less in your inference.

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


@register_adapter("filegramos")
class FileGramOSAdapter(BaseAdapter):
    """Engram-based three-channel profile extraction — formal version."""

    def __init__(self, config: dict[str, Any] | None = None, llm_fn=None):
        super().__init__(config, llm_fn=llm_fn)
        self.name = "filegramos"
        if llm_fn is None:
            raise ValueError("llm_fn is required for FileGramOSAdapter")
        cfg = config or {}
        self._normalizer = EventNormalizer()
        self._encoder = EngramEncoder(llm_fn=llm_fn)
        self._consolidator = EngramConsolidator(
            semantic_budget=cfg.get("semantic_budget", 8000),
            deviation_threshold=cfg.get("deviation_threshold", 1.5),
            llm_fn=llm_fn,
        )
        self._retriever = QueryAdaptiveRetriever()
        self._store: MemoryStore | None = None

    def ingest(self, trajectories: list[dict[str, Any]]) -> None:
        """Encode each trajectory into an Engram (parallel), then consolidate."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _encode_one(traj: dict[str, Any]):
            task_id = traj["task_id"]
            normalized = self._normalizer.normalize_all(traj["events"])
            traj_id = traj.get(
                "trajectory_id",
                f"{self._store_profile_id}_{task_id}" if hasattr(self, "_store_profile_id") else task_id,
            )
            is_perturbed = traj.get("is_perturbed", False)
            return task_id, self._encoder.encode(
                events=normalized,
                task_id=task_id,
                trajectory_id=traj_id,
                is_perturbed=is_perturbed,
            )

        max_workers = (self.config or {}).get("ingest_parallel", 4)
        engrams = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_encode_one, t): t for t in trajectories}
            for future in as_completed(futures):
                task_id, engram = future.result()
                engrams.append(engram)
                self._on_trajectory_done(task_id)

        # Sort by trajectory_id to ensure deterministic order
        engrams.sort(key=lambda e: e.trajectory_id)

        self._store = self._consolidator.consolidate(engrams)

    def infer_profile(self, profile_attributes: list[str]) -> dict[str, str]:
        """Build query-adaptive synthesis prompt."""
        if not self._store:
            return {"_prompt": "(no data ingested)", "_method": "filegramos"}

        channel_context = self._retriever.retrieve(
            self._store,
            query_type="profile",
        )

        attributes_list = "\n".join(f"- {attr}" for attr in profile_attributes)

        prompt = SYNTHESIS_PROMPT.format(
            n_trajectories=len(self._store.engrams),
            channel_context=channel_context,
            attributes_list=attributes_list,
        )

        return {"_prompt": prompt, "_method": "filegramos"}

    def _get_ingest_state(self) -> dict[str, Any]:
        """Serialize store for pickle caching."""
        if not self._store:
            return {}
        return {"store": self._store}

    def _set_ingest_state(self, state: dict[str, Any]) -> None:
        """Restore store from pickle cache."""
        self._store = state.get("store")

    def get_store(self) -> MemoryStore | None:
        """Return the consolidated MemoryStore for external use (e.g., QA eval)."""
        return self._store

    def reset(self) -> None:
        super().reset()
        self._store = None
