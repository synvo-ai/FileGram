"""Core data model for FileGramOS formal memory architecture.

Engram = the atomic memory unit per trajectory. MemoryStore = consolidated
cross-engram state with three channels (procedural, semantic, episodic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContentSample:
    """A sampled piece of user-created content."""

    path: str
    content_length: int
    sample_type: str  # "create" | "major_edit" | "minor_edit"
    content_preview: str  # Truncated excerpt for writing-style analysis
    file_type: str
    full_content: str = ""  # Complete document for content retrieval
    trajectory_id: str = ""
    importance: float = 0.0


@dataclass
class EditChainSample:
    """A sampled edit operation."""

    path: str
    lines_added: int
    lines_deleted: int
    diff_preview: str  # Truncated diff excerpt for style analysis
    full_diff: str = ""  # Complete diff for content retrieval
    trajectory_id: str = ""


@dataclass
class CrossRef:
    """A cross-file reference detected in a trajectory."""

    source_file: str
    target_file: str
    reference_type: str


@dataclass
class SemanticUnit:
    """Semantic channel data for a single trajectory."""

    created_files: list[ContentSample] = field(default_factory=list)
    edit_chains: list[EditChainSample] = field(default_factory=list)
    cross_file_refs: list[CrossRef] = field(default_factory=list)
    created_filenames: list[str] = field(default_factory=list)
    dir_structure_diff: list[str] = field(default_factory=list)
    llm_encoding: dict[str, Any] | None = None


@dataclass
class Engram:
    """Atomic memory unit: one trajectory's encoded behavioral data."""

    trajectory_id: str
    task_id: str
    procedural: dict[str, Any]  # From FeatureExtractor.extract_all()
    auxiliary: dict[str, Any]  # From FeatureExtractor.extract_auxiliary()
    semantic: SemanticUnit
    fingerprint: list[float]  # Fixed-dimension behavioral vector
    event_count: int = 0
    behavioral_event_count: int = 0
    importance_score: float = 0.0
    is_perturbed: bool = False


@dataclass
class MemoryStore:
    """Consolidated memory across all engrams for a profile.

    Three channels:
      1. Procedural: aggregated behavioral statistics + patterns + L/M/R classifications
      2. Semantic: representative content samples + filenames + dir structure
      3. Episodic: fingerprint centroid + per-session deviations + consistency + absences
    """

    profile_id: str
    engrams: list[Engram] = field(default_factory=list)

    # Channel 1: Procedural aggregate
    procedural_aggregate: dict[str, dict[str, Any]] = field(default_factory=dict)
    behavioral_patterns: list[str] = field(default_factory=list)
    dimension_classifications: list[str] = field(default_factory=list)

    # Channel 2: Semantic consolidated
    representative_samples: list[ContentSample] = field(default_factory=list)
    all_filenames: list[str] = field(default_factory=list)
    dir_structure_union: list[str] = field(default_factory=list)
    llm_narratives: dict[str, dict] = field(default_factory=dict)  # trajectory_id -> LLM behavioral encoding

    # Channel 3: Episodic
    centroid: list[float] = field(default_factory=list)
    per_session_distances: dict[str, float] = field(default_factory=dict)
    deviation_flags: dict[str, bool] = field(default_factory=dict)
    deviation_details: dict[str, list[dict]] = field(default_factory=dict)
    consistency_flags: dict[str, Any] = field(default_factory=dict)
    absence_flags: list[str] = field(default_factory=list)

    def filter_by_tasks(self, task_ids: set[str]) -> MemoryStore:
        """Return a shallow copy filtered to engrams matching task_ids."""
        filtered_engrams = [e for e in self.engrams if e.task_id in task_ids]
        filtered_ids = {e.trajectory_id for e in filtered_engrams}

        return MemoryStore(
            profile_id=self.profile_id,
            engrams=filtered_engrams,
            procedural_aggregate=self.procedural_aggregate,
            behavioral_patterns=self.behavioral_patterns,
            dimension_classifications=self.dimension_classifications,
            representative_samples=[s for s in self.representative_samples if s.trajectory_id in filtered_ids],
            all_filenames=self.all_filenames,
            dir_structure_union=self.dir_structure_union,
            llm_narratives={k: v for k, v in self.llm_narratives.items() if k in filtered_ids},
            centroid=self.centroid,
            per_session_distances={k: v for k, v in self.per_session_distances.items() if k in filtered_ids},
            deviation_flags={k: v for k, v in self.deviation_flags.items() if k in filtered_ids},
            deviation_details={k: v for k, v in self.deviation_details.items() if k in filtered_ids},
            consistency_flags=self.consistency_flags,
            absence_flags=self.absence_flags,
        )
