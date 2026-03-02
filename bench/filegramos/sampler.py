"""Stratified content sampling with configurable token budget.

Selects representative content samples from engrams for the semantic channel,
balancing across trajectories and sample types (create, major_edit, minor_edit).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .engram import ContentSample
from .tuning import SEMANTIC_BUDGET

if TYPE_CHECKING:
    from .engram import Engram


class StratifiedSampler:
    """Select representative content samples within a token budget.

    Strategy:
    - Allocate budget proportionally across trajectories (importance-weighted)
    - Per trajectory, prefer: one create + one major_edit + one minor_edit
    - Truncate previews to fit within per-trajectory budget
    - Prioritize high-importance engrams and diverse file types
    """

    def __init__(self, total_token_budget: int = SEMANTIC_BUDGET):
        self.total_budget = total_token_budget

    def select_samples(self, engrams: list[Engram]) -> list[ContentSample]:
        """Select stratified content samples from all engrams.

        Args:
            engrams: List of Engram objects with semantic data.

        Returns:
            List of ContentSample with truncated previews.
        """
        if not engrams:
            return []

        # Sort by importance (descending) — high-importance engrams get priority
        sorted_engrams = sorted(engrams, key=lambda e: -e.importance_score)

        # Compute per-trajectory budget (importance-weighted)
        total_importance = sum(e.importance_score for e in sorted_engrams) or 1.0
        per_traj_budgets: dict[str, int] = {}
        for eng in sorted_engrams:
            weight = eng.importance_score / total_importance
            per_traj_budgets[eng.trajectory_id] = max(
                200,  # minimum budget per trajectory
                int(self.total_budget * weight),
            )

        # Cap total to budget
        total_allocated = sum(per_traj_budgets.values())
        if total_allocated > self.total_budget:
            scale = self.total_budget / total_allocated
            per_traj_budgets = {k: max(200, int(v * scale)) for k, v in per_traj_budgets.items()}

        samples: list[ContentSample] = []
        seen_file_types: set[str] = set()

        for eng in sorted_engrams:
            budget = per_traj_budgets.get(eng.trajectory_id, 400)
            traj_samples = self._select_from_engram(eng, budget, seen_file_types)
            samples.extend(traj_samples)
            for s in traj_samples:
                seen_file_types.add(s.file_type)

        return samples

    def _select_from_engram(
        self,
        engram: Engram,
        budget_chars: int,
        seen_types: set[str],
    ) -> list[ContentSample]:
        """Select up to 3 samples from a single engram."""
        sem = engram.semantic
        candidates: dict[str, list[ContentSample]] = {
            "create": [],
            "major_edit": [],
            "minor_edit": [],
        }

        for cf in sem.created_files:
            candidates["create"].append(cf)

        for ec in sem.edit_chains:
            lines_changed = ec.lines_added + ec.lines_deleted
            sample_type = "major_edit" if lines_changed > 10 else "minor_edit"
            candidates[sample_type].append(_edit_to_sample(ec, sample_type, engram.trajectory_id))

        selected: list[ContentSample] = []
        used_chars = 0

        for stype in ("create", "major_edit", "minor_edit"):
            if used_chars >= budget_chars:
                break
            pool = candidates[stype]
            if not pool:
                continue

            # Prefer unseen file types for diversity
            pool.sort(key=lambda s: (s.file_type in seen_types, -s.content_length))
            pick = pool[0]

            remaining = budget_chars - used_chars
            truncated = ContentSample(
                path=pick.path,
                content_length=pick.content_length,
                sample_type=pick.sample_type,
                content_preview=pick.content_preview[:remaining],
                file_type=pick.file_type,
                full_content=pick.full_content,
                trajectory_id=engram.trajectory_id,
                importance=engram.importance_score,
            )
            selected.append(truncated)
            used_chars += len(truncated.content_preview)

        return selected


def _edit_to_sample(
    ec: EditChainSample,
    sample_type: str,
    trajectory_id: str,
) -> ContentSample:
    """Convert an EditChainSample to a ContentSample for unified handling."""
    from .engram import ContentSample

    return ContentSample(
        path=ec.path,
        content_length=ec.lines_added + ec.lines_deleted,
        sample_type=sample_type,
        content_preview=ec.diff_preview,
        file_type="diff",
        full_content=ec.full_diff,
        trajectory_id=trajectory_id,
    )
