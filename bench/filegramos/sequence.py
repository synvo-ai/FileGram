"""Channel 1 enrichment: deterministic sequence analysis.

Adds temporal/sequential features that flat event counting misses:
  - Transition matrix: P(action_j | action_i) for consecutive behavioral events
  - Phase detection: sliding-window classification into exploration/production/organization
  - File access graph: consecutive file access pairs

All deterministic — no LLM, no embeddings.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .schema import ConsumerEventType, NormalizedEvent, SIMULATION_TYPES


# Phase classifications based on event type
_EXPLORATION_TYPES = frozenset({
    ConsumerEventType.FILE_READ,
    ConsumerEventType.FILE_BROWSE,
    ConsumerEventType.FILE_SEARCH,
})

_PRODUCTION_TYPES = frozenset({
    ConsumerEventType.FILE_WRITE,
    ConsumerEventType.FILE_EDIT,
})

_ORGANIZATION_TYPES = frozenset({
    ConsumerEventType.FILE_MOVE,
    ConsumerEventType.FILE_RENAME,
    ConsumerEventType.DIR_CREATE,
    ConsumerEventType.FILE_DELETE,
    ConsumerEventType.FILE_COPY,
})


def _classify_event(et: ConsumerEventType) -> str:
    """Classify an event type into a phase category."""
    if et in _EXPLORATION_TYPES:
        return "exploration"
    if et in _PRODUCTION_TYPES:
        return "production"
    if et in _ORGANIZATION_TYPES:
        return "organization"
    return "other"


class SequenceAnalyzer:
    """Deterministic sequence analysis for Channel 1 enrichment.

    Operates on NormalizedEvent lists (behavioral events only).
    """

    def __init__(self, events: list[NormalizedEvent]):
        # Filter to behavioral events only
        self._events = [
            e for e in events
            if e.event_type.value not in SIMULATION_TYPES
        ]

    def compute_transition_matrix(self) -> dict[str, dict[str, int]]:
        """P(action_j | action_i) — consecutive behavioral event pairs.

        Returns:
            {from_type: {to_type: count}}
        """
        matrix: dict[str, dict[str, int]] = {}
        for i in range(len(self._events) - 1):
            from_t = self._events[i].event_type.value
            to_t = self._events[i + 1].event_type.value
            if from_t not in matrix:
                matrix[from_t] = {}
            matrix[from_t][to_t] = matrix[from_t].get(to_t, 0) + 1
        return matrix

    def detect_phases(self, window: int = 5) -> list[dict[str, Any]]:
        """Sliding-window phase detection.

        Classifies each window by the dominant event category:
        'exploration' (read/browse/search), 'production' (write/edit),
        'organization' (move/rename/dir_create/delete/copy).

        Returns:
            [{phase, start_idx, end_idx, dominant_actions}]
        """
        if not self._events:
            return []

        phases: list[dict[str, Any]] = []
        n = len(self._events)

        # Classify each event
        event_classes = [_classify_event(e.event_type) for e in self._events]

        # Merge consecutive windows with same dominant phase
        i = 0
        while i < n:
            end = min(i + window, n)
            window_classes = event_classes[i:end]
            counts = Counter(window_classes)
            # Remove 'other' from consideration for dominant phase
            non_other = {k: v for k, v in counts.items() if k != "other"}
            if non_other:
                dominant = max(non_other, key=non_other.get)  # type: ignore[arg-type]
            else:
                dominant = "other"

            # Count dominant actions in this window
            action_counts: dict[str, int] = {}
            for e in self._events[i:end]:
                t = e.event_type.value
                action_counts[t] = action_counts.get(t, 0) + 1

            # Try to merge with previous phase if same dominant
            if phases and phases[-1]["phase"] == dominant:
                phases[-1]["end_idx"] = end - 1
                # Merge action counts
                for k, v in action_counts.items():
                    phases[-1]["dominant_actions"][k] = phases[-1]["dominant_actions"].get(k, 0) + v
            else:
                phases.append({
                    "phase": dominant,
                    "start_idx": i,
                    "end_idx": end - 1,
                    "dominant_actions": action_counts,
                })

            i = end

        return phases

    def compute_file_access_graph(self) -> dict[str, list[str]]:
        """file_i -> file_j if accessed consecutively.

        Returns:
            {file_path: [next_file_paths]}
        """
        graph: dict[str, list[str]] = {}
        prev_path: str = ""
        for e in self._events:
            path = e.file_path
            if not path:
                continue
            if prev_path and prev_path != path:
                if prev_path not in graph:
                    graph[prev_path] = []
                graph[prev_path].append(path)
            prev_path = path
        return graph

    def summarize(self) -> dict[str, Any]:
        """Package all sequence features into a dict for Engram storage.

        Returns:
            {transition_matrix, phases, phase_counts, file_access_graph,
             dominant_transition, phase_pattern}
        """
        transition_matrix = self.compute_transition_matrix()
        phases = self.detect_phases()
        file_access_graph = self.compute_file_access_graph()

        # Phase counts
        phase_counts: dict[str, int] = {}
        for p in phases:
            phase_counts[p["phase"]] = phase_counts.get(p["phase"], 0) + 1

        # Dominant transition (most frequent pair)
        dominant_transition = ""
        max_count = 0
        for from_t, targets in transition_matrix.items():
            for to_t, count in targets.items():
                if count > max_count:
                    max_count = count
                    dominant_transition = f"{from_t} -> {to_t} ({count})"

        # Phase pattern: ordered sequence of phase labels
        phase_pattern = " -> ".join(p["phase"] for p in phases) if phases else ""

        return {
            "transition_matrix": transition_matrix,
            "phases": phases,
            "phase_counts": phase_counts,
            "file_access_graph": file_access_graph,
            "dominant_transition": dominant_transition,
            "phase_pattern": phase_pattern,
        }
