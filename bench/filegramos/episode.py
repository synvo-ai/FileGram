"""Channel 2 core: LLM-based episode segmentation.

Segments a trajectory's events into 2-5 coherent episodes, each with
LLM-generated narratives (title + content + summary). Episodes are the
semantic unit for cross-session clustering in the consolidator.

Narrative format follows EverMemos: per-episode factual third-person
behavioral narrative with specific file names and operation patterns.

Fallback: entire trajectory = 1 episode if LLM fails.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .schema import NormalizedEvent, SIMULATION_TYPES
from .tuning import (
    EPISODE_MAX_PER_TRAJECTORY,
    EPISODE_MIN_EVENTS,
)


@dataclass
class Episode:
    """A coherent behavioral episode within a trajectory."""

    episode_id: str
    trajectory_id: str
    start_idx: int
    end_idx: int
    event_count: int
    summary: str  # 1-sentence summary
    title: str = ""  # 10-20 word descriptive title
    content: str = ""  # 3-8 sentence third-person behavioral narrative
    dominant_actions: dict[str, int] = field(default_factory=dict)
    files_involved: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)  # populated during consolidation


class EpisodeSegmenter:
    """Segment a trajectory into coherent behavioral episodes via LLM.

    LLM calls per trajectory:
    1. Boundary detection: identify 2-5 episode boundaries (1 call)
    2. Per-episode narrative: EverMemos-style title + content + summary (N calls)

    Fallback: entire trajectory = 1 episode if LLM fails.
    """

    def __init__(
        self,
        llm_fn: Callable | None = None,
        min_events: int = EPISODE_MIN_EVENTS,
        max_episodes: int = EPISODE_MAX_PER_TRAJECTORY,
    ):
        self._llm_fn = llm_fn
        self._min_events = min_events
        self._max_episodes = max_episodes

    def segment(
        self,
        events: list[NormalizedEvent],
        trajectory_id: str,
    ) -> list[Episode]:
        """Segment events into episodes.

        Args:
            events: Normalized behavioral events for one trajectory.
            trajectory_id: Unique trajectory identifier.

        Returns:
            List of Episode objects (2-5, or 1 on fallback).
        """
        # Filter to behavioral events
        behavioral = [
            e for e in events
            if e.event_type.value not in SIMULATION_TYPES
        ]

        if len(behavioral) < self._min_events:
            return [self._make_single_episode(behavioral, trajectory_id)]

        # Build compact timeline for LLM
        timeline = self._build_timeline(behavioral)

        if not self._llm_fn:
            return [self._make_single_episode(behavioral, trajectory_id)]

        # Step 1: LLM boundary detection
        boundaries = self._detect_boundaries(timeline, len(behavioral))
        if not boundaries:
            return [self._make_single_episode(behavioral, trajectory_id)]

        # Step 2: Split events and generate narratives
        episodes = self._build_episodes(behavioral, boundaries, trajectory_id)
        return episodes

    def _build_timeline(self, events: list[NormalizedEvent]) -> str:
        """Build compact timeline text (~50 chars/event)."""
        lines = []
        for i, e in enumerate(events):
            et = e.event_type.value
            path = e.file_path
            if path:
                # Keep only filename
                path = path.rsplit("/", 1)[-1] if "/" in path else path
                path = path[:30]
            extra = ""
            if et == "file_write":
                extra = f" ({e.operation}, {e.content_length}ch)"
            elif et == "file_edit":
                extra = f" (+{e.lines_added}/-{e.lines_deleted})"
            elif et == "file_read":
                extra = f" ({e.content_length}ch)"
            elif et == "file_search":
                extra = ""
            lines.append(f"[{i}] {et} {path}{extra}")
        return "\n".join(lines)

    def _detect_boundaries(self, timeline: str, n_events: int) -> list[int] | None:
        """Use LLM to detect episode boundaries.

        Returns list of boundary indices (start of each new episode),
        or None on failure.
        """
        prompt = (
            "You are analyzing a file-system operation timeline. "
            f"There are {n_events} events. "
            f"Identify 2-{self._max_episodes} natural episode boundaries "
            "where the user shifts focus (e.g., from reading to writing, "
            "from one set of files to another, from creating to organizing).\n\n"
            f"## Timeline\n{timeline}\n\n"
            "## Instructions\n"
            "Return a JSON object with a single key \"boundaries\" containing "
            "a list of event indices where new episodes BEGIN. "
            "The first episode always starts at index 0 (do not include 0). "
            "Example: {\"boundaries\": [8, 15, 22]}\n"
            "Only return the JSON, nothing else."
        )

        try:
            result = self._llm_fn(prompt, system_prompt=None)
            text = result.strip()
            # Extract JSON
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                return None
            data = json.loads(text[start:end + 1])
            boundaries = data.get("boundaries", [])
            if not isinstance(boundaries, list):
                return None
            # Validate and sort
            valid = sorted(set(
                int(b) for b in boundaries
                if isinstance(b, (int, float)) and 0 < int(b) < n_events
            ))
            if not valid:
                return None
            # Limit to max_episodes - 1 boundaries
            return valid[:self._max_episodes - 1]
        except Exception:
            return None

    def _build_episodes(
        self,
        events: list[NormalizedEvent],
        boundaries: list[int],
        trajectory_id: str,
    ) -> list[Episode]:
        """Split events by boundaries and generate LLM summaries."""
        # Build segments
        starts = [0] + boundaries
        ends = boundaries + [len(events)]
        segments: list[tuple[int, int]] = list(zip(starts, ends))

        # Filter out too-small segments by merging with previous
        merged: list[tuple[int, int]] = []
        for s, e in segments:
            if merged and (e - s) < self._min_events:
                # Merge with previous
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        segments = merged

        # Generate narratives (EverMemos-style: title + content + summary)
        narratives = self._generate_narratives(events, segments)

        episodes = []
        for idx, (s, e) in enumerate(segments):
            seg_events = events[s:e]
            # Dominant actions
            action_counts: dict[str, int] = {}
            for ev in seg_events:
                t = ev.event_type.value
                action_counts[t] = action_counts.get(t, 0) + 1

            # Files involved (deduplicated)
            files: list[str] = []
            seen: set[str] = set()
            for ev in seg_events:
                if ev.file_path and ev.file_path not in seen:
                    seen.add(ev.file_path)
                    files.append(ev.file_path)

            narr = narratives[idx] if idx < len(narratives) else {
                "title": f"Episode {idx + 1}",
                "content": "",
                "summary": f"Episode {idx + 1}: events {s}-{e - 1}",
            }

            episodes.append(Episode(
                episode_id=str(uuid.uuid4())[:8],
                trajectory_id=trajectory_id,
                start_idx=s,
                end_idx=e - 1,
                event_count=e - s,
                summary=narr["summary"],
                title=narr["title"],
                content=narr["content"],
                dominant_actions=action_counts,
                files_involved=files[:10],  # cap file list
            ))

        return episodes

    @staticmethod
    def _anonymize_path(path: str) -> str:
        """Strip sandbox prefix from path."""
        if not path:
            return ""
        if "/sandbox/" in path:
            m = re.search(r"/sandbox/[^/]+/(.*)", path)
            return m.group(1) if m else path.rsplit("/", 1)[-1]
        return path

    def _events_to_narrative(self, events: list[NormalizedEvent]) -> str:
        """Convert NormalizedEvent list to natural language narrative for LLM.

        Uses only fields available on NormalizedEvent (not raw dicts).
        """
        lines = []
        for i, e in enumerate(events, 1):
            et = e.event_type.value
            path = self._anonymize_path(e.file_path)

            if et == "file_read":
                lines.append(
                    f"[{i}] Read file: {path} "
                    f"(view #{e.view_count}, {e.content_length} chars)"
                )
            elif et == "file_write":
                op = e.operation.capitalize() if e.operation else "Write"
                lines.append(
                    f"[{i}] {op} file: {path} "
                    f"({e.content_length} chars)"
                )
            elif et == "file_edit":
                lines.append(
                    f"[{i}] Edit file: {path} "
                    f"(+{e.lines_added}/-{e.lines_deleted} lines)"
                )
            elif et == "file_search":
                lines.append(f"[{i}] Search files")
            elif et == "file_browse":
                lines.append(f"[{i}] Browse directory: {path}")
            elif et == "file_rename":
                src = self._anonymize_path(e.source_file)
                tgt = self._anonymize_path(e.target_file)
                lines.append(f"[{i}] Rename: {src or path} -> {tgt}")
            elif et == "file_move":
                src = self._anonymize_path(e.source_file)
                tgt = self._anonymize_path(e.target_file)
                lines.append(f"[{i}] Move: {src or path} -> {tgt}")
            elif et == "dir_create":
                lines.append(f"[{i}] Create directory: {path} (depth={e.depth})")
            elif et == "file_delete":
                lines.append(f"[{i}] Delete file: {path}")
            elif et == "file_copy":
                src = self._anonymize_path(e.source_file)
                tgt = self._anonymize_path(e.target_file)
                lines.append(f"[{i}] Copy: {src or path} -> {tgt} (backup={e.is_backup})")
            elif et == "cross_file_reference":
                src = self._anonymize_path(e.source_file)
                tgt = self._anonymize_path(e.target_file)
                lines.append(f"[{i}] Cross-file ref: {src} -> {tgt} ({e.reference_type})")
            elif et == "context_switch":
                lines.append(f"[{i}] Context switch")
            else:
                lines.append(f"[{i}] {et}: {path}")
        return "\n".join(lines)

    def _generate_narratives(
        self,
        events: list[NormalizedEvent],
        segments: list[tuple[int, int]],
    ) -> list[dict[str, str]]:
        """Generate EverMemos-style narratives for each segment.

        Returns list of {title, content, summary} dicts, one per segment.
        """
        if not self._llm_fn:
            return [
                {"title": f"Episode {i + 1}", "content": "", "summary": f"Episode {i + 1}"}
                for i in range(len(segments))
            ]

        results: list[dict[str, str]] = []
        for i, (s, e) in enumerate(segments):
            seg_events = events[s:e]
            narrative_text = self._events_to_narrative(seg_events)

            prompt = (
                "You are converting file-system behavioral events into an episodic memory. "
                "Create a concise factual record of what the user did in this behavioral episode.\n\n"
                f"Events in this episode:\n{narrative_text}\n\n"
                "Follow these principles:\n"
                "1. Each episode should be a complete, independent behavioral unit\n"
                "2. Preserve all important information: file names, operations, patterns\n"
                "3. Use declarative language describing what the user did, not event format\n"
                "4. Highlight key behavioral choices (what they read first, how they organized, "
                "what they created)\n\n"
                "Return JSON:\n"
                "{\n"
                '  "title": "A concise descriptive title summarizing the activity (10-20 words)",\n'
                '  "content": "A factual third-person narrative of the user\'s behavior in this '
                'episode. Include specific file names, operation sequences, and behavioral '
                'patterns observed. 3-8 sentences.",\n'
                '  "summary": "One-sentence summary of the episode"\n'
                "}\n"
                "Only return the JSON, nothing else."
            )

            try:
                result = self._llm_fn(prompt, system_prompt=None)
                text = result.strip()
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    json_text = text[start:end + 1]
                    json_text = re.sub(r",\s*([}\]])", r"\1", json_text)
                    data = json.loads(json_text)
                    results.append({
                        "title": str(data.get("title", f"Episode {i + 1}")),
                        "content": str(data.get("content", "")),
                        "summary": str(data.get("summary", f"Episode {i + 1}")),
                    })
                else:
                    results.append({"title": f"Episode {i + 1}", "content": "", "summary": f"Episode {i + 1}"})
            except Exception:
                results.append({"title": f"Episode {i + 1}", "content": "", "summary": f"Episode {i + 1}"})

        return results

    def _make_single_episode(
        self,
        events: list[NormalizedEvent],
        trajectory_id: str,
    ) -> Episode:
        """Fallback: entire trajectory as one episode."""
        action_counts: dict[str, int] = {}
        files: list[str] = []
        seen: set[str] = set()
        for e in events:
            t = e.event_type.value
            action_counts[t] = action_counts.get(t, 0) + 1
            if e.file_path and e.file_path not in seen:
                seen.add(e.file_path)
                files.append(e.file_path)

        return Episode(
            episode_id=str(uuid.uuid4())[:8],
            trajectory_id=trajectory_id,
            start_idx=0,
            end_idx=max(len(events) - 1, 0),
            event_count=len(events),
            summary="Full trajectory as single episode",
            dominant_actions=action_counts,
            files_involved=files[:10],
        )
