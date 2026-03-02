"""Base adapter interface for memory system evaluation.

Adapted from EverMemOS evaluation patterns but redesigned for
file-system behavioral trajectories instead of conversations.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# Default: include full file content without truncation.
# Each method's ingest/retrieval handles its own token budget.
# Set to a positive int to truncate, 0 to omit content entirely.
DEFAULT_MAX_CONTENT_CHARS = -1  # -1 = full content, no truncation


class BaseAdapter(ABC):
    """Abstract base for all memory system adapters.

    Lifecycle:
        1. __init__(config, llm_fn)  — configure the adapter
        2. ingest(trajectories)      — feed trajectory data into the memory system
        3. infer_profile(query)      — ask the system to reconstruct a user profile
        4. reset()                   — clear state between profiles
    """

    def __init__(self, config: dict[str, Any] | None = None, llm_fn=None):
        self.config = config or {}
        self.llm_fn = llm_fn  # callback: (prompt, system_prompt=None) -> str
        self.name: str = self.__class__.__name__
        # Cost tracking
        self.llm_calls_ingest: int = 0
        self.llm_calls_infer: int = 0
        self.tokens_ingest: int = 0
        self.tokens_infer: int = 0
        # Progress callback: set externally (e.g. tqdm bar.update)
        self._progress_callback = None

    def _on_trajectory_done(self, task_id: str = ""):
        """Call after processing each trajectory during ingest."""
        if self._progress_callback:
            self._progress_callback(task_id)

    @abstractmethod
    def ingest(self, trajectories: list[dict[str, Any]]) -> None:
        """Ingest trajectory data into the memory system.

        Args:
            trajectories: List of trajectory dicts, each containing:
                - "task_id": str
                - "events": list of event dicts from events.json
                - "summary": dict from summary.json (optional)
        """

    @abstractmethod
    def infer_profile(self, profile_attributes: list[str]) -> dict[str, str]:
        """Infer a user profile from stored memories.

        Args:
            profile_attributes: List of attribute names to infer
                (e.g., ["reading_strategy", "edit_strategy", ...])

        Returns:
            Dict mapping attribute name to inferred value.
        """

    def _call_llm(self, prompt: str, system_prompt: str | None = None, phase: str = "ingest") -> str:
        """Call llm_fn with cost tracking.

        Args:
            prompt: The prompt text.
            system_prompt: Optional system prompt.
            phase: "ingest" or "infer" for cost tracking.
        """
        if not self.llm_fn:
            raise RuntimeError(f"{self.name}: llm_fn not set — cannot make LLM calls")
        result = self.llm_fn(prompt, system_prompt=system_prompt)
        if phase == "ingest":
            self.llm_calls_ingest += 1
            self.tokens_ingest += len(prompt) // 4  # rough estimate
        else:
            self.llm_calls_infer += 1
            self.tokens_infer += len(prompt) // 4
        return result

    # ---- Ingest cache (pickle-based) ----

    def _get_ingest_state(self) -> dict[str, Any]:
        """Return adapter state to cache after ingest. Override in subclasses."""
        return {}

    def _set_ingest_state(self, state: dict[str, Any]) -> None:
        """Restore adapter state from cache. Override in subclasses."""

    def save_ingest_cache(self, cache_path: Path) -> None:
        """Save ingest state to disk via pickle."""
        state = self._get_ingest_state()
        if not state:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_ingest_cache(self, cache_path: Path) -> bool:
        """Load ingest state from disk. Returns True if cache hit."""
        if not cache_path.exists():
            return False
        try:
            with open(cache_path, "rb") as f:
                state = pickle.load(f)
            self._set_ingest_state(state)
            return True
        except Exception as e:
            print(f"    [cache] Failed to load {cache_path.name}: {e}")
            return False

    def reset(self) -> None:
        """Reset adapter state between profiles. Override if needed."""
        self.llm_calls_ingest = 0
        self.llm_calls_infer = 0
        self.tokens_ingest = 0
        self.tokens_infer = 0

    def get_system_info(self) -> dict[str, Any]:
        """Return metadata about this adapter for result recording."""
        return {
            "adapter": self.name,
            "config": self.config,
            "llm_calls_ingest": self.llm_calls_ingest,
            "llm_calls_infer": self.llm_calls_infer,
            "tokens_ingest": self.tokens_ingest,
            "tokens_infer": self.tokens_infer,
        }

    # ---- Utility methods for subclasses ----

    @staticmethod
    def load_trajectories(
        signals_dir: Path, profile_id: str, task_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Load trajectory data for a profile from signal/.

        Args:
            signals_dir: Path to signal/
            profile_id: e.g. "p1"
            task_ids: If provided, only load these tasks. Otherwise load all.

        Returns:
            List of {"task_id": str, "events": list, "summary": dict}
        """
        trajectories = []
        for traj_dir in sorted(signals_dir.iterdir()):
            if not traj_dir.is_dir():
                continue
            # Directory name: {profile_id}_{task_id}
            dir_name = traj_dir.name
            if not dir_name.startswith(profile_id + "_"):
                continue
            task_id = dir_name[len(profile_id) + 1 :]
            if task_ids and task_id not in task_ids:
                continue

            events_file = traj_dir / "events.json"
            summary_file = traj_dir / "summary.json"

            events = []
            summary = {}
            if events_file.exists():
                events = json.loads(events_file.read_text(encoding="utf-8"))
            if summary_file.exists():
                summary = json.loads(summary_file.read_text(encoding="utf-8"))

            # Resolve media_ref → attach actual file content to events
            BaseAdapter._resolve_media_refs(traj_dir, events)

            trajectories.append(
                {
                    "task_id": task_id,
                    "events": events,
                    "summary": summary,
                    "path": str(traj_dir),
                }
            )

        return trajectories

    @staticmethod
    def _resolve_one_ref(media_dir: Path, ref: dict | None) -> str | None:
        """Resolve a single media_ref dict to its text content."""
        if not ref:
            return None
        ref_path = ref.get("path")
        if not ref_path:
            return None
        full_path = media_dir / ref_path
        if not full_path.exists():
            return None
        try:
            return full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return None

    @staticmethod
    def _resolve_media_refs(traj_dir: Path, events: list[dict[str, Any]]) -> None:
        """Resolve media_ref fields in events to actual blob/diff content.

        Handles:
        - media_ref      → _resolved_content  (file_write create/overwrite)
        - media_ref_diff → _resolved_diff     (file_edit unified diff)
        - media_ref_old  → _resolved_content_old  (file_edit before-blob)
        - media_ref_new  → _resolved_content_new  (file_edit after-blob)

        Modifies events in-place.
        """
        media_dir = traj_dir / "media"
        if not media_dir.exists():
            return

        _resolve = BaseAdapter._resolve_one_ref
        for event in events:
            # file_write: media_ref → _resolved_content
            c = _resolve(media_dir, event.get("media_ref"))
            if c is not None:
                event["_resolved_content"] = c

            # file_edit: media_ref_diff → _resolved_diff
            c = _resolve(media_dir, event.get("media_ref_diff"))
            if c is not None:
                event["_resolved_diff"] = c

            # file_edit: media_ref_old/new → _resolved_content_old/new
            c = _resolve(media_dir, event.get("media_ref_old"))
            if c is not None:
                event["_resolved_content_old"] = c
            c = _resolve(media_dir, event.get("media_ref_new"))
            if c is not None:
                event["_resolved_content_new"] = c

    @staticmethod
    def truncate_content(content: str, max_chars: int) -> str:
        """Truncate content with a notice if it exceeds max_chars."""
        if len(content) <= max_chars:
            return content
        return content[:max_chars] + f"\n... [truncated, {len(content)} chars total]"

    @staticmethod
    def filter_behavioral_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter out simulation metadata events, keep only behavioral signals.

        Removes: tool_call, llm_response, compaction_triggered,
                 session_start, session_end, iteration_start, iteration_end,
                 error_encounter, error_response, fs_snapshot
        """
        simulation_types = {
            "tool_call",
            "llm_response",
            "compaction_triggered",
            "session_start",
            "session_end",
            "iteration_start",
            "iteration_end",
            "error_encounter",
            "error_response",
            "fs_snapshot",
        }
        return [e for e in events if e.get("event_type") not in simulation_types]

    @staticmethod
    def _anonymize_path(path: str) -> str:
        """Strip absolute sandbox prefix, returning only the relative file path.

        Removes everything up to and including the sandbox directory name
        (e.g. 'p1_methodical_T-01/') to prevent profile name leakage.
        Falls back to basename if pattern not found.
        """
        if not path or path == "?":
            return path
        import re

        # Match sandbox dir pattern: .../<profile>_<task>/rest
        m = re.search(r"/sandbox/[^/]+/(.*)", path)
        if m:
            return m.group(1) or "."
        # Fallback: just use filename
        return path.rsplit("/", 1)[-1] if "/" in path else path

    @staticmethod
    def events_to_narrative(
        events: list[dict[str, Any]],
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    ) -> str:
        """Convert behavioral events into a natural language narrative.

        This is the default serialization for baselines that expect text input.
        Includes resolved file content (from media blobs/diffs) when available.
        File paths are anonymized to prevent profile name leakage.

        Args:
            events: List of event dicts (with optional _resolved_content).
            max_content_chars: Max chars of file content to include per event.
                Set to 0 to omit content entirely.
        """
        _anon = BaseAdapter._anonymize_path
        lines = []
        for i, event in enumerate(events, 1):
            et = event.get("event_type", "unknown")

            if et == "file_read":
                lines.append(
                    f"[{i}] Read file: {_anon(event.get('file_path', '?'))} "
                    f"(lines {event.get('view_range', [])}, "
                    f"view #{event.get('view_count', 1)}, "
                    f"{event.get('content_length', 0)} chars)"
                )
            elif et == "file_write":
                op = event.get("operation", "write")
                line = (
                    f"[{i}] {op.capitalize()} file: {_anon(event.get('file_path', '?'))} "
                    f"({event.get('content_length', 0)} chars)"
                )
                content = event.get("_resolved_content")
                if content and max_content_chars != 0:
                    preview = (
                        content if max_content_chars < 0 else BaseAdapter.truncate_content(content, max_content_chars)
                    )
                    line += f"\n--- content ---\n{preview}\n--- end ---"
                lines.append(line)
            elif et == "file_edit":
                line = (
                    f"[{i}] Edit file: {_anon(event.get('file_path', '?'))} "
                    f"(+{event.get('lines_added', 0)}/-{event.get('lines_deleted', 0)} lines, "
                    f"tool: {event.get('edit_tool', '?')})"
                )
                # Prefer diff; fall back to _resolved_content for legacy data
                diff = event.get("_resolved_diff") or event.get("_resolved_content")
                if diff and max_content_chars != 0:
                    preview = diff if max_content_chars < 0 else BaseAdapter.truncate_content(diff, max_content_chars)
                    line += f"\n--- diff ---\n{preview}\n--- end ---"
                lines.append(line)
            elif et == "file_search":
                lines.append(
                    f"[{i}] Search ({event.get('search_type', '?')}): "
                    f"query='{event.get('query', '')}', "
                    f"{event.get('files_matched', 0)} matches"
                )
            elif et == "file_browse":
                lines.append(
                    f"[{i}] Browse directory: {_anon(event.get('directory_path', '?'))} "
                    f"({event.get('files_listed', 0)} files, depth={event.get('depth', 0)})"
                )
            elif et == "file_rename":
                lines.append(
                    f"[{i}] Rename: {_anon(event.get('old_path', '?'))} -> {_anon(event.get('new_path', '?'))}"
                )
            elif et == "file_move":
                lines.append(f"[{i}] Move: {_anon(event.get('old_path', '?'))} -> {_anon(event.get('new_path', '?'))}")
            elif et == "dir_create":
                lines.append(
                    f"[{i}] Create directory: {_anon(event.get('dir_path', '?'))} (depth={event.get('depth', 0)})"
                )
            elif et == "file_delete":
                lines.append(f"[{i}] Delete file: {_anon(event.get('file_path', '?'))}")
            elif et == "file_copy":
                lines.append(
                    f"[{i}] Copy: {_anon(event.get('source_path', '?'))} -> {_anon(event.get('dest_path', '?'))} "
                    f"(backup={event.get('is_backup', False)})"
                )
            elif et == "fs_snapshot":
                lines.append(
                    f"[{i}] FS Snapshot: {event.get('total_files', 0)} files, "
                    f"max_depth={event.get('max_depth', 0)}, "
                    f"types={event.get('file_count_by_type', {})}"
                )
            elif et == "context_switch":
                lines.append(
                    f"[{i}] Context switch: {_anon(event.get('from_file', '?'))} -> "
                    f"{_anon(event.get('to_file', '?'))} (trigger: {event.get('trigger', '?')})"
                )
            elif et == "cross_file_reference":
                lines.append(
                    f"[{i}] Cross-file ref: {_anon(event.get('source_file', '?'))} -> "
                    f"{_anon(event.get('target_file', '?'))} ({event.get('reference_type', '?')})"
                )
            elif et == "error_encounter":
                lines.append(f"[{i}] Error: {event.get('error_type', '?')} - {event.get('context', '?')}")
            elif et == "error_response":
                lines.append(
                    f"[{i}] Error recovery: strategy={event.get('strategy', '?')}, "
                    f"success={event.get('resolution_successful', False)}"
                )
            else:
                lines.append(
                    f"[{i}] {et}: {json.dumps({k: v for k, v in event.items() if k not in ('event_id', 'timestamp', 'session_id', 'profile_id', 'message_id', 'model_provider', 'model_name')}, ensure_ascii=False)}"
                )

        return "\n".join(lines)
