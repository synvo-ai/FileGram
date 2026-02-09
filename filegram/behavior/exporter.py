"""JSON exporter for behavioral events.

This module handles the writing of behavioral events to JSON files,
media externalization (file content snapshots), and summary generation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .events import BehaviorEvent


class BehaviorExporter:
    """
    Exports behavioral events to JSON files.

    Each session gets its own directory with:
    - events.json: Event array (real-time, formatted for readability)
    - summary.json: Session summary (written at session end)
    - summary.md: Markdown conversation log
    - media/: Externalized file content (writes, edit before/after states)

    Events are written immediately as a formatted JSON array.
    """

    def __init__(
        self,
        session_id: str,
        output_directory: Path,
    ):
        """
        Initialize the exporter.

        Args:
            session_id: Unique session identifier
            output_directory: Base directory for session data
        """
        self.session_id = session_id
        self.session_dir = output_directory / session_id
        self.events_file = self.session_dir / "events.json"
        self.summary_file = self.session_dir / "summary.json"
        self.summary_md_file = self.session_dir / "summary.md"
        self.media_dir = self.session_dir / "media"

        self._events: list[dict[str, Any]] = []
        self._media_counter: int = 0
        self._initialized = False

    def _ensure_dir(self) -> None:
        """Ensure the session directory exists."""
        if not self._initialized:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True

    def _ensure_media_dir(self) -> None:
        """Ensure the media directory exists."""
        self._ensure_dir()
        self.media_dir.mkdir(parents=True, exist_ok=True)

    def _write_events(self) -> None:
        """Write all events to JSON file."""
        with open(self.events_file, "w", encoding="utf-8") as f:
            json.dump(self._events, f, indent=2, ensure_ascii=False)

    def write_event(self, event: BehaviorEvent) -> None:
        """
        Write a single event immediately to JSON file.

        The entire events array is rewritten each time for
        proper JSON formatting.

        Args:
            event: The behavioral event to write
        """
        self._ensure_dir()

        event_dict = event.to_dict()
        self._events.append(event_dict)

        # Rewrite the entire file (ensures valid JSON at all times)
        self._write_events()

    # ============== Media Externalization ==============

    def write_media(self, label: str, file_path: str, content: str, heading: str = "") -> str:
        """
        Write content to a numbered media file.

        Args:
            label: Label for the file (e.g., "write", "old", "new")
            file_path: Original file path for reference
            content: File content to externalize
            heading: Markdown heading (e.g., "创建文件", "编辑前", "编辑后")

        Returns:
            The media filename (e.g., "0001_write.md")
        """
        self._ensure_media_dir()

        self._media_counter += 1
        filename = f"{self._media_counter:04d}_{label}.md"
        media_path = self.media_dir / filename

        # Determine code fence language from file extension
        ext = Path(file_path).suffix.lstrip(".")
        if not ext:
            ext = ""

        # Write media file in OpenCode's established format
        if not heading:
            heading = label.capitalize()

        media_content = f"# {heading}\n\n**文件**: `{file_path}`\n\n```{ext}\n{content}\n```\n"

        with open(media_path, "w", encoding="utf-8") as f:
            f.write(media_content)

        return filename

    def externalize_write(self, file_path: str, content: str) -> str:
        """
        Externalize a file write operation to media.

        Args:
            file_path: Path of the written file
            content: Content that was written

        Returns:
            The media filename reference
        """
        return self.write_media("write", file_path, content, heading="创建文件")

    def externalize_edit(self, file_path: str, before: str, after: str) -> tuple[str, str]:
        """
        Externalize a file edit operation to media (before and after states).

        Args:
            file_path: Path of the edited file
            before: Content before the edit
            after: Content after the edit

        Returns:
            Tuple of (old_ref, new_ref) media filenames
        """
        old_ref = self.write_media("old", file_path, before, heading="编辑前")
        new_ref = self.write_media("new", file_path, after, heading="编辑后")
        return old_ref, new_ref

    # ============== Summary Generation ==============

    def write_summary(self, summary: dict[str, Any]) -> None:
        """
        Write session summary.

        Args:
            summary: Dictionary containing session summary data
        """
        self._ensure_dir()

        # Add event count to summary
        summary["total_events"] = len(self._events)

        # Write summary
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def write_summary_md(
        self,
        session_id: str,
        profile_id: str,
        model_name: str,
        start_time_ms: float,
        end_time_ms: float,
    ) -> None:
        """
        Write a markdown conversation log of the session.

        Args:
            session_id: Session identifier
            profile_id: Profile used
            model_name: Model used
            start_time_ms: Session start timestamp in milliseconds
            end_time_ms: Session end timestamp in milliseconds
        """
        self._ensure_dir()

        start_dt = datetime.fromtimestamp(start_time_ms / 1000, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_time_ms / 1000, tz=timezone.utc)
        duration_s = (end_time_ms - start_time_ms) / 1000

        lines: list[str] = []
        lines.append("# Session Summary\n")
        lines.append(f"**Session ID**: {session_id}")
        lines.append(f"**Profile**: {profile_id}")
        lines.append(f"**Model**: {model_name}")
        lines.append(f"**Started**: {start_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"**Ended**: {end_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"**Duration**: {duration_s:.1f}s")
        lines.append(f"**Total Events**: {len(self._events)}")
        lines.append("")
        lines.append("---\n")

        # Build conversation log from events
        lines.append("## Event Log\n")

        current_iteration = 0
        for event in self._events:
            event_type = event.get("event_type", "")

            if event_type == "iteration_start":
                current_iteration = event.get("iteration_number", 0)
                lines.append(f"### Iteration {current_iteration}\n")

            elif event_type == "tool_call":
                tool_name = event.get("tool_name", "unknown")
                success = event.get("success", True)
                exec_time = event.get("execution_time_ms", 0)
                status = "OK" if success else "FAIL"
                lines.append(f"- **{tool_name}** ({exec_time}ms) [{status}]")

            elif event_type == "file_read":
                fp = event.get("file_path", "")
                view_count = event.get("view_count", 1)
                lines.append(f"- Read `{fp}` (view #{view_count})")

            elif event_type == "file_write":
                fp = event.get("file_path", "")
                op = event.get("operation", "write")
                media_ref = event.get("media_ref", "")
                ref_note = f" → [{media_ref}](media/{media_ref})" if media_ref else ""
                lines.append(f"- Write `{fp}` ({op}){ref_note}")

            elif event_type == "file_edit":
                fp = event.get("file_path", "")
                added = event.get("lines_added", 0)
                deleted = event.get("lines_deleted", 0)
                ref_old = event.get("media_ref_old", "")
                ref_new = event.get("media_ref_new", "")
                ref_note = ""
                if ref_old and ref_new:
                    ref_note = f" → [{ref_old}](media/{ref_old}), [{ref_new}](media/{ref_new})"
                lines.append(f"- Edit `{fp}` (+{added}/-{deleted}){ref_note}")

            elif event_type == "file_search":
                search_type = event.get("search_type", "")
                query = event.get("query", "")
                matched = event.get("files_matched", 0)
                lines.append(f"- Search ({search_type}): `{query}` → {matched} matches")

            elif event_type == "llm_response":
                input_t = event.get("input_tokens", 0)
                output_t = event.get("output_tokens", 0)
                resp_time = event.get("response_time_ms", 0)
                lines.append(f"- LLM response ({resp_time}ms, {input_t}→{output_t} tokens)")

            elif event_type == "iteration_end":
                duration = event.get("duration_ms", 0)
                tools = event.get("tools_called", 0)
                lines.append(f"- *Iteration {current_iteration} end: {duration}ms, {tools} tools*\n")

            elif event_type == "context_switch":
                from_f = event.get("from_file", "?")
                to_f = event.get("to_file", "?")
                lines.append(f"- Context switch: `{from_f}` → `{to_f}`")

            elif event_type == "compaction_triggered":
                reason = event.get("reason", "")
                tokens_saved = event.get("tokens_saved", 0)
                lines.append(f"- Compaction: {reason} (saved {tokens_saved} tokens)")

            elif event_type == "session_start":
                target_dir = event.get("target_directory", "")
                lines.append(f"- Session started (target: `{target_dir}`)")

            elif event_type == "session_end":
                total_iters = event.get("total_iterations", 0)
                total_tools = event.get("total_tool_calls", 0)
                dur = event.get("duration_ms", 0)
                lines.append(f"- Session ended ({total_iters} iterations, {total_tools} tools, {dur}ms)")

        lines.append("")

        with open(self.summary_md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @property
    def event_count(self) -> int:
        """Return the number of events written."""
        return len(self._events)


__all__ = ["BehaviorExporter"]
