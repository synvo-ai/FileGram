"""Behavioral signal collector for tracking agent execution patterns.

This module provides the BehaviorCollector class that aggregates events
during agent execution and maintains session-level statistics for
persona modeling and memory pipeline research.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .events import (
    CompactionTriggeredEvent,
    ContextSwitchEvent,
    FileEditEvent,
    FileReadEvent,
    FileSearchEvent,
    FileWriteEvent,
    IterationEndEvent,
    IterationStartEvent,
    LLMResponseEvent,
    SessionEndEvent,
    SessionStartEvent,
    ToolCallEvent,
    compute_file_hash,
    get_directory_depth,
    get_file_type,
)
from .exporter import BehaviorExporter

if TYPE_CHECKING:
    pass


@dataclass
class FileAccessStats:
    """Statistics for file access tracking."""

    view_count: int = 0
    last_view_time: float = 0.0  # timestamp in ms

    def record_view(self) -> tuple[int, int | None]:
        """Record a view and return (view_count, revisit_interval_ms)."""
        now = time.time() * 1000
        interval = None
        if self.last_view_time > 0:
            interval = int(now - self.last_view_time)

        self.view_count += 1
        self.last_view_time = now

        return self.view_count, interval


@dataclass
class SessionStats:
    """Aggregated statistics for a session."""

    # Timing
    session_start_time: float = 0.0
    session_end_time: float = 0.0

    # Iteration tracking
    total_iterations: int = 0
    current_iteration: int = 0
    iteration_start_time: float = 0.0

    # Tool tracking
    total_tool_calls: int = 0
    tool_success_count: int = 0
    tool_sequence: list[str] = field(default_factory=list)
    tool_usage_frequency: dict[str, int] = field(default_factory=dict)
    current_iteration_tool_count: int = 0
    current_iteration_has_error: bool = False

    # File tracking
    files_read: set[str] = field(default_factory=set)
    files_modified: set[str] = field(default_factory=set)
    files_created: list[str] = field(default_factory=list)
    total_lines_added: int = 0
    total_lines_deleted: int = 0

    # Token tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Context tracking
    current_file: str | None = None
    context_switch_count: int = 0

    # Compaction
    compaction_count: int = 0


class BehaviorCollector:
    """
    Collects behavioral signals during agent execution.

    This collector tracks:
    - File operations (read, write, edit, search)
    - Tool calls and their outcomes
    - Iteration timing
    - LLM response metrics
    - Context switches
    - Compaction events

    Events are written in real-time to JSONL files for later analysis.
    """

    def __init__(
        self,
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        target_directory: Path,
        output_directory: Path | None = None,
        enabled: bool = True,
    ):
        """
        Initialize the behavior collector.

        Args:
            session_id: Unique session identifier
            profile_id: Current profile ID (or "default")
            model_provider: LLM provider name
            model_name: Specific model name
            target_directory: Working directory for the session
            output_directory: Where to write behavior data (default: data/behavior/sessions/)
            enabled: Whether to collect events
        """
        self.session_id = session_id
        self.profile_id = profile_id
        self.model_provider = model_provider
        self.model_name = model_name
        self.target_directory = target_directory
        self.enabled = enabled

        # Current message context
        self._current_message_id: str = ""

        # File access tracking
        self._file_access: dict[str, FileAccessStats] = {}

        # Session statistics
        self.stats = SessionStats()
        self.stats.session_start_time = time.time() * 1000

        # Initialize exporter
        # Use project root (where filegram package is) for output, not target_directory
        if output_directory is None:
            # Find project root by looking for filegram package location
            import filegram

            project_root = Path(filegram.__file__).parent.parent
            output_directory = project_root / "data" / "behavior" / "sessions"

        self.exporter = BehaviorExporter(
            session_id=session_id,
            output_directory=output_directory,
        )

        # Emit SESSION_START event
        if self.enabled:
            event = SessionStartEvent.create(
                session_id=self.session_id,
                profile_id=self.profile_id,
                model_provider=self.model_provider,
                model_name=self.model_name,
                target_directory=str(self.target_directory),
            )
            self.exporter.write_event(event)

    def set_message_id(self, message_id: str) -> None:
        """Set the current message ID for event correlation."""
        self._current_message_id = message_id

    def update_model(self, provider: str, model: str) -> None:
        """Update the model information (for runtime switching)."""
        self.model_provider = provider
        self.model_name = model

    def update_profile(self, profile_id: str) -> None:
        """Update the profile ID (for runtime switching)."""
        self.profile_id = profile_id

    # ============== File Events ==============

    def record_file_read(
        self,
        file_path: str,
        view_range: tuple[int, int],
        content_length: int,
    ) -> None:
        """Record a file read event."""
        if not self.enabled:
            return

        # Track file access stats
        if file_path not in self._file_access:
            self._file_access[file_path] = FileAccessStats()

        view_count, revisit_interval = self._file_access[file_path].record_view()

        # Track context switch
        if self.stats.current_file and self.stats.current_file != file_path:
            self._record_context_switch(
                from_file=self.stats.current_file,
                to_file=file_path,
                trigger="file_read",
            )
        self.stats.current_file = file_path
        self.stats.files_read.add(file_path)

        # Create and export event
        event = FileReadEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            file_path=file_path,
            file_type=get_file_type(file_path),
            directory_depth=get_directory_depth(file_path, str(self.target_directory)),
            view_count=view_count,
            view_range=view_range,
            content_length=content_length,
            revisit_interval_ms=revisit_interval,
        )
        self.exporter.write_event(event)

    def record_file_write(
        self,
        file_path: str,
        operation: str,
        content_length: int,
        before_content: str | None = None,
        after_content: str | None = None,
    ) -> None:
        """Record a file write event."""
        if not self.enabled:
            return

        before_hash = compute_file_hash(before_content) if before_content else None
        after_hash = compute_file_hash(after_content) if after_content else None

        # Externalize content to media/
        media_ref = None
        if after_content is not None:
            media_ref = self.exporter.externalize_write(file_path, after_content)

        # Track stats
        if operation == "create":
            self.stats.files_created.append(file_path)
        self.stats.files_modified.add(file_path)

        event = FileWriteEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            file_path=file_path,
            file_type=get_file_type(file_path),
            directory_depth=get_directory_depth(file_path, str(self.target_directory)),
            operation=operation,
            content_length=content_length,
            before_hash=before_hash,
            after_hash=after_hash,
            media_ref=media_ref,
        )
        self.exporter.write_event(event)

    def record_file_edit(
        self,
        file_path: str,
        edit_tool: str,
        lines_added: int,
        lines_deleted: int,
        diff_summary: str,
        before_content: str | None = None,
        after_content: str | None = None,
    ) -> None:
        """Record a file edit event."""
        if not self.enabled:
            return

        before_hash = compute_file_hash(before_content) if before_content else None
        after_hash = compute_file_hash(after_content) if after_content else None

        # Externalize content to media/
        media_ref_old = None
        media_ref_new = None
        if before_content is not None and after_content is not None:
            media_ref_old, media_ref_new = self.exporter.externalize_edit(file_path, before_content, after_content)

        # Track stats
        self.stats.files_modified.add(file_path)
        self.stats.total_lines_added += lines_added
        self.stats.total_lines_deleted += lines_deleted

        event = FileEditEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            file_path=file_path,
            file_type=get_file_type(file_path),
            edit_tool=edit_tool,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            lines_modified=min(lines_added, lines_deleted),
            diff_summary=diff_summary,
            before_hash=before_hash,
            after_hash=after_hash,
            media_ref_old=media_ref_old,
            media_ref_new=media_ref_new,
        )
        self.exporter.write_event(event)

    def record_file_search(
        self,
        search_type: str,
        query: str,
        files_matched: int,
    ) -> None:
        """Record a file search event."""
        if not self.enabled:
            return

        event = FileSearchEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            search_type=search_type,
            query=query,
            files_matched=files_matched,
        )
        self.exporter.write_event(event)

    # ============== Tool Events ==============

    def record_tool_call(
        self,
        tool_name: str,
        tool_parameters: dict[str, Any],
        execution_time_ms: int,
        success: bool,
        error_type: str | None = None,
        error_message: str | None = None,
        retry_count: int = 0,
    ) -> None:
        """Record a tool call event."""
        if not self.enabled:
            return

        # Update stats
        self.stats.total_tool_calls += 1
        self.stats.current_iteration_tool_count += 1
        if success:
            self.stats.tool_success_count += 1
        else:
            self.stats.current_iteration_has_error = True

        self.stats.tool_sequence.append(tool_name)
        self.stats.tool_usage_frequency[tool_name] = self.stats.tool_usage_frequency.get(tool_name, 0) + 1

        event = ToolCallEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            tool_name=tool_name,
            tool_parameters=tool_parameters,
            execution_time_ms=execution_time_ms,
            success=success,
            error_type=error_type,
            error_message=error_message,
            retry_count=retry_count,
            sequence_position=self.stats.total_tool_calls,
        )
        self.exporter.write_event(event)

    # ============== Timing Events ==============

    def record_iteration_start(self) -> None:
        """Record the start of an iteration."""
        if not self.enabled:
            return

        self.stats.total_iterations += 1
        self.stats.current_iteration = self.stats.total_iterations
        self.stats.iteration_start_time = time.time() * 1000
        self.stats.current_iteration_tool_count = 0
        self.stats.current_iteration_has_error = False

        event = IterationStartEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            iteration_number=self.stats.current_iteration,
        )
        self.exporter.write_event(event)

    def record_iteration_end(self) -> None:
        """Record the end of an iteration."""
        if not self.enabled:
            return

        now = time.time() * 1000
        duration_ms = int(now - self.stats.iteration_start_time)

        event = IterationEndEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            iteration_number=self.stats.current_iteration,
            duration_ms=duration_ms,
            tools_called=self.stats.current_iteration_tool_count,
            has_tool_error=self.stats.current_iteration_has_error,
        )
        self.exporter.write_event(event)

    def record_llm_response(
        self,
        response_time_ms: int,
        input_tokens: int,
        output_tokens: int,
        has_reasoning: bool,
        stop_reason: str,
    ) -> None:
        """Record an LLM response event."""
        if not self.enabled:
            return

        # Update stats
        self.stats.total_input_tokens += input_tokens
        self.stats.total_output_tokens += output_tokens

        event = LLMResponseEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            response_time_ms=response_time_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            has_reasoning=has_reasoning,
            stop_reason=stop_reason,
        )
        self.exporter.write_event(event)

    # ============== Context Events ==============

    def _record_context_switch(
        self,
        from_file: str | None,
        to_file: str,
        trigger: str,
    ) -> None:
        """Record a context switch event (internal)."""
        self.stats.context_switch_count += 1

        event = ContextSwitchEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            from_file=from_file,
            to_file=to_file,
            trigger=trigger,
            switch_count=self.stats.context_switch_count,
        )
        self.exporter.write_event(event)

    def record_compaction(
        self,
        reason: str,
        messages_before: int,
        messages_after: int,
        tokens_saved: int,
    ) -> None:
        """Record a compaction event."""
        if not self.enabled:
            return

        self.stats.compaction_count += 1

        event = CompactionTriggeredEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            reason=reason,
            messages_before=messages_before,
            messages_after=messages_after,
            tokens_saved=tokens_saved,
        )
        self.exporter.write_event(event)

    # ============== Session Events ==============

    def finalize(self) -> dict[str, Any]:
        """Finalize the session and generate summary."""
        if not self.enabled:
            return {}

        self.stats.session_end_time = time.time() * 1000

        # Calculate summary
        total_duration_ms = int(self.stats.session_end_time - self.stats.session_start_time)
        tool_success_rate = (
            self.stats.tool_success_count / self.stats.total_tool_calls if self.stats.total_tool_calls > 0 else 1.0
        )

        # Emit SESSION_END event
        event = SessionEndEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            total_iterations=self.stats.total_iterations,
            total_tool_calls=self.stats.total_tool_calls,
            duration_ms=total_duration_ms,
        )
        self.exporter.write_event(event)

        summary = {
            "session_id": self.session_id,
            "profile_id": self.profile_id,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "total_duration_ms": total_duration_ms,
            "total_iterations": self.stats.total_iterations,
            "total_tool_calls": self.stats.total_tool_calls,
            "tool_success_rate": tool_success_rate,
            "tool_usage_frequency": self.stats.tool_usage_frequency,
            "tool_sequence": self.stats.tool_sequence,
            "unique_files_read": len(self.stats.files_read),
            "unique_files_modified": len(self.stats.files_modified),
            "files_created": self.stats.files_created,
            "files_modified": list(self.stats.files_modified),
            "total_lines_added": self.stats.total_lines_added,
            "total_lines_deleted": self.stats.total_lines_deleted,
            "compaction_count": self.stats.compaction_count,
            "total_input_tokens": self.stats.total_input_tokens,
            "total_output_tokens": self.stats.total_output_tokens,
            "context_switch_count": self.stats.context_switch_count,
        }

        # Write summary.json
        self.exporter.write_summary(summary)

        # Write summary.md
        self.exporter.write_summary_md(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_name=self.model_name,
            start_time_ms=self.stats.session_start_time,
            end_time_ms=self.stats.session_end_time,
        )

        return summary


__all__ = [
    "BehaviorCollector",
    "FileAccessStats",
    "SessionStats",
]
