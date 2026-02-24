"""Behavioral signal collector for tracking agent execution patterns.

This module provides the BehaviorCollector class that aggregates events
during agent execution and maintains session-level statistics for
persona modeling and memory pipeline research.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .events import (
    CompactionTriggeredEvent,
    ContextSwitchEvent,
    CrossFileReferenceEvent,
    DirCreateEvent,
    ErrorEncounterEvent,
    ErrorResponseEvent,
    FileBrowseEvent,
    FileCopyEvent,
    FileDeleteEvent,
    FileEditEvent,
    FileMoveEvent,
    FileReadEvent,
    FileRenameEvent,
    FileSearchEvent,
    FileWriteEvent,
    FsSnapshotEvent,
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

    # File organization tracking
    files_deleted: list[str] = field(default_factory=list)
    files_renamed: list[tuple[str, str]] = field(default_factory=list)
    dirs_created: list[str] = field(default_factory=list)

    # Error tracking
    error_count: int = 0
    error_recovery_count: int = 0


def _detect_naming_pattern(old_name: str, new_name: str) -> str:
    """Detect naming pattern change between old and new filenames."""
    if "_" in old_name and "-" in new_name:
        return "snake_to_kebab"
    if "-" in old_name and "_" in new_name:
        return "kebab_to_snake"
    if "_" in old_name and old_name.lower() != old_name:
        return "mixed_to_other"
    if old_name.replace("_", "") == new_name.replace("_", "").lower():
        return "case_change"
    if old_name.lower() == new_name.lower():
        return "case_change"
    return "other"


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

        # Cross-file reference tracking
        self._last_file_operation: tuple[str, str, float] | None = None  # (file_path, op_type, timestamp_ms)
        self._last_search_files: list[str] = []  # files matched in last search

        # Error tracking state
        self._pending_error: tuple[str, str, float] | None = None  # (error_event_id, tool_name, timestamp_ms)

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

            # Capture initial filesystem snapshot
            self.record_fs_snapshot()

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

        # Detect cross-file reference
        self._maybe_emit_cross_file_reference(file_path, "read")

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

        # Track for cross-file detection
        self._last_file_operation = (file_path, "read", time.time() * 1000)

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
        diff_ref = None
        if before_content is not None and after_content is not None:
            edit_refs = self.exporter.externalize_edit(file_path, before_content, after_content)
            if isinstance(edit_refs, dict):
                media_ref_old = edit_refs.get("before")
                media_ref_new = edit_refs.get("after")
                diff_ref = edit_refs.get("diff")
            else:
                # Legacy tuple format
                media_ref_old, media_ref_new = edit_refs
                diff_ref = None

        # Detect cross-file reference
        self._maybe_emit_cross_file_reference(file_path, "edit")

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
            media_ref_diff=diff_ref,
        )
        self.exporter.write_event(event)

        # Track for cross-file detection
        self._last_file_operation = (file_path, "edit", time.time() * 1000)

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

    # ============== File Organization Events ==============

    def record_file_browse(
        self,
        directory_path: str,
        files_listed: int,
    ) -> None:
        """Record a directory browse/listing event."""
        if not self.enabled:
            return

        depth = get_directory_depth(directory_path + "/x", str(self.target_directory))

        event = FileBrowseEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            directory_path=directory_path,
            files_listed=files_listed,
            depth=depth,
        )
        self.exporter.write_event(event)

    def record_file_rename(
        self,
        old_path: str,
        new_path: str,
    ) -> None:
        """Record a file rename event (same directory)."""
        if not self.enabled:
            return

        # Detect naming pattern change
        old_name = Path(old_path).stem
        new_name = Path(new_path).stem
        naming_pattern_change = _detect_naming_pattern(old_name, new_name)

        self.stats.files_renamed.append((old_path, new_path))

        event = FileRenameEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            old_path=old_path,
            new_path=new_path,
            naming_pattern_change=naming_pattern_change,
        )
        self.exporter.write_event(event)

    def record_file_move(
        self,
        old_path: str,
        new_path: str,
    ) -> None:
        """Record a file move event (across directories)."""
        if not self.enabled:
            return

        dest_depth = get_directory_depth(new_path, str(self.target_directory))

        event = FileMoveEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            old_path=old_path,
            new_path=new_path,
            destination_directory_depth=dest_depth,
        )
        self.exporter.write_event(event)

    def record_dir_create(
        self,
        dir_path: str,
    ) -> None:
        """Record a directory creation event."""
        if not self.enabled:
            return

        depth = get_directory_depth(dir_path + "/x", str(self.target_directory))

        # Count siblings
        sibling_count = 0
        parent = Path(dir_path).parent
        try:
            if parent.exists():
                sibling_count = sum(1 for p in parent.iterdir() if p.is_dir()) - 1
                sibling_count = max(0, sibling_count)
        except OSError:
            pass

        self.stats.dirs_created.append(dir_path)

        event = DirCreateEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            dir_path=dir_path,
            depth=depth,
            sibling_count=sibling_count,
        )
        self.exporter.write_event(event)

    def record_file_delete(
        self,
        file_path: str,
        was_temporary: bool = False,
    ) -> None:
        """Record a file deletion event."""
        if not self.enabled:
            return

        # Compute file age from access history
        file_age_ms = None
        if file_path in self._file_access:
            first_seen = self._file_access[file_path].last_view_time
            if first_seen > 0:
                file_age_ms = int(time.time() * 1000 - first_seen)

        self.stats.files_deleted.append(file_path)

        event = FileDeleteEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            file_path=file_path,
            file_age_ms=file_age_ms,
            was_temporary=was_temporary,
        )
        self.exporter.write_event(event)

    def record_file_copy(
        self,
        source_path: str,
        dest_path: str,
        is_backup: bool = False,
    ) -> None:
        """Record a file copy event."""
        if not self.enabled:
            return

        event = FileCopyEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            source_path=source_path,
            dest_path=dest_path,
            is_backup=is_backup,
        )
        self.exporter.write_event(event)

    def record_fs_snapshot(self) -> None:
        """Record a filesystem snapshot of the target directory."""
        if not self.enabled:
            return

        file_count_by_type: dict[str, int] = {}
        max_depth = 0
        total_files = 0
        tree: list[str] = []

        try:
            base = self.target_directory.resolve()
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    total_files += 1
                    ext = path.suffix.lstrip(".") or "unknown"
                    file_count_by_type[ext] = file_count_by_type.get(ext, 0) + 1

                # Compute depth
                try:
                    rel = path.relative_to(base)
                    depth = len(rel.parts)
                    if depth > max_depth:
                        max_depth = depth
                    tree.append(str(rel))
                except ValueError:
                    pass
        except OSError:
            pass

        # Store tree in media
        media_ref = None
        if tree:
            tree_json = json.dumps(tree, indent=2, ensure_ascii=False)
            media_ref = self.exporter.externalize_write(
                "fs_snapshot.json",
                tree_json,
            )

        event = FsSnapshotEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            file_count_by_type=file_count_by_type,
            max_depth=max_depth,
            total_files=total_files,
            media_ref=media_ref,
        )
        self.exporter.write_event(event)

    # ============== Error Events ==============

    def record_error_encounter(
        self,
        error_type: str,
        context: str,
        severity: str = "medium",
        tool_name: str | None = None,
        file_path: str | None = None,
    ) -> str:
        """Record an error encounter event.

        Returns:
            The event_id of the error event (for linking to error_response).
        """
        if not self.enabled:
            return ""

        self.stats.error_count += 1

        event = ErrorEncounterEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            error_type=error_type,
            context=context,
            severity=severity,
            tool_name=tool_name,
            file_path=file_path,
        )
        self.exporter.write_event(event)

        # Track pending error for response detection
        self._pending_error = (event.metadata.event_id, tool_name or "", time.time() * 1000)

        return event.metadata.event_id

    def record_error_response(
        self,
        strategy: str,
        latency_ms: int,
        error_event_id: str | None = None,
        resolution_successful: bool = False,
    ) -> None:
        """Record an error recovery response event."""
        if not self.enabled:
            return

        self.stats.error_recovery_count += 1

        event = ErrorResponseEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            strategy=strategy,
            latency_ms=latency_ms,
            error_event_id=error_event_id,
            resolution_successful=resolution_successful,
        )
        self.exporter.write_event(event)

        # Clear pending error
        self._pending_error = None

    def check_error_recovery(
        self,
        tool_name: str,
        success: bool,
    ) -> None:
        """Check if a tool call resolves a pending error and emit ERROR_RESPONSE."""
        if not self.enabled or self._pending_error is None:
            return

        error_event_id, error_tool_name, error_time_ms = self._pending_error
        now = time.time() * 1000
        latency_ms = int(now - error_time_ms)

        if success:
            # Determine recovery strategy
            if tool_name == error_tool_name:
                strategy = "retry"
            else:
                strategy = "rethink"
            self.record_error_response(
                strategy=strategy,
                latency_ms=latency_ms,
                error_event_id=error_event_id,
                resolution_successful=True,
            )

    # ============== Cross-File Events ==============

    def _maybe_emit_cross_file_reference(
        self,
        current_file: str,
        current_op: str,
    ) -> None:
        """Check if current operation creates a cross-file reference."""
        if self._last_file_operation is None:
            return

        last_file, last_op, last_time = self._last_file_operation
        if last_file == current_file:
            return

        now = time.time() * 1000
        interval_ms = int(now - last_time)

        # Only link operations within 30 seconds
        if interval_ms > 30_000:
            return

        # Determine reference type
        if last_op == "read" and current_op == "edit":
            ref_type = "read_then_edit"
        elif last_op == "search" and current_op == "read":
            ref_type = "search_then_read"
        elif last_op == "read" and current_op == "read":
            ref_type = "sequential_access"
        else:
            ref_type = "sequential_access"

        # Check if this was a search-driven access
        if current_file in self._last_search_files:
            ref_type = "search_then_read"
            self._last_search_files.clear()

        event = CrossFileReferenceEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            source_file=last_file,
            target_file=current_file,
            reference_type=ref_type,
            interval_ms=interval_ms,
        )
        self.exporter.write_event(event)

    def record_cross_file_reference(
        self,
        source_file: str,
        target_file: str,
        reference_type: str,
        interval_ms: int,
    ) -> None:
        """Record an explicit cross-file reference event."""
        if not self.enabled:
            return

        event = CrossFileReferenceEvent.create(
            session_id=self.session_id,
            profile_id=self.profile_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            message_id=self._current_message_id,
            source_file=source_file,
            target_file=target_file,
            reference_type=reference_type,
            interval_ms=interval_ms,
        )
        self.exporter.write_event(event)

    # ============== Session Events ==============

    def finalize(self, error: bool = False) -> dict[str, Any]:
        """Finalize the session and generate summary.

        Args:
            error: If True, the session ended due to an LLM error (e.g., quota exceeded).
                   In this case, session_end event is NOT emitted so the trajectory
                   is recognized as incomplete and can be re-run.
        """
        if not self.enabled:
            return {}

        self.stats.session_end_time = time.time() * 1000

        # Capture final filesystem snapshot
        self.record_fs_snapshot()

        # Calculate summary
        total_duration_ms = int(self.stats.session_end_time - self.stats.session_start_time)
        tool_success_rate = (
            self.stats.tool_success_count / self.stats.total_tool_calls if self.stats.total_tool_calls > 0 else 1.0
        )

        # Only emit SESSION_END event for successful completions
        if not error:
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
            "files_deleted": self.stats.files_deleted,
            "files_renamed": [(old, new) for old, new in self.stats.files_renamed],
            "dirs_created": self.stats.dirs_created,
            "error_count": self.stats.error_count,
            "error_recovery_count": self.stats.error_recovery_count,
            "completed": not error,
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
