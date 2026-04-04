"""Behavioral event definitions for signal collection.

This module defines all event types that can be collected during agent execution.
Events are designed to capture behavioral signals for persona modeling and
memory pipeline research.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(str, Enum):
    """Types of behavioral events."""

    # File events
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_EDIT = "file_edit"
    FILE_SEARCH = "file_search"

    # Tool events
    TOOL_CALL = "tool_call"

    # Timing events
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"
    LLM_RESPONSE = "llm_response"

    # Context events
    CONTEXT_SWITCH = "context_switch"
    COMPACTION_TRIGGERED = "compaction_triggered"

    # Planning events
    TODO_CREATED = "todo_created"
    TODO_STATUS_CHANGE = "todo_status_change"

    # Session events
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # File organization events
    FILE_BROWSE = "file_browse"
    FILE_RENAME = "file_rename"
    FILE_MOVE = "file_move"
    DIR_CREATE = "dir_create"
    FILE_DELETE = "file_delete"
    FILE_COPY = "file_copy"
    FS_SNAPSHOT = "fs_snapshot"

    # Error events
    ERROR_ENCOUNTER = "error_encounter"
    ERROR_RESPONSE = "error_response"

    # Cross-file events
    CROSS_FILE_REFERENCE = "cross_file_reference"


@dataclass
class EventMetadata:
    """Common metadata for all events."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: float = field(default_factory=lambda: time.time() * 1000)  # ms
    session_id: str = ""
    profile_id: str = ""
    message_id: str = ""
    model_provider: str = ""
    model_name: str = ""


@dataclass
class BehaviorEvent:
    """Base class for all behavioral events."""

    metadata: EventMetadata
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "event_id": self.metadata.event_id,
            "event_type": self.metadata.event_type,
            "timestamp": self.metadata.timestamp,
            "session_id": self.metadata.session_id,
            "profile_id": self.metadata.profile_id,
            "message_id": self.metadata.message_id,
            "model_provider": self.metadata.model_provider,
            "model_name": self.metadata.model_name,
            **self.data,
        }


# ============== File Events ==============


@dataclass
class FileReadEvent(BehaviorEvent):
    """Event for file read operations."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        file_path: str,
        file_type: str,
        directory_depth: int,
        view_count: int,
        view_range: tuple[int, int],
        content_length: int,
        revisit_interval_ms: int | None = None,
    ) -> FileReadEvent:
        """Create a file read event."""
        metadata = EventMetadata(
            event_type=EventType.FILE_READ.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return FileReadEvent(
            metadata=metadata,
            data={
                "file_path": file_path,
                "file_type": file_type,
                "directory_depth": directory_depth,
                "view_count": view_count,
                "view_range": list(view_range),
                "content_length": content_length,
                "revisit_interval_ms": revisit_interval_ms,
            },
        )


@dataclass
class FileWriteEvent(BehaviorEvent):
    """Event for file write operations."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        file_path: str,
        file_type: str,
        directory_depth: int,
        operation: str,  # "create" or "overwrite"
        content_length: int,
        before_hash: str | None = None,
        after_hash: str | None = None,
        media_ref: str | dict | None = None,
    ) -> FileWriteEvent:
        """Create a file write event."""
        metadata = EventMetadata(
            event_type=EventType.FILE_WRITE.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        data = {
            "file_path": file_path,
            "file_type": file_type,
            "directory_depth": directory_depth,
            "operation": operation,
            "content_length": content_length,
            "before_hash": before_hash,
            "after_hash": after_hash,
        }
        if media_ref is not None:
            data["media_ref"] = media_ref
        return FileWriteEvent(metadata=metadata, data=data)


@dataclass
class FileEditEvent(BehaviorEvent):
    """Event for file edit operations."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        file_path: str,
        file_type: str,
        edit_tool: str,  # "edit", "multiedit", "apply_patch"
        lines_added: int,
        lines_deleted: int,
        lines_modified: int,
        diff_summary: str,
        before_hash: str | None = None,
        after_hash: str | None = None,
        media_ref_old: str | dict | None = None,
        media_ref_new: str | dict | None = None,
        media_ref_diff: str | dict | None = None,
    ) -> FileEditEvent:
        """Create a file edit event."""
        metadata = EventMetadata(
            event_type=EventType.FILE_EDIT.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        data = {
            "file_path": file_path,
            "file_type": file_type,
            "edit_tool": edit_tool,
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "lines_modified": lines_modified,
            "diff_summary": diff_summary,
            "before_hash": before_hash,
            "after_hash": after_hash,
        }
        if media_ref_old is not None:
            data["media_ref_old"] = media_ref_old
        if media_ref_new is not None:
            data["media_ref_new"] = media_ref_new
        if media_ref_diff is not None:
            data["media_ref_diff"] = media_ref_diff
        return FileEditEvent(metadata=metadata, data=data)


@dataclass
class FileSearchEvent(BehaviorEvent):
    """Event for file search operations."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        search_type: str,  # "grep", "glob", "codesearch"
        query: str,
        files_matched: int,
        files_opened_after: list[str] | None = None,
    ) -> FileSearchEvent:
        """Create a file search event."""
        metadata = EventMetadata(
            event_type=EventType.FILE_SEARCH.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return FileSearchEvent(
            metadata=metadata,
            data={
                "search_type": search_type,
                "query": query,
                "files_matched": files_matched,
                "files_opened_after": files_opened_after or [],
            },
        )


# ============== Tool Events ==============


@dataclass
class ToolCallEvent(BehaviorEvent):
    """Event for tool call operations."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        tool_name: str,
        tool_parameters: dict[str, Any],
        execution_time_ms: int,
        success: bool,
        error_type: str | None = None,
        error_message: str | None = None,
        retry_count: int = 0,
        sequence_position: int = 0,
    ) -> ToolCallEvent:
        """Create a tool call event."""
        metadata = EventMetadata(
            event_type=EventType.TOOL_CALL.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return ToolCallEvent(
            metadata=metadata,
            data={
                "tool_name": tool_name,
                "tool_parameters": tool_parameters,
                "execution_time_ms": execution_time_ms,
                "success": success,
                "error_type": error_type,
                "error_message": error_message,
                "retry_count": retry_count,
                "sequence_position": sequence_position,
            },
        )


# ============== Timing Events ==============


@dataclass
class IterationStartEvent(BehaviorEvent):
    """Event for iteration start."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        iteration_number: int,
    ) -> IterationStartEvent:
        """Create an iteration start event."""
        metadata = EventMetadata(
            event_type=EventType.ITERATION_START.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return IterationStartEvent(
            metadata=metadata,
            data={"iteration_number": iteration_number},
        )


@dataclass
class IterationEndEvent(BehaviorEvent):
    """Event for iteration end."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        iteration_number: int,
        duration_ms: int,
        tools_called: int,
        has_tool_error: bool,
    ) -> IterationEndEvent:
        """Create an iteration end event."""
        metadata = EventMetadata(
            event_type=EventType.ITERATION_END.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return IterationEndEvent(
            metadata=metadata,
            data={
                "iteration_number": iteration_number,
                "duration_ms": duration_ms,
                "tools_called": tools_called,
                "has_tool_error": has_tool_error,
            },
        )


@dataclass
class LLMResponseEvent(BehaviorEvent):
    """Event for LLM response."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        response_time_ms: int,
        input_tokens: int,
        output_tokens: int,
        has_reasoning: bool,
        stop_reason: str,
    ) -> LLMResponseEvent:
        """Create an LLM response event."""
        metadata = EventMetadata(
            event_type=EventType.LLM_RESPONSE.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return LLMResponseEvent(
            metadata=metadata,
            data={
                "response_time_ms": response_time_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "has_reasoning": has_reasoning,
                "stop_reason": stop_reason,
            },
        )


# ============== Context Events ==============


@dataclass
class ContextSwitchEvent(BehaviorEvent):
    """Event for context switch (file navigation)."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        from_file: str | None,
        to_file: str,
        trigger: str,
        switch_count: int,
    ) -> ContextSwitchEvent:
        """Create a context switch event."""
        metadata = EventMetadata(
            event_type=EventType.CONTEXT_SWITCH.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return ContextSwitchEvent(
            metadata=metadata,
            data={
                "from_file": from_file,
                "to_file": to_file,
                "trigger": trigger,
                "switch_count": switch_count,
            },
        )


@dataclass
class CompactionTriggeredEvent(BehaviorEvent):
    """Event for compaction trigger."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        reason: str,
        messages_before: int,
        messages_after: int,
        tokens_saved: int,
    ) -> CompactionTriggeredEvent:
        """Create a compaction triggered event."""
        metadata = EventMetadata(
            event_type=EventType.COMPACTION_TRIGGERED.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return CompactionTriggeredEvent(
            metadata=metadata,
            data={
                "reason": reason,
                "messages_before": messages_before,
                "messages_after": messages_after,
                "tokens_saved": tokens_saved,
            },
        )


# ============== Session Events ==============


@dataclass
class SessionStartEvent(BehaviorEvent):
    """Event for session start."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        target_directory: str,
    ) -> SessionStartEvent:
        """Create a session start event."""
        metadata = EventMetadata(
            event_type=EventType.SESSION_START.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
        )
        return SessionStartEvent(
            metadata=metadata,
            data={
                "target_directory": target_directory,
            },
        )


@dataclass
class SessionEndEvent(BehaviorEvent):
    """Event for session end."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        total_iterations: int,
        total_tool_calls: int,
        duration_ms: int,
    ) -> SessionEndEvent:
        """Create a session end event."""
        metadata = EventMetadata(
            event_type=EventType.SESSION_END.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
        )
        return SessionEndEvent(
            metadata=metadata,
            data={
                "total_iterations": total_iterations,
                "total_tool_calls": total_tool_calls,
                "duration_ms": duration_ms,
            },
        )


# ============== File Organization Events ==============


@dataclass
class FileBrowseEvent(BehaviorEvent):
    """Event for directory browsing operations."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        directory_path: str,
        files_listed: int,
        depth: int,
    ) -> FileBrowseEvent:
        """Create a file browse event."""
        metadata = EventMetadata(
            event_type=EventType.FILE_BROWSE.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return FileBrowseEvent(
            metadata=metadata,
            data={
                "directory_path": directory_path,
                "files_listed": files_listed,
                "depth": depth,
            },
        )


@dataclass
class FileRenameEvent(BehaviorEvent):
    """Event for file rename operations (same directory)."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        old_path: str,
        new_path: str,
        naming_pattern_change: str,
    ) -> FileRenameEvent:
        """Create a file rename event."""
        metadata = EventMetadata(
            event_type=EventType.FILE_RENAME.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return FileRenameEvent(
            metadata=metadata,
            data={
                "old_path": old_path,
                "new_path": new_path,
                "naming_pattern_change": naming_pattern_change,
            },
        )


@dataclass
class FileMoveEvent(BehaviorEvent):
    """Event for file move operations (across directories)."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        old_path: str,
        new_path: str,
        destination_directory_depth: int,
    ) -> FileMoveEvent:
        """Create a file move event."""
        metadata = EventMetadata(
            event_type=EventType.FILE_MOVE.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return FileMoveEvent(
            metadata=metadata,
            data={
                "old_path": old_path,
                "new_path": new_path,
                "destination_directory_depth": destination_directory_depth,
            },
        )


@dataclass
class DirCreateEvent(BehaviorEvent):
    """Event for directory creation."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        dir_path: str,
        depth: int,
        sibling_count: int,
    ) -> DirCreateEvent:
        """Create a directory creation event."""
        metadata = EventMetadata(
            event_type=EventType.DIR_CREATE.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return DirCreateEvent(
            metadata=metadata,
            data={
                "dir_path": dir_path,
                "depth": depth,
                "sibling_count": sibling_count,
            },
        )


@dataclass
class FileDeleteEvent(BehaviorEvent):
    """Event for file deletion."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        file_path: str,
        file_age_ms: int | None,
        was_temporary: bool,
    ) -> FileDeleteEvent:
        """Create a file delete event."""
        metadata = EventMetadata(
            event_type=EventType.FILE_DELETE.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return FileDeleteEvent(
            metadata=metadata,
            data={
                "file_path": file_path,
                "file_age_ms": file_age_ms,
                "was_temporary": was_temporary,
            },
        )


@dataclass
class FileCopyEvent(BehaviorEvent):
    """Event for file copy operations."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        source_path: str,
        dest_path: str,
        is_backup: bool,
    ) -> FileCopyEvent:
        """Create a file copy event."""
        metadata = EventMetadata(
            event_type=EventType.FILE_COPY.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return FileCopyEvent(
            metadata=metadata,
            data={
                "source_path": source_path,
                "dest_path": dest_path,
                "is_backup": is_backup,
            },
        )


@dataclass
class FsSnapshotEvent(BehaviorEvent):
    """Event for filesystem snapshot (directory tree state)."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        file_count_by_type: dict[str, int],
        max_depth: int,
        total_files: int,
        media_ref: str | dict | None = None,
    ) -> FsSnapshotEvent:
        """Create a filesystem snapshot event."""
        metadata = EventMetadata(
            event_type=EventType.FS_SNAPSHOT.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        data: dict[str, Any] = {
            "file_count_by_type": file_count_by_type,
            "max_depth": max_depth,
            "total_files": total_files,
        }
        if media_ref is not None:
            data["media_ref"] = media_ref
        return FsSnapshotEvent(metadata=metadata, data=data)


# ============== Error Events ==============


@dataclass
class ErrorEncounterEvent(BehaviorEvent):
    """Event for error encounters during execution."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        error_type: str,
        context: str,
        severity: str,  # "low", "medium", "high"
        tool_name: str | None = None,
        file_path: str | None = None,
    ) -> ErrorEncounterEvent:
        """Create an error encounter event."""
        metadata = EventMetadata(
            event_type=EventType.ERROR_ENCOUNTER.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return ErrorEncounterEvent(
            metadata=metadata,
            data={
                "error_type": error_type,
                "context": context,
                "severity": severity,
                "tool_name": tool_name,
                "file_path": file_path,
            },
        )


@dataclass
class ErrorResponseEvent(BehaviorEvent):
    """Event for error recovery response."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        strategy: str,  # "retry", "skip", "rethink", "ignore", "fix"
        latency_ms: int,
        error_event_id: str | None = None,
        resolution_successful: bool = False,
    ) -> ErrorResponseEvent:
        """Create an error response event."""
        metadata = EventMetadata(
            event_type=EventType.ERROR_RESPONSE.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return ErrorResponseEvent(
            metadata=metadata,
            data={
                "strategy": strategy,
                "latency_ms": latency_ms,
                "error_event_id": error_event_id,
                "resolution_successful": resolution_successful,
            },
        )


# ============== Cross-File Events ==============


@dataclass
class CrossFileReferenceEvent(BehaviorEvent):
    """Event for cross-file causal references."""

    @staticmethod
    def create(
        session_id: str,
        profile_id: str,
        model_provider: str,
        model_name: str,
        message_id: str,
        source_file: str,
        target_file: str,
        reference_type: str,  # "import", "read_then_edit", "search_then_read", "sequential_access"
        interval_ms: int,
    ) -> CrossFileReferenceEvent:
        """Create a cross-file reference event."""
        metadata = EventMetadata(
            event_type=EventType.CROSS_FILE_REFERENCE.value,
            session_id=session_id,
            profile_id=profile_id,
            model_provider=model_provider,
            model_name=model_name,
            message_id=message_id,
        )
        return CrossFileReferenceEvent(
            metadata=metadata,
            data={
                "source_file": source_file,
                "target_file": target_file,
                "reference_type": reference_type,
                "interval_ms": interval_ms,
            },
        )


# ============== Helper Functions ==============


def compute_file_hash(content: str) -> str:
    """Compute SHA-256 hash of file content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def get_directory_depth(file_path: str, base_path: str) -> int:
    """Calculate directory depth relative to base path."""
    try:
        rel_path = Path(file_path).relative_to(Path(base_path))
        return len(rel_path.parts) - 1
    except ValueError:
        return len(Path(file_path).parts) - 1


def get_file_type(file_path: str) -> str:
    """Get file extension/type."""
    suffix = Path(file_path).suffix
    return suffix.lstrip(".") if suffix else "unknown"


# ============== Exports ==============

__all__ = [
    "EventType",
    "EventMetadata",
    "BehaviorEvent",
    "FileReadEvent",
    "FileWriteEvent",
    "FileEditEvent",
    "FileSearchEvent",
    "ToolCallEvent",
    "IterationStartEvent",
    "IterationEndEvent",
    "LLMResponseEvent",
    "ContextSwitchEvent",
    "CompactionTriggeredEvent",
    "SessionStartEvent",
    "SessionEndEvent",
    # File organization events
    "FileBrowseEvent",
    "FileRenameEvent",
    "FileMoveEvent",
    "DirCreateEvent",
    "FileDeleteEvent",
    "FileCopyEvent",
    "FsSnapshotEvent",
    # Error events
    "ErrorEncounterEvent",
    "ErrorResponseEvent",
    # Cross-file events
    "CrossFileReferenceEvent",
    # Helpers
    "compute_file_hash",
    "get_directory_depth",
    "get_file_type",
]
