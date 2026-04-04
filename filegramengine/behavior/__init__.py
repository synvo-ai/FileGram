"""Behavioral signal collection module.

This module provides tools for collecting behavioral signals during agent execution.
Collected data supports persona modeling and memory pipeline research.

Usage:
    from filegramengine.behavior import BehaviorCollector

    collector = BehaviorCollector(
        session_id="abc123",
        profile_id="default",
        model_provider="azure_openai",
        model_name="gpt-4",
        target_directory=Path("/path/to/workspace"),
    )

    # Record events
    collector.record_file_read(file_path, view_range, content_length)
    collector.record_tool_call(tool_name, params, duration_ms, success)

    # Finalize session
    summary = collector.finalize()
"""

from .collector import BehaviorCollector, FileAccessStats, SessionStats
from .events import (
    BehaviorEvent,
    CompactionTriggeredEvent,
    ContextSwitchEvent,
    CrossFileReferenceEvent,
    DirCreateEvent,
    ErrorEncounterEvent,
    ErrorResponseEvent,
    EventMetadata,
    EventType,
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
from .media_store import MediaRef, MediaStore, SnapshotRefs

__all__ = [
    # Main collector
    "BehaviorCollector",
    "BehaviorExporter",
    # Stats
    "FileAccessStats",
    "SessionStats",
    # Event types
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
    # Media store
    "MediaStore",
    "MediaRef",
    "SnapshotRefs",
]
