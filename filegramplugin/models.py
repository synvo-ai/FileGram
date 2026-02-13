"""Event model helpers aligned with FileGram behavior/events.py schema.

All events are plain dicts to stay lightweight. The factory functions
guarantee field names match the FileGram BehaviorEvent.to_dict() output.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import Any

HASH_PREFIX_LEN = 16


def _base(event_type: str, session_id: str, profile_id: str) -> dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": time.time() * 1000,
        "session_id": session_id,
        "profile_id": profile_id,
        "message_id": None,
        "model_provider": None,
        "model_name": None,
    }


def compute_file_hash(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:HASH_PREFIX_LEN]


def get_file_type(file_path: str) -> str:
    suffix = Path(file_path).suffix
    return suffix.lstrip(".") if suffix else "unknown"


def get_directory_depth(file_path: str, base_path: str) -> int:
    try:
        rel = Path(file_path).relative_to(Path(base_path))
        return len(rel.parts) - 1
    except ValueError:
        return len(Path(file_path).parts) - 1


# ---------- event factories ----------


def session_start_event(session_id: str, profile_id: str, target_directory: str) -> dict[str, Any]:
    e = _base("session_start", session_id, profile_id)
    e["target_directory"] = target_directory
    return e


def session_end_event(session_id: str, profile_id: str, duration_ms: int) -> dict[str, Any]:
    e = _base("session_end", session_id, profile_id)
    e["total_iterations"] = 0
    e["total_tool_calls"] = 0
    e["duration_ms"] = duration_ms
    return e


def file_write_event(
    session_id: str,
    profile_id: str,
    file_path: str,
    base_path: str,
    operation: str,
    content_length: int,
    before_hash: str | None,
    after_hash: str | None,
    media_ref: dict | None = None,
) -> dict[str, Any]:
    e = _base("file_write", session_id, profile_id)
    e["file_path"] = file_path
    e["file_type"] = get_file_type(file_path)
    e["directory_depth"] = get_directory_depth(file_path, base_path)
    e["operation"] = operation
    e["content_length"] = content_length
    e["before_hash"] = before_hash
    e["after_hash"] = after_hash
    if media_ref is not None:
        e["media_ref"] = media_ref
    return e


def file_edit_event(
    session_id: str,
    profile_id: str,
    file_path: str,
    base_path: str,
    lines_added: int,
    lines_deleted: int,
    lines_modified: int,
    before_hash: str | None,
    after_hash: str | None,
    diff_media: dict | None = None,
) -> dict[str, Any]:
    e = _base("file_edit", session_id, profile_id)
    e["file_path"] = file_path
    e["file_type"] = get_file_type(file_path)
    e["directory_depth"] = get_directory_depth(file_path, base_path)
    e["edit_tool"] = "external"
    e["lines_added"] = lines_added
    e["lines_deleted"] = lines_deleted
    e["lines_modified"] = lines_modified
    e["diff_summary"] = f"+{lines_added} -{lines_deleted} ~{lines_modified}"
    e["before_hash"] = before_hash
    e["after_hash"] = after_hash
    if diff_media is not None:
        e["diff_media"] = diff_media
    return e


def file_delete_event(
    session_id: str,
    profile_id: str,
    file_path: str,
    file_age_ms: int | None,
    was_temporary: bool,
) -> dict[str, Any]:
    e = _base("file_delete", session_id, profile_id)
    e["file_path"] = file_path
    e["file_type"] = get_file_type(file_path)
    e["directory_depth"] = 0
    e["file_age_ms"] = file_age_ms
    e["was_temporary"] = was_temporary
    return e


def file_rename_event(
    session_id: str,
    profile_id: str,
    old_path: str,
    new_path: str,
    naming_pattern_change: str,
) -> dict[str, Any]:
    e = _base("file_rename", session_id, profile_id)
    e["old_path"] = old_path
    e["new_path"] = new_path
    e["naming_pattern_change"] = naming_pattern_change
    return e


def file_move_event(
    session_id: str,
    profile_id: str,
    old_path: str,
    new_path: str,
    destination_directory_depth: int,
) -> dict[str, Any]:
    e = _base("file_move", session_id, profile_id)
    e["old_path"] = old_path
    e["new_path"] = new_path
    e["destination_directory_depth"] = destination_directory_depth
    return e


def dir_create_event(
    session_id: str,
    profile_id: str,
    dir_path: str,
    depth: int,
    sibling_count: int,
) -> dict[str, Any]:
    e = _base("dir_create", session_id, profile_id)
    e["dir_path"] = dir_path
    e["depth"] = depth
    e["sibling_count"] = sibling_count
    return e


def file_copy_event(
    session_id: str,
    profile_id: str,
    source_path: str,
    dest_path: str,
    is_backup: bool,
) -> dict[str, Any]:
    e = _base("file_copy", session_id, profile_id)
    e["source_path"] = source_path
    e["dest_path"] = dest_path
    e["is_backup"] = is_backup
    return e


def fs_snapshot_event(
    session_id: str,
    profile_id: str,
    directory_tree: dict,
    file_count_by_type: dict[str, int],
    max_depth: int,
    total_files: int,
) -> dict[str, Any]:
    e = _base("fs_snapshot", session_id, profile_id)
    e["directory_tree"] = directory_tree
    e["file_count_by_type"] = file_count_by_type
    e["max_depth"] = max_depth
    e["total_files"] = total_files
    return e
