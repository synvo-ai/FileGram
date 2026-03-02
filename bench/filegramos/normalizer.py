"""Event normalisation layer: raw dict → NormalizedEvent.

Sits between events.json loading and FeatureExtractor / EngramEncoder.
Responsibilities:
  - Validate event_type against ConsumerEventType
  - Unify field names (e.g. source_path → source_file)
  - Resolve media_ref → resolved_content (if not already done)
  - Skip unknown event types silently (forward-compatible)
  - Use dataclass defaults for missing fields (no crash)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .schema import (
    _CONSUMER_EVENT_VALUES,
    ConsumerEventType,
    NormalizedEvent,
)

logger = logging.getLogger(__name__)


class EventNormalizer:
    """Convert raw event dicts into ``NormalizedEvent`` instances.

    Parameters
    ----------
    media_dir : Path | None
        If provided, used to resolve ``media_ref`` fields that have not
        already been resolved by ``BaseAdapter._resolve_media_refs``.
    """

    def __init__(self, media_dir: Path | None = None):
        self._media_dir = media_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_all(self, raw_events: list[dict[str, Any]]) -> list[NormalizedEvent]:
        """Normalize a full trajectory's worth of raw events.

        Events whose ``event_type`` is not in ``ConsumerEventType`` are
        silently skipped (e.g. tool_call, llm_response, session_start).
        """
        result: list[NormalizedEvent] = []
        for raw in raw_events:
            evt = self._normalize_one(raw)
            if evt is not None:
                result.append(evt)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize_one(self, raw: dict[str, Any]) -> NormalizedEvent | None:
        et_str = raw.get("event_type", "")
        if et_str not in _CONSUMER_EVENT_VALUES:
            return None

        et = ConsumerEventType(et_str)

        # --- Resolve media content if not already injected ---------------
        resolved_content = raw.get("_resolved_content", "")
        resolved_diff = raw.get("_resolved_diff", "")

        if not resolved_content and self._media_dir is not None:
            resolved_content = self._resolve_ref(raw.get("media_ref")) or ""
        if not resolved_diff and self._media_dir is not None:
            resolved_diff = self._resolve_ref(raw.get("media_ref_diff")) or ""

        # --- Unify source/target fields ----------------------------------
        # file_copy uses source_path / dest_path in producer
        # cross_file_reference uses source_file / target_file
        source_file = raw.get("source_file", "") or raw.get("source_path", "")
        target_file = raw.get("target_file", "") or raw.get("dest_path", "")

        return NormalizedEvent(
            event_type=et,
            file_path=raw.get("file_path", ""),
            file_type=raw.get("file_type", ""),
            content_length=raw.get("content_length", 0),
            view_count=raw.get("view_count", 1),
            operation=raw.get("operation", ""),
            lines_added=raw.get("lines_added", 0),
            lines_deleted=raw.get("lines_deleted", 0),
            depth=raw.get("depth", 0),
            max_depth=raw.get("max_depth", 0),
            directory_tree=raw.get("directory_tree"),
            is_backup=raw.get("is_backup", False),
            source_file=source_file,
            target_file=target_file,
            reference_type=raw.get("reference_type", ""),
            tools_called=raw.get("tools_called", 0),
            has_tool_error=raw.get("has_tool_error", False),
            resolved_content=resolved_content,
            resolved_diff=resolved_diff,
        )

    def _resolve_ref(self, ref: dict | None) -> str | None:
        """Resolve a single media_ref dict to text content."""
        if not ref or not self._media_dir:
            return None
        ref_path = ref.get("path")
        if not ref_path:
            return None
        full_path = self._media_dir / ref_path
        if not full_path.exists():
            return None
        try:
            return full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return None
