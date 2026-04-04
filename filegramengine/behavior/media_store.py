"""Content-addressable media storage for behavioral signals.

This module provides a content-addressable store (CAS) for externalizing
file content snapshots and diffs produced during agent execution. Files are
stored by their SHA-256 hash, giving natural deduplication.

Directory layout under each session:
    media/
        blobs/{hash}.blob    # Raw file content snapshots
        diffs/{hash}.diff    # Standard unified diffs
        manifest.json        # Hash -> metadata index
"""

from __future__ import annotations

import difflib
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Storage limits
MAX_BLOB_SIZE = 1_048_576  # 1 MB
HASH_PREFIX_LEN = 16  # Match existing compute_file_hash


def _content_hash(content: str) -> str:
    """Compute SHA-256 hash prefix of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:HASH_PREFIX_LEN]


def _is_binary(content: str) -> bool:
    """Heuristic: content is binary if it contains null bytes."""
    return "\x00" in content[:8192]


@dataclass
class MediaRef:
    """Reference to a stored media file."""

    type: str  # "blob" or "diff"
    hash: str  # Content hash (16-char SHA-256 prefix)
    path: str  # Relative path within media/ (e.g. "blobs/a1b2c3d4.blob")
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for inclusion in events.json."""
        return {"type": self.type, "hash": self.hash, "path": self.path}


@dataclass
class SnapshotRefs:
    """References produced by a write or edit operation."""

    before_blob: MediaRef | None = None  # None for creates
    after_blob: MediaRef | None = None
    diff: MediaRef | None = None  # None for creates


@dataclass
class ManifestEntry:
    """Metadata about a stored media file."""

    type: str
    original_path: str
    size_bytes: int
    stored_at: float
    references: int = 1


class MediaStore:
    """Content-addressable media storage for a single session.

    Usage::

        store = MediaStore(session_dir)
        ref = store.store_blob(content, "src/main.py")
        snapshot = store.store_snapshot("src/main.py", before, after)
    """

    def __init__(self, session_dir: Path):
        self._session_dir = session_dir
        self._blobs_dir = session_dir / "media" / "blobs"
        self._diffs_dir = session_dir / "media" / "diffs"
        self._manifest_path = session_dir / "media" / "manifest.json"

        self._manifest: dict[str, dict[str, Any]] = {}
        self._stats = {
            "total_blobs": 0,
            "total_diffs": 0,
            "total_bytes": 0,
            "deduplicated_saves": 0,
        }
        self._initialized = False

    # ---- public API ----

    def store_blob(self, content: str, file_path: str) -> MediaRef | None:
        """Store file content as a blob. Returns MediaRef, or None if skipped."""
        if _is_binary(content):
            return None

        if len(content.encode("utf-8")) > MAX_BLOB_SIZE:
            content = content[:MAX_BLOB_SIZE] + "\n[truncated at 1MB]"

        h = _content_hash(content)
        rel_path = f"blobs/{h}.blob"

        if h in self._manifest:
            # Deduplicated — just bump reference count
            self._manifest[h]["references"] += 1
            self._stats["deduplicated_saves"] += 1
            self._write_manifest()
            return MediaRef(type="blob", hash=h, path=rel_path, size_bytes=self._manifest[h]["size_bytes"])

        self._ensure_dirs()
        abs_path = self._blobs_dir / f"{h}.blob"
        abs_path.write_text(content, encoding="utf-8")

        size = len(content.encode("utf-8"))
        self._manifest[h] = {
            "type": "blob",
            "original_path": file_path,
            "size_bytes": size,
            "stored_at": time.time() * 1000,
            "references": 1,
        }
        self._stats["total_blobs"] += 1
        self._stats["total_bytes"] += size
        self._write_manifest()

        return MediaRef(type="blob", hash=h, path=rel_path, size_bytes=size)

    def store_diff(self, before: str, after: str, file_path: str) -> MediaRef | None:
        """Generate and store a unified diff. Returns MediaRef, or None if empty."""
        diff_lines = list(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
            )
        )
        if not diff_lines:
            return None

        diff_text = "".join(diff_lines)
        h = _content_hash(diff_text)
        rel_path = f"diffs/{h}.diff"

        if h in self._manifest:
            self._manifest[h]["references"] += 1
            self._stats["deduplicated_saves"] += 1
            self._write_manifest()
            return MediaRef(type="diff", hash=h, path=rel_path, size_bytes=self._manifest[h]["size_bytes"])

        self._ensure_dirs()
        abs_path = self._diffs_dir / f"{h}.diff"
        abs_path.write_text(diff_text, encoding="utf-8")

        size = len(diff_text.encode("utf-8"))
        self._manifest[h] = {
            "type": "diff",
            "original_path": file_path,
            "size_bytes": size,
            "stored_at": time.time() * 1000,
            "references": 1,
        }
        self._stats["total_diffs"] += 1
        self._stats["total_bytes"] += size
        self._write_manifest()

        return MediaRef(type="diff", hash=h, path=rel_path, size_bytes=size)

    def store_snapshot(
        self,
        file_path: str,
        before: str | None,
        after: str,
    ) -> SnapshotRefs:
        """Store a complete write/edit snapshot (blobs + diff).

        For creates (before is None): stores only after blob.
        For edits: stores before blob, after blob, and diff.
        """
        refs = SnapshotRefs()

        if before is not None:
            refs.before_blob = self.store_blob(before, file_path)
            refs.diff = self.store_diff(before, after, file_path)

        refs.after_blob = self.store_blob(after, file_path)
        return refs

    def get_stats(self) -> dict[str, Any]:
        """Return storage statistics."""
        return dict(self._stats)

    # ---- internal ----

    def _ensure_dirs(self) -> None:
        if not self._initialized:
            self._blobs_dir.mkdir(parents=True, exist_ok=True)
            self._diffs_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True

    def _write_manifest(self) -> None:
        self._ensure_dirs()
        data = {
            "version": 1,
            "entries": self._manifest,
            "stats": self._stats,
        }
        self._manifest_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


__all__ = [
    "MediaStore",
    "MediaRef",
    "SnapshotRefs",
]
