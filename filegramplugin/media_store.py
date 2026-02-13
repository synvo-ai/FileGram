"""Content-addressable media storage — mirrors FileGram's behavior/media_store.py."""

from __future__ import annotations

import difflib
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_BLOB_SIZE = 1_048_576  # 1 MB
HASH_PREFIX_LEN = 16


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:HASH_PREFIX_LEN]


def _content_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:HASH_PREFIX_LEN]


@dataclass
class MediaRef:
    type: str  # "blob" or "diff"
    hash: str
    path: str
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "hash": self.hash, "path": self.path}


class MediaStore:
    """CAS storage compatible with FileGram's media/ layout."""

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

    def store_blob(self, content: str, file_path: str) -> MediaRef | None:
        if "\x00" in content[:8192]:
            return None
        if len(content.encode("utf-8")) > MAX_BLOB_SIZE:
            content = content[:MAX_BLOB_SIZE] + "\n[truncated at 1MB]"
        h = _content_hash(content)
        rel_path = f"blobs/{h}.blob"
        if h in self._manifest:
            self._manifest[h]["references"] += 1
            self._stats["deduplicated_saves"] += 1
            self._write_manifest()
            return MediaRef(type="blob", hash=h, path=rel_path, size_bytes=self._manifest[h]["size_bytes"])
        self._ensure_dirs()
        (self._blobs_dir / f"{h}.blob").write_text(content, encoding="utf-8")
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
        (self._diffs_dir / f"{h}.diff").write_text(diff_text, encoding="utf-8")
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

    def get_stats(self) -> dict[str, Any]:
        return dict(self._stats)

    # -- internal --

    def _ensure_dirs(self) -> None:
        if not self._initialized:
            self._blobs_dir.mkdir(parents=True, exist_ok=True)
            self._diffs_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True

    def _write_manifest(self) -> None:
        self._ensure_dirs()
        data = {"version": 1, "entries": self._manifest, "stats": self._stats}
        self._manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
