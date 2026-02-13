"""Shadow-copy manager and diff calculator."""

from __future__ import annotations

import difflib
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .media_store import MediaStore

HASH_PREFIX_LEN = 16


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:HASH_PREFIX_LEN]


@dataclass
class DiffResult:
    """Result of comparing current file to its shadow copy."""

    before_hash: str | None
    after_hash: str
    lines_added: int
    lines_deleted: int
    lines_modified: int
    diff_media: dict[str, Any] | None  # MediaRef dict or None
    after_media: dict[str, Any] | None  # MediaRef dict for the new blob
    is_new: bool  # True if no shadow existed (first write)


class Differ:
    """Maintains shadow copies and computes diffs on modification."""

    def __init__(self, shadow_dir: Path, media_store: MediaStore, text_extensions: list[str]):
        self._shadow_dir = shadow_dir
        self._media_store = media_store
        self._text_exts = set(ext.lower().lstrip(".") for ext in text_extensions)

    def is_text_file(self, file_path: str) -> bool:
        ext = Path(file_path).suffix.lstrip(".").lower()
        return ext in self._text_exts

    def init_shadows(self, watch_dirs: list[str]) -> None:
        """Walk watch_dirs and create initial shadow copies."""
        for wd in watch_dirs:
            wd_resolved = str(Path(wd).resolve())
            for root, _dirs, files in os.walk(wd_resolved):
                for fname in files:
                    abs_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(abs_path, wd_resolved)
                    self._save_shadow(wd_resolved, rel_path, abs_path)

    def compute_diff(self, abs_path: str, rel_path: str, watch_dir: str) -> DiffResult | None:
        """Compare *abs_path* with its shadow and return a DiffResult.

        Returns ``None`` if the file cannot be read or has not changed.
        """
        try:
            current_bytes = Path(abs_path).read_bytes()
        except (OSError, PermissionError):
            return None

        after_hash = _hash_bytes(current_bytes)
        shadow = self._shadow_path(watch_dir, rel_path)
        is_text = self.is_text_file(abs_path)

        if not shadow.exists():
            # First time seeing this file — treat as create
            after_media = None
            if is_text:
                content = current_bytes.decode("utf-8", errors="replace")
                ref = self._media_store.store_blob(content, rel_path)
                after_media = ref.to_dict() if ref else None
            self._save_shadow(watch_dir, rel_path, abs_path)
            return DiffResult(
                before_hash=None,
                after_hash=after_hash,
                lines_added=0,
                lines_deleted=0,
                lines_modified=0,
                diff_media=None,
                after_media=after_media,
                is_new=True,
            )

        old_bytes = shadow.read_bytes()
        before_hash = _hash_bytes(old_bytes)

        if before_hash == after_hash:
            return None  # No change

        diff_media = None
        after_media = None
        lines_added = 0
        lines_deleted = 0
        lines_modified = 0

        if is_text:
            try:
                old_text = old_bytes.decode("utf-8", errors="replace")
                new_text = current_bytes.decode("utf-8", errors="replace")
            except Exception:
                # Fallback: treat as binary
                self._save_shadow(watch_dir, rel_path, abs_path)
                return DiffResult(
                    before_hash=before_hash,
                    after_hash=after_hash,
                    lines_added=0,
                    lines_deleted=0,
                    lines_modified=0,
                    diff_media=None,
                    after_media=None,
                    is_new=False,
                )

            ref = self._media_store.store_diff(old_text, new_text, rel_path)
            if ref:
                diff_media = ref.to_dict()

            blob_ref = self._media_store.store_blob(new_text, rel_path)
            if blob_ref:
                after_media = blob_ref.to_dict()

            lines_added, lines_deleted, lines_modified = self._count_changes(old_text, new_text)

        self._save_shadow(watch_dir, rel_path, abs_path)
        return DiffResult(
            before_hash=before_hash,
            after_hash=after_hash,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            lines_modified=lines_modified,
            diff_media=diff_media,
            after_media=after_media,
            is_new=False,
        )

    def remove_shadow(self, watch_dir: str, rel_path: str) -> None:
        shadow = self._shadow_path(watch_dir, rel_path)
        if shadow.exists():
            shadow.unlink()

    def move_shadow(self, watch_dir: str, old_rel: str, new_rel: str) -> None:
        old_shadow = self._shadow_path(watch_dir, old_rel)
        new_shadow = self._shadow_path(watch_dir, new_rel)
        if old_shadow.exists():
            new_shadow.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_shadow), str(new_shadow))

    # -- internal --

    def _shadow_path(self, watch_dir: str, rel_path: str) -> Path:
        # Use watch_dir basename as namespace to avoid collisions
        ns = Path(watch_dir).name
        return self._shadow_dir / ns / rel_path

    def _save_shadow(self, watch_dir: str, rel_path: str, abs_path: str) -> None:
        shadow = self._shadow_path(watch_dir, rel_path)
        shadow.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(abs_path, shadow)
        except (OSError, PermissionError):
            pass

    @staticmethod
    def _count_changes(old_text: str, new_text: str) -> tuple[int, int, int]:
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        added = 0
        deleted = 0
        modified = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "insert":
                added += j2 - j1
            elif tag == "delete":
                deleted += i2 - i1
            elif tag == "replace":
                modified += max(i2 - i1, j2 - j1)
        return added, deleted, modified
