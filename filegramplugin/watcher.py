"""Watchdog event handler — translates FS events to FileGram events."""

from __future__ import annotations

import fnmatch
import os
import threading
import time
from pathlib import Path
from typing import Any

from watchdog.events import (
    DirCreatedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)

from .config import PluginConfig
from .copy_detector import CopyDetector
from .differ import Differ
from .exporter import Exporter
from .models import (
    compute_file_hash,
    dir_create_event,
    file_copy_event,
    file_delete_event,
    file_edit_event,
    file_move_event,
    file_rename_event,
    file_write_event,
)


class FileGramHandler(FileSystemEventHandler):
    """Translates watchdog events into FileGram-compatible event dicts."""

    def __init__(
        self,
        differ: Differ,
        exporter: Exporter,
        copy_detector: CopyDetector,
        config: PluginConfig,
        watch_dir: str,
    ):
        super().__init__()
        self._differ = differ
        self._exporter = exporter
        self._copy_detector = copy_detector
        self._config = config
        # Resolve symlinks so macOS /var vs /private/var doesn't break relpath
        self._watch_dir = str(Path(watch_dir).resolve())

        # Dedup state: path -> (scheduled_timer, timestamp)
        self._pending_modifies: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

        # Track create timestamps for file_age_ms on delete
        self._file_birth: dict[str, float] = {}

    # ---- helpers: path resolution ----

    def _resolve(self, path: str) -> str:
        """Resolve symlinks so relpath against watch_dir works correctly."""
        return str(Path(path).resolve())

    # ---- watchdog overrides ----

    def on_created(self, event: Any) -> None:
        src = self._resolve(event.src_path)
        if self._should_ignore(src):
            return
        if isinstance(event, DirCreatedEvent):
            self._handle_dir_create(src)
        elif isinstance(event, FileCreatedEvent):
            self._handle_file_create(src)

    def on_modified(self, event: Any) -> None:
        if isinstance(event, FileModifiedEvent):
            src = self._resolve(event.src_path)
            if self._should_ignore(src):
                return
            self._schedule_modify(src)

    def on_deleted(self, event: Any) -> None:
        src = self._resolve(event.src_path)
        if self._should_ignore(src):
            return
        if isinstance(event, FileDeletedEvent):
            # Suppress delete if file reappears quickly (atomic save)
            # Use a short delay to check
            rel_path = os.path.relpath(src, self._watch_dir)
            threading.Timer(0.5, self._deferred_delete, args=[src, rel_path]).start()

    def on_moved(self, event: Any) -> None:
        src = self._resolve(event.src_path)
        dest = self._resolve(event.dest_path) if hasattr(event, "dest_path") else None
        src_base = os.path.basename(src)
        dest_base = os.path.basename(dest) if dest else ""

        # macOS atomic save step 1: original.txt → original.txt.sb-xxx
        # Don't move the shadow — keep it for the real filename so we can diff later
        if ".sb-" in dest_base:
            return

        # macOS atomic save step 2: temp.sb-xxx → original.txt
        # Treat as a modify on the destination
        if ".sb-" in src_base or src_base.startswith("~") or src_base.endswith("~"):
            if dest and not self._should_ignore(dest):
                self._schedule_modify(dest)
            return

        if self._should_ignore(src):
            return
        if isinstance(event, DirMovedEvent):
            return
        if isinstance(event, FileMovedEvent):
            if dest and self._should_ignore(dest):
                return
            self._handle_file_moved(src, dest)

    # ---- event handlers ----

    def _handle_file_create(self, abs_path: str) -> None:
        if not os.path.isfile(abs_path):
            return

        rel_path = os.path.relpath(abs_path, self._watch_dir)

        # If a shadow exists for this path, this is likely an atomic-save recreate
        # → treat as modify (debounced) instead of create
        shadow = self._differ._shadow_path(self._watch_dir, rel_path)
        if shadow.exists():
            self._schedule_modify(abs_path)
            return

        self._file_birth[abs_path] = time.time() * 1000

        # Check if this is a copy of an existing file
        source = self._copy_detector.check_copy(abs_path)
        if source:
            src_rel = os.path.relpath(source, self._watch_dir)
            evt = file_copy_event(
                session_id=self._exporter.session_id,
                profile_id=self._config.profile_id,
                source_path=src_rel,
                dest_path=rel_path,
                is_backup=self._looks_like_backup(rel_path),
            )
            self._exporter.append(evt)
            return

        # Genuinely new file
        content_length = 0
        after_hash = None
        media_ref = None
        try:
            data = Path(abs_path).read_bytes()
            content_length = len(data)
            after_hash = compute_file_hash(data)
            if self._differ.is_text_file(abs_path):
                text = data.decode("utf-8", errors="replace")
                ref = self._differ._media_store.store_blob(text, rel_path)
                if ref:
                    media_ref = ref.to_dict()
        except (OSError, PermissionError):
            pass

        # Initialize shadow
        self._differ._save_shadow(self._watch_dir, rel_path, abs_path)
        self._copy_detector.register(abs_path)

        evt = file_write_event(
            session_id=self._exporter.session_id,
            profile_id=self._config.profile_id,
            file_path=rel_path,
            base_path=self._watch_dir,
            operation="create",
            content_length=content_length,
            before_hash=None,
            after_hash=after_hash,
            media_ref=media_ref,
        )
        self._exporter.append(evt)

    def _handle_dir_create(self, abs_path: str) -> None:
        rel_path = os.path.relpath(abs_path, self._watch_dir)
        depth = len(Path(rel_path).parts)
        parent = os.path.dirname(abs_path)
        try:
            siblings = len([e for e in os.listdir(parent) if os.path.isdir(os.path.join(parent, e))])
        except OSError:
            siblings = 0

        evt = dir_create_event(
            session_id=self._exporter.session_id,
            profile_id=self._config.profile_id,
            dir_path=rel_path,
            depth=depth,
            sibling_count=siblings,
        )
        self._exporter.append(evt)

    def _schedule_modify(self, abs_path: str) -> None:
        """Debounce rapid modify events within dedup_window_sec."""
        with self._lock:
            if abs_path in self._pending_modifies:
                self._pending_modifies[abs_path].cancel()

            timer = threading.Timer(
                self._config.dedup_window_sec,
                self._flush_modify,
                args=[abs_path],
            )
            timer.daemon = True
            self._pending_modifies[abs_path] = timer
            timer.start()

    def _flush_modify(self, abs_path: str) -> None:
        """Process a debounced modify event."""
        with self._lock:
            self._pending_modifies.pop(abs_path, None)

        if not os.path.isfile(abs_path):
            return

        rel_path = os.path.relpath(abs_path, self._watch_dir)
        result = self._differ.compute_diff(abs_path, rel_path, self._watch_dir)
        if result is None:
            return

        if result.is_new:
            # Shadow didn't exist — emit as create
            content_length = 0
            try:
                content_length = os.path.getsize(abs_path)
            except OSError:
                pass
            evt = file_write_event(
                session_id=self._exporter.session_id,
                profile_id=self._config.profile_id,
                file_path=rel_path,
                base_path=self._watch_dir,
                operation="create",
                content_length=content_length,
                before_hash=None,
                after_hash=result.after_hash,
                media_ref=result.after_media,
            )
        else:
            evt = file_edit_event(
                session_id=self._exporter.session_id,
                profile_id=self._config.profile_id,
                file_path=rel_path,
                base_path=self._watch_dir,
                lines_added=result.lines_added,
                lines_deleted=result.lines_deleted,
                lines_modified=result.lines_modified,
                before_hash=result.before_hash,
                after_hash=result.after_hash,
                diff_media=result.diff_media,
            )

        self._copy_detector.register(abs_path)
        self._exporter.append(evt)

    def _deferred_delete(self, abs_path: str, rel_path: str) -> None:
        """Only emit delete if the file hasn't reappeared (i.e., not an atomic save)."""
        if os.path.exists(abs_path):
            return  # File reappeared — atomic save, not a real delete
        self._handle_file_delete(abs_path)

    def _handle_file_delete(self, abs_path: str) -> None:
        rel_path = os.path.relpath(abs_path, self._watch_dir)

        file_age_ms = None
        birth = self._file_birth.pop(abs_path, None)
        if birth is not None:
            file_age_ms = int(time.time() * 1000 - birth)

        was_temp = self._looks_temporary(rel_path)

        evt = file_delete_event(
            session_id=self._exporter.session_id,
            profile_id=self._config.profile_id,
            file_path=rel_path,
            file_age_ms=file_age_ms,
            was_temporary=was_temp,
        )
        self._exporter.append(evt)
        self._differ.remove_shadow(self._watch_dir, rel_path)

    def _handle_file_moved(self, src: str, dest: str) -> None:
        old_rel = os.path.relpath(src, self._watch_dir)
        new_rel = os.path.relpath(dest, self._watch_dir)

        old_dir = os.path.dirname(old_rel)
        new_dir = os.path.dirname(new_rel)

        if old_dir == new_dir:
            # Same directory — rename
            pattern = self._detect_naming_pattern(os.path.basename(old_rel), os.path.basename(new_rel))
            evt = file_rename_event(
                session_id=self._exporter.session_id,
                profile_id=self._config.profile_id,
                old_path=old_rel,
                new_path=new_rel,
                naming_pattern_change=pattern,
            )
        else:
            # Different directory — move
            dest_depth = len(Path(new_dir).parts) if new_dir else 0
            evt = file_move_event(
                session_id=self._exporter.session_id,
                profile_id=self._config.profile_id,
                old_path=old_rel,
                new_path=new_rel,
                destination_directory_depth=dest_depth,
            )

        self._exporter.append(evt)
        self._differ.move_shadow(self._watch_dir, old_rel, new_rel)

        # Update birth tracking
        if src in self._file_birth:
            self._file_birth[dest] = self._file_birth.pop(src)

    # ---- helpers ----

    def _should_ignore(self, path: str) -> bool:
        basename = os.path.basename(path)
        # Skip hidden files
        if basename.startswith("."):
            return True
        # Skip macOS atomic-save temp files (.sb-* used by TextEdit, Xcode, etc.)
        if ".sb-" in basename:
            return True
        # Skip common editor temp/swap files
        if basename.endswith("~") or basename.startswith("~"):
            return True
        for pattern in self._config.ignore_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
            try:
                rel = os.path.relpath(path, self._watch_dir)
                if fnmatch.fnmatch(rel, pattern):
                    return True
            except ValueError:
                pass
        return False

    @staticmethod
    def _looks_temporary(rel_path: str) -> bool:
        name = os.path.basename(rel_path).lower()
        return (
            name.startswith("~")
            or name.endswith(".tmp")
            or name.endswith(".swp")
            or name.endswith(".bak")
            or "untitled" in name
        )

    @staticmethod
    def _looks_like_backup(rel_path: str) -> bool:
        name = os.path.basename(rel_path).lower()
        return "backup" in name or "copy" in name or name.endswith(".bak") or " copy" in name

    @staticmethod
    def _detect_naming_pattern(old_name: str, new_name: str) -> str:
        old_stem = Path(old_name).stem
        new_stem = Path(new_name).stem
        old_ext = Path(old_name).suffix
        new_ext = Path(new_name).suffix

        if old_ext != new_ext:
            return "extension_change"

        # Date prefix detection
        import re

        date_re = re.compile(r"^\d{4}[-_]\d{2}")
        old_has_date = bool(date_re.match(old_stem))
        new_has_date = bool(date_re.match(new_stem))
        if not old_has_date and new_has_date:
            return "added_date_prefix"
        if old_has_date and not new_has_date:
            return "removed_date_prefix"

        # Case change
        if old_stem.lower() == new_stem.lower() and old_stem != new_stem:
            return "case_change"

        # Snake <-> kebab
        if "_" in old_stem and "-" in new_stem:
            return "snake_to_kebab"
        if "-" in old_stem and "_" in new_stem:
            return "kebab_to_snake"

        return "other"
