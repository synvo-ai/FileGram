"""Periodic directory-tree snapshotter."""

from __future__ import annotations

import os
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from .exporter import Exporter
from .models import fs_snapshot_event


class Snapshotter:
    """Takes periodic fs_snapshot events of watched directories."""

    def __init__(
        self,
        watch_dirs: list[str],
        exporter: Exporter,
        interval_sec: int,
        ignore_patterns: list[str] | None = None,
    ):
        self._watch_dirs = watch_dirs
        self._exporter = exporter
        self._interval = interval_sec
        self._ignore = set(ignore_patterns or [])
        self._timer: threading.Timer | None = None
        self._stopped = threading.Event()

    def take_snapshot(self) -> None:
        """Build and emit a snapshot for every watched directory."""
        for wd in self._watch_dirs:
            tree, counts, max_depth, total = self._scan(wd)
            event = fs_snapshot_event(
                session_id=self._exporter.session_id,
                profile_id=self._exporter.profile_id,
                directory_tree=tree,
                file_count_by_type=dict(counts),
                max_depth=max_depth,
                total_files=total,
            )
            self._exporter.append(event)

    def start(self) -> None:
        """Start periodic snapshotting in a background thread."""
        self._schedule()

    def stop(self) -> None:
        self._stopped.set()
        if self._timer:
            self._timer.cancel()

    # -- internal --

    def _schedule(self) -> None:
        if self._stopped.is_set():
            return
        self._timer = threading.Timer(self._interval, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self) -> None:
        if self._stopped.is_set():
            return
        self.take_snapshot()
        self._schedule()

    def _scan(self, root: str) -> tuple[dict, Counter, int, int]:
        """Walk *root* and return (tree, type_counts, max_depth, total_files)."""
        counts: Counter[str] = Counter()
        max_depth = 0
        total = 0

        def _build(path: str, depth: int) -> dict[str, Any]:
            nonlocal max_depth, total
            name = os.path.basename(path)
            if os.path.isfile(path):
                ext = Path(path).suffix.lstrip(".") or "unknown"
                counts[ext] += 1
                total += 1
                if depth > max_depth:
                    max_depth = depth
                return {"name": name, "type": "file"}
            # directory
            if depth > max_depth:
                max_depth = depth
            children = []
            try:
                entries = sorted(os.listdir(path))
            except PermissionError:
                entries = []
            for entry in entries:
                full = os.path.join(path, entry)
                # Simple ignore: skip hidden dirs and common noise
                if entry.startswith(".") or entry in (
                    "node_modules",
                    "__pycache__",
                    ".git",
                ):
                    continue
                children.append(_build(full, depth + 1))
            return {"name": name, "type": "directory", "children": children}

        tree = _build(root, 0)
        return tree, counts, max_depth, total
