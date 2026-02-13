"""Heuristic copy detection.

Watches for create events whose content hash matches a recently-seen file,
indicating the new file is likely a copy.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

HASH_PREFIX_LEN = 16
WINDOW_SEC = 10.0  # Look-back window for copy detection


class CopyDetector:
    """Track file content hashes to detect copy operations.

    macOS (and most OSes) emit separate create + modify events for a copy.
    We detect copies by matching the content hash of a newly-created file
    against hashes of existing files that were recently read or modified.
    """

    def __init__(self) -> None:
        # hash -> (abs_path, timestamp)
        self._known: dict[str, tuple[str, float]] = {}

    def register(self, abs_path: str) -> None:
        """Register a file's content hash."""
        try:
            data = Path(abs_path).read_bytes()
        except (OSError, PermissionError):
            return
        h = hashlib.sha256(data).hexdigest()[:HASH_PREFIX_LEN]
        self._known[h] = (abs_path, time.time())

    def check_copy(self, abs_path: str) -> str | None:
        """Return the source path if *abs_path* looks like a copy, else None."""
        try:
            data = Path(abs_path).read_bytes()
        except (OSError, PermissionError):
            return None
        h = hashlib.sha256(data).hexdigest()[:HASH_PREFIX_LEN]
        if h in self._known:
            src, ts = self._known[h]
            if src != abs_path and (time.time() - ts) < WINDOW_SEC:
                return src
        # Register regardless
        self._known[h] = (abs_path, time.time())
        return None

    def prune(self, max_age: float = 300.0) -> None:
        """Remove stale entries older than *max_age* seconds."""
        cutoff = time.time() - max_age
        self._known = {h: (p, t) for h, (p, t) in self._known.items() if t > cutoff}
