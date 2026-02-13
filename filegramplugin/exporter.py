"""Event exporter — writes events.json as a JSON array, flushing after each event."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class Exporter:
    """Thread-safe event writer.

    Events are kept in memory and flushed to ``events.json`` after every
    append so that data survives unexpected exits.
    """

    def __init__(self, events_path: Path, session_id: str, profile_id: str):
        self._path = events_path
        self.session_id = session_id
        self.profile_id = profile_id
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)
            self._flush()

    def _flush(self) -> None:
        self._path.write_text(
            json.dumps(self._events, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._fix_owner(self._path)

    @staticmethod
    def _fix_owner(path: Path) -> None:
        """If running as root via sudo, chown file to the real user."""
        if os.geteuid() == 0:
            uid = os.environ.get("SUDO_UID")
            gid = os.environ.get("SUDO_GID")
            if uid and gid:
                try:
                    os.chown(path, int(uid), int(gid))
                except OSError:
                    pass

    def write_summary(self, summary: dict[str, Any]) -> None:
        summary_path = self._path.parent / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @property
    def event_count(self) -> int:
        with self._lock:
            return len(self._events)
