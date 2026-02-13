"""File-open monitor using macOS fs_usage.

Runs ``sudo fs_usage -w -f filesys`` in a subprocess and parses ``open``
syscalls to emit ``file_read`` events.  Requires root privileges.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .exporter import Exporter
from .models import get_directory_depth, get_file_type

# fs_usage open line example:
#   18:02:54.022703  open  F=24  (R_____)  /tmp/filename  0.000003  Preview.12345
# We care about: operation == "open", flags contain "R", and the path.
# The process name is at the end of the line (word.PID or word).
_OPEN_RE = re.compile(
    r"^\s*[\d:.]+\s+"  # timestamp
    r"open\s+"  # operation
    r"F=\d+\s+"  # file descriptor
    r"\(([^)]+)\)\s+"  # flags group (e.g., R_____)
    r"(/\S+)"  # absolute file path
    r".*?\s+(\S+)\s*$"  # process name at end of line
)

# Processes to ignore — system daemons, indexers, thumbnail generators, etc.
_IGNORE_PROCS = {
    # System daemons
    "mds",
    "mds_stores",
    "mdworker",
    "mdworker_shared",
    "fseventsd",
    "fseventsctl",
    "distnoted",
    "cfprefsd",
    "lsd",
    "trustd",
    "secinitd",
    "launchd",
    "syslogd",
    "logd",
    "notifyd",
    "watchdogd",
    "kernelmanagerd",
    "endpointsecurityd",
    "sandboxd",
    "syspolicyd",
    "runningboardd",
    "containermanagerd",
    "filecoordinationd",
    "com.apple.bird",
    "bird",
    "cloudd",
    # Thumbnail / preview generators (cause false file_read events)
    "quicklookd",
    "quicklooksatellite",
    "qlmanage",
    "com.apple.quicklook.thumbnailsagent",
    "thumbnailextension",
    "thumbnailagent",
    "iconservicesagent",
    "finder",  # Finder opens files for thumbnails/previews
    # Our own process
    "python",
    "python3",
    "python3.12",
    "python3.11",
    "python3.10",
}


class OpenMonitor:
    """Monitors file opens via fs_usage and emits file_read events."""

    def __init__(
        self,
        watch_dirs: list[str],
        exporter: Exporter,
        text_extensions: list[str],
        dedup_window_sec: float = 10.0,
    ):
        self._watch_dirs = [str(Path(d).resolve()) for d in watch_dirs]
        self._exporter = exporter
        self._text_exts = set(ext.lower().lstrip(".") for ext in text_extensions)
        self._dedup_window = dedup_window_sec

        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()

        # Track per-file view stats: path -> (view_count, last_view_time_ms)
        self._file_views: dict[str, tuple[int, float]] = {}
        # The file the user is currently viewing (last emitted file_read)
        self._current_file: str | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start fs_usage subprocess and parsing thread."""
        try:
            self._proc = subprocess.Popen(
                ["sudo", "-n", "fs_usage", "-w", "-f", "filesys"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except (OSError, PermissionError) as e:
            print(f"[OpenMonitor] Failed to start fs_usage: {e}")
            print("[OpenMonitor] Run with: sudo python -m filegramplugin")
            return

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        print("[OpenMonitor] fs_usage file-open monitoring active")

    def stop(self) -> None:
        self._stopped.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def _reader_loop(self) -> None:
        """Read fs_usage stdout line by line and parse open() calls."""
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if self._stopped.is_set():
                break
            self._parse_line(line)

    def _parse_line(self, line: str) -> None:
        m = _OPEN_RE.search(line)
        if not m:
            return

        flags = m.group(1)
        file_path = m.group(2)
        proc_raw = m.group(3) if m.lastindex >= 3 else ""

        # Only care about read opens
        if "R" not in flags:
            return

        # Extract process name (fs_usage format: "ProcessName.PID")
        proc_name = proc_raw.rsplit(".", 1)[0].lower() if proc_raw else ""
        # Also handle spaces in process names (fs_usage truncates them)
        proc_name = proc_name.strip()
        if proc_name in _IGNORE_PROCS:
            return
        # Filter any process containing these substrings
        if any(
            s in proc_name
            for s in (
                "quicklook",
                "thumbnail",
                "mdworker",
                "spotlight",
                "iconservices",
                "filecoordination",
            )
        ):
            return

        # Must be under a watched directory
        watch_dir = self._match_watch_dir(file_path)
        if not watch_dir:
            return

        # Skip directories, non-existent, hidden files
        if not os.path.isfile(file_path):
            return
        basename = os.path.basename(file_path)
        if basename.startswith(".") or ".sb-" in basename:
            return

        # Skip system/noise extensions
        ext = Path(file_path).suffix.lstrip(".").lower()
        if ext in ("plist", "db", "sqlite", "lock", "pid"):
            return

        rel_path = os.path.relpath(file_path, watch_dir)
        now_ms = time.time() * 1000

        with self._lock:
            # If this is the same file the user is already viewing, skip.
            # App is just re-reading (rendering pages, loading metadata, etc.)
            if rel_path == self._current_file:
                return

            view_count = 1
            revisit_interval_ms = None

            if rel_path in self._file_views:
                prev_count, prev_time = self._file_views[rel_path]
                view_count = prev_count + 1
                revisit_interval_ms = int(now_ms - prev_time)

            self._file_views[rel_path] = (view_count, now_ms)
            self._current_file = rel_path

        # Get content length
        content_length = 0
        try:
            content_length = os.path.getsize(file_path)
        except OSError:
            pass

        event = self._make_file_read_event(rel_path, watch_dir, view_count, content_length, revisit_interval_ms)
        self._exporter.append(event)

    def _match_watch_dir(self, abs_path: str) -> str | None:
        """Return the watch_dir that contains abs_path, or None."""
        resolved = str(Path(abs_path).resolve())
        for wd in self._watch_dirs:
            if resolved.startswith(wd + "/") or resolved == wd:
                return wd
        return None

    def _make_file_read_event(
        self,
        rel_path: str,
        watch_dir: str,
        view_count: int,
        content_length: int,
        revisit_interval_ms: int | None,
    ) -> dict[str, Any]:
        import uuid

        return {
            "event_id": str(uuid.uuid4()),
            "event_type": "file_read",
            "timestamp": time.time() * 1000,
            "session_id": self._exporter.session_id,
            "profile_id": self._exporter.profile_id,
            "message_id": None,
            "model_provider": None,
            "model_name": None,
            "file_path": rel_path,
            "file_type": get_file_type(rel_path),
            "directory_depth": get_directory_depth(os.path.join(watch_dir, rel_path), watch_dir),
            "view_count": view_count,
            "view_range": None,
            "content_length": content_length,
            "revisit_interval_ms": revisit_interval_ms,
        }
