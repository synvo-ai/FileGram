"""FileGramPlugin entry point.

Usage::

    python -m filegramplugin.main [config.yaml]
    # or
    python filegramplugin/main.py [config.yaml]

Press Ctrl+C to stop. A final snapshot and summary are written on exit.
"""

from __future__ import annotations

import os
import signal
import sys
import time
import uuid
from pathlib import Path

from watchdog.observers import Observer

from .config import load_config
from .copy_detector import CopyDetector
from .differ import Differ
from .exporter import Exporter
from .media_store import MediaStore
from .models import session_end_event, session_start_event
from .open_monitor import OpenMonitor
from .snapshotter import Snapshotter
from .watcher import FileGramHandler


def main(config_path: str | None = None) -> None:
    # --- config ---
    config = load_config(config_path)
    if not config.watch_dirs:
        print("ERROR: no watch_dirs configured. Check config.yaml.")
        sys.exit(1)

    # Resolve paths
    config.watch_dirs = [str(Path(d).expanduser().resolve()) for d in config.watch_dirs]
    for wd in config.watch_dirs:
        if not Path(wd).is_dir():
            print(f"ERROR: watch_dir does not exist: {wd}")
            sys.exit(1)

    session_id = str(uuid.uuid4())
    data_dir = Path(__file__).parent / "data"
    session_dir = data_dir / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    shadow_dir = data_dir / ".shadow"

    # If running as root (sudo), fix ownership so files are accessible without sudo
    if os.geteuid() == 0:
        real_user = os.environ.get("SUDO_UID")
        real_group = os.environ.get("SUDO_GID")
        if real_user and real_group:
            uid, gid = int(real_user), int(real_group)
            for d in (data_dir, session_dir, shadow_dir):
                d.mkdir(parents=True, exist_ok=True)
                os.chown(d, uid, gid)
                for root, dirs, files in os.walk(d):
                    for name in dirs + files:
                        try:
                            os.chown(os.path.join(root, name), uid, gid)
                        except OSError:
                            pass

    print(f"Session: {session_id}")
    print(f"Watching: {config.watch_dirs}")
    print(f"Data dir: {session_dir}")

    # --- components ---
    media_store = MediaStore(session_dir)
    exporter = Exporter(session_dir / "events.json", session_id, config.profile_id)
    differ = Differ(shadow_dir, media_store, config.text_extensions)
    copy_detector = CopyDetector()
    snapshotter = Snapshotter(
        config.watch_dirs,
        exporter,
        config.snapshot_interval_sec,
        config.ignore_patterns,
    )

    start_time = time.time() * 1000

    # --- session start ---
    for wd in config.watch_dirs:
        evt = session_start_event(session_id, config.profile_id, wd)
        exporter.append(evt)

    # Initial shadow copies
    print("Initializing shadow copies...")
    differ.init_shadows(config.watch_dirs)
    print("Taking initial snapshot...")
    snapshotter.take_snapshot()

    # --- watchdog ---
    observer = Observer()
    for wd in config.watch_dirs:
        handler = FileGramHandler(differ, exporter, copy_detector, config, wd)
        observer.schedule(handler, wd, recursive=True)
    observer.start()
    snapshotter.start()

    # --- file-open monitoring (requires sudo) ---
    open_monitor = OpenMonitor(config.watch_dirs, exporter, config.text_extensions)
    open_monitor.start()

    print("Monitoring file system events. Press Ctrl+C to stop.\n")

    # --- main loop ---
    def _shutdown(signum: int | None = None, frame: object = None) -> None:
        print("\nShutting down...")
        open_monitor.stop()
        snapshotter.stop()
        observer.stop()

        # Final snapshot
        print("Taking final snapshot...")
        snapshotter.take_snapshot()

        duration_ms = int(time.time() * 1000 - start_time)
        end_evt = session_end_event(session_id, config.profile_id, duration_ms)
        exporter.append(end_evt)

        # Summary
        summary = {
            "session_id": session_id,
            "profile_id": config.profile_id,
            "watch_dirs": config.watch_dirs,
            "start_time": start_time,
            "end_time": time.time() * 1000,
            "duration_ms": duration_ms,
            "total_events": exporter.event_count,
            "media_stats": media_store.get_stats(),
        }
        exporter.write_summary(summary)
        print(f"Session saved: {exporter.event_count} events → {session_dir}")

        observer.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown()


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg)
