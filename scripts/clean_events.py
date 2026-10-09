#!/usr/bin/env python3
"""Clean raw trajectories into the released behavioral event logs.

For every trajectory under signal/, reads events_raw.json and writes events.json
with only the 12 atomic behavioral event types, dropping the simulation metadata
fields (message_id, model_provider, model_name).

Usage:
    python scripts/clean_events.py                       # write signal/*/events.json
    python scripts/clean_events.py --check               # compare with existing events.json, write nothing
    python scripts/clean_events.py --signal-dir /path/to/signal
"""

import argparse
import json
from pathlib import Path

BEHAVIORAL_TYPES = {
    "file_read",
    "file_write",
    "file_edit",
    "file_delete",
    "file_rename",
    "file_move",
    "file_copy",
    "file_search",
    "file_browse",
    "dir_create",
    "context_switch",
    "cross_file_reference",
}
METADATA_FIELDS = ("message_id", "model_provider", "model_name")


def clean(raw_events: list[dict]) -> list[dict]:
    return [
        {k: v for k, v in e.items() if k not in METADATA_FIELDS}
        for e in raw_events
        if e.get("event_type") in BEHAVIORAL_TYPES
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean events_raw.json into events.json")
    parser.add_argument("--signal-dir", type=Path, default=Path(__file__).resolve().parent.parent / "signal")
    parser.add_argument("--check", action="store_true", help="compare with existing events.json instead of writing")
    args = parser.parse_args()

    raw_paths = sorted(args.signal_dir.glob("*/events_raw.json"))
    mismatched = []
    for raw_path in raw_paths:
        out_path = raw_path.with_name("events.json")
        events = clean(json.loads(raw_path.read_text(encoding="utf-8")))
        text = json.dumps(events, indent=2, ensure_ascii=False)
        if args.check:
            if not out_path.exists() or out_path.read_text(encoding="utf-8") != text:
                mismatched.append(raw_path.parent.name)
        else:
            out_path.write_text(text, encoding="utf-8")

    if args.check:
        print(f"{len(raw_paths) - len(mismatched)}/{len(raw_paths)} trajectories match events.json")
        for name in mismatched:
            print(f"  MISMATCH {name}")
        raise SystemExit(1 if mismatched else 0)
    print(f"Wrote events.json for {len(raw_paths)} trajectories")


if __name__ == "__main__":
    main()
