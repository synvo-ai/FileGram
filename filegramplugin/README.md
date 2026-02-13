# FileGramPlugin

**FileGramPlugin** is a real-time file system behavioral data collector for macOS. It monitors file operations on your local machine and generates the exact same `events.json` format as FileGram agent simulations, enabling direct comparison between real user behavior and agent-generated behavior.

## Overview

FileGramPlugin runs as a background daemon that watches specified directories for file system changes. It captures writes, edits, moves, renames, deletions, directory creations, and file opens, producing behavioral traces compatible with FileGram's evaluation pipeline.

### What It Does

- **Monitors file system operations** using macOS FSEvents (via watchdog)
- **Captures file opens** using `fs_usage` syscall monitoring (requires sudo)
- **Tracks content changes** via shadow copies and unified diffs
- **Detects file copies** heuristically based on content hash matching
- **Takes periodic directory snapshots** for full file system state
- **Stores content in CAS** (content-addressable storage) for deduplication
- **Outputs schema-compatible events.json** matching FileGram agent format

### Why It Exists

FileGram generates behavioral data from AI agents performing tasks under controlled user profiles. FileGramPlugin collects the same behavioral signals from **real users** on their actual machines, enabling:

1. **Validation** — Compare agent behavior against ground-truth human behavior
2. **Augmentation** — Mix real user traces with synthetic agent data
3. **Calibration** — Tune agent profiles to match real user behavioral patterns
4. **Evaluation** — Benchmark memory systems on mixed real+synthetic data

## Features

### File System Event Capture

| Operation | Watchdog Event | FileGram Event Type | Description |
|-----------|---------------|-------------------|-------------|
| File created | `FileCreatedEvent` | `file_write` (create) | New file creation with content snapshot |
| File modified | `FileModifiedEvent` | `file_edit` | File content changes with unified diff |
| File deleted | `FileDeletedEvent` | `file_delete` | File deletion with age tracking |
| File renamed | `FileMovedEvent` (same dir) | `file_rename` | Filename change with pattern detection |
| File moved | `FileMovedEvent` (diff dir) | `file_move` | File relocation across directories |
| Directory created | `DirCreatedEvent` | `dir_create` | New directory with depth info |
| File copied | Heuristic detection | `file_copy` | Copy operation detected by content hash |
| File opened | `fs_usage` syscall | `file_read` | File access with view count tracking |

### Content Preservation

- **Shadow copies** — Maintains a mirror of watched files to enable diff computation
- **Content-addressable storage (CAS)** — Deduplicates file content by SHA-256 hash
- **Unified diffs** — Stores text file changes as patches in `media/diffs/`
- **Full content snapshots** — Stores complete file content in `media/blobs/`
- **Directory tree snapshots** — Periodic full directory structure captures

### Smart Filtering

- **Debouncing** — Merges rapid-fire modify events (e.g., autosave) within 2-second window
- **Atomic save handling** — Detects and filters macOS atomic save temp files (`.sb-*`)
- **System noise filtering** — Ignores `.git/`, `.DS_Store`, `node_modules/`, etc.
- **Process filtering** — Filters out system daemons, indexers, thumbnail generators from file_read events

## Installation

### Requirements

- **macOS** (uses FSEvents and fs_usage)
- **Python >= 3.10**
- **sudo access** (optional, required only for file-open monitoring)

### Install Dependencies

```bash
cd /Users/choiszt/Desktop/code/Synvo/FileGram/filegramplugin
pip install -r requirements.txt
```

Dependencies:
- `watchdog>=4.0.0` — File system event monitoring
- `pyyaml>=6.0` — Configuration loading

## Configuration

Edit `config.yaml` to customize monitoring behavior:

```yaml
# Directories to monitor (absolute paths)
watch_dirs:
  - /Users/username/Desktop/test

# Glob patterns to ignore
ignore_patterns:
  - "**/.git/**"
  - "**/.DS_Store"
  - "**/node_modules/**"
  - "**/__pycache__/**"

# Dedup window: merge rapid-fire modify events within this window (seconds)
dedup_window_sec: 2.0

# Directory snapshot interval (seconds)
snapshot_interval_sec: 300

# Binary file size threshold — files above this only record hash (bytes)
binary_size_threshold: 1048576  # 1 MB

# Text file extensions (others treated as binary)
text_extensions:
  - txt
  - md
  - py
  - json
  - yaml
  # ... (see config.yaml for full list)

# User identifier (appears in events.json as profile_id)
profile_id: "real_user"
```

## Usage

### Basic Usage (No File-Open Monitoring)

```bash
python -m filegramplugin
# or
python filegramplugin/main.py
```

This monitors file writes, edits, moves, renames, deletions, and directory creations. **File-open events are not captured** without sudo.

### Full Monitoring (With File Opens)

```bash
sudo python -m filegramplugin
```

This enables `fs_usage` syscall monitoring to capture `file_read` events.

**Note**: `fs_usage` requires root privileges. The plugin automatically fixes file ownership so output files remain accessible to the original user.

### Custom Configuration

```bash
python -m filegramplugin /path/to/custom_config.yaml
```

### Stopping the Plugin

Press `Ctrl+C` to stop. The plugin will:
1. Take a final directory snapshot
2. Write session summary to `summary.json`
3. Flush all events to `events.json`

## Output Format

All data is stored under `filegramplugin/data/sessions/{session_id}/`:

```
data/sessions/{session_id}/
├── events.json          # Behavioral event log (JSON array)
├── summary.json         # Session statistics
└── media/               # Content-addressable storage
    ├── blobs/           # Full file content snapshots ({hash}.blob)
    ├── diffs/           # Unified diffs ({hash}.diff)
    └── manifest.json    # CAS metadata and deduplication stats
```

### events.json Schema

FileGramPlugin produces the **exact same schema** as FileGram agent simulations. Each event has:

**Common fields** (present in all events):
```json
{
  "event_id": "uuid",
  "event_type": "file_write",
  "timestamp": 1770542208476.703,
  "session_id": "uuid",
  "profile_id": "real_user",
  "message_id": null,
  "model_provider": null,
  "model_name": null
}
```

**Type-specific fields** (examples):

- `file_write`: `file_path`, `file_type`, `operation`, `content_length`, `before_hash`, `after_hash`, `media_ref`
- `file_edit`: `file_path`, `lines_added`, `lines_deleted`, `lines_modified`, `diff_summary`, `before_hash`, `after_hash`, `diff_media`
- `file_delete`: `file_path`, `file_age_ms`, `was_temporary`
- `file_rename`: `old_path`, `new_path`, `naming_pattern_change`
- `file_move`: `old_path`, `new_path`, `destination_directory_depth`
- `dir_create`: `dir_path`, `depth`, `sibling_count`
- `file_copy`: `source_path`, `dest_path`, `is_backup`
- `file_read`: `file_path`, `view_count`, `content_length`, `revisit_interval_ms`
- `fs_snapshot`: `directory_tree`, `file_count_by_type`, `max_depth`, `total_files`

See FileGram `CLAUDE.md` § "Behavioral Signal Schema" for complete field definitions.

### summary.json

Session-level statistics:
```json
{
  "session_id": "uuid",
  "profile_id": "real_user",
  "watch_dirs": ["/path/to/watched/dir"],
  "start_time": 1770542208476.703,
  "end_time": 1770545208476.703,
  "duration_ms": 3000000,
  "total_events": 127,
  "media_stats": {
    "total_blobs": 23,
    "total_diffs": 18,
    "total_bytes": 456789,
    "deduplicated_saves": 5
  }
}
```

## Architecture

### Components

| Module | Purpose |
|--------|---------|
| `main.py` | Entry point, session management, signal handling |
| `watcher.py` | Watchdog event handler — translates FS events to FileGram events |
| `differ.py` | Shadow copy manager and diff calculator |
| `media_store.py` | Content-addressable storage (CAS) for blobs and diffs |
| `exporter.py` | Thread-safe JSON event writer with real-time flushing |
| `snapshotter.py` | Periodic directory tree snapshot generator |
| `copy_detector.py` | Heuristic file copy detection via content hash matching |
| `open_monitor.py` | fs_usage syscall parser for file-open events |
| `models.py` | Event factory functions matching FileGram schema |
| `config.py` | YAML configuration loader |

### Event Flow

```
File system operation (user action)
  ↓
Watchdog event OR fs_usage syscall
  ↓
FileGramHandler or OpenMonitor
  ↓
Differ (compute diff if edit) + MediaStore (store content)
  ↓
Event dict created via models.py factory
  ↓
Exporter.append() → events.json (flushed immediately)
```

### Shadow Copy Strategy

For every file in the watched directories, FileGramPlugin maintains a "shadow copy" in `data/.shadow/{watch_dir_basename}/{relative_path}`. When a file is modified:

1. Read current file content
2. Compare with shadow copy
3. Generate unified diff (text files) or record hash change (binary files)
4. Update shadow copy
5. Store diff and new content in CAS

This enables precise change tracking even when files are modified by external editors.

### Copy Detection Strategy

macOS emits `FileCreatedEvent` for both genuinely new files and copied files. To distinguish:

1. When a file is created, compute its content hash
2. Check if this hash matches a recently-seen file (within 10 seconds)
3. If match found → emit `file_copy` event
4. Otherwise → emit `file_write` (create) event

### File-Open Monitoring Strategy

`fs_usage -f filesys` outputs every file system syscall with the format:
```
18:02:54.022703  open  F=24  (R_____)  /path/to/file  0.000003  ProcessName.12345
```

`open_monitor.py` parses these lines, filters out system processes (mdworker, Spotlight, QuickLook, etc.), and emits `file_read` events for user-initiated file opens.

## Limitations

### What FileGramPlugin Does NOT Capture

The following signals are specific to FileGram agent simulations and cannot be captured from a real user's file system:

| Signal | Why Not Captured |
|--------|-----------------|
| `file_search` | Requires hooking into grep/Spotlight API calls (not available via FS events) |
| `file_browse` | Would need application-level hooks (Finder, terminal `ls`, etc.) |
| `iteration_start/end` | Agent-specific concept (conversation turns) |
| `context_switch` | Requires inferring causal relationships from read sequences |
| `cross_file_reference` | Requires content analysis to detect references |
| `error_encounter/response` | Requires application-level error reporting |
| `tool_call`, `llm_response` | Agent simulation metadata |

FileGramPlugin captures **write operations, file organization behavior, and file access patterns** — sufficient for evaluating profile dimensions B (Production), C (Organization), D (Iteration), and partially F (Cross-Modal).

### macOS-Specific Behaviors

- **Atomic saves** — Apps like TextEdit and Xcode save via temporary `.sb-*` files. FileGramPlugin detects and filters these, treating the final rename as a single edit.
- **Finder thumbnails** — Finder opens files to generate thumbnails. FileGramPlugin filters these by process name.
- **Time Machine, Spotlight** — System indexers are filtered out.

## Validation

To verify the plugin is working correctly:

1. **Start the plugin** in a terminal:
   ```bash
   sudo python -m filegramplugin
   ```

2. **Perform test operations** in the watched directory:
   - Create a new file
   - Edit a file (save multiple times to test debouncing)
   - Rename a file
   - Move a file to a subdirectory
   - Copy a file
   - Delete a file
   - Open a file in an editor

3. **Stop the plugin** with `Ctrl+C`

4. **Check output**:
   ```bash
   cd data/sessions/{latest-session-id}/
   cat events.json | jq '.[] | {event_type, file_path}' # View event types and paths
   ls media/blobs/  # Check content snapshots
   ls media/diffs/  # Check diff files
   ```

5. **Verify schema compatibility** — Load `events.json` into FileGram's evaluation pipeline and confirm no schema errors.

## Integration with FileGram

FileGramPlugin outputs are **drop-in compatible** with FileGram's behavioral data pipeline:

```
FileGramPlugin events.json
  ↓
FileGramBench evaluation tasks (same format as agent-generated data)
  ↓
Memory system (FileGramOS, Mem0, Zep, etc.)
  ↓
Profile reconstruction or behavioral prediction metrics
```

Use real user traces to:
- **Validate** agent profiles — Do synthetic traces match real user behavior?
- **Augment** training data — Mix real traces into agent-generated datasets
- **Calibrate** profile dimensions — Tune L/M/R behavioral specs to match real distributions
- **Benchmark** memory systems on mixed real+synthetic data

## Troubleshooting

### "Failed to start fs_usage: Permission denied"

File-open monitoring requires sudo:
```bash
sudo python -m filegramplugin
```

If you can't use sudo, file-open events will be skipped but all other events (writes, edits, moves, etc.) will still be captured.

### "Too many file_read events from system processes"

Edit `open_monitor.py` and add process names to `_IGNORE_PROCS`. Common offenders:
- `Finder` (thumbnail generation)
- `mdworker*` (Spotlight indexing)
- `quicklook*` (preview generation)

### "Events missing for rapid file edits"

This is **by design** — rapid-fire modify events within `dedup_window_sec` (default 2 seconds) are merged into a single `file_edit` event. This prevents autosave from generating hundreds of duplicate events.

To disable debouncing, set `dedup_window_sec: 0` in `config.yaml`.

### "Directory snapshot events missing"

Check `snapshot_interval_sec` in `config.yaml` (default 300 seconds = 5 minutes). Snapshots are also taken at session start and end.

### "File content not saved in media/blobs/"

Binary files larger than `binary_size_threshold` (default 1 MB) only record their hash, not full content. Text files are always saved (truncated at 1 MB).

To save larger files, increase `binary_size_threshold` in `config.yaml`.

## License

Part of the FileGram project. See main repository for license information.

## Related Documentation

- **FileGram CLAUDE.md** — Full project design document
- **FileGram behavior/events.py** — Event type definitions (source of truth for schema)
- **FileGram behavior/media_store.py** — Agent-side CAS implementation (mirrored here)
