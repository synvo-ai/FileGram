#!/usr/bin/env python3
"""FileGram Bench Dashboard Server.

Reads results from test_results/*.json and serves an interactive dashboard.
Usage: python bench/dashboard/server.py
"""

import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import urllib.parse
import webbrowser

# Add bench/ to sys.path so pickle can resolve filegramos dataclasses
_bench_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _bench_dir not in sys.path:
    sys.path.insert(0, _bench_dir)

try:
    import yaml
except ImportError:
    yaml = None

try:
    import numpy as np
except ImportError:
    np = None
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

PORT = 8765
BENCH_DIR = Path(__file__).parent.parent  # bench/
DASHBOARD_DIR = Path(__file__).parent


# Auto-detect model-slug directories, or accept --cache-dir / --results-dir
def _detect_dir(prefix: str) -> Path:
    """Find the first bench/{prefix}_{slug}/ directory, or fall back to bench/{prefix}/."""
    fallback = BENCH_DIR / prefix
    candidates = sorted(BENCH_DIR.glob(f"{prefix}_*"))
    dirs = [c for c in candidates if c.is_dir()]
    return dirs[0] if dirs else fallback


RESULTS_DIR = _detect_dir("test_results")
INGEST_CACHE_DIR = _detect_dir("ingest_cache")
PROJECT_ROOT = BENCH_DIR.parent  # FileGram/
TASKS_DIR = PROJECT_ROOT / "tasks"
PROFILES_DIR = PROJECT_ROOT / "profiles"
SIGNAL_DIR = PROJECT_ROOT / "signal"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
ANNOTATIONS_DIR = BENCH_DIR / "annotations"

# Progress monitor config
PROGRESS_PROFILES = [f"p{i}" for i in range(1, 21)]
PROGRESS_PROFILE_IDS = {
    "p1": "p1_methodical",
    "p2": "p2_thorough_reviser",
    "p3": "p3_efficient_executor",
    "p4": "p4_structured_analyst",
    "p5": "p5_balanced_organizer",
    "p6": "p6_quick_curator",
    "p7": "p7_visual_reader",
    "p8": "p8_minimal_editor",
    "p9": "p9_visual_organizer",
    "p10": "p10_silent_auditor",
    "p11": "p11_meticulous_planner",
    "p12": "p12_prolific_scanner",
    "p13": "p13_visual_architect",
    "p14": "p14_concise_organizer",
    "p15": "p15_thorough_surveyor",
    "p16": "p16_phased_minimalist",
    "p17": "p17_creative_archivist",
    "p18": "p18_decisive_scanner",
    "p19": "p19_agile_pragmatist",
    "p20": "p20_visual_auditor",
}
PROGRESS_TASKS = [f"T-{i:02d}" for i in range(1, 33)]

# Perturbation config (mirrors run_all_200.py)
PERTURBATION_CANDIDATES = ["t12", "t18", "t22", "t24", "t26", "t29", "t31", "t16"]
PERTURB_PER_PROFILE = 5


def get_perturbed_tasks(profile_short: str) -> set[str]:
    """Return the set of task shorts (e.g. 't12') perturbed for this profile."""
    seed_str = f"{profile_short}_perturbation_assignment"
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    candidates = list(PERTURBATION_CANDIDATES)
    rng.shuffle(candidates)
    return set(candidates[:PERTURB_PER_PROFILE])


def build_perturbation_map():
    """Return {profile_short: set of 'T-XX' task IDs that are perturbed}."""
    result = {}
    for p in PROGRESS_PROFILES:
        shorts = get_perturbed_tasks(p)
        # Convert 't12' → 'T-12'
        result[p] = [f"T-{s[1:].zfill(2)}" for s in shorts]
    return result


# Event types to keep in trajectory viewer (behavioral events only)
BEHAVIORAL_EVENTS = {
    "file_read",
    "file_write",
    "file_edit",
    "file_rename",
    "file_move",
    "dir_create",
    "file_delete",
    "file_copy",
    "file_browse",
    "file_search",
    "context_switch",
    "cross_file_reference",
}
STRIP_FIELDS = {"message_id", "model_provider", "model_name", "session_id", "event_id"}

TEXT_EXTENSIONS = {".md", ".txt", ".eml", ".csv", ".ics", ".json", ".yaml", ".yml", ".log"}
BINARY_MIME = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

# Static file MIME types
STATIC_MIME = {
    ".css": "text/css",
    ".js": "application/javascript",
    ".html": "text/html",
    ".json": "application/json",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".pdf": "application/pdf",
}


def list_files_recursive(dirpath):
    result = []
    dirpath = Path(dirpath)
    if not dirpath.exists():
        return result
    for f in sorted(dirpath.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(dirpath))
            result.append(
                {
                    "path": rel,
                    "size": f.stat().st_size,
                    "type": f.suffix.lstrip(".") or "unknown",
                }
            )
    return result


def filter_events(events, sandbox_prefix):
    prefix_no_slash = sandbox_prefix.rstrip("/")
    alt_prefix = sandbox_prefix.replace("/FileGram/", "/Claude Code/")
    alt_prefix_no_slash = alt_prefix.rstrip("/")
    filtered = []
    for evt in events:
        if evt.get("event_type") not in BEHAVIORAL_EVENTS:
            continue
        clean = {k: v for k, v in evt.items() if k not in STRIP_FIELDS}
        for key in (
            "file_path",
            "old_path",
            "new_path",
            "dir_path",
            "target_directory",
            "from_file",
            "to_file",
            "source_file",
            "target_file",
            "source_path",
            "dest_path",
            "directory_path",
        ):
            if key in clean and isinstance(clean[key], str):
                clean[key] = clean[key].replace(sandbox_prefix, "")
                clean[key] = clean[key].replace(prefix_no_slash, "")
                clean[key] = clean[key].replace(alt_prefix, "")
                clean[key] = clean[key].replace(alt_prefix_no_slash, "")
                if clean[key].startswith("/"):
                    clean[key] = clean[key][1:]
        if "files_listed" in clean and isinstance(clean["files_listed"], list):
            clean["files_listed"] = [
                f.replace(sandbox_prefix, "").lstrip("/") if isinstance(f, str) else f for f in clean["files_listed"]
            ]
        filtered.append(clean)
    return filtered


def task_id_to_workspace(task_id):
    num = task_id.replace("T-", "").replace("t-", "")
    return f"t{num.zfill(2)}_workspace"


def load_all_data():
    data = {"profiles": {}, "tasks": [], "coverage": {}, "profile_matrix": {}, "meta": {}}

    for pdir in sorted(RESULTS_DIR.iterdir()):
        if pdir.is_dir():
            rpath = pdir / "results.json"
            if rpath.exists():
                with open(rpath) as f:
                    data["profiles"][pdir.name] = json.load(f)

    if not data["profiles"]:
        all_path = RESULTS_DIR / "all_results.json"
        if all_path.exists():
            with open(all_path) as f:
                data["profiles"] = json.load(f)

    tasks_path = TASKS_DIR / "all_tasks.json"
    if not tasks_path.exists():
        tasks_path = BENCH_DIR.parent / "tasks" / "all_tasks.json"
    if tasks_path.exists():
        with open(tasks_path) as f:
            data["tasks"] = json.load(f)

    cov_path = TASKS_DIR / "coverage.json"
    if not cov_path.exists():
        cov_path = BENCH_DIR.parent / "tasks" / "coverage.json"
    if cov_path.exists():
        with open(cov_path) as f:
            data["coverage"] = json.load(f)

    matrix_path = PROFILES_DIR / "profile_matrix.json"
    if matrix_path.exists():
        with open(matrix_path) as f:
            data["profile_matrix"] = json.load(f)

    data["prompts"] = {}
    for pdir in sorted(RESULTS_DIR.iterdir()):
        if pdir.is_dir():
            pname = pdir.name
            data["prompts"][pname] = {}
            for pfile in pdir.glob("*_prompt.txt"):
                method = pfile.stem.replace("_prompt", "")
                with open(pfile, encoding="utf-8") as f:
                    data["prompts"][pname][method] = f.read()

    # Prefer v2 curve results (has dim_scores), fall back to v1
    curve_path = RESULTS_DIR / "curve_results_v2.json"
    if not curve_path.exists():
        curve_path = RESULTS_DIR / "curve_results.json"
    if curve_path.exists():
        with open(curve_path) as f:
            data["curve"] = json.load(f)
    else:
        data["curve"] = {}

    data["meta"] = {
        "results_dir": str(RESULTS_DIR),
        "profiles_with_data": list(data["profiles"].keys()),
        "total_methods": 9,
    }

    return data


# ── Annotation helpers ──────────────────────────────────────────────
# Task-type → which dimensions it primarily activates
TASK_TYPE_DIM_MAP = {
    "understand": ["A", "B", "E"],
    "create": ["B", "C", "F"],
    "organize": ["C", "A"],
    "iterate": ["D", "B"],
    "synthesize": ["A", "B", "F"],
    "maintain": ["D", "E", "C"],
}

# Dimension definitions for annotation UI
DIMENSION_DEFS = {
    "A": {
        "name": "Consumption Pattern",
        "question": "How does this user explore and read files?",
        "options": {
            "L": {
                "label": "Sequential",
                "desc": "Reads files one-by-one in order, full content, revisits often. No grep/search.",
            },
            "M": {
                "label": "Targeted",
                "desc": "Searches first (grep/glob), reads only matched sections. Skips irrelevant files.",
            },
            "R": {
                "label": "Breadth-first",
                "desc": "Scans broadly with ls/glob, reads only first 10-20 lines. Coverage over depth.",
            },
        },
    },
    "B": {
        "name": "Production Style",
        "question": "How does this user produce output content?",
        "options": {
            "L": {
                "label": "Comprehensive",
                "desc": "Multi-level headings, tables, appendices, 200+ lines, 3+ files per task.",
            },
            "M": {"label": "Balanced", "desc": "2 heading levels, occasional tables, 80-150 lines, 1-2 files."},
            "R": {"label": "Minimal", "desc": "Flat bullet lists, no deep headings, one file only, under 60 lines."},
        },
    },
    "C": {
        "name": "Organization Preference",
        "question": "How does this user organize files and directories?",
        "options": {
            "L": {
                "label": "Deeply Nested",
                "desc": "3+ level dirs, descriptive names with prefixes, backups, never deletes.",
            },
            "M": {
                "label": "Adaptive",
                "desc": "1-2 level dirs when needed, mixed naming, keeps old versions sometimes.",
            },
            "R": {
                "label": "Flat",
                "desc": "All files in root, no subdirs, short names, overwrites directly, deletes temp files.",
            },
        },
    },
    "D": {
        "name": "Iteration Strategy",
        "question": "How does this user modify and refine existing work?",
        "options": {
            "L": {
                "label": "Incremental",
                "desc": "Many small edits (few lines each), reviews after each, creates backups.",
            },
            "M": {"label": "Balanced", "desc": "Moderate edits of reasonable scope, occasional review."},
            "R": {"label": "Rewrite", "desc": "Overwrites entire files, delete-recreate pattern, one decisive pass."},
        },
    },
    "E": {
        "name": "Curation",
        "question": "How does this user manage workspace artifacts?",
        "options": {
            "L": {
                "label": "Selective",
                "desc": "Actively prunes unnecessary files, deletes temp/intermediate outputs, backs up before pruning.",
            },
            "M": {
                "label": "Pragmatic",
                "desc": "Occasionally removes unneeded files, moderate cleanup but no systematic curation.",
            },
            "R": {
                "label": "Preservative",
                "desc": "Never deletes anything, keeps all intermediate files, workspace only grows.",
            },
        },
    },
    "F": {
        "name": "Cross-Modal Behavior",
        "question": "Does this user work with visual materials?",
        "options": {
            "L": {
                "label": "Visual-heavy",
                "desc": "Generates charts/diagrams, figure references with captions, figures/ directory.",
            },
            "M": {
                "label": "Balanced",
                "desc": "Uses markdown tables where warranted, structured formatting. No standalone images.",
            },
            "R": {
                "label": "Text-only",
                "desc": "Pure text. No tables, no images, no charts. Prose and bullet lists only.",
            },
        },
    },
}


def select_annotation_traces(profile_id: str, tasks_dir: Path) -> list[dict]:
    """Select 5 traces per profile: one per task type (understand, create, organize, iterate, synthesize).
    Uses deterministic seed. Picks the trajectory with the most behavioral events for signal richness."""
    target_types = ["understand", "create", "organize", "iterate", "synthesize"]
    rng = random.Random(hash(profile_id) & 0xFFFFFFFF)

    # Load all task definitions
    task_defs = {}
    for tf in sorted(tasks_dir.glob("t??.json")):
        with open(tf) as f:
            td = json.load(f)
        task_defs[td["task_id"]] = td

    # Group available trajectories by task type
    type_to_tasks = {}
    for tid, td in task_defs.items():
        tt = td.get("type", "")
        if tt in target_types:
            type_to_tasks.setdefault(tt, []).append(tid)

    selected = []
    for tt in target_types:
        candidates = type_to_tasks.get(tt, [])
        if not candidates:
            continue
        # For each candidate, check if trajectory exists and count events
        scored = []
        for tid in candidates:
            sig_dir = SIGNAL_DIR / f"{profile_id}_{tid}"
            events_path = sig_dir / "events.json"
            if events_path.exists():
                try:
                    with open(events_path) as f:
                        evts = json.load(f)
                    beh_count = sum(1 for e in evts if e.get("event_type") in BEHAVIORAL_EVENTS)
                    scored.append((tid, beh_count))
                except Exception:
                    pass
        if not scored:
            continue
        # Pick the one with the most behavioral events (deterministic via sort + seed)
        scored.sort(key=lambda x: (-x[1], x[0]))
        # If tied, use rng to pick among top candidates
        max_count = scored[0][1]
        top = [s for s in scored if s[1] == max_count]
        if len(top) > 1:
            rng.shuffle(top)
        selected.append(
            {
                "task_id": top[0][0],
                "task_type": tt,
                "task_def": task_defs[top[0][0]],
                "event_count": top[0][1],
            }
        )

    return selected


def compute_trace_stats(events: list[dict]) -> dict:
    """Compute summary statistics from filtered behavioral events."""
    reads = writes = edits = dirs = deletes = moves = browses = searches = cross_refs = 0
    files_read = set()
    files_created = set()
    for e in events:
        et = e.get("event_type", "")
        fp = e.get("file_path", "")
        if et == "file_read":
            reads += 1
            if fp:
                files_read.add(fp)
        elif et == "file_write":
            writes += 1
            if fp:
                files_created.add(fp)
        elif et == "file_edit":
            edits += 1
        elif et == "dir_create":
            dirs += 1
        elif et == "file_delete":
            deletes += 1
        elif et in ("file_move", "file_rename"):
            moves += 1
        elif et == "file_browse":
            browses += 1
        elif et == "file_search":
            searches += 1
        elif et == "cross_file_reference":
            cross_refs += 1
    return {
        "total_events": len(events),
        "reads": reads,
        "writes": writes,
        "edits": edits,
        "dirs_created": dirs,
        "deletes": deletes,
        "moves": moves,
        "browses": browses,
        "searches": searches,
        "cross_refs": cross_refs,
        "unique_files_read": len(files_read),
        "files_created": len(files_created),
    }


def compute_dimension_hints(events: list[dict], stats: dict) -> dict:
    """Compute per-dimension statistical hints to assist annotators."""
    hints = {}
    total = stats["total_events"] or 1

    # A: Consumption — read ratio, search ratio, browse ratio
    read_ratio = round(stats["reads"] / total, 2) if total else 0
    search_ratio = round(stats["searches"] / total, 2) if total else 0
    browse_ratio = round(stats["browses"] / total, 2) if total else 0
    hints["A"] = (
        f"Read: {stats['reads']} ({read_ratio}), Search: {stats['searches']} ({search_ratio}), Browse: {stats['browses']} ({browse_ratio})"
    )

    # B: Production — files created, avg content length
    content_lengths = [
        e.get("content_length", 0) for e in events if e.get("event_type") == "file_write" and e.get("content_length")
    ]
    avg_len = round(sum(content_lengths) / len(content_lengths)) if content_lengths else 0
    hints["B"] = f"Files created: {stats['files_created']}, Avg output length: {avg_len} chars"

    # C: Organization — dirs created, moves, deletes
    max_depth = max(
        (e.get("directory_depth", 0) or 0 for e in events if e.get("event_type") == "dir_create"), default=0
    )
    hints["C"] = (
        f"Dirs: {stats['dirs_created']}, Max depth: {max_depth}, Moves: {stats['moves']}, Deletes: {stats['deletes']}"
    )

    # D: Iteration — edits, avg lines changed
    edit_events = [e for e in events if e.get("event_type") == "file_edit"]
    avg_lines = (
        round(sum((e.get("lines_added", 0) + e.get("lines_deleted", 0)) for e in edit_events) / len(edit_events))
        if edit_events
        else 0
    )
    hints["D"] = (
        f"Edits: {stats['edits']}, Avg lines changed: {avg_lines}, Writes (overwrite): {sum(1 for e in events if e.get('event_type') == 'file_write' and e.get('operation') == 'overwrite')}"
    )

    # E: Rhythm — context switches, phasing
    switches = sum(1 for e in events if e.get("event_type") == "context_switch")
    hints["E"] = f"Context switches: {switches}, Event distribution across timeline"

    # F: Cross-modal — cross-file refs, image/media file creation
    media_files = sum(
        1
        for e in events
        if e.get("event_type") == "file_write" and e.get("file_type", "") in ("png", "jpg", "svg", "csv")
    )
    hints["F"] = f"Cross-file refs: {stats['cross_refs']}, Media files created: {media_files}"

    return hints


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.serve_file(DASHBOARD_DIR / "index.html", "text/html")
        elif self.path.startswith("/css/") or self.path.startswith("/js/") or self.path.startswith("/img/"):
            self._serve_static()
        elif self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = load_all_data()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        elif self.path.startswith("/api/prompt/"):
            parts = self.path.split("/")
            if len(parts) == 5:
                profile, method = parts[3], parts[4]
                prompt_path = RESULTS_DIR / profile / f"{method}_prompt.txt"
                if prompt_path.exists():
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    with open(prompt_path, encoding="utf-8") as f:
                        self.wfile.write(f.read().encode("utf-8"))
                else:
                    self.send_error(404, f"Prompt not found: {prompt_path}")
            else:
                self.send_error(400)
        elif self.path == "/api/instructions":
            claude_md = PROJECT_ROOT / "CLAUDE.md"
            if claude_md.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(claude_md, encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            else:
                self.send_error(404, "CLAUDE.md not found")
        elif self.path == "/api/prompts":
            result = {}
            for pdir in sorted(RESULTS_DIR.iterdir()):
                if pdir.is_dir():
                    result[pdir.name] = {}
                    for pfile in pdir.glob("*_prompt.txt"):
                        method = pfile.stem.replace("_prompt", "")
                        result[pdir.name][method] = {
                            "length": pfile.stat().st_size,
                        }
            self._send_json(result)
        elif self.path == "/api/progress":
            self._serve_progress()
        elif self.path == "/api/trajectories":
            self._serve_trajectory_index()
        elif self.path.startswith("/api/trajectory/"):
            self._serve_trajectory_detail()
        elif self.path.startswith("/api/workspace-file/"):
            self._serve_workspace_file()
        elif self.path.startswith("/api/profile/"):
            self._serve_profile()
        elif self.path.startswith("/api/media/"):
            self._serve_media_file()
        elif self.path == "/api/memory-store/sources":
            self._serve_memory_store_sources()
        elif self.path.startswith("/api/memory-store/index"):
            self._serve_memory_store_index()
        elif self.path.startswith("/api/memory-store/"):
            self._serve_memory_store_detail()
        elif self.path.startswith("/api/pipeline/"):
            self._serve_pipeline_data()
        elif self.path == "/annotate" or self.path == "/annotate.html":
            self.serve_file(DASHBOARD_DIR / "annotate.html", "text/html")
        elif self.path.startswith("/api/annotate/init"):
            self._serve_annotate_init()
        elif self.path.startswith("/api/annotate/traces/"):
            self._serve_annotate_traces()
        elif self.path.startswith("/api/annotate/truth/"):
            self._serve_annotate_truth()
        elif self.path == "/api/annotate/results":
            self._serve_annotate_results()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/annotate/save":
            self._handle_annotate_save()
        else:
            self.send_error(404)

    def _serve_static(self):
        """Serve CSS/JS files from dashboard directory."""
        clean_path = urllib.parse.unquote(self.path).lstrip("/")
        file_path = DASHBOARD_DIR / clean_path
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, f"Static file not found: {clean_path}")
            return
        # Security: ensure path is within dashboard dir
        try:
            file_path.resolve().relative_to(DASHBOARD_DIR.resolve())
        except ValueError:
            self.send_error(403, "Access denied")
            return
        ext = file_path.suffix.lower()
        content_type = STATIC_MIME.get(ext, "application/octet-stream")
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))

    def _serve_progress(self):
        """GET /api/progress -- real-time progress of trajectory run."""
        done = {}
        for p in PROGRESS_PROFILES:
            pid = PROGRESS_PROFILE_IDS[p]
            done[p] = []
            for t in PROGRESS_TASKS:
                sig = SIGNAL_DIR / f"{pid}_{t}"
                if sig.is_dir() and (sig / "events.json").exists():
                    done[p].append(t)

        # Count active filegram processes
        active = 0
        try:
            result = subprocess.run(["pgrep", "-f", "filegram -1"], capture_output=True, text=True, timeout=5)
            if result.stdout.strip():
                active = len(result.stdout.strip().split("\n"))
        except Exception:
            pass

        total_done = sum(len(v) for v in done.values())
        perturbed = build_perturbation_map()
        self._send_json(
            {
                "done": done,
                "profiles": PROGRESS_PROFILES,
                "profile_ids": PROGRESS_PROFILE_IDS,
                "tasks": PROGRESS_TASKS,
                "total_done": total_done,
                "total_target": len(PROGRESS_PROFILES) * len(PROGRESS_TASKS),
                "active_processes": active,
                "perturbed": perturbed,
            }
        )

    def _serve_trajectory_index(self):
        trajectories = []
        tasks = {}
        if SIGNAL_DIR.exists():
            for d in sorted(SIGNAL_DIR.iterdir()):
                if not d.is_dir() or d.name.startswith("_"):
                    continue
                idx = d.name.rfind("_T-")
                if idx == -1:
                    continue
                profile = d.name[:idx]
                task = d.name[idx + 1 :]
                summary_path = d / "summary.json"
                summary = {}
                if summary_path.exists():
                    with open(summary_path) as f:
                        summary = json.load(f)
                trajectories.append(
                    {
                        "profile": profile,
                        "task": task,
                        "total_events": summary.get("total_events", 0),
                        "total_iterations": summary.get("total_iterations", 0),
                        "duration_ms": summary.get("total_duration_ms", 0),
                        "files_created": len(summary.get("files_created", [])),
                        "dirs_created": len(summary.get("dirs_created", [])),
                        "unique_files_read": summary.get("unique_files_read", 0),
                    }
                )
        for tf in sorted(TASKS_DIR.glob("t??.json")):
            with open(tf) as f:
                td = json.load(f)
                tasks[td["task_id"]] = {
                    "name": td.get("name", ""),
                    "name_en": td.get("name_en", ""),
                    "type": td.get("type", ""),
                    "dimensions": td.get("dimensions", []),
                }
        self._send_json({"trajectories": trajectories, "tasks": tasks})

    def _serve_trajectory_detail(self):
        parts = self.path.split("/")
        if len(parts) < 5:
            self.send_error(400, "Expected /api/trajectory/{profile}/{task}")
            return
        profile = parts[3]
        task = parts[4]
        sig_dir = SIGNAL_DIR / f"{profile}_{task}"
        if not sig_dir.exists():
            self.send_error(404, f"Trajectory not found: {profile}_{task}")
            return

        # Construct sandbox prefix from profile/task (deterministic path)
        # session_start event was removed during signal cleaning
        sandbox_prefix = str(PROJECT_ROOT / "sandbox" / f"{profile}_{task}") + "/"

        events_path = sig_dir / "events.json"
        events = []
        if events_path.exists():
            with open(events_path, encoding="utf-8") as f:
                raw = json.load(f)
            events = filter_events(raw, sandbox_prefix)

        summary_path = sig_dir / "summary.json"
        summary = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

        task_num = task.replace("T-", "").replace("t-", "")
        task_path = TASKS_DIR / f"t{task_num.zfill(2)}.json"
        task_def = {}
        if task_path.exists():
            with open(task_path) as f:
                task_def = json.load(f)

        ws_name = task_id_to_workspace(task)
        ws_dir = WORKSPACE_DIR / ws_name
        workspace_files = list_files_recursive(ws_dir) if ws_dir.exists() else []

        manifest_path = sig_dir / "media" / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

        self._send_json(
            {
                "profile": profile,
                "task": task,
                "events": events,
                "summary": summary,
                "task_def": task_def,
                "workspace_files": workspace_files,
                "media_manifest": manifest.get("entries", {}),
            }
        )

    def _serve_workspace_file(self):
        raw_path = urllib.parse.unquote(self.path)
        prefix = "/api/workspace-file/"
        rest = raw_path[len(prefix) :]
        slash = rest.find("/")
        if slash == -1:
            self.send_error(400, "Expected /api/workspace-file/{task}/{path}")
            return
        task = rest[:slash]
        file_rel = rest[slash + 1 :]
        ws_name = task_id_to_workspace(task)
        file_path = WORKSPACE_DIR / ws_name / file_rel
        if not file_path.exists():
            self.send_error(404, f"File not found: {file_rel}")
            return
        ext = file_path.suffix.lower()
        if ext in TEXT_EXTENSIONS:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))
            except UnicodeDecodeError:
                self.send_error(415, "Cannot read as text")
        elif ext in BINARY_MIME:
            with open(file_path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", BINARY_MIME[ext])
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(415, f"Unsupported file type: {ext}")

    def _serve_profile(self):
        """GET /api/profile/{profile_name} -- return profile YAML as JSON."""
        parts = self.path.split("/")
        if len(parts) < 4:
            self.send_error(400, "Expected /api/profile/{profile_name}")
            return
        profile_name = parts[3]
        # Find matching YAML file
        profile_path = None
        for f in PROFILES_DIR.glob("*.yaml"):
            if f.stem == profile_name:
                profile_path = f
                break
        if not profile_path:
            self.send_error(404, f"Profile not found: {profile_name}")
            return
        if not yaml:
            # Fallback: return raw text
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(profile_path.read_text(encoding="utf-8").encode("utf-8"))
            return
        with open(profile_path, encoding="utf-8") as f:
            profile = yaml.safe_load(f)
        # Extract L/M/R vector from first comment lines
        vector = {}
        with open(profile_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("# L/M/R Vector:"):
                    vec_str = line.split(":", 1)[1].strip()
                    for pair in vec_str.split():
                        if ":" in pair:
                            dim, val = pair.split(":", 1)
                            vector[dim] = val
                    break
        profile["_vector"] = vector
        self._send_json(profile)

    def _serve_media_file(self):
        raw_path = urllib.parse.unquote(self.path)
        parts = raw_path.split("/")
        if len(parts) < 6:
            self.send_error(400, "Expected /api/media/{profile}/{task}/{hash}")
            return
        profile = parts[3]
        task = parts[4]
        hash_id = parts[5]
        sig_dir = SIGNAL_DIR / f"{profile}_{task}"
        blob_path = sig_dir / "media" / "blobs" / f"{hash_id}.blob"
        diff_path = sig_dir / "media" / "diffs" / f"{hash_id}.diff"
        target = blob_path if blob_path.exists() else diff_path if diff_path.exists() else None
        if not target:
            self.send_error(404, f"Media not found: {hash_id}")
            return
        try:
            with open(target, encoding="utf-8") as f:
                content = f.read()
            ct = "text/x-diff" if target.suffix == ".diff" else "text/plain"
            self.send_response(200)
            self.send_header("Content-Type", f"{ct}; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
        except UnicodeDecodeError:
            self.send_error(415, "Binary media content")

    def _resolve_cache_dir(self) -> Path:
        """Resolve cache dir from ?source= query param, falling back to default."""
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        source = qs.get("source", [None])[0]
        if source:
            p = Path(source)
            if p.is_absolute() and p.exists():
                return p
            candidate = BENCH_DIR / f"ingest_cache_{source}"
            if candidate.exists():
                return candidate
        return INGEST_CACHE_DIR

    def _serve_memory_store_sources(self):
        """GET /api/memory-store/sources -- list all available ingest_cache_* dirs."""
        sources = []
        for d in sorted(BENCH_DIR.glob("ingest_cache_*")):
            if d.is_dir():
                slug = d.name.replace("ingest_cache_", "")
                profile_count = sum(1 for p in d.iterdir() if p.is_dir())
                sources.append({"slug": slug, "path": str(d), "profiles": profile_count})
        # Also include bare ingest_cache/ if it exists
        bare = BENCH_DIR / "ingest_cache"
        if bare.exists() and bare.is_dir():
            profile_count = sum(1 for p in bare.iterdir() if p.is_dir())
            sources.insert(0, {"slug": "default", "path": str(bare), "profiles": profile_count})
        self._send_json(sources)

    def _serve_memory_store_index(self):
        """GET /api/memory-store/index?source=slug -- list cached profiles and methods."""
        cache_dir = self._resolve_cache_dir()
        index = {}
        if cache_dir.exists():
            for pdir in sorted(cache_dir.iterdir()):
                if not pdir.is_dir():
                    continue
                methods = {}
                for pkl in sorted(pdir.glob("*.pkl")):
                    try:
                        with open(pkl, "rb") as f:
                            state = pickle.load(f)
                        summary = {}
                        for k, v in state.items():
                            if isinstance(v, list):
                                summary[k] = len(v)
                            elif isinstance(v, bool):
                                summary[k] = v
                            else:
                                summary[k] = type(v).__name__
                        methods[pkl.stem] = {
                            "size_kb": round(pkl.stat().st_size / 1024, 1),
                            "fields": summary,
                        }
                    except Exception as e:
                        methods[pkl.stem] = {"error": str(e)}
                index[pdir.name] = methods
        self._send_json(index)

    def _serve_memory_store_detail(self):
        """GET /api/memory-store/{profile}/{method}?source=slug -- return full memory contents."""
        parsed = urllib.parse.urlparse(self.path)
        parts = parsed.path.split("/")
        if len(parts) < 5:
            self.send_error(400, "Expected /api/memory-store/{profile}/{method}")
            return
        profile = parts[3]
        method = parts[4]
        cache_dir = self._resolve_cache_dir()
        pkl_path = cache_dir / profile / f"{method}.pkl"
        if not pkl_path.exists():
            self.send_error(404, f"Cache not found: {profile}/{method}")
            return
        try:
            with open(pkl_path, "rb") as f:
                state = pickle.load(f)
        except Exception as e:
            self.send_error(500, f"Failed to load: {e}")
            return

        # Convert to JSON-serializable format
        # ?full=1 disables all truncation
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        full_mode = qs.get("full", ["0"])[0] == "1"
        MAX_STR = 0 if full_mode else 500  # 0 = no truncation
        MAX_ITEMS = 0 if full_mode else 200
        MAX_DEPTH = 0 if full_mode else 5  # 0 = unlimited

        def sanitize(obj, depth=0):
            if MAX_DEPTH and depth > MAX_DEPTH:
                return str(obj)[:MAX_STR] if MAX_STR else str(obj)
            if isinstance(obj, dict):
                return {k: sanitize(v, depth + 1) for k, v in obj.items()}
            if isinstance(obj, list):
                limit = len(obj) if not MAX_ITEMS else MAX_ITEMS
                items = [sanitize(x, depth + 1) for x in obj[:limit]]
                if MAX_ITEMS and len(obj) > MAX_ITEMS:
                    items.append(f"... ({len(obj) - MAX_ITEMS} more)")
                return items
            if np and isinstance(obj, np.ndarray):
                return obj.tolist() if full_mode else f"ndarray shape={obj.shape} dtype={obj.dtype}"
            if isinstance(obj, (int, float, bool)):
                return obj
            if obj is None:
                return None
            s = str(obj)
            if MAX_STR and len(s) > MAX_STR:
                return s[:MAX_STR] + "..."
            return s

        # Check for FileGramOS MemoryStore — render with channel structure
        memory_store = state.get("store")
        if memory_store and type(memory_store).__name__ == "MemoryStore":
            rendered = self._render_memory_store(memory_store, sanitize)
            self._send_json(
                {
                    "profile": profile,
                    "method": method,
                    "full_mode": full_mode,
                    "data": rendered["data"],
                    "per_trajectory_keys": [],
                    "task_ids": [e.task_id for e in memory_store.engrams],
                    "size_kb": round(pkl_path.stat().st_size / 1024, 1),
                    "memory_store": rendered,
                }
            )
            return

        result = {}
        task_ids_order = []  # track trajectory order for per-traj methods

        for key, val in state.items():
            if isinstance(val, list):
                limit = len(val) if not MAX_ITEMS else MAX_ITEMS
                items = []
                for item in val[:limit]:
                    items.append(sanitize(item))
                    # Extract task_id if present
                    if isinstance(item, dict):
                        tid = item.get("task_id")
                        if not tid and isinstance(item.get("metadata"), dict):
                            tid = item["metadata"].get("task_id")
                        if tid and tid not in task_ids_order:
                            task_ids_order.append(tid)
                if MAX_ITEMS and len(val) > MAX_ITEMS:
                    items.append(f"... ({len(val) - MAX_ITEMS} more)")
                result[key] = {"type": "list", "count": len(val), "items": items}
            elif isinstance(val, bool):
                result[key] = {"type": "bool", "value": val}
            elif hasattr(val, "shape"):  # numpy
                if full_mode:
                    result[key] = {
                        "type": "ndarray",
                        "shape": list(val.shape),
                        "dtype": str(val.dtype),
                        "data": val.tolist(),
                    }
                else:
                    result[key] = {"type": "ndarray", "shape": list(val.shape), "dtype": str(val.dtype)}
            else:
                result[key] = {"type": type(val).__name__, "value": sanitize(val)}

        # Determine if this is per-trajectory (20 items = 20 tasks)
        per_traj_keys = [k for k, v in result.items() if v.get("type") == "list" and v.get("count") == 20]

        self._send_json(
            {
                "profile": profile,
                "method": method,
                "full_mode": full_mode,
                "data": result,
                "per_trajectory_keys": per_traj_keys,
                "task_ids": task_ids_order,
                "size_kb": round(pkl_path.stat().st_size / 1024, 1),
            }
        )

    def _render_memory_store(self, store, sanitize):
        """Convert a FileGramOS MemoryStore to structured channel data for the dashboard."""
        # Generate the clean rendered profile text
        rendered_text = ""
        try:
            from filegramos.retriever import QueryAdaptiveRetriever

            retriever = QueryAdaptiveRetriever()
            rendered_text = retriever.retrieve(store, "profile")
        except Exception:
            rendered_text = "(retriever unavailable)"

        # Channel 1: Procedural
        ch1 = {
            "dimension_classifications": store.dimension_classifications,
            "behavioral_patterns": store.behavioral_patterns,
            "aggregate": sanitize(store.procedural_aggregate),
        }

        # Channel 2: Semantic — full data
        ch2_filenames = store.all_filenames
        ch2_dirs = store.dir_structure_union
        ch2_samples = []
        for s in store.representative_samples:
            ch2_samples.append(
                {
                    "path": s.path,
                    "size": s.content_length,
                    "type": s.file_type,
                    "sample_type": s.sample_type,
                    "trajectory_id": s.trajectory_id,
                    "importance": s.importance,
                    "preview": s.content_preview or "",
                }
            )

        ch2 = {
            "filenames": ch2_filenames,
            "total_files": len(store.all_filenames),
            "directories": ch2_dirs,
            "content_samples": ch2_samples,
            "llm_narratives": sanitize(store.llm_narratives),
        }

        # Channel 3: Episodic — full data
        ch3 = {
            "centroid": store.centroid,
            "centroid_dims": len(store.centroid),
            "per_session_distances": store.per_session_distances,
            "deviation_flags": store.deviation_flags,
            "deviation_details": sanitize(store.deviation_details),
            "consistency_flags": sanitize(store.consistency_flags),
            "absence_flags": store.absence_flags,
        }

        # Per-engram full data
        engrams_data = []
        for eng in store.engrams:
            is_dev = store.deviation_flags.get(eng.trajectory_id, False)
            distance = store.per_session_distances.get(eng.trajectory_id, 0.0)

            # Semantic unit
            created_files = []
            for cf in eng.semantic.created_files:
                created_files.append(
                    {
                        "path": cf.path,
                        "content_length": cf.content_length,
                        "sample_type": cf.sample_type,
                        "file_type": cf.file_type,
                        "preview": (cf.content_preview or "")[:500],
                    }
                )
            edit_chains = []
            for ec in eng.semantic.edit_chains:
                edit_chains.append(
                    {
                        "path": ec.path,
                        "lines_added": ec.lines_added,
                        "lines_deleted": ec.lines_deleted,
                        "diff_preview": (ec.diff_preview or "")[:400],
                    }
                )
            cross_refs = []
            for cr in eng.semantic.cross_file_refs:
                cross_refs.append(
                    {
                        "source_file": cr.source_file,
                        "target_file": cr.target_file,
                        "reference_type": cr.reference_type,
                    }
                )

            engrams_data.append(
                {
                    "trajectory_id": eng.trajectory_id,
                    "task_id": eng.task_id,
                    "event_count": eng.event_count,
                    "behavioral_event_count": eng.behavioral_event_count,
                    "importance_score": eng.importance_score,
                    "is_perturbed": eng.is_perturbed,
                    "is_deviant": is_dev,
                    "distance_from_centroid": round(distance, 4),
                    "fingerprint": eng.fingerprint,
                    "procedural": sanitize(eng.procedural),
                    "auxiliary": sanitize(eng.auxiliary),
                    "semantic": {
                        "created_files": created_files,
                        "edit_chains": edit_chains,
                        "cross_file_refs": cross_refs,
                        "created_filenames": eng.semantic.created_filenames,
                        "dir_structure_diff": eng.semantic.dir_structure_diff,
                        "llm_encoding": sanitize(eng.semantic.llm_encoding) if eng.semantic.llm_encoding else None,
                    },
                }
            )

        # Legacy session summary (kept for backward compat)
        sessions = []
        for eng in store.engrams:
            rs = eng.procedural.get("reading_strategy", {})
            od = eng.procedural.get("output_detail", {})
            ds = eng.procedural.get("directory_style", {})
            es = eng.procedural.get("edit_strategy", {})
            is_dev = store.deviation_flags.get(eng.trajectory_id, False)
            fnames = [fn.rsplit("/", 1)[-1] if "/" in fn else fn for fn in eng.semantic.created_filenames[:3]]
            sessions.append(
                {
                    "task_id": eng.task_id,
                    "is_deviant": is_dev,
                    "reads": rs.get("total_reads", 0),
                    "search_pct": round(rs.get("search_ratio", 0) * 100),
                    "output_chars": od.get("avg_output_length", 0),
                    "files_created": od.get("files_created", 0),
                    "dirs_created": ds.get("dirs_created", 0),
                    "max_depth": ds.get("max_dir_depth", 0),
                    "edits": es.get("total_edits", 0),
                    "filenames": fnames,
                }
            )

        return {
            "rendered_profile": rendered_text,
            "n_sessions": len(store.engrams),
            "n_deviant": sum(1 for v in store.deviation_flags.values() if v),
            "channel_1_procedural": ch1,
            "channel_2_semantic": ch2,
            "channel_3_episodic": ch3,
            "engrams": engrams_data,
            "sessions": sessions,
            "data": {},  # empty — channels replace generic data view
        }

    def _serve_pipeline_data(self):
        """GET /api/pipeline/{profile}?source=slug -- walk through the full FileGramOS pipeline with real data."""
        parsed = urllib.parse.urlparse(self.path)
        parts = parsed.path.split("/")
        if len(parts) < 4:
            self.send_error(400, "Expected /api/pipeline/{profile}")
            return
        profile = parts[3]
        cache_dir = self._resolve_cache_dir()

        # Load the MemoryStore from filegramos pickle
        pkl_path = cache_dir / profile / "filegramos.pkl"
        if not pkl_path.exists():
            # Try filegramos_simple
            pkl_path = cache_dir / profile / "filegramos_simple.pkl"
        if not pkl_path.exists():
            self.send_error(404, f"No FileGramOS cache found for {profile}")
            return

        try:
            with open(pkl_path, "rb") as f:
                state = pickle.load(f)
        except Exception as e:
            self.send_error(500, f"Failed to load: {e}")
            return

        store = state.get("store")
        if not store or type(store).__name__ != "MemoryStore":
            self.send_error(404, f"No MemoryStore in {pkl_path.name}")
            return

        # Pick one engram as the example trajectory
        example_engram = store.engrams[0] if store.engrams else None
        example_task_id = example_engram.task_id if example_engram else None

        # Step 1: Raw events sample
        raw_events_sample = []
        if example_task_id and SIGNAL_DIR.exists():
            # Find the signal dir for this profile + task
            sig_name = f"{profile}_{example_task_id}"
            sig_dir = SIGNAL_DIR / sig_name
            if sig_dir.exists():
                events_path = sig_dir / "events.json"
                if events_path.exists():
                    try:
                        with open(events_path, encoding="utf-8") as f:
                            raw = json.load(f)
                        # Take first 8 behavioral events as sample
                        count = 0
                        for evt in raw:
                            if evt.get("event_type") in BEHAVIORAL_EVENTS:
                                # Strip heavy fields
                                sample = {
                                    k: v
                                    for k, v in evt.items()
                                    if k not in STRIP_FIELDS and k not in ("_resolved_content", "_resolved_diff")
                                }
                                raw_events_sample.append(sample)
                                count += 1
                                if count >= 8:
                                    break
                    except Exception:
                        pass

        # Step 2: NormalizedEvent sample (show what normalization produces)
        normalized_sample = []
        if raw_events_sample:
            try:
                from filegramos.normalizer import EventNormalizer

                normalizer = EventNormalizer()
                # Build minimal raw events for normalization
                normalized = normalizer.normalize_all(raw_events_sample)
                for ne in normalized[:8]:
                    # Convert NormalizedEvent to dict for JSON
                    normalized_sample.append(
                        {
                            "event_type": ne.event_type.value,
                            "file_path": ne.file_path,
                            "file_type": ne.file_type,
                            "content_length": ne.content_length,
                            "view_count": ne.view_count,
                            "operation": ne.operation,
                            "lines_added": ne.lines_added,
                            "lines_deleted": ne.lines_deleted,
                            "depth": ne.depth,
                            "source_file": ne.source_file,
                            "target_file": ne.target_file,
                            "reference_type": ne.reference_type,
                            "is_backup": ne.is_backup,
                        }
                    )
            except Exception:
                pass

        # Step 3: FeatureExtractor output (from example engram)
        feature_extraction = {}
        if example_engram:
            feature_extraction = example_engram.procedural

        # Step 4: Semantic channel (from example engram)
        semantic_data = {}
        if example_engram:
            sem = example_engram.semantic
            semantic_data = {
                "created_files": [
                    {
                        "path": cf.path,
                        "content_length": cf.content_length,
                        "file_type": cf.file_type,
                        "preview": (cf.content_preview or "")[:300],
                    }
                    for cf in sem.created_files[:6]
                ],
                "edit_chains": [
                    {
                        "path": ec.path,
                        "lines_added": ec.lines_added,
                        "lines_deleted": ec.lines_deleted,
                        "diff_preview": (ec.diff_preview or "")[:200],
                    }
                    for ec in sem.edit_chains[:4]
                ],
                "cross_file_refs": [
                    {
                        "source_file": cr.source_file,
                        "target_file": cr.target_file,
                        "reference_type": cr.reference_type,
                    }
                    for cr in sem.cross_file_refs[:6]
                ],
                "created_filenames": sem.created_filenames,
                "dir_structure_diff": sem.dir_structure_diff,
                "llm_encoding": sem.llm_encoding,
            }

        # Step 5: Complete Engram structure
        engram_overview = {}
        if example_engram:
            engram_overview = {
                "trajectory_id": example_engram.trajectory_id,
                "task_id": example_engram.task_id,
                "event_count": example_engram.event_count,
                "behavioral_event_count": example_engram.behavioral_event_count,
                "importance_score": example_engram.importance_score,
                "is_perturbed": example_engram.is_perturbed,
                "fingerprint": example_engram.fingerprint,
                "fingerprint_dims": len(example_engram.fingerprint),
                "procedural_keys": list(example_engram.procedural.keys()),
                "semantic_files_count": len(example_engram.semantic.created_files),
                "semantic_edits_count": len(example_engram.semantic.edit_chains),
                "semantic_crossrefs_count": len(example_engram.semantic.cross_file_refs),
            }

        # Step 6: Consolidated MemoryStore — 3 channels overview
        consolidation = {
            "n_engrams": len(store.engrams),
            "channel_1_procedural": {
                "aggregate_dimensions": list(store.procedural_aggregate.keys()),
                "n_classifications": len(store.dimension_classifications),
                "classifications": store.dimension_classifications,
                "n_patterns": len(store.behavioral_patterns),
                "patterns": store.behavioral_patterns,
            },
            "channel_2_semantic": {
                "n_representative_samples": len(store.representative_samples),
                "n_filenames": len(store.all_filenames),
                "n_directories": len(store.dir_structure_union),
                "n_llm_narratives": len(store.llm_narratives),
                "filenames_sample": store.all_filenames[:20],
                "directories": store.dir_structure_union,
            },
            "channel_3_episodic": {
                "centroid_dims": len(store.centroid),
                "n_deviant": sum(1 for v in store.deviation_flags.values() if v),
                "deviation_flags": store.deviation_flags,
                "per_session_distances": store.per_session_distances,
                "n_absence_flags": len(store.absence_flags),
                "absence_flags": store.absence_flags,
                "consistency_keys": list(store.consistency_flags.keys())
                if isinstance(store.consistency_flags, dict)
                else [],
            },
        }

        # Step 7: Rendered profile text
        rendered_text = ""
        try:
            from filegramos.retriever import QueryAdaptiveRetriever

            retriever = QueryAdaptiveRetriever()
            rendered_text = retriever.retrieve(store, "profile")
        except Exception:
            rendered_text = "(retriever unavailable)"

        # All engrams summary for selection
        all_engrams = [
            {
                "trajectory_id": e.trajectory_id,
                "task_id": e.task_id,
                "event_count": e.event_count,
                "behavioral_event_count": e.behavioral_event_count,
                "importance_score": e.importance_score,
                "is_perturbed": e.is_perturbed,
                "is_deviant": store.deviation_flags.get(e.trajectory_id, False),
            }
            for e in store.engrams
        ]

        self._send_json(
            {
                "profile": profile,
                "method": pkl_path.stem,
                "example_task_id": example_task_id,
                "steps": {
                    "step1_raw_events": raw_events_sample,
                    "step2_normalized": normalized_sample,
                    "step3_features": feature_extraction,
                    "step4_semantic": semantic_data,
                    "step5_engram": engram_overview,
                    "step6_consolidation": consolidation,
                    "step7_rendered": rendered_text,
                },
                "all_engrams": all_engrams,
            }
        )

    # ── Annotation endpoints ───────────────────────────────────────
    def _serve_annotate_init(self):
        """GET /api/annotate/init?annotator={id} — profile list (shuffled) + dimension defs + saved progress."""
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        annotator = qs.get("annotator", [""])[0].strip()
        if not annotator:
            self.send_error(400, "annotator parameter required")
            return

        # Load profile matrix
        matrix_path = PROFILES_DIR / "profile_matrix.json"
        if not matrix_path.exists():
            self.send_error(404, "profile_matrix.json not found")
            return
        with open(matrix_path) as f:
            matrix = json.load(f)

        # Build profile list (blind mode: shuffled by annotator ID)
        profiles = list(matrix["profiles"].keys())
        seed = int(hashlib.sha256(f"annotator_{annotator}".encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        rng.shuffle(profiles)

        blind_profiles = []
        for i, pid in enumerate(profiles):
            blind_profiles.append(
                {
                    "index": i + 1,
                    "profile_id": pid,
                    "label": f"Profile #{i + 1}",
                }
            )

        # Load saved progress
        progress = {}
        ann_dir = ANNOTATIONS_DIR / annotator
        if ann_dir.exists():
            for f in ann_dir.glob("*.json"):
                try:
                    with open(f) as fh:
                        progress[f.stem] = json.load(fh)
                except Exception:
                    pass

        # Ground truth per profile (revealed client-side only after submit)
        gt_map = {}
        for pid, pdata in matrix.get("profiles", {}).items():
            gt_map[pid] = pdata.get("dimensions", {})

        self._send_json(
            {
                "annotator": annotator,
                "profiles": blind_profiles,
                "dimensions": DIMENSION_DEFS,
                "progress": progress,
                "total": len(profiles),
                "completed": len(progress),
                "ground_truth": gt_map,
            }
        )

    def _serve_annotate_traces(self):
        """GET /api/annotate/traces/{profile_id} — 5 traces with filtered events + stats + hints."""
        parsed = urllib.parse.urlparse(self.path)
        parts = parsed.path.split("/")
        if len(parts) < 5:
            self.send_error(400, "Expected /api/annotate/traces/{profile_id}")
            return
        profile_id = parts[4]

        # Select 5 traces
        trace_picks = select_annotation_traces(profile_id, TASKS_DIR)
        if not trace_picks:
            self._send_json({"profile_id": profile_id, "traces": [], "error": "No trajectories found"})
            return

        traces = []
        for pick in trace_picks:
            task_id = pick["task_id"]
            sig_dir = SIGNAL_DIR / f"{profile_id}_{task_id}"
            sandbox_prefix = str(PROJECT_ROOT / "sandbox" / f"{profile_id}_{task_id}") + "/"

            events_path = sig_dir / "events.json"
            events = []
            if events_path.exists():
                with open(events_path, encoding="utf-8") as f:
                    raw = json.load(f)
                events = filter_events(raw, sandbox_prefix)

            stats = compute_trace_stats(events)
            hints = compute_dimension_hints(events, stats)

            # Get activated dimensions from task def
            task_def = pick["task_def"]
            activated_dims = task_def.get("dimensions", [])

            traces.append(
                {
                    "task_id": task_id,
                    "task_type": pick["task_type"],
                    "task_name": task_def.get("name_en", task_def.get("name", "")),
                    "task_description": task_def.get("description_en", task_def.get("description", ""))[:200],
                    "activated_dimensions": activated_dims,
                    "events": events,
                    "stats": stats,
                    "hints": hints,
                }
            )

        self._send_json(
            {
                "profile_id": profile_id,
                "traces": traces,
            }
        )

    def _handle_annotate_save(self):
        """POST /api/annotate/save — save annotation (6 dims L/M/R + confidence + notes)."""
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            self.send_error(400, "Empty body")
            return
        body = self.rfile.read(content_len)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        annotator = data.get("annotator", "").strip()
        profile_id = data.get("profile_id", "").strip()
        if not annotator or not profile_id:
            self.send_error(400, "annotator and profile_id required")
            return

        # Save to bench/annotations/{annotator}/{profile_id}.json
        ann_dir = ANNOTATIONS_DIR / annotator
        ann_dir.mkdir(parents=True, exist_ok=True)
        save_path = ann_dir / f"{profile_id}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self._send_json({"status": "ok", "saved": str(save_path)})

    def _serve_annotate_truth(self):
        """GET /api/annotate/truth/{profile_id} — return ground truth dims for one profile."""
        parts = urllib.parse.urlparse(self.path).path.split("/")
        profile_id = parts[4] if len(parts) >= 5 else ""
        matrix_path = PROFILES_DIR / "profile_matrix.json"
        if not matrix_path.exists():
            self._send_json({})
            return
        with open(matrix_path) as mf:
            matrix = json.load(mf)
        dims = matrix.get("profiles", {}).get(profile_id, {}).get("dimensions", {})
        self._send_json(dims)

    def _serve_annotate_results(self):
        """GET /api/annotate/results — accuracy + inter-annotator stats."""
        if not ANNOTATIONS_DIR.exists():
            self._send_json({"error": "No annotations yet", "annotators": []})
            return

        # Load ground truth
        matrix_path = PROFILES_DIR / "profile_matrix.json"
        ground_truth = {}
        if matrix_path.exists():
            with open(matrix_path) as f:
                matrix = json.load(f)
            for pid, pdata in matrix["profiles"].items():
                ground_truth[pid] = pdata.get("dimensions", {})

        # Load all annotations
        annotators_data = {}
        for ann_dir in sorted(ANNOTATIONS_DIR.iterdir()):
            if not ann_dir.is_dir():
                continue
            annotator = ann_dir.name
            annotators_data[annotator] = {}
            for f in sorted(ann_dir.glob("*.json")):
                try:
                    with open(f) as fh:
                        annotators_data[annotator][f.stem] = json.load(fh)
                except Exception:
                    pass

        # Compute per-annotator, per-dimension accuracy
        dim_keys = ["A", "B", "C", "D", "E", "F"]
        annotator_results = {}
        all_predictions = {d: [] for d in dim_keys}  # for inter-annotator

        for annotator, anns in annotators_data.items():
            correct = {d: 0 for d in dim_keys}
            total = {d: 0 for d in dim_keys}
            for pid, ann in anns.items():
                gt = ground_truth.get(pid, {})
                dims = ann.get("dimensions", {})
                for d in dim_keys:
                    pred = dims.get(d, {}).get("value")
                    truth = gt.get(d)
                    if pred and truth:
                        total[d] += 1
                        if pred == truth:
                            correct[d] += 1

            per_dim = {}
            for d in dim_keys:
                per_dim[d] = {
                    "correct": correct[d],
                    "total": total[d],
                    "accuracy": round(correct[d] / total[d], 3) if total[d] else 0,
                }
            overall_correct = sum(correct.values())
            overall_total = sum(total.values())
            annotator_results[annotator] = {
                "per_dimension": per_dim,
                "overall_accuracy": round(overall_correct / overall_total, 3) if overall_total else 0,
                "profiles_annotated": len(anns),
            }

        # Compute confusion matrices per dimension
        confusion = {}
        for d in dim_keys:
            mat = {"L": {"L": 0, "M": 0, "R": 0}, "M": {"L": 0, "M": 0, "R": 0}, "R": {"L": 0, "M": 0, "R": 0}}
            for annotator, anns in annotators_data.items():
                for pid, ann in anns.items():
                    gt = ground_truth.get(pid, {})
                    truth = gt.get(d)
                    pred = ann.get("dimensions", {}).get(d, {}).get("value")
                    if truth in mat and pred in mat[truth]:
                        mat[truth][pred] += 1
            confusion[d] = mat

        # Inter-annotator agreement (Cohen's kappa for pairs)
        kappa_pairs = []
        ann_list = list(annotators_data.keys())
        for i in range(len(ann_list)):
            for j in range(i + 1, len(ann_list)):
                a1, a2 = ann_list[i], ann_list[j]
                agree = 0
                total_compared = 0
                for pid in annotators_data[a1]:
                    if pid in annotators_data[a2]:
                        for d in dim_keys:
                            v1 = annotators_data[a1][pid].get("dimensions", {}).get(d, {}).get("value")
                            v2 = annotators_data[a2][pid].get("dimensions", {}).get(d, {}).get("value")
                            if v1 and v2:
                                total_compared += 1
                                if v1 == v2:
                                    agree += 1
                if total_compared > 0:
                    po = agree / total_compared
                    pe = 1.0 / 3.0  # 3 classes, assume uniform
                    kappa = (po - pe) / (1 - pe) if pe < 1 else 0
                    kappa_pairs.append(
                        {
                            "annotators": [a1, a2],
                            "agreement": round(po, 3),
                            "kappa": round(kappa, 3),
                            "n": total_compared,
                        }
                    )

        self._send_json(
            {
                "annotators": annotator_results,
                "confusion": confusion,
                "inter_annotator": kappa_pairs,
                "ground_truth_profiles": len(ground_truth),
            }
        )

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def serve_file(self, filepath, content_type):
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404, f"File not found: {filepath}")

    def log_message(self, format, *args):
        pass


def main():
    import argparse

    global RESULTS_DIR, INGEST_CACHE_DIR
    parser = argparse.ArgumentParser(description="FileGram Bench Dashboard")
    parser.add_argument(
        "--cache-dir", type=str, default=None, help="Ingest cache directory (slug like 'gemini_2.5_flash' or full path)"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None, help="Results directory (slug like 'gemini_2.5_flash' or full path)"
    )
    parser.add_argument("--port", type=int, default=PORT, help="Server port")
    args = parser.parse_args()
    if args.cache_dir:
        p = Path(args.cache_dir)
        INGEST_CACHE_DIR = p if (p.is_absolute() or p.exists()) else BENCH_DIR / f"ingest_cache_{args.cache_dir}"
    if args.results_dir:
        p = Path(args.results_dir)
        RESULTS_DIR = p if (p.is_absolute() or p.exists()) else BENCH_DIR / f"test_results_{args.results_dir}"

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    url = f"http://localhost:{args.port}"
    print("FileGram Bench Dashboard")
    print(f"  URL: {url}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Cache:   {INGEST_CACHE_DIR}")
    print("  Auto-refresh: 5s polling")
    print("  Press Ctrl+C to stop\n")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
        server.server_close()


if __name__ == "__main__":
    main()
