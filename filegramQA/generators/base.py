"""Base utilities for FileGramQA question generation.

Provides: data loading, behavioral stats extraction, description templates,
choice building, path anonymization.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from collections import defaultdict
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SIGNAL_DIR = PROJECT_ROOT / "signal"
PROFILES_DIR = PROJECT_ROOT / "profiles"
MATRIX_PATH = PROFILES_DIR / "profile_matrix.json"

# ---------------------------------------------------------------------------
# Dimension definitions
# ---------------------------------------------------------------------------
DIMENSIONS = ["A", "B", "C", "D", "E", "F"]

DIM_NAMES = {
    "A": "Consumption Pattern",
    "B": "Production Style",
    "C": "Organization Preference",
    "D": "Iteration Strategy",
    "E": "Curation",
    "F": "Cross-Modal Behavior",
}

TIER_LABELS = {
    "A": {"L": "sequential", "M": "targeted", "R": "breadth-first"},
    "B": {"L": "comprehensive", "M": "balanced", "R": "minimal"},
    "C": {"L": "deeply_nested", "M": "adaptive", "R": "flat"},
    "D": {"L": "incremental", "M": "balanced", "R": "rewrite"},
    "E": {"L": "selective", "M": "pragmatic", "R": "preservative"},
    "F": {"L": "visual-heavy", "M": "balanced", "R": "text-only"},
}

# ---------------------------------------------------------------------------
# Paper taxonomy: sub_type → (track, sub_track, channel)
# ---------------------------------------------------------------------------
QUESTION_TAXONOMY = {
    # Track 1: Understanding
    "dimension_classification":     ("T1", "Attribute Recognition",      "procedural"),
    "semantic_attribute":           ("T1", "Attribute Recognition",      "semantic"),
    "behavioral_fingerprint":       ("T1", "Behavioral Fingerprint",    "procedural"),
    # Track 2: Reasoning
    "counterfactual":               ("T2", "Behavioral Inference",      "procedural"),
    "multi_user_attribution":       ("T2", "Trace Disentanglement",     "procedural"),
    "single_user_task_separation":  ("T2", "Trace Disentanglement",     "procedural"),
    # Track 3: Detection
    "anomaly_detection":            ("T3", "Anomaly Detection",         "procedural"),
    "impersonation_detection":      ("T3", "Anomaly Detection",         "episodic"),
    "perturbation_detection":       ("T3", "Anomaly Detection",         "episodic"),
    "shifted_dimension_id":         ("T3", "Shift Analysis",            "episodic"),
    "shift_direction":              ("T3", "Shift Analysis",            "episodic"),
    "behavioral_change_description": ("T3", "Shift Analysis",            "episodic"),
    # Track 4: Multimodal
    "file_grounding":               ("T4", "File Grounding",            "procedural"),
    "visual_grounding":             ("T4", "Visual Grounding",          "procedural"),
}

def enrich_taxonomy(question: dict) -> dict:
    """Add taxonomy fields and reorder keys: question_id first, then taxonomy."""
    old_sub_type = question.pop("sub_type", "")
    question.pop("setting", None)  # no longer needed
    if old_sub_type in QUESTION_TAXONOMY:
        track, sub_track, channel = QUESTION_TAXONOMY[old_sub_type]
        # Rebuild dict with desired key order
        ordered = {"question_id": question.pop("question_id", "")}
        ordered["track"] = track
        ordered["sub_track"] = sub_track
        ordered["sub_class"] = old_sub_type
        ordered["channel"] = channel
        ordered.update(question)
        return ordered
    # Fallback: put sub_type back if not in taxonomy
    question["sub_type"] = old_sub_type
    return question


# Pre-defined profile pairs from profile_matrix.json validation
FINE_GRAINED_PAIRS = [
    ("p15_thorough_surveyor", "p5_balanced_organizer", ["B"], 1),
    ("p16_phased_minimalist", "p19_agile_pragmatist", ["E"], 1),
    ("p1_methodical", "p2_thorough_reviser", ["C", "D"], 2),
    ("p1_methodical", "p17_creative_archivist", ["E", "F"], 2),
    ("p8_minimal_editor", "p10_silent_auditor", ["A", "C", "D"], 3),
]

COARSE_PAIRS = [
    ("p1_methodical", "p3_efficient_executor", ["A", "B", "C", "D", "E"], 5),
    ("p17_creative_archivist", "p18_decisive_scanner", ["A", "B", "C", "D", "F"], 5),
]

# Dimension activation thresholds (per-trajectory)
DIM_ACTIVATION_THRESHOLDS = {
    "A": lambda s: s["reads"] >= 3,
    "B": lambda s: s["writes"] >= 1,
    "C": lambda s: s["org_ops"] >= 2,
    "D": lambda s: (s["edits"] + s["overwrites"]) >= 1,
    "E": lambda s: s["writes"] >= 1,  # curation signal exists whenever files are created (delete or not-delete is the signal)
    "F": lambda s: s["cross_refs"] >= 2,
}

# Task type mapping
TASK_TYPES = {
    "T-01": "understand", "T-02": "understand", "T-03": "create",
    "T-04": "create", "T-05": "organize", "T-06": "synthesize",
    "T-07": "synthesize", "T-08": "create", "T-09": "iterate",
    "T-10": "maintain", "T-11": "iterate", "T-12": "iterate",
    "T-13": "iterate", "T-14": "organize", "T-15": "synthesize",
    "T-16": "understand", "T-17": "understand", "T-18": "maintain",
    "T-19": "iterate", "T-20": "create",
    # Multimodal tasks (T-21 to T-32)
    "T-21": "organize", "T-22": "understand", "T-23": "organize",
    "T-24": "synthesize", "T-25": "understand", "T-26": "organize",
    "T-27": "create", "T-28": "synthesize", "T-29": "organize",
    "T-30": "iterate", "T-31": "create", "T-32": "maintain",
}

TASK_NAMES = {
    "T-01": "Investment Analyst Work Summary",
    "T-02": "Legal Case Materials & Timeline",
    "T-03": "Personal Knowledge Base Setup",
    "T-04": "Meeting Minutes & Follow-up Docs",
    "T-05": "Messy Folder Reorganization",
    "T-06": "Multi-source Research Report",
    "T-07": "Diary & Notes Personal Profile",
    "T-08": "Quarterly Work Summary",
    "T-09": "Report Revision & Condensation",
    "T-10": "Knowledge Base Content Update",
    "T-11": "Cross-file Error Detection & Fix",
    "T-12": "Document Format Standardization",
    "T-13": "Review Feedback Integration",
    "T-14": "Version Management & Archival",
    "T-15": "Conflicting Reports Reconciliation",
    "T-16": "Time-constrained Triage",
    "T-17": "File System Health Check",
    "T-18": "KB Three-round Incremental Update",
    "T-19": "Document Audience Rewrite",
    "T-20": "Weekly Report System Setup",
    # Multimodal tasks (T-21 to T-32)
    "T-21": "File System Cleanup & Deduplication",
    "T-22": "Film & Animation Collection Catalog",
    "T-23": "Travel Photo Organization & Album",
    "T-24": "Investment Earnings Call Analysis",
    "T-25": "Legal Case Multimedia Evidence Review",
    "T-26": "Personal Digital Asset Archiving",
    "T-27": "Student Portfolio Compilation",
    "T-28": "Pet Care Documentation Synthesis",
    "T-29": "Company Registration DB Construction",
    "T-30": "Voice Memo Organization & Archiving",
    "T-31": "Nature Scenery Video Collection Curation",
    "T-32": "Cross-modal Archive Consistency Check",
}


# ---------------------------------------------------------------------------
# ProfileMatrix — L/M/R assignments for all profiles
# ---------------------------------------------------------------------------
class ProfileMatrix:
    """Loaded profile matrix with L/M/R assignments and metadata."""

    def __init__(self, path: Path = MATRIX_PATH):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.profiles: dict[str, dict] = {}
        self.dim_groups: dict[str, dict[str, list[str]]] = {
            d: {"L": [], "M": [], "R": []} for d in DIMENSIONS
        }
        for pid, pdata in data["profiles"].items():
            dims = pdata["dimensions"]
            self.profiles[pid] = {
                "name": pdata.get("name", ""),
                "role": pdata.get("role", ""),
                "language": pdata.get("language", ""),
                "dimensions": dims,
            }
            for d in DIMENSIONS:
                tier = dims.get(d, "M")
                self.dim_groups[d][tier].append(pid)

    def get_tier(self, profile_id: str, dim: str) -> str:
        return self.profiles[profile_id]["dimensions"].get(dim, "M")

    def get_profiles_by_tier(self, dim: str, tier: str) -> list[str]:
        return list(self.dim_groups[dim][tier])

    def dim_distance(self, p1: str, p2: str) -> int:
        d1 = self.profiles[p1]["dimensions"]
        d2 = self.profiles[p2]["dimensions"]
        return sum(1 for d in DIMENSIONS if d1.get(d) != d2.get(d))

    def differing_dims(self, p1: str, p2: str) -> list[str]:
        d1 = self.profiles[p1]["dimensions"]
        d2 = self.profiles[p2]["dimensions"]
        return [d for d in DIMENSIONS if d1.get(d) != d2.get(d)]

    def shared_dims(self, p1: str, p2: str) -> list[str]:
        d1 = self.profiles[p1]["dimensions"]
        d2 = self.profiles[p2]["dimensions"]
        return [d for d in DIMENSIONS if d1.get(d) == d2.get(d)]


# ---------------------------------------------------------------------------
# ProfileYAMLLoader — load work_habits and personality from YAML profiles
# ---------------------------------------------------------------------------
class ProfileYAMLLoader:
    """Load full YAML profile data for semantic attribute questions."""

    def __init__(self, profiles_dir: Path = PROFILES_DIR):
        self.profiles: dict[str, dict[str, str]] = {}
        for ypath in sorted(profiles_dir.glob("p*.yaml")):
            pid = ypath.stem
            try:
                data = yaml.safe_load(ypath.read_text(encoding="utf-8"))
            except Exception:
                continue
            pers = data.get("personality", {})
            wh = data.get("work_habits", {})
            self.profiles[pid] = {
                "tone": pers.get("tone", ""),
                "verbosity": pers.get("verbosity", ""),
                "working_style": wh.get("working_style", ""),
                "thoroughness": wh.get("thoroughness", ""),
                "naming_convention": wh.get("naming_convention", ""),
                "version_strategy": wh.get("version_strategy", ""),
                "error_handling": wh.get("error_handling", ""),
                "organization_style": wh.get("organization_style", ""),
            }

    def get_attr(self, profile_id: str, attr: str) -> str:
        return self.profiles.get(profile_id, {}).get(attr, "")

    def all_values(self, attr: str) -> dict[str, str]:
        """Return {profile_id: value} for a given attribute."""
        return {
            pid: vals[attr]
            for pid, vals in self.profiles.items()
            if vals.get(attr)
        }


# ---------------------------------------------------------------------------
# TrajectoryStore — load and index all trajectory data
# ---------------------------------------------------------------------------
class TrajectoryStore:
    """Load and cache all completed trajectory data."""

    def __init__(self, signal_dir: Path = SIGNAL_DIR):
        self.signal_dir = signal_dir
        self.by_profile: dict[str, list[dict]] = defaultdict(list)
        self.by_task: dict[str, list[dict]] = defaultdict(list)
        self.by_key: dict[tuple[str, str], dict] = {}
        self._load_all()

    def _load_all(self):
        for d in sorted(self.signal_dir.iterdir()):
            if not d.is_dir() or d.name.startswith((".", "__")):
                continue
            m = re.match(r"^(.+)_(T-\d+)$", d.name)
            if not m:
                continue
            profile_id, task_id = m.group(1), m.group(2)
            events_file = d / "events.json"
            if not events_file.exists():
                continue

            summary_file = d / "summary.json"
            summary: dict = {}
            if summary_file.exists():
                try:
                    summary = json.loads(summary_file.read_text(encoding="utf-8"))
                    if not summary.get("completed", True):
                        continue
                except Exception:
                    continue

            events = json.loads(events_file.read_text(encoding="utf-8"))

            # Load perturbation metadata if present
            perturb_file = d / "perturbation.json"
            perturbation: dict | None = None
            if perturb_file.exists():
                try:
                    perturbation = json.loads(perturb_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            traj = {
                "profile_id": profile_id,
                "task_id": task_id,
                "path": str(d),
                "dir_name": d.name,
                "events": events,
                "summary": summary,
                "perturbation": perturbation,
            }
            self.by_profile[profile_id].append(traj)
            self.by_task[task_id].append(traj)
            self.by_key[(profile_id, task_id)] = traj

    def get_profile_trajectories(self, profile_id: str) -> list[dict]:
        return self.by_profile.get(profile_id, [])

    def get_task_trajectories(self, task_id: str) -> list[dict]:
        return self.by_task.get(task_id, [])

    def get_trajectory(self, profile_id: str, task_id: str) -> dict | None:
        return self.by_key.get((profile_id, task_id))

    def get_common_tasks(self, p1: str, p2: str) -> list[str]:
        t1 = {t["task_id"] for t in self.by_profile.get(p1, [])}
        t2 = {t["task_id"] for t in self.by_profile.get(p2, [])}
        return sorted(t1 & t2, key=lambda x: int(x.split("-")[1]))

    @property
    def all_profiles(self) -> list[str]:
        return sorted(self.by_profile.keys())

    @property
    def all_tasks(self) -> list[str]:
        return sorted(self.by_task.keys(), key=lambda x: int(x.split("-")[1]))

    def get_perturbed_trajectories(self, profile_id: str) -> list[dict]:
        """Return trajectories with applied perturbation for a profile."""
        return [
            t for t in self.by_profile.get(profile_id, [])
            if t.get("perturbation") and t["perturbation"].get("applied")
        ]

    def get_standard_trajectories(self, profile_id: str) -> list[dict]:
        """Return trajectories without perturbation for a profile."""
        return [
            t for t in self.by_profile.get(profile_id, [])
            if not (t.get("perturbation") and t["perturbation"].get("applied"))
        ]


# ---------------------------------------------------------------------------
# Behavioral stats extraction
# ---------------------------------------------------------------------------
def compute_dim_stats(events: list[dict]) -> dict[str, Any]:
    """Compute per-dimension behavioral statistics from a list of events."""
    reads = [e for e in events if e.get("event_type") == "file_read"]
    writes = [e for e in events if e.get("event_type") == "file_write"]
    edits = [e for e in events if e.get("event_type") == "file_edit"]
    searches = [e for e in events if e.get("event_type") == "file_search"]
    browses = [e for e in events if e.get("event_type") == "file_browse"]
    dir_creates = [e for e in events if e.get("event_type") == "dir_create"]
    moves = [e for e in events if e.get("event_type") == "file_move"]
    renames = [e for e in events if e.get("event_type") == "file_rename"]
    copies = [e for e in events if e.get("event_type") == "file_copy"]
    deletes = [e for e in events if e.get("event_type") == "file_delete"]
    ctx_switches = [e for e in events if e.get("event_type") == "context_switch"]
    cross_refs = [e for e in events if e.get("event_type") == "cross_file_reference"]

    total = len(events)
    first_write_idx = next(
        (i for i, e in enumerate(events) if e["event_type"] in ("file_write", "file_edit")),
        total,
    )
    content_lengths = [e.get("content_length", 0) for e in writes]
    lines_modified = [e.get("lines_modified", 0) for e in edits]
    depths = [e.get("depth", 0) for e in dir_creates]

    return {
        # A: Consumption
        "reads": len(reads),
        "searches": len(searches),
        "browses": len(browses),
        "revisits": sum(1 for e in reads if e.get("view_count", 1) > 1),
        "unique_files_read": len({e.get("file_path", "") for e in reads}),
        "search_ratio": len(searches) / max(len(reads), 1),
        # B: Production
        "writes": len(writes),
        "files_written": len({e.get("file_path", "") for e in writes}),
        "mean_content_length": sum(content_lengths) / max(len(content_lengths), 1),
        "total_content_length": sum(content_lengths),
        # C: Organization
        "dir_creates": len(dir_creates),
        "max_depth": max(depths) if depths else 0,
        "moves": len(moves),
        "renames": len(renames),
        "copies": len(copies),
        "deletes": len(deletes),
        "org_ops": len(dir_creates) + len(moves) + len(renames) + len(copies) + len(deletes),
        # D: Iteration
        "edits": len(edits),
        "mean_lines_modified": sum(lines_modified) / max(len(lines_modified), 1),
        "overwrites": sum(1 for e in writes if e.get("operation") == "overwrite"),
        # E: Curation
        "total_events": total,
        "delete_count": len(deletes),
        "delete_to_create_ratio": len(deletes) / max(len(writes) + len(dir_creates), 1),
        "copy_count": len(copies),
        # F: Cross-Modal
        "cross_refs": len(cross_refs),
        "media_writes": sum(
            1 for e in writes
            if any(ext in e.get("file_path", "") for ext in (".png", ".jpg", ".svg", ".csv", ".mermaid"))
        ),
    }


def aggregate_stats(trajectories: list[dict]) -> dict[str, Any]:
    """Aggregate dim stats across multiple trajectories.

    Returns per-task means + total sums with _total / _per_task suffixes.
    """
    if not trajectories:
        return {**compute_dim_stats([]), "n_tasks": 0}

    all_stats = [compute_dim_stats(t["events"]) for t in trajectories]
    n = len(all_stats)

    # Per-task mean for each key
    agg: dict[str, Any] = {}
    for key in all_stats[0]:
        values = [s[key] for s in all_stats]
        agg[key] = sum(values) / n

    # Add total sums and explicit per_task aliases
    sum_keys = [
        "reads", "searches", "browses", "writes", "edits", "dir_creates",
        "moves", "renames", "copies", "deletes", "cross_refs",
        "media_writes", "revisits", "overwrites", "org_ops", "delete_count",
        "copy_count",
    ]
    for key in sum_keys:
        agg[f"{key}_total"] = sum(s[key] for s in all_stats)
        agg[f"{key}_per_task"] = agg[key]

    agg["n_tasks"] = n
    return agg


def dim_is_active(events: list[dict], dim: str) -> bool:
    """Check if a dimension is activated in a single trajectory."""
    s = compute_dim_stats(events)
    return DIM_ACTIVATION_THRESHOLDS[dim](s)


def profile_dim_activation_rate(trajectories: list[dict], dim: str) -> float:
    """Fraction of trajectories that activate a given dimension."""
    if not trajectories:
        return 0.0
    active = sum(1 for t in trajectories if dim_is_active(t["events"], dim))
    return active / len(trajectories)


# ---------------------------------------------------------------------------
# Description templates — filled with real stats, no LLM needed
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Semantic behavioral descriptions — NO raw statistics, pure natural language
# ---------------------------------------------------------------------------
SEMANTIC_DIM_DESCS = {
    "A": {
        "L": (
            "Sequential deep reader: reads files one by one in logical order, "
            "going through each document completely before moving to the next. "
            "Rarely uses search tools, preferring to read all available content "
            "systematically. Frequently revisits files to cross-reference details."
        ),
        "M": (
            "Targeted searcher: searches for specific keywords or file types first, "
            "then reads only the matched files and relevant sections. Takes an efficient, "
            "focused approach — skipping files that don't match search criteria rather "
            "than reading everything."
        ),
        "R": (
            "Breadth-first scanner: quickly scans directory listings and file headers "
            "to get an overview of all available content. Browses many files briefly, "
            "reading only the first few lines of each. Prioritizes broad coverage over "
            "deep understanding of any single file."
        ),
    },
    "B": {
        "L": (
            "Comprehensive producer: creates detailed, well-structured output with "
            "multi-level headings (three or more levels), data tables, and appendices. "
            "Typically produces multiple files per task including auxiliary documents "
            "like README files, indexes, and reference lists."
        ),
        "M": (
            "Balanced producer: creates moderately detailed output with clear sections "
            "and two levels of headings. Occasionally includes supporting tables. "
            "Produces one to two files per task with a practical balance of "
            "thoroughness and conciseness."
        ),
        "R": (
            "Minimal producer: writes concise, focused output — typically a single "
            "file with flat structure and minimal formatting. Uses bullet points rather "
            "than deep heading hierarchies, keeping documents short and to the point "
            "without auxiliary materials."
        ),
    },
    "C": {
        "L": (
            "Deeply nested organizer: creates multi-level directory structures, "
            "systematically organizing files by category with long descriptive names "
            "and date prefixes. Moves files into appropriate subdirectories, creates "
            "backups before modifications, and never deletes files."
        ),
        "M": (
            "Adaptive organizer: creates directory structure as needed — adding one "
            "or two levels of organization when multiple similar files exist. Uses "
            "practical, mixed naming conventions and keeps old versions only when "
            "particularly important."
        ),
        "R": (
            "Flat organizer: keeps all files in the root directory without creating "
            "subdirectories. Uses short, abbreviated filenames. Overwrites existing "
            "files directly without creating backups, and occasionally deletes "
            "temporary files to keep the workspace clean."
        ),
    },
    "D": {
        "L": (
            "Incremental editor: makes many small, focused edits — changing only a "
            "few lines at a time and reviewing after each modification. Creates backups "
            "before making changes and prefers multiple refinement passes over "
            "large-scale rewrites."
        ),
        "M": (
            "Balanced editor: makes moderate-scope edits, balancing between targeted "
            "changes and broader rewrites. Occasionally reviews work between edits "
            "and maintains a practical approach to file modification."
        ),
        "R": (
            "Decisive rewriter: prefers to overwrite entire files rather than making "
            "incremental edits. When significant changes are needed, rewrites the "
            "whole document in one decisive pass, sometimes deleting and recreating "
            "files rather than editing them."
        ),
    },
    "E": {
        "L": (
            "Selective: actively curates the workspace by deleting temporary files, "
            "removing redundant content, and pruning intermediate outputs as they work. "
            "Maintains a lean file system, discarding files that are no longer needed "
            "after processing."
        ),
        "M": (
            "Pragmatic: occasionally removes clearly unnecessary files but "
            "does not actively maintain workspace hygiene. Deletes some temporary "
            "files opportunistically but keeps most intermediate outputs, balancing "
            "between accumulation and tidiness."
        ),
        "R": (
            "Preservative: rarely deletes any files, preferring to accumulate "
            "everything created during the work session. Preserves all intermediate "
            "outputs and temporary files, resulting in a workspace that grows "
            "steadily without pruning."
        ),
    },
    "F": {
        "L": (
            "Visual-heavy: actively creates visual materials like charts, diagrams, "
            "and figures to accompany text. Pairs data with visual representations, "
            "maintains a dedicated figures or assets directory, and references figures "
            "with numbered captions in written output."
        ),
        "M": (
            "Balanced cross-modal: uses markdown tables and structured formatting "
            "where data warrants it, but does not create standalone image or media "
            "files. Balances visual structure with text content through formatting "
            "rather than separate media."
        ),
        "R": (
            "Text-only: produces pure text output without any visual elements — "
            "no tables, charts, images, or diagrams. Uses prose paragraphs and "
            "bullet lists exclusively, describing data inline rather than creating "
            "visual representations."
        ),
    },
}

# Keep old stat-based templates for analytics / internal use only
_STAT_TEMPLATES = {
    "A": {
        "L": "reads_total={reads_total:.0f} searches={searches_total:.0f} revisits={revisits_total:.0f} search_ratio={search_ratio:.2f}",
        "M": "searches_total={searches_total:.0f} reads_per_task={reads_per_task:.1f} search_ratio={search_ratio:.2f}",
        "R": "browses_total={browses_total:.0f} reads_per_task={reads_per_task:.1f} search_ratio={search_ratio:.2f}",
    },
    "B": {
        "L": "writes_total={writes_total:.0f} mean_content_length={mean_content_length:.0f}",
        "M": "writes_total={writes_total:.0f} mean_content_length={mean_content_length:.0f}",
        "R": "writes_total={writes_total:.0f} mean_content_length={mean_content_length:.0f}",
    },
    "C": {
        "L": "dir_creates={dir_creates_total:.0f} max_depth={max_depth:.0f} moves={moves_total:.0f}",
        "M": "dir_creates={dir_creates_total:.0f} max_depth={max_depth:.0f}",
        "R": "dir_creates={dir_creates_total:.0f} deletes={deletes_total:.0f}",
    },
    "D": {
        "L": "edits={edits_total:.0f} mean_lines={mean_lines_modified:.1f}",
        "M": "edits={edits_total:.0f} mean_lines={mean_lines_modified:.1f}",
        "R": "overwrites={overwrites_total:.0f} edits={edits_total:.0f}",
    },
    "E": {
        "L": "deletes={delete_count_total:.0f} del_ratio={delete_to_create_ratio:.2f}",
        "M": "deletes={delete_count_total:.0f} del_ratio={delete_to_create_ratio:.2f}",
        "R": "deletes={delete_count_total:.0f} del_ratio={delete_to_create_ratio:.2f}",
    },
    "F": {
        "L": "cross_refs={cross_refs_total:.0f} media_writes={media_writes_total:.0f}",
        "M": "cross_refs={cross_refs_total:.0f}",
        "R": "cross_refs={cross_refs_total:.0f} media_writes={media_writes_total:.0f}",
    },
}


def render_description(dim: str, tier: str, stats: dict[str, Any] | None = None) -> str:
    """Return semantic behavioral description for a dimension/tier.

    The ``stats`` argument is accepted for backward compatibility but ignored.
    """
    return SEMANTIC_DIM_DESCS[dim][tier]


# ---------------------------------------------------------------------------
# Profile portraits — combine all 6 dimensions into a holistic description
# ---------------------------------------------------------------------------
_DIM_PORTRAIT_FRAGMENTS = {
    "A": {
        "L": "reads files thoroughly and sequentially from start to finish",
        "M": "searches for specific information before reading targeted sections",
        "R": "quickly scans many files for a broad overview without deep reading",
    },
    "B": {
        "L": "produces detailed, well-structured documents with multi-level headings and auxiliary files",
        "M": "creates moderately detailed output with clear sections",
        "R": "writes concise, minimal output in a single file",
    },
    "C": {
        "L": "organizes files into deep nested directory structures with systematic naming",
        "M": "creates directory structure as needed with practical naming",
        "R": "keeps all files flat in the root directory with short names",
    },
    "D": {
        "L": "makes many small incremental edits with frequent reviews",
        "M": "edits with moderate scope, balancing refinement and efficiency",
        "R": "rewrites entire files in one decisive pass rather than editing incrementally",
    },
    "E": {
        "L": "actively curates the workspace by deleting unnecessary files and pruning intermediate outputs",
        "M": "preserves nearly all files created during work, rarely deleting anything",
        "R": "occasionally removes clearly unneeded files but mostly lets outputs accumulate",
    },
    "F": {
        "L": "actively creates visual materials like charts and diagrams",
        "M": "uses tables and structured formatting where data warrants",
        "R": "produces pure text without any visual elements",
    },
}


def build_profile_portrait(dims: dict[str, str]) -> str:
    """Build a concise behavioral portrait from a profile's L/M/R assignments."""
    parts = [_DIM_PORTRAIT_FRAGMENTS[d][dims.get(d, "M")] for d in DIMENSIONS]
    return (
        f"This user {parts[0]}, and {parts[1]}. "
        f"They {parts[2]}, {parts[3]}, and {parts[4]}. "
        f"They {parts[5]}."
    )


def build_counterfactual_prediction(dims: dict[str, str], task_type: str) -> str:
    """Build a semantic prediction of how a profile would approach a task type."""
    a, b, c, d = dims.get("A", "M"), dims.get("B", "M"), dims.get("C", "M"), dims.get("D", "M")
    e, f = dims.get("E", "M"), dims.get("F", "M")

    reading = {
        "L": "read all input files sequentially and thoroughly",
        "M": "search for relevant files by keyword, then read targeted sections",
        "R": "quickly scan file headers and skim broadly",
    }[a]

    output = {
        "L": "a comprehensive, multi-section report with detailed headings and supplementary materials",
        "M": "a moderately detailed document with clear sections",
        "R": "a brief, concise summary in a single file",
    }[b]

    org = {
        "L": "creating a nested directory structure to organize all outputs",
        "M": "creating a simple folder structure if needed",
        "R": "keeping everything in the root directory",
    }[c]

    editing = {
        "L": "refining output through multiple small incremental edits",
        "M": "making moderate edits to polish the output",
        "R": "writing the output in one pass without iterative refinement",
    }[d]

    curation = {
        "L": "Afterwards, they would actively clean up temporary and intermediate files",
        "M": "They would leave most files in place, only removing obviously unnecessary ones",
        "R": "They would preserve all files created during the task without any cleanup",
    }[e]

    crossmodal = {
        "L": "and include visual materials like charts or diagrams alongside the text output",
        "M": "and use markdown tables where data warrants structured presentation",
        "R": "and produce pure text without any visual elements or tables",
    }[f]

    return (
        f"Would {reading}, then produce {output}, "
        f"{org}, and {editing}. "
        f"{curation}, {crossmodal}."
    )


# ---------------------------------------------------------------------------
# Choice builder
# ---------------------------------------------------------------------------
def build_choices(
    correct_text: str, distractors: list[str], rng: random.Random,
) -> tuple[dict[str, str], str]:
    """Shuffle correct + distractors into labeled A/B/C/D choices.

    Returns (choices_dict, correct_letter).
    """
    options = [correct_text] + list(distractors)
    rng.shuffle(options)

    labels = ["A", "B", "C", "D", "E"][:len(options)]
    choices: dict[str, str] = {}
    correct_letter = "A"
    for i, opt in enumerate(options):
        choices[labels[i]] = opt
        if opt == correct_text:
            correct_letter = labels[i]

    return choices, correct_letter


# ---------------------------------------------------------------------------
# Path anonymization
# ---------------------------------------------------------------------------
_STRIP_PATTERNS = [
    re.compile(r"/Users/[^/]+/Desktop/code/Synvo/FileGram/"),
    re.compile(r"sandbox/[^/]+/"),
    re.compile(r"pilot/sandbox/[^/]+/"),
]


def anonymize_path(path: str) -> str:
    """Remove profile/task identifiers from a file path."""
    for pat in _STRIP_PATTERNS:
        path = pat.sub("workspace/", path)
    return path


def anonymize_events(events: list[dict]) -> list[dict]:
    """Strip identifying fields and anonymize paths."""
    clean = []
    for e in events:
        ce = dict(e)
        for k in ("session_id", "profile_id", "message_id", "model_provider", "model_name"):
            ce.pop(k, None)
        for pk in ("file_path", "dir_path", "old_path", "new_path",
                    "source_path", "dest_path", "from_file", "to_file"):
            if pk in ce:
                ce[pk] = anonymize_path(ce[pk])
        clean.append(ce)
    return clean
