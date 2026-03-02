"""Centralized schema configuration for FileGramOS.

All attribute definitions, event types, file types, fingerprint features,
classification dimensions, thresholds, friendly names, and prompt generators
live here. Consumer modules import from this single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# ─── 1. Event Types ──────────────────────────────────────────────────────────

SIMULATION_TYPES: frozenset[str] = frozenset(
    {
        "tool_call",
        "llm_response",
        "compaction_triggered",
        "session_start",
        "session_end",
        "iteration_start",
        "iteration_end",
        "error_encounter",
        "error_response",
        "fs_snapshot",
    }
)

BEHAVIORAL_TYPES: frozenset[str] = frozenset(
    {
        "file_read",
        "file_write",
        "file_edit",
        "file_search",
        "file_browse",
        "file_rename",
        "file_move",
        "dir_create",
        "file_delete",
        "file_copy",
        "fs_snapshot",
        "context_switch",
        "cross_file_reference",
    }
)


class ConsumerEventType(str, Enum):
    """Consumer-side event type registry (14 values).

    Decoupled from producer ``EventType`` in ``filegram.behavior.events``.
    Only event types that the consumer (feature extraction / encoding)
    actually understands are listed here.  Unknown types are silently
    skipped during normalisation for forward-compatibility.
    """

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_EDIT = "file_edit"
    FILE_SEARCH = "file_search"
    FILE_BROWSE = "file_browse"
    FILE_RENAME = "file_rename"
    FILE_MOVE = "file_move"
    DIR_CREATE = "dir_create"
    FILE_DELETE = "file_delete"
    FILE_COPY = "file_copy"
    FS_SNAPSHOT = "fs_snapshot"
    CONTEXT_SWITCH = "context_switch"
    CROSS_FILE_REFERENCE = "cross_file_reference"
    ITERATION_END = "iteration_end"


# Lookup set for fast membership testing during normalisation
_CONSUMER_EVENT_VALUES: frozenset[str] = frozenset(e.value for e in ConsumerEventType)


@dataclass(frozen=True)
class NormalizedEvent:
    """Typed representation of every field that the consumer pipeline needs.

    Field sources are traced from every ``.get()`` call in
    ``feature_extraction.py`` and ``encoder.py``.
    """

    event_type: ConsumerEventType
    # File identity
    file_path: str = ""
    file_type: str = ""
    # file_read
    content_length: int = 0
    view_count: int = 1
    # file_write
    operation: str = ""  # "create" | "overwrite"
    # file_edit
    lines_added: int = 0
    lines_deleted: int = 0
    # dir_create
    depth: int = 0
    # fs_snapshot
    max_depth: int = 0
    directory_tree: Any = None
    # file_copy  (producer uses source_path / dest_path)
    is_backup: bool = False
    # cross_file_reference + file_copy (unified source/target)
    source_file: str = ""
    target_file: str = ""
    reference_type: str = ""
    # iteration_end
    tools_called: int = 0
    has_tool_error: bool = False
    # Resolved media content (injected by BaseAdapter._resolve_media_refs)
    resolved_content: str = ""  # was _resolved_content
    resolved_diff: str = ""  # was _resolved_diff


# ─── 2. File Types ───────────────────────────────────────────────────────────

STRUCTURED_FILE_TYPES: frozenset[str] = frozenset(
    {
        "csv",
        "json",
        "xlsx",
        "yaml",
        "xml",
        "tsv",
    }
)

PRIMARY_OUTPUT_TYPES: frozenset[str] = frozenset(
    {
        "md",
        "txt",
        "docx",
        "pdf",
    }
)

IMAGE_FILE_TYPES: frozenset[str] = frozenset(
    {
        "png",
        "jpg",
        "jpeg",
        "svg",
        "gif",
    }
)

TEXT_PARSEABLE_TYPES: frozenset[str] = frozenset(
    {
        "md",
        "txt",
        "csv",
        "json",
        "yaml",
        "yml",
        "xml",
        "tsv",
        "log",
        "ini",
        "toml",
        "rst",
    }
)

MULTIMODAL_TYPES: frozenset[str] = frozenset(
    {
        "png",
        "jpg",
        "jpeg",
        "gif",
        "svg",
        "pdf",
        "pptx",
        "docx",
        "xlsx",
    }
)

# ─── 3. Attribute Registry ───────────────────────────────────────────────────


@dataclass(frozen=True)
class AttributeDef:
    """Definition of a single profile attribute."""

    name: str  # e.g. "reading_strategy"
    display_name: str  # e.g. "Consumption Pattern"
    dimension: str  # e.g. "A"
    dimension_label: str  # e.g. "Consumption"
    is_primary: bool  # True = 9 primary attrs, False = 2 auxiliary
    extractor_method: str  # e.g. "extract_reading_strategy"


ATTRIBUTES: list[AttributeDef] = [
    # 9 primary attributes
    AttributeDef("reading_strategy", "Consumption Pattern", "A", "Consumption", True, "extract_reading_strategy"),
    AttributeDef("output_detail", "Production Volume", "B", "Production", True, "extract_output_detail"),
    AttributeDef("output_structure", "Output Structure", "B", "Production", True, "extract_output_structure"),
    AttributeDef("directory_style", "Organization Style", "C", "Organization", True, "extract_directory_style"),
    AttributeDef("naming", "Naming Convention", "C", "Organization", True, "extract_naming"),
    AttributeDef("edit_strategy", "Iteration Strategy", "D", "Iteration", True, "extract_edit_strategy"),
    AttributeDef("version_strategy", "Version Management", "D", "Iteration", True, "extract_version_strategy"),
    AttributeDef("tone", "Document Tone", "B", "Production", True, "extract_tone_features"),
    AttributeDef(
        "cross_modal_behavior", "Cross-Modal Behavior", "F", "Cross-Modal", True, "extract_cross_modal_behavior"
    ),
    # 2 auxiliary attributes
    AttributeDef("verbosity", "Verbosity", "B", "Production", False, "extract_verbosity"),
    AttributeDef("work_rhythm", "Curation", "E", "Curation", False, "extract_work_rhythm"),
]

# Derived lookup tables
PRIMARY_ATTRIBUTES: list[AttributeDef] = [a for a in ATTRIBUTES if a.is_primary]
AUXILIARY_ATTRIBUTES: list[AttributeDef] = [a for a in ATTRIBUTES if not a.is_primary]
ATTRIBUTE_BY_NAME: dict[str, AttributeDef] = {a.name: a for a in ATTRIBUTES}
PRIMARY_ATTRIBUTE_NAMES: list[str] = [a.name for a in PRIMARY_ATTRIBUTES]
AUXILIARY_ATTRIBUTE_NAMES: list[str] = [a.name for a in AUXILIARY_ATTRIBUTES]

# ─── 4. Fingerprint Features ─────────────────────────────────────────────────

# 17-feature vector covering dimensions A–F
FINGERPRINT_FEATURES: list[tuple[str, str]] = [
    # A: Consumption Pattern (3 features)
    ("reading_strategy", "search_ratio"),
    ("reading_strategy", "browse_ratio"),
    ("reading_strategy", "revisit_ratio"),
    # B: Production Style (3 features)
    ("output_detail", "avg_output_length"),
    ("output_detail", "files_created"),
    ("output_detail", "total_output_chars"),
    # C: Organization Preference (3 features)
    ("directory_style", "dirs_created"),
    ("directory_style", "max_dir_depth"),
    ("directory_style", "files_moved"),
    # D: Iteration Strategy (3 features)
    ("edit_strategy", "total_edits"),
    ("edit_strategy", "avg_lines_changed"),
    ("edit_strategy", "small_edit_ratio"),
    # E: Curation (2 features)
    ("version_strategy", "total_deletes"),
    ("version_strategy", "delete_to_create_ratio"),
    # F: Cross-Modal Behavior (3 features)
    ("cross_modal_behavior", "structured_files_created"),
    ("cross_modal_behavior", "markdown_table_rows"),
    ("cross_modal_behavior", "image_files_created"),
]

FEATURE_TO_DIMENSION: dict[str, str] = {
    "reading_strategy": "A (Consumption)",
    "output_detail": "B (Production)",
    "directory_style": "C (Organization)",
    "edit_strategy": "D (Iteration)",
    "cross_modal_behavior": "F (Cross-Modal)",
}

# ─── 5. Classification Dimensions ────────────────────────────────────────────


@dataclass(frozen=True)
class Category:
    """A single classification category within a dimension."""

    name: str
    description: str


@dataclass(frozen=True)
class DimensionDef:
    """Classification dimension with 3 L/M/R categories."""

    attribute: str
    display_name: str
    categories: tuple[Category, Category, Category]
    pattern_prefixes: tuple[str, ...]


DIMENSIONS: list[DimensionDef] = [
    DimensionDef(
        attribute="reading_strategy",
        display_name="How the user explores information",
        categories=(
            Category("sequential_deep", "thorough sequential reading with revisits, minimal search usage"),
            Category("targeted_search", "actively uses search/grep to locate information, reads selectively"),
            Category("breadth_first", "broadly scans directories, reads many files shallowly"),
        ),
        pattern_prefixes=("READING",),
    ),
    DimensionDef(
        attribute="output_detail",
        display_name="Content production volume and granularity",
        categories=(
            Category("detailed", "high average content length, multiple files per task"),
            Category("balanced", "moderate length and file count"),
            Category("concise", "short outputs, typically single file per task"),
        ),
        pattern_prefixes=("OUTPUT",),
    ),
    DimensionDef(
        attribute="directory_style",
        display_name="File system organization approach",
        categories=(
            Category("nested_by_topic", "creates multi-level directory hierarchies (2+ levels)"),
            Category("by_type", "creates some directories as needed (1-2 levels)"),
            Category("flat", "rarely or never creates directories"),
        ),
        pattern_prefixes=("ORGANIZATION",),
    ),
    DimensionDef(
        attribute="edit_strategy",
        display_name="How the user refines work after initial creation",
        categories=(
            Category("incremental_small", "many small edits, gradual refinement"),
            Category("balanced", "moderate editing frequency and scope"),
            Category("bulk_rewrite", "few or no edits, writes files in near-final form"),
        ),
        pattern_prefixes=("EDITING",),
    ),
    DimensionDef(
        attribute="version_strategy",
        display_name="How the user manages file versions",
        categories=(
            Category("keep_history", "creates backups before modifying, preserves old versions"),
            Category("balanced", "occasional version management"),
            Category("overwrite", "overwrites directly, no backups"),
        ),
        pattern_prefixes=("VERSIONING",),
    ),
    DimensionDef(
        attribute="working_style",
        display_name="Workspace curation and artifact management",
        categories=(
            Category("selective", "actively deletes unnecessary files, prunes intermediate outputs"),
            Category("pragmatic", "occasionally removes unneeded files, moderate cleanup"),
            Category("preservative", "rarely deletes anything, accumulates all files"),
        ),
        pattern_prefixes=("CURATION",),
    ),
    DimensionDef(
        attribute="thoroughness",
        display_name="Reading depth and coverage",
        categories=(
            Category("exhaustive", "reads files multiple times, high revisit rate"),
            Category("balanced", "moderate reading depth"),
            Category("minimal", "single-pass reading, rarely revisits"),
        ),
        pattern_prefixes=(),
    ),
    DimensionDef(
        attribute="error_handling",
        display_name="Caution level in file operations",
        categories=(
            Category("defensive", "creates backups, cautious about destructive operations"),
            Category("balanced", "standard caution"),
            Category("optimistic", "overwrites and deletes freely without backup"),
        ),
        pattern_prefixes=(),
    ),
    DimensionDef(
        attribute="cross_modal_behavior",
        display_name="Use of non-text structured content",
        categories=(
            Category("tables_and_references", "frequently uses tables, creates images or data files"),
            Category("minimal_tables", "occasional table use"),
            Category("text_only", "pure text output, no tables, images, or structured data"),
        ),
        pattern_prefixes=("CROSS-MODAL",),
    ),
]

DIMENSION_BY_ATTRIBUTE: dict[str, DimensionDef] = {d.attribute: d for d in DIMENSIONS}

# All valid pattern prefixes (for prompt generation)
ALL_PATTERN_PREFIXES: tuple[str, ...] = (
    "READING",
    "OUTPUT",
    "ORGANIZATION",
    "EDITING",
    "VERSIONING",
    "CROSS-MODAL",
    "STRUCTURE",
    "CURATION",
)

# ─── 6. Thresholds ───────────────────────────────────────────────────────────

# Fingerprint / deviation
DEVIATION_THRESHOLD: float = 1.5
TOP_K_SHIFTED_DIMS: int = 3

# Semantic / token budgets
SEMANTIC_TOKEN_BUDGET: int = 8000
MIN_TRAJECTORY_BUDGET: int = 200

# Edit classification
SMALL_EDIT_LINE_THRESHOLD: int = 10
MAJOR_EDIT_LINE_THRESHOLD: int = 10

# Importance scoring
MAX_DIVERSITY_ATTRIBUTES: int = 9

# ── Pattern detection thresholds ──

# Reading strategy
SEARCH_RATIO_NEVER: float = 0.05
SEARCH_RATIO_HEAVY: float = 0.2
REVISIT_RATIO_FREQUENT: float = 0.25
REVISIT_RATIO_RARE: float = 0.05
BROWSE_RATIO_FREQUENT: float = 0.3
READS_PER_FILE_DEEP: float = 1.5
READS_PER_FILE_SHALLOW: float = 1.1

# Output detail
AVG_OUTPUT_DETAILED: float = 3000
AVG_OUTPUT_CONCISE: float = 800
FILES_CREATED_MANY: float = 3
FILES_CREATED_FEW: float = 1.5

# Directory style
DIRS_CREATED_MULTIPLE: float = 3
DIRS_CREATED_RARE: float = 0.5
FILES_MOVED_ACTIVE: float = 1

# Edit strategy
TOTAL_EDITS_MANY: float = 3
SMALL_EDIT_RATIO_HIGH: float = 0.6
TOTAL_EDITS_RARE: float = 1
AVG_LINES_LARGE: float = 30

# Version strategy
BACKUP_RATIO_HIGH: float = 0.5
OVERWRITE_NO_BACKUP_THRESH: float = 0.5
BACKUP_RATIO_LOW: float = 0.1
DELETES_ACTIVE: float = 1

# Cross-modal
TABLES_RATIO_FREQUENT: float = 0.5
TABLE_ROWS_MANY: float = 5
IMAGES_RATIO_PRESENT: float = 0.3
STRUCTURED_FILES_PRESENT: float = 0.5
TABLES_RATIO_ABSENT: float = 0.2
IMAGES_RATIO_ABSENT: float = 0.1

# Tone / structure
HEADING_DEPTH_DEEP: float = 3
PROSE_RATIO_HEAVY: float = 3
PROSE_RATIO_LIGHT: float = 0.5

# ─── 7. Consistency Keys + CV Threshold ───────────────────────────────────────

CONSISTENCY_KEYS: dict[str, list[str]] = {
    "reading_strategy": ["total_reads", "revisit_ratio", "search_ratio"],
    "output_detail": ["total_output_chars", "avg_output_length", "files_created"],
    "edit_strategy": ["total_edits", "avg_lines_changed", "small_edit_ratio"],
    "directory_style": ["dirs_created", "max_dir_depth"],
    "tone": ["heading_count", "heading_max_depth", "prose_to_structure_ratio"],
}

CV_STABILITY_THRESHOLD: float = 0.3
CV_VARIABLE_DISPLAY_THRESHOLD: float = 0.5

# Language detection (from filenames)
LANGUAGE_RATIO_DOMINANT: float = 0.6
LANGUAGE_RATIO_MIXED: float = 0.2

# Naming convention detection
NAMING_PATTERN_THRESHOLD: float = 0.3
NAMING_UNDERSCORE_THRESHOLD: float = 0.7

# ─── 8. Friendly Names ───────────────────────────────────────────────────────

FRIENDLY_NAMES: dict[tuple[str, str], str] = {
    ("reading_strategy", "total_reads"): "read count",
    ("reading_strategy", "search_ratio"): "search usage",
    ("reading_strategy", "revisit_ratio"): "revisit rate",
    ("reading_strategy", "browse_ratio"): "browse rate",
    ("reading_strategy", "context_switch_rate"): "context switching",
    ("output_detail", "total_output_chars"): "output volume",
    ("output_detail", "avg_output_length"): "output length",
    ("output_detail", "files_created"): "file count",
    ("directory_style", "dirs_created"): "dir creation",
    ("directory_style", "max_dir_depth"): "dir depth",
    ("directory_style", "files_moved"): "file moves",
    ("edit_strategy", "total_edits"): "edit count",
    ("edit_strategy", "avg_lines_changed"): "edit size",
    ("tone", "heading_max_depth"): "heading depth",
    ("tone", "heading_count"): "heading count",
}

# ─── 9. Prompt Generator ─────────────────────────────────────────────────────


def build_classification_prompt(n_trajectories: int | str, stats_text: str) -> str:
    """Generate the LLM classification prompt from DIMENSIONS config.

    Replaces the hand-written 88-line CLASSIFICATION_PROMPT constant.
    The output is character-for-character equivalent to the original.
    """
    dim_lines: list[str] = []
    for d in DIMENSIONS:
        dim_lines.append(f"- **{d.attribute}** ({d.display_name}):")
        for cat in d.categories:
            dim_lines.append(f"  - {cat.name}: {cat.description}")

    prefixes = ", ".join(ALL_PATTERN_PREFIXES)

    return f"""\
You are analyzing aggregated behavioral statistics from {n_trajectories} \
file-system operation trajectories by a single user. These statistics \
capture how the user interacts with files across multiple tasks.

## Aggregated Statistics

{stats_text}

## Task

Based on these statistics, perform two analyses:

### 1. Dimension Classification

Classify the user's behavior on each dimension below. For each, provide \
the classification, confidence level (HIGH/MODERATE/LOW), and cite \
specific statistics as evidence.

Dimensions and categories:
{chr(10).join(dim_lines)}

### 2. Behavioral Patterns

Identify 5-12 specific, factual behavioral patterns grounded in the statistics. \
Each pattern should reference concrete numbers. Use category prefixes: \
{prefixes}.

## Output

Output a JSON object:
{{
  "classifications": [
    {{"attribute": "reading_strategy", "classification": "sequential_deep", "confidence": "HIGH", "evidence": "search_ratio=0.02, revisit_ratio=0.35"}},
    {{"attribute": "output_detail", "classification": "detailed", "confidence": "HIGH", "evidence": "avg_output_length=3500, files_created=4.2"}},
    ...
  ],
  "patterns": [
    "READING: Never uses search tools (search_ratio=0.02). Relies on sequential file access.",
    "OUTPUT: Produces very detailed output (avg 3500 chars/file). Creates supplementary materials.",
    ...
  ]
}}"""
