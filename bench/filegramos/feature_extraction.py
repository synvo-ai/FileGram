"""Step 1: Deterministic per-attribute feature extraction from trajectories.

For each profile attribute, defines what statistics to extract from events.json.
No LLM involved — pure computation on structured event data.

Supports both legacy ``list[dict]`` input and typed ``list[NormalizedEvent]``.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from pathlib import PurePosixPath
from typing import Any

from .schema import (
    IMAGE_FILE_TYPES,
    PRIMARY_OUTPUT_TYPES,
    SIMULATION_TYPES,
    STRUCTURED_FILE_TYPES,
    ConsumerEventType,
    NormalizedEvent,
)
from .tuning import PREVIEW_DIFF_CHARS, PREVIEW_FILE_CHARS

# Short alias used in typed helpers
CT = ConsumerEventType


class FeatureExtractor:
    """Extract structured features from a single trajectory's events."""

    def __init__(self, events: list[dict[str, Any]] | list[NormalizedEvent]):
        if events and isinstance(events[0], NormalizedEvent):
            # ── Typed path ──
            self._typed: list[NormalizedEvent] | None = events  # type: ignore[assignment]
            self._behavioral_typed: list[NormalizedEvent] = [
                e for e in self._typed if e.event_type.value not in SIMULATION_TYPES
            ]
            # Keep legacy fields as empty so callers that inspect them don't crash
            self.events: list[dict[str, Any]] = []
            self._behavioral: list[dict[str, Any]] = []
        else:
            # ── Legacy path (raw dicts) ──
            self._typed = None
            self._behavioral_typed = []
            self.events = events  # type: ignore[assignment]
            self._behavioral = [e for e in self.events if e.get("event_type") not in SIMULATION_TYPES]

    # ------------------------------------------------------------------
    # Public API (dispatch to typed or legacy)
    # ------------------------------------------------------------------

    def extract_all(self) -> dict[str, Any]:
        """Extract features for all 9 profile attributes."""
        return {
            "reading_strategy": self.extract_reading_strategy(),
            "output_detail": self.extract_output_detail(),
            "output_structure": self.extract_output_structure(),
            "directory_style": self.extract_directory_style(),
            "naming": self.extract_naming(),
            "edit_strategy": self.extract_edit_strategy(),
            "version_strategy": self.extract_version_strategy(),
            "tone": self.extract_tone_features(),
            "cross_modal_behavior": self.extract_cross_modal_behavior(),
        }

    def extract_auxiliary(self) -> dict[str, Any]:
        """Extract auxiliary features not in the 9-attribute profile matrix."""
        return {
            "verbosity": self.extract_verbosity(),
            "work_rhythm": self.extract_work_rhythm(),
        }

    # ==================================================================
    # Per-attribute extractors (each dispatches typed vs legacy)
    # ==================================================================

    def extract_reading_strategy(self) -> dict[str, Any]:
        if self._typed is not None:
            return self._extract_reading_strategy_typed()
        return self._extract_reading_strategy_legacy()

    def extract_output_detail(self) -> dict[str, Any]:
        if self._typed is not None:
            return self._extract_output_detail_typed()
        return self._extract_output_detail_legacy()

    def extract_output_structure(self) -> dict[str, Any]:
        if self._typed is not None:
            return self._extract_output_structure_typed()
        return self._extract_output_structure_legacy()

    def extract_directory_style(self) -> dict[str, Any]:
        if self._typed is not None:
            return self._extract_directory_style_typed()
        return self._extract_directory_style_legacy()

    def extract_naming(self) -> dict[str, Any]:
        if self._typed is not None:
            return self._extract_naming_typed()
        return self._extract_naming_legacy()

    def extract_edit_strategy(self) -> dict[str, Any]:
        if self._typed is not None:
            return self._extract_edit_strategy_typed()
        return self._extract_edit_strategy_legacy()

    def extract_version_strategy(self) -> dict[str, Any]:
        if self._typed is not None:
            return self._extract_version_strategy_typed()
        return self._extract_version_strategy_legacy()

    def extract_cross_modal_behavior(self) -> dict[str, Any]:
        if self._typed is not None:
            return self._extract_cross_modal_behavior_typed()
        return self._extract_cross_modal_behavior_legacy()

    def extract_tone_features(self) -> dict[str, Any]:
        if self._typed is not None:
            return self._extract_tone_features_typed()
        return self._extract_tone_features_legacy()

    def extract_verbosity(self) -> dict[str, Any]:
        if self._typed is not None:
            return self._extract_verbosity_typed()
        return self._extract_verbosity_legacy()

    def extract_work_rhythm(self) -> dict[str, Any]:
        if self._typed is not None:
            return self._extract_work_rhythm_typed()
        return self._extract_work_rhythm_legacy()

    def extract_semantic_channel(self) -> dict[str, Any]:
        if self._typed is not None:
            return self._extract_semantic_channel_typed()
        return self._extract_semantic_channel_legacy()

    # ==================================================================
    # ── TYPED implementations ──
    # ==================================================================

    def _extract_reading_strategy_typed(self) -> dict[str, Any]:
        b = self._behavioral_typed
        reads = [e for e in b if e.event_type == CT.FILE_READ]
        browses = [e for e in b if e.event_type == CT.FILE_BROWSE]
        searches = [e for e in b if e.event_type == CT.FILE_SEARCH]
        switches = [e for e in b if e.event_type == CT.CONTEXT_SWITCH]

        total_access = len(reads) + len(browses) + len(searches)
        browse_ratio = len(browses) / max(total_access, 1)
        search_ratio = len(searches) / max(total_access, 1)
        read_ratio = len(reads) / max(total_access, 1)

        revisits = [e for e in reads if e.view_count > 1]
        revisit_ratio = len(revisits) / max(len(reads), 1)

        content_lengths = [e.content_length for e in reads]
        avg_content_length = statistics.mean(content_lengths) if content_lengths else 0

        unique_files = len(set(e.file_path for e in reads))
        switch_rate = len(switches) / max(len(reads), 1)

        return {
            "total_reads": len(reads),
            "total_browses": len(browses),
            "total_searches": len(searches),
            "browse_ratio": round(browse_ratio, 3),
            "search_ratio": round(search_ratio, 3),
            "read_ratio": round(read_ratio, 3),
            "revisit_ratio": round(revisit_ratio, 3),
            "avg_content_length": round(avg_content_length, 1),
            "unique_files_read": unique_files,
            "context_switch_rate": round(switch_rate, 3),
        }

    def _extract_output_detail_typed(self) -> dict[str, Any]:
        writes = [e for e in self._behavioral_typed if e.event_type == CT.FILE_WRITE and e.operation == "create"]

        content_lengths = [e.content_length for e in writes]
        total_output_chars = sum(content_lengths)
        avg_output_length = statistics.mean(content_lengths) if content_lengths else 0
        max_output_length = max(content_lengths) if content_lengths else 0

        return {
            "files_created": len(writes),
            "total_output_chars": total_output_chars,
            "avg_output_length": round(avg_output_length, 1),
            "max_output_length": max_output_length,
            "output_length_std": round(statistics.stdev(content_lengths), 1) if len(content_lengths) > 1 else 0,
        }

    def _extract_output_structure_typed(self) -> dict[str, Any]:
        writes = [e for e in self._behavioral_typed if e.event_type == CT.FILE_WRITE and e.operation == "create"]

        file_types = Counter(e.file_type or "unknown" for e in writes)
        has_markdown = file_types.get("md", 0) > 0
        has_structured_file = any(ft in file_types for ft in ("json", "yaml", "csv", "xml"))

        markdown_table_rows = 0
        for e in writes:
            content = e.resolved_content
            if content:
                markdown_table_rows += len(re.findall(r"^\|.*\|", content, re.MULTILINE))

        has_markdown_tables = markdown_table_rows > 0

        return {
            "file_types_created": dict(file_types),
            "has_markdown": has_markdown,
            "has_structured_data": has_structured_file or has_markdown_tables,
            "has_markdown_tables": has_markdown_tables,
            "markdown_table_rows": markdown_table_rows,
            "distinct_file_types": len(file_types),
        }

    def _extract_directory_style_typed(self) -> dict[str, Any]:
        b = self._behavioral_typed
        dir_creates = [e for e in b if e.event_type == CT.DIR_CREATE]
        moves = [e for e in b if e.event_type == CT.FILE_MOVE]
        snapshots = [e for e in b if e.event_type == CT.FS_SNAPSHOT]

        depths = [e.depth for e in dir_creates]
        max_depth = max(depths) if depths else 0
        avg_depth = statistics.mean(depths) if depths else 0

        dirs_created = len(dir_creates)

        final_max_depth = 0
        if snapshots:
            final_max_depth = snapshots[-1].max_depth

        return {
            "dirs_created": dirs_created,
            "max_dir_depth": max_depth,
            "avg_dir_depth": round(avg_depth, 2),
            "final_fs_max_depth": final_max_depth,
            "files_moved": len(moves),
        }

    def _extract_naming_typed(self) -> dict[str, Any]:
        writes = [e for e in self._behavioral_typed if e.event_type == CT.FILE_WRITE and e.operation == "create"]
        renames = [e for e in self._behavioral_typed if e.event_type == CT.FILE_RENAME]

        filenames = []
        for e in writes:
            fp = e.file_path
            if fp:
                filenames.append(PurePosixPath(fp).stem)

        if not filenames:
            return {
                "avg_filename_length": 0,
                "has_date_prefix": False,
                "has_numeric_prefix": False,
                "has_underscores": False,
                "has_hyphens": False,
                "avg_word_count": 0,
                "total_renames": len(renames),
            }

        avg_len = statistics.mean(len(f) for f in filenames)

        date_pattern = re.compile(r"^\d{4}[-_]?\d{2}[-_]?\d{2}")
        has_date = any(date_pattern.match(f) for f in filenames)
        numeric_prefix_pattern = re.compile(r"^\d{1,3}[-_]")
        has_numeric_prefix = any(numeric_prefix_pattern.match(f) for f in filenames)
        has_underscores = any("_" in f for f in filenames)
        has_hyphens = any("-" in f for f in filenames)

        word_counts = []
        for f in filenames:
            words = re.split(r"[-_.\s]+", f)
            word_counts.append(len([w for w in words if w]))
        avg_words = statistics.mean(word_counts) if word_counts else 0

        return {
            "avg_filename_length": round(avg_len, 1),
            "has_date_prefix": has_date,
            "has_numeric_prefix": has_numeric_prefix,
            "has_underscores": has_underscores,
            "has_hyphens": has_hyphens,
            "avg_word_count": round(avg_words, 1),
            "total_renames": len(renames),
        }

    def _extract_edit_strategy_typed(self) -> dict[str, Any]:
        edits = [e for e in self._behavioral_typed if e.event_type == CT.FILE_EDIT]

        if not edits:
            return {
                "total_edits": 0,
                "avg_lines_changed": 0,
                "max_lines_changed": 0,
                "edits_per_file": 0,
                "small_edit_ratio": 0,
            }

        lines_per_edit = [e.lines_added + e.lines_deleted for e in edits]
        avg_lines = statistics.mean(lines_per_edit)
        max_lines = max(lines_per_edit)

        files_edited = Counter(e.file_path for e in edits)
        avg_edits_per_file = statistics.mean(files_edited.values())

        small_edits = sum(1 for lc in lines_per_edit if lc < 10)
        small_ratio = small_edits / len(edits)

        return {
            "total_edits": len(edits),
            "avg_lines_changed": round(avg_lines, 1),
            "max_lines_changed": max_lines,
            "edits_per_file": round(avg_edits_per_file, 2),
            "small_edit_ratio": round(small_ratio, 3),
        }

    def _extract_version_strategy_typed(self) -> dict[str, Any]:
        b = self._behavioral_typed
        copies = [e for e in b if e.event_type == CT.FILE_COPY]
        deletes = [e for e in b if e.event_type == CT.FILE_DELETE]
        overwrites = [e for e in b if e.event_type == CT.FILE_WRITE and e.operation == "overwrite"]

        backups = [e for e in copies if e.is_backup]

        creates = [e for e in b if e.event_type == CT.FILE_WRITE and e.operation == "create"]

        return {
            "total_copies": len(copies),
            "backup_copies": len(backups),
            "total_deletes": len(deletes),
            "total_overwrites": len(overwrites),
            "has_backup_behavior": len(backups) > 0,
            "delete_to_create_ratio": round(
                len(deletes) / max(len(creates), 1),
                3,
            ),
        }

    def _extract_cross_modal_behavior_typed(self) -> dict[str, Any]:
        writes = [e for e in self._behavioral_typed if e.event_type == CT.FILE_WRITE and e.operation == "create"]

        file_types = Counter(e.file_type or "unknown" for e in writes)

        structured_files = sum(file_types.get(ft, 0) for ft in STRUCTURED_FILE_TYPES)
        primary_files = sum(file_types.get(ft, 0) for ft in PRIMARY_OUTPUT_TYPES)

        total_created = len(writes)
        supplementary_ratio = structured_files / max(total_created, 1)

        image_files = sum(file_types.get(ft, 0) for ft in IMAGE_FILE_TYPES)

        markdown_table_rows = 0
        for e in writes:
            content = e.resolved_content
            if content:
                markdown_table_rows += len(re.findall(r"^\|.*\|", content, re.MULTILINE))

        return {
            "structured_files_created": structured_files,
            "primary_files_created": primary_files,
            "image_files_created": image_files,
            "supplementary_ratio": round(supplementary_ratio, 3),
            "total_files_created": total_created,
            "has_tables_or_data": (structured_files > 0) or (markdown_table_rows > 0),
            "markdown_table_rows": markdown_table_rows,
            "has_images": image_files > 0,
        }

    def _extract_tone_features_typed(self) -> dict[str, Any]:
        writes = [e for e in self._behavioral_typed if e.event_type == CT.FILE_WRITE and e.operation == "create"]

        total_sentences = 0
        total_words = 0
        heading_count = 0
        heading_max_depth = 0
        table_count = 0
        list_count = 0
        prose_sentences = 0
        has_content = False

        for e in writes:
            content = e.resolved_content
            if not content:
                continue
            has_content = True
            headings = re.findall(r"^(#{1,6})\s", content, re.MULTILINE)
            heading_count += len(headings)
            for h in headings:
                heading_max_depth = max(heading_max_depth, len(h))
            table_count += len(re.findall(r"^\|.*\|", content, re.MULTILINE))
            list_items = len(re.findall(r"^[\s]*[-*+]\s", content, re.MULTILINE))
            list_items += len(re.findall(r"^[\s]*\d+\.\s", content, re.MULTILINE))
            list_count += list_items
            sentences = re.split(r"[.!?。！？]\s", content)
            total_sentences += len(sentences)
            total_words += len(content.split())
            for line in content.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    continue
                if stripped.startswith("|"):
                    continue
                if re.match(r"^[-*+]\s", stripped) or re.match(r"^\d+\.\s", stripped):
                    continue
                if len(stripped) > 20:
                    prose_sentences += 1

        avg_sentence_length = total_words / max(total_sentences, 1)
        structure_items = list_count + table_count
        prose_to_structure_ratio = round(prose_sentences / max(structure_items, 1), 3)

        return {
            "has_content_analysis": has_content,
            "heading_count": heading_count,
            "heading_max_depth": heading_max_depth,
            "table_row_count": table_count,
            "list_item_count": list_count,
            "avg_sentence_length_words": round(avg_sentence_length, 1),
            "total_words": total_words,
            "prose_to_structure_ratio": prose_to_structure_ratio,
        }

    def _extract_verbosity_typed(self) -> dict[str, Any]:
        writes = [e for e in self._behavioral_typed if e.event_type == CT.FILE_WRITE and e.operation == "create"]

        lengths = [e.content_length for e in writes]
        total = sum(lengths)

        return {
            "total_output_chars": total,
            "files_created": len(writes),
            "avg_chars_per_file": round(total / max(len(writes), 1), 1),
        }

    def _extract_work_rhythm_typed(self) -> dict[str, Any]:
        # iteration_end uses all events (not just behavioral)
        assert self._typed is not None
        iterations = [e for e in self._typed if e.event_type == CT.ITERATION_END]
        switches = [e for e in self._behavioral_typed if e.event_type == CT.CONTEXT_SWITCH]

        total_iterations = len(iterations)

        tools_per_iter = [e.tools_called for e in iterations]
        avg_tools = statistics.mean(tools_per_iter) if tools_per_iter else 0

        error_iters = sum(1 for e in iterations if e.has_tool_error)
        error_rate = error_iters / max(total_iterations, 1)

        return {
            "total_iterations": total_iterations,
            "avg_tools_per_iteration": round(avg_tools, 1),
            "iteration_error_rate": round(error_rate, 3),
            "total_context_switches": len(switches),
        }

    def _extract_semantic_channel_typed(self) -> dict[str, Any]:
        assert self._typed is not None
        _anon = self._anonymize_path

        created_files = []
        created_filenames = []
        writes = [e for e in self._behavioral_typed if e.event_type == CT.FILE_WRITE and e.operation == "create"]
        for e in writes:
            fp = e.file_path
            anon_path = _anon(fp)
            content = e.resolved_content or ""
            created_files.append(
                {
                    "path": anon_path,
                    "content_length": e.content_length,
                    "preview": content[:PREVIEW_FILE_CHARS],  # style analysis
                    "content": content,  # full doc for retrieval
                }
            )
            if fp:
                created_filenames.append(PurePosixPath(fp).name)

        edit_diffs = []
        edits = [e for e in self._behavioral_typed if e.event_type == CT.FILE_EDIT]
        for e in edits:
            fp = e.file_path
            content = e.resolved_content or ""
            edit_diffs.append(
                {
                    "path": _anon(fp),
                    "lines_added": e.lines_added,
                    "lines_deleted": e.lines_deleted,
                    "diff_preview": content[:PREVIEW_DIFF_CHARS],  # style analysis
                    "diff_content": content,  # full diff for retrieval
                }
            )

        snapshots = [e for e in self._behavioral_typed if e.event_type == CT.FS_SNAPSHOT]
        dir_structure_diff: list[str] = []
        if len(snapshots) >= 2:
            first_tree = set(snapshots[0].directory_tree or [])
            last_tree = set(snapshots[-1].directory_tree or [])
            new_entries = sorted(last_tree - first_tree)
            dir_structure_diff = [_anon(p) for p in new_entries]

        return {
            "created_files": created_files,
            "edit_diffs": edit_diffs,
            "dir_structure_diff": dir_structure_diff,
            "created_filenames": created_filenames,
        }

    # ==================================================================
    # ── LEGACY implementations (original .get() code, unchanged) ──
    # ==================================================================

    def _extract_reading_strategy_legacy(self) -> dict[str, Any]:
        reads = [e for e in self._behavioral if e.get("event_type") == "file_read"]
        browses = [e for e in self._behavioral if e.get("event_type") == "file_browse"]
        searches = [e for e in self._behavioral if e.get("event_type") == "file_search"]
        switches = [e for e in self._behavioral if e.get("event_type") == "context_switch"]

        total_access = len(reads) + len(browses) + len(searches)
        browse_ratio = len(browses) / max(total_access, 1)
        search_ratio = len(searches) / max(total_access, 1)
        read_ratio = len(reads) / max(total_access, 1)

        revisits = [e for e in reads if e.get("view_count", 1) > 1]
        revisit_ratio = len(revisits) / max(len(reads), 1)

        content_lengths = [e.get("content_length", 0) for e in reads]
        avg_content_length = statistics.mean(content_lengths) if content_lengths else 0

        unique_files = len(set(e.get("file_path", "") for e in reads))
        switch_rate = len(switches) / max(len(reads), 1)

        return {
            "total_reads": len(reads),
            "total_browses": len(browses),
            "total_searches": len(searches),
            "browse_ratio": round(browse_ratio, 3),
            "search_ratio": round(search_ratio, 3),
            "read_ratio": round(read_ratio, 3),
            "revisit_ratio": round(revisit_ratio, 3),
            "avg_content_length": round(avg_content_length, 1),
            "unique_files_read": unique_files,
            "context_switch_rate": round(switch_rate, 3),
        }

    def _extract_output_detail_legacy(self) -> dict[str, Any]:
        writes = [e for e in self._behavioral if e.get("event_type") == "file_write" and e.get("operation") == "create"]

        content_lengths = [e.get("content_length", 0) for e in writes]
        total_output_chars = sum(content_lengths)
        avg_output_length = statistics.mean(content_lengths) if content_lengths else 0
        max_output_length = max(content_lengths) if content_lengths else 0

        return {
            "files_created": len(writes),
            "total_output_chars": total_output_chars,
            "avg_output_length": round(avg_output_length, 1),
            "max_output_length": max_output_length,
            "output_length_std": round(statistics.stdev(content_lengths), 1) if len(content_lengths) > 1 else 0,
        }

    def _extract_output_structure_legacy(self) -> dict[str, Any]:
        writes = [e for e in self._behavioral if e.get("event_type") == "file_write" and e.get("operation") == "create"]

        file_types = Counter(e.get("file_type", "unknown") for e in writes)
        has_markdown = file_types.get("md", 0) > 0
        has_structured_file = any(ft in file_types for ft in ("json", "yaml", "csv", "xml"))

        markdown_table_rows = 0
        for e in writes:
            content = e.get("_resolved_content", "")
            if content:
                markdown_table_rows += len(re.findall(r"^\|.*\|", content, re.MULTILINE))

        has_markdown_tables = markdown_table_rows > 0

        return {
            "file_types_created": dict(file_types),
            "has_markdown": has_markdown,
            "has_structured_data": has_structured_file or has_markdown_tables,
            "has_markdown_tables": has_markdown_tables,
            "markdown_table_rows": markdown_table_rows,
            "distinct_file_types": len(file_types),
        }

    def _extract_directory_style_legacy(self) -> dict[str, Any]:
        dir_creates = [e for e in self._behavioral if e.get("event_type") == "dir_create"]
        moves = [e for e in self._behavioral if e.get("event_type") == "file_move"]
        snapshots = [e for e in self._behavioral if e.get("event_type") == "fs_snapshot"]

        depths = [e.get("depth", 0) for e in dir_creates]
        max_depth = max(depths) if depths else 0
        avg_depth = statistics.mean(depths) if depths else 0

        dirs_created = len(dir_creates)

        final_max_depth = 0
        if snapshots:
            final_max_depth = snapshots[-1].get("max_depth", 0)

        return {
            "dirs_created": dirs_created,
            "max_dir_depth": max_depth,
            "avg_dir_depth": round(avg_depth, 2),
            "final_fs_max_depth": final_max_depth,
            "files_moved": len(moves),
        }

    def _extract_naming_legacy(self) -> dict[str, Any]:
        writes = [e for e in self._behavioral if e.get("event_type") == "file_write" and e.get("operation") == "create"]
        renames = [e for e in self._behavioral if e.get("event_type") == "file_rename"]

        filenames = []
        for e in writes:
            fp = e.get("file_path", "")
            if fp:
                filenames.append(PurePosixPath(fp).stem)

        if not filenames:
            return {
                "avg_filename_length": 0,
                "has_date_prefix": False,
                "has_numeric_prefix": False,
                "has_underscores": False,
                "has_hyphens": False,
                "avg_word_count": 0,
                "total_renames": len(renames),
            }

        avg_len = statistics.mean(len(f) for f in filenames)

        date_pattern = re.compile(r"^\d{4}[-_]?\d{2}[-_]?\d{2}")
        has_date = any(date_pattern.match(f) for f in filenames)
        numeric_prefix_pattern = re.compile(r"^\d{1,3}[-_]")
        has_numeric_prefix = any(numeric_prefix_pattern.match(f) for f in filenames)
        has_underscores = any("_" in f for f in filenames)
        has_hyphens = any("-" in f for f in filenames)

        word_counts = []
        for f in filenames:
            words = re.split(r"[-_.\s]+", f)
            word_counts.append(len([w for w in words if w]))
        avg_words = statistics.mean(word_counts) if word_counts else 0

        return {
            "avg_filename_length": round(avg_len, 1),
            "has_date_prefix": has_date,
            "has_numeric_prefix": has_numeric_prefix,
            "has_underscores": has_underscores,
            "has_hyphens": has_hyphens,
            "avg_word_count": round(avg_words, 1),
            "total_renames": len(renames),
        }

    def _extract_edit_strategy_legacy(self) -> dict[str, Any]:
        edits = [e for e in self._behavioral if e.get("event_type") == "file_edit"]

        if not edits:
            return {
                "total_edits": 0,
                "avg_lines_changed": 0,
                "max_lines_changed": 0,
                "edits_per_file": 0,
                "small_edit_ratio": 0,
            }

        lines_per_edit = [e.get("lines_added", 0) + e.get("lines_deleted", 0) for e in edits]
        avg_lines = statistics.mean(lines_per_edit)
        max_lines = max(lines_per_edit)

        files_edited = Counter(e.get("file_path", "") for e in edits)
        avg_edits_per_file = statistics.mean(files_edited.values())

        small_edits = sum(1 for lc in lines_per_edit if lc < 10)
        small_ratio = small_edits / len(edits)

        return {
            "total_edits": len(edits),
            "avg_lines_changed": round(avg_lines, 1),
            "max_lines_changed": max_lines,
            "edits_per_file": round(avg_edits_per_file, 2),
            "small_edit_ratio": round(small_ratio, 3),
        }

    def _extract_version_strategy_legacy(self) -> dict[str, Any]:
        copies = [e for e in self._behavioral if e.get("event_type") == "file_copy"]
        deletes = [e for e in self._behavioral if e.get("event_type") == "file_delete"]
        overwrites = [
            e for e in self._behavioral if e.get("event_type") == "file_write" and e.get("operation") == "overwrite"
        ]

        backups = [e for e in copies if e.get("is_backup", False)]

        return {
            "total_copies": len(copies),
            "backup_copies": len(backups),
            "total_deletes": len(deletes),
            "total_overwrites": len(overwrites),
            "has_backup_behavior": len(backups) > 0,
            "delete_to_create_ratio": round(
                len(deletes)
                / max(
                    len(
                        [
                            e
                            for e in self._behavioral
                            if e.get("event_type") == "file_write" and e.get("operation") == "create"
                        ]
                    ),
                    1,
                ),
                3,
            ),
        }

    def _extract_cross_modal_behavior_legacy(self) -> dict[str, Any]:
        writes = [e for e in self._behavioral if e.get("event_type") == "file_write" and e.get("operation") == "create"]

        file_types = Counter(e.get("file_type", "unknown") for e in writes)

        structured_files = sum(file_types.get(ft, 0) for ft in STRUCTURED_FILE_TYPES)
        primary_files = sum(file_types.get(ft, 0) for ft in PRIMARY_OUTPUT_TYPES)

        total_created = len(writes)
        supplementary_ratio = structured_files / max(total_created, 1)

        image_files = sum(file_types.get(ft, 0) for ft in IMAGE_FILE_TYPES)

        markdown_table_rows = 0
        for e in writes:
            content = e.get("_resolved_content", "")
            if content:
                markdown_table_rows += len(re.findall(r"^\|.*\|", content, re.MULTILINE))

        return {
            "structured_files_created": structured_files,
            "primary_files_created": primary_files,
            "image_files_created": image_files,
            "supplementary_ratio": round(supplementary_ratio, 3),
            "total_files_created": total_created,
            "has_tables_or_data": (structured_files > 0) or (markdown_table_rows > 0),
            "markdown_table_rows": markdown_table_rows,
            "has_images": image_files > 0,
        }

    def _extract_tone_features_legacy(self) -> dict[str, Any]:
        writes = [e for e in self._behavioral if e.get("event_type") == "file_write" and e.get("operation") == "create"]

        total_sentences = 0
        total_words = 0
        heading_count = 0
        heading_max_depth = 0
        table_count = 0
        list_count = 0
        prose_sentences = 0
        has_content = False

        for e in writes:
            content = e.get("_resolved_content", "")
            if not content:
                continue
            has_content = True
            headings = re.findall(r"^(#{1,6})\s", content, re.MULTILINE)
            heading_count += len(headings)
            for h in headings:
                heading_max_depth = max(heading_max_depth, len(h))
            table_count += len(re.findall(r"^\|.*\|", content, re.MULTILINE))
            list_items = len(re.findall(r"^[\s]*[-*+]\s", content, re.MULTILINE))
            list_items += len(re.findall(r"^[\s]*\d+\.\s", content, re.MULTILINE))
            list_count += list_items
            sentences = re.split(r"[.!?。！？]\s", content)
            total_sentences += len(sentences)
            total_words += len(content.split())
            for line in content.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    continue
                if stripped.startswith("|"):
                    continue
                if re.match(r"^[-*+]\s", stripped) or re.match(r"^\d+\.\s", stripped):
                    continue
                if len(stripped) > 20:
                    prose_sentences += 1

        avg_sentence_length = total_words / max(total_sentences, 1)
        structure_items = list_count + table_count
        prose_to_structure_ratio = round(prose_sentences / max(structure_items, 1), 3)

        return {
            "has_content_analysis": has_content,
            "heading_count": heading_count,
            "heading_max_depth": heading_max_depth,
            "table_row_count": table_count,
            "list_item_count": list_count,
            "avg_sentence_length_words": round(avg_sentence_length, 1),
            "total_words": total_words,
            "prose_to_structure_ratio": prose_to_structure_ratio,
        }

    def _extract_verbosity_legacy(self) -> dict[str, Any]:
        writes = [e for e in self._behavioral if e.get("event_type") == "file_write" and e.get("operation") == "create"]

        lengths = [e.get("content_length", 0) for e in writes]
        total = sum(lengths)

        return {
            "total_output_chars": total,
            "files_created": len(writes),
            "avg_chars_per_file": round(total / max(len(writes), 1), 1),
        }

    def _extract_work_rhythm_legacy(self) -> dict[str, Any]:
        iterations = [e for e in self.events if e.get("event_type") == "iteration_end"]
        switches = [e for e in self._behavioral if e.get("event_type") == "context_switch"]

        total_iterations = len(iterations)

        tools_per_iter = [e.get("tools_called", 0) for e in iterations]
        avg_tools = statistics.mean(tools_per_iter) if tools_per_iter else 0

        error_iters = sum(1 for e in iterations if e.get("has_tool_error", False))
        error_rate = error_iters / max(total_iterations, 1)

        return {
            "total_iterations": total_iterations,
            "avg_tools_per_iteration": round(avg_tools, 1),
            "iteration_error_rate": round(error_rate, 3),
            "total_context_switches": len(switches),
        }

    def _extract_semantic_channel_legacy(self) -> dict[str, Any]:
        _anon = self._anonymize_path

        created_files = []
        created_filenames = []
        writes = [e for e in self._behavioral if e.get("event_type") == "file_write" and e.get("operation") == "create"]
        for e in writes:
            fp = e.get("file_path", "")
            anon_path = _anon(fp)
            content = e.get("_resolved_content", "") or ""
            created_files.append(
                {
                    "path": anon_path,
                    "content_length": e.get("content_length", 0),
                    "preview": content[:PREVIEW_FILE_CHARS],  # style analysis
                    "content": content,  # full doc for retrieval
                }
            )
            if fp:
                created_filenames.append(PurePosixPath(fp).name)

        edit_diffs = []
        edits = [e for e in self._behavioral if e.get("event_type") == "file_edit"]
        for e in edits:
            fp = e.get("file_path", "")
            content = e.get("_resolved_content", "") or ""
            edit_diffs.append(
                {
                    "path": _anon(fp),
                    "lines_added": e.get("lines_added", 0),
                    "lines_deleted": e.get("lines_deleted", 0),
                    "diff_preview": content[:PREVIEW_DIFF_CHARS],  # style analysis
                    "diff_content": content,  # full diff for retrieval
                }
            )

        snapshots = [e for e in self._behavioral if e.get("event_type") == "fs_snapshot"]
        dir_structure_diff: list[str] = []
        if len(snapshots) >= 2:
            first_tree = set(snapshots[0].get("directory_tree", []))
            last_tree = set(snapshots[-1].get("directory_tree", []))
            new_entries = sorted(last_tree - first_tree)
            dir_structure_diff = [_anon(p) for p in new_entries]

        return {
            "created_files": created_files,
            "edit_diffs": edit_diffs,
            "dir_structure_diff": dir_structure_diff,
            "created_filenames": created_filenames,
        }

    # ---- Utility ----

    @staticmethod
    def _anonymize_path(path: str) -> str:
        """Strip absolute sandbox prefix, returning only the relative file path."""
        if not path or path == "?":
            return path
        m = re.search(r"/sandbox/[^/]+/(.*)", path)
        if m:
            return m.group(1) or "."
        return path.rsplit("/", 1)[-1] if "/" in path else path
