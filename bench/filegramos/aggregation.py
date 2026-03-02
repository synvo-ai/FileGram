"""Step 2: Cross-trajectory aggregation.

Aggregates features from 10 trajectories per profile into
mean/mode/distribution statistics per attribute.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from . import schema as S


class FeatureAggregator:
    """Aggregate per-trajectory features into profile-level statistics."""

    def __init__(self, per_trajectory_features: list[dict[str, Any]]):
        """
        Args:
            per_trajectory_features: List of dicts, one per trajectory.
                Each dict maps attribute_name -> feature_dict.
        """
        self.features = per_trajectory_features
        self.n = len(per_trajectory_features)

    def aggregate_all(self) -> dict[str, dict[str, Any]]:
        """Aggregate all attributes across trajectories."""
        if not self.features:
            return {}

        # Get attribute names from first trajectory
        attr_names = list(self.features[0].keys())

        aggregated = {}
        for attr in attr_names:
            per_traj = [f.get(attr, {}) for f in self.features]
            aggregated[attr] = self._aggregate_attribute(attr, per_traj)

        return aggregated

    def _aggregate_attribute(self, attr_name: str, per_traj: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate a single attribute across trajectories."""
        if not per_traj:
            return {}

        # Collect all numeric keys and aggregate them
        all_keys = set()
        for ft in per_traj:
            all_keys.update(ft.keys())

        result: dict[str, Any] = {"_n_trajectories": self.n}

        for key in sorted(all_keys):
            if key.startswith("_"):
                continue

            values = [ft.get(key) for ft in per_traj if key in ft]

            if not values:
                continue

            if all(isinstance(v, (int, float)) for v in values):
                result[f"{key}_mean"] = round(statistics.mean(values), 3)
                result[f"{key}_median"] = round(statistics.median(values), 3)
                if len(values) > 1:
                    result[f"{key}_std"] = round(statistics.stdev(values), 3)
                result[f"{key}_min"] = min(values)
                result[f"{key}_max"] = max(values)

            elif all(isinstance(v, bool) for v in values):
                true_count = sum(1 for v in values if v)
                result[f"{key}_true_ratio"] = round(true_count / len(values), 3)

            elif all(isinstance(v, dict) for v in values):
                # Merge counter-like dicts
                merged = Counter()
                for v in values:
                    for k2, v2 in v.items():
                        if isinstance(v2, (int, float)):
                            merged[k2] += v2
                result[f"{key}_merged"] = dict(merged)

        return result

    def to_summary_text(self) -> str:
        """Convert aggregated features to a human-readable summary.

        This is the input for Step 3 (LLM synthesis).
        Includes both raw statistics AND pattern-level behavioral interpretations.
        """
        aggregated = self.aggregate_all()
        lines = []

        for attr_name, features in aggregated.items():
            lines.append(f"\n### {attr_name}")
            lines.append(f"  (based on {features.get('_n_trajectories', '?')} trajectories)")

            for key, value in sorted(features.items()):
                if key.startswith("_"):
                    continue
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.3f}")
                elif isinstance(value, dict):
                    lines.append(f"  {key}: {value}")
                else:
                    lines.append(f"  {key}: {value}")

        # Add pattern-level behavioral interpretations
        patterns = self._detect_behavioral_patterns(aggregated)
        if patterns:
            lines.append("\n### Detected Behavioral Patterns")
            for p in patterns:
                lines.append(f"  - {p}")

        return "\n".join(lines)

    def _detect_behavioral_patterns(self, aggregated: dict[str, dict[str, Any]]) -> list[str]:
        """Detect high-level behavioral patterns from aggregated statistics.

        Translates raw numbers into interpretable behavioral descriptions
        that are more useful for profile inference than raw statistics alone.
        """
        patterns: list[str] = []
        n = self.n

        def _g(attr: str, key: str, default: float = 0) -> float:
            return aggregated.get(attr, {}).get(key, default)

        # --- Reading Strategy patterns ---
        search_ratio = _g("reading_strategy", "search_ratio_mean")
        browse_ratio = _g("reading_strategy", "browse_ratio_mean")
        revisit_ratio = _g("reading_strategy", "revisit_ratio_mean")
        switch_rate = _g("reading_strategy", "context_switch_rate_mean")
        total_reads = _g("reading_strategy", "total_reads_mean")
        unique_files = _g("reading_strategy", "unique_files_read_mean")

        if search_ratio < S.SEARCH_RATIO_NEVER:
            patterns.append(
                f"READING: Never/rarely uses search tools (search_ratio={search_ratio:.2f}). "
                f"Relies on sequential file access rather than keyword search."
            )
        elif search_ratio > S.SEARCH_RATIO_HEAVY:
            patterns.append(
                f"READING: Heavily uses search tools (search_ratio={search_ratio:.2f}). "
                f"Targeted information-seeking strategy."
            )

        if revisit_ratio > S.REVISIT_RATIO_FREQUENT:
            patterns.append(
                f"READING: Frequently revisits files (revisit_ratio={revisit_ratio:.2f}). "
                f"Cross-checks and verifies by re-reading."
            )
        elif revisit_ratio < S.REVISIT_RATIO_RARE:
            patterns.append(
                f"READING: Rarely revisits files (revisit_ratio={revisit_ratio:.2f}). "
                f"Single-pass reading, moves forward without re-checking."
            )

        if browse_ratio > S.BROWSE_RATIO_FREQUENT:
            patterns.append(
                f"READING: Frequently browses directories (browse_ratio={browse_ratio:.2f}). "
                f"Surveys file landscape before diving in."
            )

        if total_reads > 0 and unique_files > 0:
            reads_per_file = total_reads / unique_files
            if reads_per_file > S.READS_PER_FILE_DEEP:
                patterns.append(
                    f"READING: Reads each file ~{reads_per_file:.1f} times on average. "
                    f"Deep, thorough reading with multiple passes."
                )
            elif reads_per_file < S.READS_PER_FILE_SHALLOW:
                patterns.append(
                    f"READING: Reads each file ~{reads_per_file:.1f} times. Breadth-oriented scanning, rarely re-reads."
                )

        # --- Output Detail patterns ---
        avg_output = _g("output_detail", "avg_output_length_mean")
        files_created = _g("output_detail", "files_created_mean")
        total_output = _g("output_detail", "total_output_chars_mean")

        if avg_output > S.AVG_OUTPUT_DETAILED:
            patterns.append(
                f"OUTPUT: Produces very detailed output (avg {avg_output:.0f} chars/file). "
                f"Comprehensive, lengthy documents."
            )
        elif avg_output < S.AVG_OUTPUT_CONCISE:
            patterns.append(
                f"OUTPUT: Produces concise output (avg {avg_output:.0f} chars/file). Minimal, to-the-point writing."
            )

        if files_created > S.FILES_CREATED_MANY:
            patterns.append(
                f"OUTPUT: Creates many files per task (avg {files_created:.1f}). "
                f"Produces supplementary materials beyond the main deliverable."
            )
        elif files_created <= S.FILES_CREATED_FEW:
            patterns.append(
                f"OUTPUT: Creates few files per task (avg {files_created:.1f}). Focuses on a single main output file."
            )

        # --- Directory Style patterns ---
        dirs_created = _g("directory_style", "dirs_created_mean")
        max_depth = _g("directory_style", "max_dir_depth_mean")
        files_moved = _g("directory_style", "files_moved_mean")

        if dirs_created >= S.DIRS_CREATED_MULTIPLE:
            patterns.append(
                f"ORGANIZATION: Creates multiple directories (avg {dirs_created:.1f}/task, "
                f"max depth {max_depth:.1f}). Builds structured hierarchical organization."
            )
        elif dirs_created < S.DIRS_CREATED_RARE:
            patterns.append(
                f"ORGANIZATION: Rarely creates directories (avg {dirs_created:.1f}/task). "
                f"Keeps files flat, minimal folder structure."
            )

        if files_moved > S.FILES_MOVED_ACTIVE:
            patterns.append(f"ORGANIZATION: Actively moves/reorganizes files (avg {files_moved:.1f} moves/task).")

        # --- Edit Strategy patterns ---
        total_edits = _g("edit_strategy", "total_edits_mean")
        avg_lines = _g("edit_strategy", "avg_lines_changed_mean")
        small_ratio = _g("edit_strategy", "small_edit_ratio_mean")

        if total_edits > S.TOTAL_EDITS_MANY and small_ratio > S.SMALL_EDIT_RATIO_HIGH:
            patterns.append(
                f"EDITING: Many small incremental edits (avg {total_edits:.1f} edits/task, "
                f"{small_ratio:.0%} are small <{S.SMALL_EDIT_LINE_THRESHOLD} lines). Refines work gradually."
            )
        elif total_edits <= S.TOTAL_EDITS_RARE:
            patterns.append(
                f"EDITING: Rarely edits after initial creation (avg {total_edits:.1f} edits/task). "
                f"Creates files in near-final form."
            )
        if avg_lines > S.AVG_LINES_LARGE:
            patterns.append(
                f"EDITING: Large edit scope when editing (avg {avg_lines:.0f} lines/edit). Bulk rewrite approach."
            )

        # --- Version Strategy patterns ---
        backups = _g("version_strategy", "backup_copies_mean")
        deletes = _g("version_strategy", "total_deletes_mean")
        overwrites = _g("version_strategy", "total_overwrites_mean")
        has_backup = _g("version_strategy", "has_backup_behavior_true_ratio")

        if has_backup > S.BACKUP_RATIO_HIGH:
            patterns.append(
                f"VERSIONING: Creates backups before modifying (backup ratio={has_backup:.0%}). Preserves file history."
            )
        elif overwrites > S.OVERWRITE_NO_BACKUP_THRESH and backups < S.BACKUP_RATIO_LOW:
            patterns.append(
                f"VERSIONING: Overwrites files directly (avg {overwrites:.1f} overwrites/task), "
                f"no backups. Destructive update pattern."
            )

        if deletes > S.DELETES_ACTIVE:
            patterns.append(
                f"VERSIONING: Actively deletes files (avg {deletes:.1f}/task). Cleans up as part of workflow."
            )

        # --- Cross-Modal patterns ---
        cm = aggregated.get("cross_modal_behavior", {})
        has_tables = cm.get("has_tables_or_data_true_ratio", 0)
        table_rows = cm.get("markdown_table_rows_mean", 0)
        has_images = cm.get("has_images_true_ratio", 0)
        structured_files = cm.get("structured_files_created_mean", 0)

        if has_tables > S.TABLES_RATIO_FREQUENT or table_rows > S.TABLE_ROWS_MANY:
            patterns.append(
                f"CROSS-MODAL: Frequently uses tables ({has_tables:.0%} of tasks, "
                f"avg {table_rows:.0f} table rows). Data-presentation oriented."
            )
        if has_images > S.IMAGES_RATIO_PRESENT:
            patterns.append(f"CROSS-MODAL: Creates image/visual files ({has_images:.0%} of tasks).")
        if structured_files > S.STRUCTURED_FILES_PRESENT:
            patterns.append(f"CROSS-MODAL: Creates structured data files (avg {structured_files:.1f}/task).")
        if (
            has_tables < S.TABLES_RATIO_ABSENT
            and has_images < S.IMAGES_RATIO_ABSENT
            and structured_files < S.TABLES_RATIO_ABSENT
        ):
            patterns.append("CROSS-MODAL: Primarily text-only output. No tables, images, or structured data files.")

        # --- Tone (structural proxy) ---
        heading_depth = _g("tone", "heading_max_depth_mean")
        heading_count = _g("tone", "heading_count_mean")
        prose_ratio = _g("tone", "prose_to_structure_ratio_mean")

        if heading_depth >= S.HEADING_DEPTH_DEEP:
            patterns.append(
                f"STRUCTURE: Deep heading hierarchy (up to H{heading_depth:.0f}). "
                f"Formal, well-organized document structure."
            )
        if prose_ratio > S.PROSE_RATIO_HEAVY:
            patterns.append(
                f"STRUCTURE: Prose-heavy output (prose:structure ratio={prose_ratio:.1f}). "
                f"Narrative writing style over bullet points."
            )
        elif prose_ratio < S.PROSE_RATIO_LIGHT and heading_count > 0:
            patterns.append(
                f"STRUCTURE: Structure-heavy output (prose:structure ratio={prose_ratio:.1f}). "
                f"Lists and tables dominate over prose."
            )

        # --- Cross-trajectory consistency ---
        self._add_consistency_patterns(patterns)

        return patterns

    def _add_consistency_patterns(self, patterns: list[str]) -> None:
        """Add cross-trajectory consistency observations."""
        if self.n < 2:
            return

        # Check if key behaviors are consistent across all trajectories
        for feat in self.features:
            rs = feat.get("reading_strategy", {})
            # Check search usage consistency
        all_no_search = all(f.get("reading_strategy", {}).get("total_searches", 0) == 0 for f in self.features)
        if all_no_search:
            patterns.append(
                f"CONSISTENCY: Never used search in any of {self.n} trajectories. "
                f"This is a stable behavioral trait, not task-dependent."
            )

        all_no_dirs = all(f.get("directory_style", {}).get("dirs_created", 0) == 0 for f in self.features)
        if all_no_dirs:
            patterns.append(
                f"CONSISTENCY: Never created directories in any of {self.n} trajectories. "
                f"Consistently flat file organization."
            )

        all_dirs = all(f.get("directory_style", {}).get("dirs_created", 0) >= 2 for f in self.features)
        if all_dirs:
            patterns.append(
                f"CONSISTENCY: Created directories in every trajectory ({self.n}/{self.n}). "
                f"Consistently organizes files into folder structures."
            )

        all_no_edit = all(f.get("edit_strategy", {}).get("total_edits", 0) == 0 for f in self.features)
        if all_no_edit:
            patterns.append(
                f"CONSISTENCY: Zero post-creation edits across all {self.n} trajectories. "
                f"Writes files in final form on first attempt."
            )

    # ---- Semantic channel aggregation ----

    @staticmethod
    def aggregate_semantic(
        per_trajectory_semantic: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Concatenate semantic data across trajectories.

        Args:
            per_trajectory_semantic: List of dicts from extract_semantic_channel(),
                one per trajectory.

        Returns:
            Merged dict with all created_files, edit_diffs, dir entries, filenames.
        """
        all_created_files: list[dict] = []
        all_edit_diffs: list[dict] = []
        all_dir_entries: list[str] = []
        all_filenames: list[str] = []

        for i, sem in enumerate(per_trajectory_semantic):
            label = f"T{i + 1}"
            for cf in sem.get("created_files", []):
                cf_copy = dict(cf)
                cf_copy["trajectory"] = label
                all_created_files.append(cf_copy)
            for ed in sem.get("edit_diffs", []):
                ed_copy = dict(ed)
                ed_copy["trajectory"] = label
                all_edit_diffs.append(ed_copy)
            all_dir_entries.extend(sem.get("dir_structure_diff", []))
            all_filenames.extend(sem.get("created_filenames", []))

        # Deduplicate dir entries
        seen_dirs: set[str] = set()
        unique_dirs: list[str] = []
        for d in all_dir_entries:
            if d not in seen_dirs:
                seen_dirs.add(d)
                unique_dirs.append(d)

        return {
            "created_files": all_created_files,
            "edit_diffs": all_edit_diffs,
            "dir_structure_diff": unique_dirs,
            "created_filenames": list(dict.fromkeys(all_filenames)),  # dedup, preserve order
        }

    @staticmethod
    def to_semantic_text(semantic: dict[str, Any]) -> str:
        """Format semantic channel data as readable text for prompt."""
        lines: list[str] = []

        # Files Created
        created = semantic.get("created_files", [])
        if created:
            lines.append("### Files Created")
            for cf in created:
                path = cf.get("path", "?")
                length = cf.get("content_length", 0)
                preview = cf.get("preview", "")
                lines.append(f"  - {path} ({length} chars)")
                if preview:
                    # Indent preview, limit to ~4 lines
                    preview_lines = preview.split("\n")[:4]
                    for pl in preview_lines:
                        lines.append(f"    {pl}")
                    if len(preview) > 200:
                        lines.append("    ...")

        # Directory Structure Created
        dirs = semantic.get("dir_structure_diff", [])
        if dirs:
            lines.append("### Directory Structure Created")
            lines.append("  " + " → ".join(dirs))

        # Edit Summary
        edits = semantic.get("edit_diffs", [])
        if edits:
            lines.append("### Edit Summary")
            for ed in edits:
                path = ed.get("path", "?")
                added = ed.get("lines_added", 0)
                deleted = ed.get("lines_deleted", 0)
                lines.append(f"  - Edited {path} (+{added}/-{deleted} lines)")
                diff_preview = ed.get("diff_preview", "")
                if diff_preview:
                    diff_lines = diff_preview.split("\n")[:3]
                    for dl in diff_lines:
                        lines.append(f"    {dl}")

        # Created Filenames
        filenames = semantic.get("created_filenames", [])
        if filenames:
            lines.append("### Created Filenames")
            for fn in filenames:
                lines.append(f"  - {fn}")

        return "\n".join(lines)

    # ---- Episodic channel aggregation ----

    @staticmethod
    def aggregate_episodic(
        per_trajectory_features: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute cross-trajectory consistency metrics.

        For key numeric features, computes coefficient of variation (CV).
        Flags as 'stable' when CV < 0.3.

        Returns note when n_trajectories <= 1.
        """
        n = len(per_trajectory_features)
        if n <= 1:
            return {
                "n_trajectories": n,
                "note": "Requires 2+ trajectories for episodic consistency analysis.",
            }

        # Key metrics to check consistency across
        consistency_keys = S.CONSISTENCY_KEYS

        results: dict[str, Any] = {"n_trajectories": n}

        for attr, keys in consistency_keys.items():
            attr_consistency: dict[str, Any] = {}
            for key in keys:
                values = []
                for feat in per_trajectory_features:
                    attr_dict = feat.get(attr, {})
                    v = attr_dict.get(key)
                    if isinstance(v, (int, float)):
                        values.append(v)
                if len(values) >= 2:
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values)
                    cv = std_val / mean_val if mean_val != 0 else 0
                    attr_consistency[key] = {
                        "mean": round(mean_val, 3),
                        "std": round(std_val, 3),
                        "cv": round(cv, 3),
                        "stable": cv < S.CV_STABILITY_THRESHOLD,
                    }
            if attr_consistency:
                results[attr] = attr_consistency

        return results

    @staticmethod
    def to_episodic_text(episodic: dict[str, Any]) -> str:
        """Format episodic channel data as readable text for prompt."""
        n = episodic.get("n_trajectories", 0)
        note = episodic.get("note")
        if note:
            return f"(requires 2+ trajectories; only {n} available)"

        lines = [f"Cross-trajectory consistency ({n} trajectories):"]
        for attr, metrics in episodic.items():
            if attr in ("n_trajectories", "note"):
                continue
            if not isinstance(metrics, dict):
                continue
            lines.append(f"  {attr}:")
            for key, info in metrics.items():
                if not isinstance(info, dict):
                    continue
                stable_flag = "STABLE" if info.get("stable") else "VARIABLE"
                lines.append(f"    {key}: mean={info['mean']}, std={info['std']}, cv={info['cv']} [{stable_flag}]")

        return "\n".join(lines)
