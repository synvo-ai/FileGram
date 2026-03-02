"""Query-adaptive three-channel retriever for FileGramOS.

Produces clean, channel-separated behavioral profiles from MemoryStore.
Design principles:
  - Clear three-channel structure (Procedural → Semantic → Episodic)
  - Natural language descriptions, not abbreviated stat dumps
  - Per-session evidence in readable tabular format
  - No redundancy: each signal appears once in its primary channel
"""

from __future__ import annotations

from .engram import MemoryStore
from .schema import (
    CV_VARIABLE_DISPLAY_THRESHOLD,
    FRIENDLY_NAMES,
    LANGUAGE_RATIO_DOMINANT,
    LANGUAGE_RATIO_MIXED,
    NAMING_PATTERN_THRESHOLD,
    NAMING_UNDERSCORE_THRESHOLD,
)
from .tuning import RETRIEVER_DISPLAY_CHARS


class QueryAdaptiveRetriever:
    """Compose three-channel memory into clean behavioral profiles."""

    def retrieve(
        self,
        store: MemoryStore,
        query_type: str = "profile",
        attribute: str | None = None,
    ) -> str:
        n = len(store.engrams)
        n_dev = sum(1 for v in store.deviation_flags.values() if v)

        parts: list[str] = []
        parts.append(f"## Behavioral Memory Profile ({n} sessions, {n_dev} anomalous)")
        parts.append(self._channel_procedural(store))
        parts.append(self._channel_semantic(store))
        parts.append(self._channel_episodic(store, n_dev))
        parts.append(self._session_table(store))

        return "\n\n".join(p for p in parts if p)

    # ── Channel 1: Procedural ───────────────────────────────────────

    def _channel_procedural(self, store: MemoryStore) -> str:
        """Behavioral dimension profiles with natural-language descriptions."""
        cls_map = self._build_classification_map(store)
        agg = store.procedural_aggregate
        lines = ["### Channel 1: Procedural Patterns"]

        # A: Consumption
        rs = agg.get("reading_strategy", {})
        label = self._label(cls_map.get("reading_strategy", "unknown"))
        reads = rs.get("total_reads_mean", 0)
        search = rs.get("search_ratio_mean", 0)
        revisit = rs.get("revisit_ratio_mean", 0)
        browse = rs.get("browse_ratio_mean", 0)
        desc_parts = [f"avg {reads:.1f} reads/task"]
        if search > 0.01:
            desc_parts.append(f"search ratio {search:.0%}")
        else:
            desc_parts.append("no search tools used")
        if revisit > 0.01:
            desc_parts.append(f"revisit rate {revisit:.0%}")
        if browse > 0.01:
            desc_parts.append(f"browse rate {browse:.0%}")
        lines.append(f"**A. Consumption — {label}**")
        lines.append(self._join_sentences(desc_parts))

        # B: Production
        od = agg.get("output_detail", {})
        tn = agg.get("tone", {})
        label = self._label(cls_map.get("output_detail", "unknown"))
        avg_len = od.get("avg_output_length_mean", 0)
        files = od.get("files_created_mean", 0)
        h_depth = tn.get("heading_max_depth_mean", 0)
        lines.append(f"**B. Production — {label}**")
        lines.append(f"Avg {avg_len:,.0f} chars across {files:.1f} files/task. Heading depth: H{h_depth:.0f}.")

        # C: Organization
        ds = agg.get("directory_style", {})
        nm = agg.get("naming", {})
        label = self._label(cls_map.get("directory_style", "unknown"))
        dirs = ds.get("dirs_created_mean", 0)
        depth = ds.get("max_dir_depth_mean", 0)
        moves = ds.get("files_moved_mean", 0)
        name_len = nm.get("avg_filename_length_mean", 0)
        lines.append(f"**C. Organization — {label}**")
        org_parts = [f"{dirs:.1f} dirs/task, depth {depth:.1f}"]
        if moves > 0.1:
            org_parts.append(f"{moves:.1f} file moves/task")
        org_parts.append(f"avg filename {name_len:.0f} chars")
        lines.append(self._join_sentences(org_parts))

        # D: Iteration
        es = agg.get("edit_strategy", {})
        vs = agg.get("version_strategy", {})
        label = self._label(cls_map.get("edit_strategy", "unknown"))
        edits = es.get("total_edits_mean", 0)
        avg_lines = es.get("avg_lines_changed_mean", 0)
        backup = vs.get("has_backup_behavior_true_ratio", 0)
        deletes = vs.get("total_deletes_mean", 0)
        lines.append(f"**D. Iteration — {label}**")
        iter_parts = [f"{edits:.1f} edits/task, avg {avg_lines:.0f} lines/edit"]
        if backup > 0.01:
            iter_parts.append(f"backup in {backup:.0%} of tasks")
        else:
            iter_parts.append("no backup behavior")
        if deletes > 0.1:
            iter_parts.append(f"{deletes:.1f} deletions/task")
        lines.append(self._join_sentences(iter_parts))

        # E: Rhythm
        label = self._label(cls_map.get("working_style", "unknown"))
        switch_rate = rs.get("context_switch_rate_mean", 0)
        lines.append(f"**E. Rhythm — {label}**")
        lines.append(f"Context switch rate: {switch_rate:.2f}.")

        # F: Cross-Modal
        cm = agg.get("cross_modal_behavior", {})
        label = self._label(cls_map.get("cross_modal_behavior", "unknown"))
        tables = cm.get("has_tables_true_ratio", 0)
        images = cm.get("creates_image_files_true_ratio", 0)
        lines.append(f"**F. Cross-Modal — {label}**")
        cm_parts = []
        if tables > 0.01:
            cm_parts.append(f"tables in {tables:.0%} of tasks")
        if images > 0.01:
            cm_parts.append(f"images in {images:.0%} of tasks")
        if not cm_parts:
            cm_parts.append("no tables or images created")
        lines.append(self._join_sentences(cm_parts))

        return "\n".join(lines)

    # ── Channel 2: Semantic ─────────────────────────────────────────

    def _channel_semantic(self, store: MemoryStore) -> str:
        """Content characteristics: language, naming, file samples."""
        lines = ["### Channel 2: Semantic Content"]

        # Detect language from filenames
        if store.all_filenames:
            cn_count = sum(1 for fn in store.all_filenames if any("\u4e00" <= c <= "\u9fff" for c in fn))
            total = len(store.all_filenames)
            if cn_count > total * LANGUAGE_RATIO_DOMINANT:
                lines.append("**Language:** Chinese (中文) primary")
            elif cn_count > total * LANGUAGE_RATIO_MIXED:
                lines.append("**Language:** Mixed Chinese/English")
            else:
                lines.append("**Language:** English primary")

        # Naming convention detection
        if store.engrams:
            agg_nm = store.procedural_aggregate.get("naming", {})
            conventions = []
            date_ratio = agg_nm.get("has_date_prefix_mean", 0)
            num_ratio = agg_nm.get("has_numeric_prefix_mean", 0)
            underscore = agg_nm.get("has_underscores_mean", 0)
            if date_ratio > NAMING_PATTERN_THRESHOLD:
                conventions.append("date prefixes")
            if num_ratio > NAMING_PATTERN_THRESHOLD:
                conventions.append("numeric prefixes (01_, 02_)")
            if underscore > NAMING_UNDERSCORE_THRESHOLD:
                conventions.append("underscore-separated")
            if not conventions:
                conventions.append("descriptive names")
            lines.append(f"**Naming:** {', '.join(conventions)}")

        # File type distribution
        if store.engrams:
            type_counts: dict[str, int] = {}
            for eng in store.engrams:
                for cs in eng.semantic.created_files:
                    ext = cs.file_type or "unknown"
                    type_counts[ext] = type_counts.get(ext, 0) + 1
            if type_counts:
                sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
                type_str = ", ".join(f".{t} ({c})" for t, c in sorted_types[:4])
                lines.append(f"**File types:** {type_str}")

        # Representative filenames (deduplicated, concise)
        if store.all_filenames:
            seen: set[str] = set()
            unique: list[str] = []
            for fn in store.all_filenames:
                base = fn.rsplit("/", 1)[-1] if "/" in fn else fn
                if base not in seen and base != "README.md":
                    seen.add(base)
                    unique.append(base)
            total = len(store.all_filenames)
            sample = unique[:15]
            # Truncate long filenames
            sample = [fn[:40] + "…" if len(fn) > 40 else fn for fn in sample]
            lines.append(f"**Files ({total} total):** {', '.join(sample)}")

        # Directory structure
        if store.dir_structure_union:
            dirs = store.dir_structure_union[:6]
            lines.append(f"**Directories:** {', '.join(dirs)}")

        # Content previews (most informative for semantic questions)
        if store.representative_samples:
            lines.append(f"**Content samples ({len(store.representative_samples)}):**")
            for s in store.representative_samples[:12]:
                preview = ""
                if s.content_preview:
                    # Use preview (style excerpt) for profile rendering
                    preview = s.content_preview.replace("\n", " ").strip()[:RETRIEVER_DISPLAY_CHARS]
                basename = s.path.rsplit("/", 1)[-1] if "/" in s.path else s.path
                size_k = s.content_length / 1000
                lines.append(f"  - {basename} ({size_k:.0f}K): {preview}")

        # Writing style and content analysis from LLM narratives
        if store.llm_narratives:
            styles = set()
            structures = set()
            detail_levels = set()
            approaches = set()
            habits = set()
            for narrative in store.llm_narratives.values():
                for val, target, limit in [
                    (narrative.get("writing_style", ""), styles, 6),
                    (narrative.get("content_structure", ""), structures, 4),
                    (narrative.get("detail_level", ""), detail_levels, 4),
                    (narrative.get("work_approach", ""), approaches, 4),
                    (narrative.get("notable_habits", ""), habits, 6),
                ]:
                    if isinstance(val, list):
                        val = "; ".join(str(v) for v in val)
                    if val and isinstance(val, str) and len(val) < 150 and len(target) < limit:
                        target.add(val)
            if styles:
                lines.append(f"**Writing style:** {'; '.join(list(styles)[:4])}")
            if structures:
                lines.append(f"**Content structure:** {'; '.join(list(structures)[:3])}")
            if detail_levels:
                lines.append(f"**Detail level:** {'; '.join(list(detail_levels)[:3])}")
            if approaches:
                lines.append(f"**Work approach:** {'; '.join(list(approaches)[:3])}")
            if habits:
                lines.append(f"**Notable habits:** {'; '.join(list(habits)[:4])}")

        return "\n".join(lines)

    # ── Channel 3: Episodic ─────────────────────────────────────────

    def _channel_episodic(self, store: MemoryStore, n_dev: int) -> str:
        """Behavioral consistency and anomaly detection."""
        lines = ["### Channel 3: Episodic Consistency"]

        # Stable vs variable behaviors
        if store.consistency_flags:
            stable, variable = [], []
            for attr, metrics in store.consistency_flags.items():
                if attr in ("n_trajectories", "note") or not isinstance(metrics, dict):
                    continue
                for key, info in metrics.items():
                    if not isinstance(info, dict):
                        continue
                    friendly = self._friendly_metric(attr, key)
                    if info.get("stable"):
                        stable.append(friendly)
                    else:
                        cv = info.get("cv", 0)
                        if cv > CV_VARIABLE_DISPLAY_THRESHOLD:
                            variable.append(f"{friendly} (cv={cv:.2f})")
            if stable:
                lines.append(f"**Stable:** {', '.join(stable[:6])}")
            if variable:
                lines.append(f"**Variable:** {', '.join(variable[:6])}")

        # Absences
        if store.absence_flags:
            lines.append(f"**Never observed:** {'; '.join(store.absence_flags)}")

        # Deviations with detail
        if n_dev > 0:
            lines.append(f"**Anomalous sessions ({n_dev}):**")
            for tid, is_dev in store.deviation_flags.items():
                if not is_dev:
                    continue
                details = store.deviation_details.get(tid, [])
                if details:
                    top = details[0]
                    dim = top["dimension"].split("(")[-1].rstrip(")") if "(" in top["dimension"] else top["dimension"]
                    feat = top["feature"].split(".")[-1]
                    delta = top["delta"]
                    lines.append(f"  - {tid}: shifted in {dim} ({feat}, δ={delta:.1f})")
                else:
                    lines.append(f"  - {tid}: deviation detected")

        return "\n".join(lines)

    # ── Session Evidence Table ──────────────────────────────────────

    def _session_table(self, store: MemoryStore) -> str:
        """Per-session behavioral data in detailed format."""
        lines = ["### Session Evidence"]

        for eng in store.engrams:
            proc = eng.procedural
            rs = proc.get("reading_strategy", {})
            od = proc.get("output_detail", {})
            ds = proc.get("directory_style", {})
            es = proc.get("edit_strategy", {})
            vs = proc.get("version_strategy", {})
            tn = proc.get("tone", {})
            cm = proc.get("cross_modal_behavior", {})

            # Task ID with deviation marker
            tid = eng.task_id
            is_dev = store.deviation_flags.get(eng.trajectory_id, False)
            if is_dev:
                tid += "*"

            # Core metrics
            reads = rs.get("total_reads", 0)
            search_r = rs.get("search_ratio", 0)
            browse_r = rs.get("browse_ratio", 0)
            revisit_r = rs.get("revisit_ratio", 0)
            avg_out = od.get("avg_output_length", 0)
            n_files = od.get("files_created", 0)
            n_dirs = ds.get("dirs_created", 0)
            depth = ds.get("max_dir_depth", 0)
            moves = ds.get("files_moved", 0)
            n_edits = es.get("total_edits", 0)
            avg_lines = es.get("avg_lines_changed", 0)
            backups = vs.get("backup_copies", 0)
            deletes = vs.get("total_deletes", 0)
            overwrites = vs.get("total_overwrites", 0)
            h_depth = tn.get("heading_max_depth", 0)
            tables = tn.get("table_row_count", 0)
            has_images = cm.get("image_files_created", 0)

            # Build compact per-task description
            parts = [f"**[{tid}]**"]

            # Reading
            read_desc = f"reads={reads}"
            if search_r > 0.01:
                read_desc += f", search={search_r:.0%}"
            if browse_r > 0.01:
                read_desc += f", browse={browse_r:.0%}"
            if revisit_r > 0.01:
                read_desc += f", revisit={revisit_r:.0%}"
            parts.append(read_desc)

            # Output
            out_k = f"{avg_out / 1000:.1f}K" if avg_out >= 1000 else f"{avg_out:.0f}"
            out_desc = f"output={out_k}×{n_files}files, H{h_depth:.0f}"
            if tables > 0:
                out_desc += f", {tables}table-rows"
            parts.append(out_desc)

            # Organization
            if n_dirs > 0 or moves > 0:
                org_desc = f"dirs={n_dirs}/d{depth}"
                if moves > 0:
                    org_desc += f", {moves}moves"
                parts.append(org_desc)

            # Iteration
            if n_edits > 0:
                edit_desc = f"edits={n_edits}×{avg_lines:.0f}lines"
                if backups > 0:
                    edit_desc += f", {backups}backups"
                if overwrites > 0:
                    edit_desc += f", {overwrites}overwrites"
                parts.append(edit_desc)
            if deletes > 0:
                parts.append(f"deletes={deletes}")

            # Cross-modal
            if has_images > 0:
                parts.append(f"images={has_images}")

            # Key filenames
            fnames = []
            for fn in eng.semantic.created_filenames[:3]:
                base = fn.rsplit("/", 1)[-1] if "/" in fn else fn
                base = base[:35] + "…" if len(base) > 35 else base
                fnames.append(base)
            if fnames:
                parts.append(f"files: {', '.join(fnames)}")

            lines.append(" | ".join(parts))

        return "\n".join(lines)

    # ── Helpers ──────────────────────────────────────────────────────

    def _build_classification_map(self, store: MemoryStore) -> dict[str, str]:
        """Build lookup from dimension_classifications."""
        cls_map: dict[str, str] = {}
        for c in store.dimension_classifications:
            if " → " in c:
                attr, rest = c.split(" → ", 1)
                cls_map[attr.strip()] = rest
        return cls_map

    def _label(self, classification: str) -> str:
        """Extract clean label from classification string."""
        if "[" in classification:
            return classification.split("[")[0].strip()
        if ":" in classification:
            return classification.split(":")[0].strip()
        return classification.strip()

    _FRIENDLY_NAMES = FRIENDLY_NAMES

    @staticmethod
    def _join_sentences(parts: list[str]) -> str:
        """Join sentence parts with proper capitalization."""
        return ". ".join(p[0].upper() + p[1:] if p else p for p in parts) + "."

    def _friendly_metric(self, attr: str, key: str) -> str:
        """Convert technical metric names to friendly labels."""
        return self._FRIENDLY_NAMES.get((attr, key), f"{attr}.{key}")
