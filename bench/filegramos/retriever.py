"""Query-adaptive three-channel retriever for FileGramOS.

Produces clean, channel-separated behavioral profiles from MemoryStore.
Design principles:
  - Clear three-channel structure (Procedural → Semantic → Episodic)
  - Natural language descriptions, not abbreviated stat dumps
  - Per-session evidence in readable tabular format
  - No redundancy: each signal appears once in its primary channel
"""

from __future__ import annotations

from .embedder import TextEmbedder
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

# Default number of episodes returned for query-adaptive retrieval
EPISODE_TOP_K = 5


class QueryAdaptiveRetriever:
    """Compose three-channel memory into clean behavioral profiles.

    When ``query`` is provided, episode retrieval becomes query-adaptive:
    the query is embedded and compared against stored episode embeddings
    to return only the most relevant episodes.
    """

    def retrieve(
        self,
        store: MemoryStore,
        query_type: str = "profile",
        attribute: str | None = None,
        query: str | None = None,
        episode_top_k: int = EPISODE_TOP_K,
        disabled_channels: set[str] | None = None,
    ) -> str:
        """Compose three-channel memory into a behavioral profile.

        Args:
            disabled_channels: Set of channel names to skip for ablation.
                Valid values: ``{"procedural", "semantic", "episodic"}``.
        """
        n = len(store.engrams)
        n_dev = sum(1 for v in store.deviation_flags.values() if v)
        skip = disabled_channels or set()

        parts: list[str] = []
        if "episodic" not in skip:
            parts.append(f"## Behavioral Memory Profile ({n} sessions, {n_dev} anomalous)")
        else:
            parts.append(f"## Behavioral Memory Profile ({n} sessions)")
        if "procedural" not in skip:
            parts.append(self._channel_procedural(store, skip_labels="semantic" in skip))
        if "semantic" not in skip:
            parts.append(self._channel_semantic(store, query=query))
        if "episodic" not in skip:
            parts.append(self._channel_episodic(store, n_dev, query=query, episode_top_k=episode_top_k))
        if "procedural" not in skip and "episodic" not in skip:
            parts.append(self._session_table(store, skip_semantic="semantic" in skip))

        return "\n\n".join(p for p in parts if p)

    # ── Channel 1: Procedural ───────────────────────────────────────

    def _channel_procedural(self, store: MemoryStore, skip_labels: bool = False) -> str:
        """Behavioral dimension profiles with natural-language descriptions.

        When *skip_labels* is True (Semantic channel disabled), output compact
        stats-only format — no tier labels like "sequential" / "comprehensive"
        and no workflow phase / dominant transition text (those are semantic
        interpretations of raw operation sequences).
        """
        cls_map = self._build_classification_map(store)
        agg = store.procedural_aggregate
        lines = ["### Channel 1: Procedural Patterns"]

        # A: Consumption
        rs = agg.get("reading_strategy", {})
        reads = rs.get("total_reads_mean", 0)
        search = rs.get("search_ratio_mean", 0)
        revisit = rs.get("revisit_ratio_mean", 0)
        browse = rs.get("browse_ratio_mean", 0)
        if skip_labels:
            lines.append("**A. Consumption**")
            lines.append(
                f"total_reads={reads:.1f}, search_ratio={search:.2f}, "
                f"revisit_ratio={revisit:.2f}, browse_ratio={browse:.2f}"
            )
        else:
            label = self._label(cls_map.get("reading_strategy", "unknown"))
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
        avg_len = od.get("avg_output_length_mean", 0)
        files = od.get("files_created_mean", 0)
        h_depth = tn.get("heading_max_depth_mean", 0)
        if skip_labels:
            lines.append("**B. Production**")
            lines.append(
                f"avg_output_length={avg_len:.0f}, files_created={files:.1f}, "
                f"heading_max_depth={h_depth:.0f}"
            )
        else:
            label = self._label(cls_map.get("output_detail", "unknown"))
            lines.append(f"**B. Production — {label}**")
            lines.append(f"Avg {avg_len:,.0f} chars across {files:.1f} files/task. Heading depth: H{h_depth:.0f}.")

        # C: Organization
        ds = agg.get("directory_style", {})
        nm = agg.get("naming", {})
        dirs = ds.get("dirs_created_mean", 0)
        depth = ds.get("max_dir_depth_mean", 0)
        moves = ds.get("files_moved_mean", 0)
        name_len = nm.get("avg_filename_length_mean", 0)
        if skip_labels:
            lines.append("**C. Organization**")
            lines.append(
                f"dirs_created={dirs:.1f}, max_dir_depth={depth:.1f}, "
                f"files_moved={moves:.1f}, avg_filename_length={name_len:.0f}"
            )
        else:
            label = self._label(cls_map.get("directory_style", "unknown"))
            lines.append(f"**C. Organization — {label}**")
            org_parts = [f"{dirs:.1f} dirs/task, depth {depth:.1f}"]
            if moves > 0.1:
                org_parts.append(f"{moves:.1f} file moves/task")
            org_parts.append(f"avg filename {name_len:.0f} chars")
            lines.append(self._join_sentences(org_parts))

        # D: Iteration
        es = agg.get("edit_strategy", {})
        vs = agg.get("version_strategy", {})
        edits = es.get("total_edits_mean", 0)
        avg_lines = es.get("avg_lines_changed_mean", 0)
        backup = vs.get("has_backup_behavior_true_ratio", 0)
        deletes = vs.get("total_deletes_mean", 0)
        if skip_labels:
            lines.append("**D. Iteration**")
            lines.append(
                f"total_edits={edits:.1f}, avg_lines_changed={avg_lines:.0f}, "
                f"backup_ratio={backup:.2f}, total_deletes={deletes:.1f}"
            )
        else:
            label = self._label(cls_map.get("edit_strategy", "unknown"))
            lines.append(f"**D. Iteration — {label}**")
            iter_parts = [f"{edits:.1f} edits/task, avg {avg_lines:.0f} lines/edit"]
            if backup > 0.01:
                iter_parts.append(f"backup in {backup:.0%} of tasks")
            else:
                iter_parts.append("no backup behavior")
            if deletes > 0.1:
                iter_parts.append(f"{deletes:.1f} deletions/task")
            lines.append(self._join_sentences(iter_parts))

        # E: Curation
        switch_rate = rs.get("context_switch_rate_mean", 0)
        if skip_labels:
            lines.append("**E. Curation**")
            lines.append(f"context_switch_rate={switch_rate:.2f}, total_deletes={deletes:.1f}")
        else:
            label = self._label(cls_map.get("working_style", "unknown"))
            lines.append(f"**E. Curation — {label}**")
            lines.append(f"Context switch rate: {switch_rate:.2f}.")

        # F: Cross-Modal
        cm = agg.get("cross_modal_behavior", {})
        tables = cm.get("has_tables_true_ratio", 0)
        images = cm.get("creates_image_files_true_ratio", 0)
        if skip_labels:
            lines.append("**F. Cross-Modal**")
            lines.append(f"table_ratio={tables:.2f}, image_ratio={images:.2f}")
        else:
            label = self._label(cls_map.get("cross_modal_behavior", "unknown"))
            lines.append(f"**F. Cross-Modal — {label}**")
            cm_parts = []
            if tables > 0.01:
                cm_parts.append(f"tables in {tables:.0%} of tasks")
            if images > 0.01:
                cm_parts.append(f"images in {images:.0%} of tasks")
            if not cm_parts:
                cm_parts.append("no tables or images created")
            lines.append(self._join_sentences(cm_parts))

        # Sequence summary (aggregated across sessions)
        # Skip when labels are stripped — phases and transitions are semantic interpretations
        if not skip_labels:
            phase_counts_agg: dict[str, int] = {}
            transition_counts: dict[str, int] = {}
            for eng in store.engrams:
                sf = eng.sequence_features
                if not sf:
                    continue
                for phase, count in sf.get("phase_counts", {}).items():
                    phase_counts_agg[phase] = phase_counts_agg.get(phase, 0) + count
                dom = sf.get("dominant_transition", "")
                if dom:
                    transition_counts[dom] = transition_counts.get(dom, 0) + 1

            if phase_counts_agg:
                phase_str = ", ".join(
                    f"{p}({c})" for p, c in
                    sorted(phase_counts_agg.items(), key=lambda x: -x[1])
                )
                lines.append(f"**Workflow phases:** {phase_str}")

            if transition_counts:
                top_transition = max(transition_counts, key=transition_counts.get)  # type: ignore[arg-type]
                lines.append(f"**Dominant transition:** {top_transition}")

        return "\n".join(lines)

    # ── Channel 2: Semantic ─────────────────────────────────────────

    def _channel_semantic(self, store: MemoryStore, query: str | None = None) -> str:
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

        # Unified content profile (LLM-synthesized from all trajectories)
        if store.content_profile:
            lines.append(f"**Content profile:** {store.content_profile}")

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

        # Query-adaptive content retrieval from embedded chunks
        if store.content_chunks and query:
            lines.append(self._retrieve_content_chunks(store, query))

        return "\n".join(lines)

    # ── Content chunk retrieval ──────────────────────────────────

    @staticmethod
    def _retrieve_content_chunks(
        store: MemoryStore,
        query: str,
        top_k: int = 5,
    ) -> str:
        """Retrieve content chunks most relevant to the query via cosine similarity."""
        embedder = TextEmbedder()
        query_vec = embedder.embed_query(query)

        scored: list[tuple[float, object]] = []
        for chunk in store.content_chunks:
            if chunk.embedding:
                sim = embedder.cosine_similarity(query_vec, chunk.embedding)
                scored.append((sim, chunk))

        scored.sort(key=lambda x: -x[0])

        lines = [f"**Relevant content ({min(top_k, len(scored))} chunks, query-adaptive):**"]
        for sim, chunk in scored[:top_k]:
            basename = chunk.source_path.rsplit("/", 1)[-1] if "/" in chunk.source_path else chunk.source_path
            preview = chunk.text.replace("\n", " ").strip()[:RETRIEVER_DISPLAY_CHARS]
            lines.append(f"  - [{chunk.chunk_type}] {basename} (sim={sim:.3f}): {preview}")

        return "\n".join(lines)

    # ── Episode retrieval helpers ──────────────────────────────────

    @staticmethod
    def _query_adaptive_episodes(
        all_eps: list[tuple],
        query: str,
        top_k: int,
    ) -> str:
        """Rank episodes by cosine similarity to query, return top-K full narratives."""
        embedder = TextEmbedder()
        query_vec = embedder.embed_query(query)

        # Use pre-computed embeddings if available, otherwise embed content now
        scored: list[tuple[float, object, str]] = []
        eps_needing_embed: list[int] = []
        for i, (ep, task_id) in enumerate(all_eps):
            if ep.embedding and any(v != 0 for v in ep.embedding):
                sim = embedder.cosine_similarity(query_vec, ep.embedding)
                scored.append((sim, ep, task_id))
            else:
                eps_needing_embed.append(i)

        # Batch-embed episodes without embeddings
        if eps_needing_embed:
            texts = [all_eps[i][0].content or all_eps[i][0].summary for i in eps_needing_embed]
            vecs = embedder.embed(texts)
            for idx, vec in zip(eps_needing_embed, vecs):
                ep, task_id = all_eps[idx]
                ep.embedding = vec  # cache for future use
                sim = embedder.cosine_similarity(query_vec, vec)
                scored.append((sim, ep, task_id))

        # Sort by similarity descending
        scored.sort(key=lambda x: -x[0])

        lines = [f"**Relevant episodes (top-{min(top_k, len(scored))}, query-adaptive):**"]
        for sim, ep, task_id in scored[:top_k]:
            title = ep.title or ep.summary
            lines.append(f"**[{task_id}] [{ep.start_idx}–{ep.end_idx}] ({ep.event_count} events) — {title}** (relevance: {sim:.3f})")
            if ep.content:
                lines.append(f"  {ep.content}")

        return "\n".join(lines)

    @staticmethod
    def _cluster_based_episodes(
        all_eps: list[tuple],
        episode_clusters: list[dict],
    ) -> str:
        """Group episodes by cluster, show top-K per cluster with full narratives."""
        # Build episode lookup: episode_id -> (episode, task_id)
        ep_lookup: dict[str, tuple] = {}
        for ep, task_id in all_eps:
            ep_lookup[ep.episode_id] = (ep, task_id)

        top_k = 3
        lines = [f"**Behavioral episodes ({len(episode_clusters)} clusters):**"]
        for cluster in episode_clusters[:5]:
            theme = cluster.get("theme", "")
            size = cluster.get("size", 0)
            lines.append(f"**Cluster {cluster.get('cluster_id', '?')} — {theme} ({size} episodes):**")

            # Collect episodes in this cluster, sorted by event_count desc
            cluster_eps = []
            for eid in cluster.get("episode_ids", []):
                if eid in ep_lookup:
                    cluster_eps.append(ep_lookup[eid])
            cluster_eps.sort(key=lambda x: -x[0].event_count)

            # Top-K, prefer diverse trajectories
            seen_traj: set[str] = set()
            selected = []
            for ep, task_id in cluster_eps:
                if ep.trajectory_id not in seen_traj and len(selected) < top_k:
                    seen_traj.add(ep.trajectory_id)
                    selected.append((ep, task_id))
            for ep, task_id in cluster_eps:
                if len(selected) >= top_k:
                    break
                if (ep, task_id) not in selected:
                    selected.append((ep, task_id))

            for ep, task_id in selected:
                title = ep.title or ep.summary
                lines.append(f"  [{task_id}] [{ep.start_idx}–{ep.end_idx}] ({ep.event_count} events) — {title}")
                if ep.content:
                    lines.append(f"  {ep.content}")

        return "\n".join(lines)

    # ── Channel 3: Episodic ─────────────────────────────────────────

    def _channel_episodic(
        self,
        store: MemoryStore,
        n_dev: int,
        query: str | None = None,
        episode_top_k: int = EPISODE_TOP_K,
    ) -> str:
        """Behavioral consistency, episode/trace clustering, and anomaly detection."""
        lines = ["### Channel 3: Episodic Consistency"]

        # Episode clustering (action-level) — query-adaptive or cluster-based
        all_eps: list[tuple] = []
        for eng in store.engrams:
            for ep in eng.semantic.episodes:
                all_eps.append((ep, eng.task_id))

        if all_eps and query:
            lines.append(self._query_adaptive_episodes(all_eps, query, episode_top_k))
        elif store.episode_clusters:
            lines.append(self._cluster_based_episodes(all_eps, store.episode_clusters))

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

        # Behavioral clusters
        if store.behavioral_clusters:
            lines.append(f"**Behavioral clusters ({len(store.behavioral_clusters)}):**")
            for cluster in store.behavioral_clusters:
                tids = cluster.get("trajectory_ids", [])
                task_ids = cluster.get("task_ids", [])
                tasks_str = ",".join(task_ids) if task_ids else ",".join(tids[:6])
                lines.append(f"  - Cluster {cluster.get('cluster_id', '?')} ({cluster.get('size', 0)} sessions): {tasks_str}")

        # Deviations with detail + LLM verification
        if n_dev > 0:
            lines.append(f"**Anomalous sessions ({n_dev}):**")
            for tid, is_dev in store.deviation_flags.items():
                if not is_dev:
                    continue
                details = store.deviation_details.get(tid, [])
                verification = store.llm_deviation_analysis.get(tid, "")
                if details:
                    top = details[0]
                    dim = top["dimension"].split("(")[-1].rstrip(")") if "(" in top["dimension"] else top["dimension"]
                    feat = top["feature"].split(".")[-1]
                    delta = top["delta"]
                    suffix = f" [{verification}]" if verification else ""
                    lines.append(f"  - {tid}: shifted in {dim} ({feat}, δ={delta:.1f}){suffix}")
                else:
                    suffix = f" [{verification}]" if verification else ""
                    lines.append(f"  - {tid}: deviation detected{suffix}")

        return "\n".join(lines)

    # ── Session Evidence Table ──────────────────────────────────────

    def _session_table(self, store: MemoryStore, skip_semantic: bool = False) -> str:
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

            # Key filenames (skip when semantic channel is disabled)
            if not skip_semantic:
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
