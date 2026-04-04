"""Stage 2: Cross-engram consolidation with perturbation detection.

Consolidates multiple Engrams into a MemoryStore with three channels:
  1. Procedural: aggregated statistics + patterns + L/M/R classifications
  2. Semantic: stratified content samples + episode clustering + merged filenames/dirs
  3. Episodic: fingerprint-based deviation scoring + behavioral clustering + LLM verification

Classification and pattern detection are performed by LLM (required).
"""

from __future__ import annotations

import json
import math
import statistics
from typing import Any

from .aggregation import FeatureAggregator
from .embedder import TextEmbedder
from .engram import ContentChunk, Engram, MemoryStore
from .fingerprint import (
    compute_deviations,
    detect_absences,
    locate_shifted_dimensions,
    normalize_fingerprints,
)
from .sampler import StratifiedSampler
from .schema import FINGERPRINT_FEATURES, build_classification_prompt
from .tuning import (
    BEHAVIORAL_CLUSTER_MAX,
    CONTENT_CHUNK_MAX,
    CONTENT_CHUNK_SIZE,
    EPISODE_CLUSTER_THRESHOLD,
    SEMANTIC_BUDGET,
)


class EngramConsolidator:
    """Consolidate engrams into a three-channel MemoryStore.

    Uses LLM for dimension classification and behavioral pattern detection.
    llm_fn is required — no deterministic fallback.
    """

    def __init__(
        self,
        semantic_budget: int = SEMANTIC_BUDGET,
        deviation_threshold: float = 1.5,
        llm_fn=None,
    ):
        if llm_fn is None:
            raise ValueError("llm_fn is required for EngramConsolidator")
        self._semantic_budget = semantic_budget
        self._deviation_threshold = deviation_threshold
        self._llm_fn = llm_fn
        self._sampler = StratifiedSampler(total_token_budget=semantic_budget)

    def consolidate(
        self,
        engrams: list[Engram],
        profile_id: str = "",
    ) -> MemoryStore:
        """Consolidate a list of engrams into a unified MemoryStore.

        Args:
            engrams: List of Engram objects (one per trajectory).
            profile_id: Profile identifier for the store.

        Returns:
            A fully populated MemoryStore.
        """
        if not engrams:
            return MemoryStore(profile_id=profile_id)

        # --- Episodic channel (compute first, needed for weighting) ---
        raw_fps = {e.trajectory_id: e.fingerprint for e in engrams}
        norm_fps = normalize_fingerprints(raw_fps)
        centroid, distances, deviation_flags = compute_deviations(
            norm_fps,
            threshold=self._deviation_threshold,
        )

        # Locate shifted dimensions for deviant engrams
        deviation_details: dict[str, list[dict]] = {}
        for eng in engrams:
            tid = eng.trajectory_id
            if deviation_flags.get(tid, False) and centroid:
                details = locate_shifted_dimensions(
                    norm_fps[tid],
                    centroid,
                    top_k=3,
                )
                deviation_details[tid] = details

        # --- Channel 1: Procedural ---
        # Weight deviant engrams lower for aggregation
        weighted_features = self._weight_features(engrams, deviation_flags)
        aggregator = FeatureAggregator(weighted_features)
        procedural_aggregate = aggregator.aggregate_all()

        # Classification via LLM
        dimension_classifications, behavioral_patterns = self._llm_classify(
            procedural_aggregate,
        )
        if dimension_classifications is None:
            raise RuntimeError("LLM classification failed — check llm_fn and model availability")

        # --- Channel 2: Semantic ---
        # Prefer non-deviant engrams for representative samples
        non_deviant = [e for e in engrams if not deviation_flags.get(e.trajectory_id, False)]
        sample_pool = non_deviant if non_deviant else engrams
        representative_samples = self._sampler.select_samples(sample_pool)

        # Merge all filenames and dir structure (deduplicated)
        all_filenames = self._merge_filenames(engrams)
        dir_structure_union = self._merge_dir_structure(engrams)

        # Collect LLM behavioral narratives (skip deviant trajectories)
        llm_narratives: dict[str, dict] = {}
        for eng in engrams:
            if eng.semantic.llm_encoding and not deviation_flags.get(eng.trajectory_id, False):
                llm_narratives[eng.trajectory_id] = eng.semantic.llm_encoding

        # --- Channel 2 (continued): Content chunking + embedding ---
        content_chunks = self._chunk_and_embed_content(engrams, deviation_flags)

        # --- Channel 2 (continued): LLM content profile synthesis ---
        content_profile = self._llm_synthesize_content_profile(llm_narratives)

        # --- Channel 3: Episodic ---
        all_features = [e.procedural for e in engrams]
        consistency_flags = FeatureAggregator.aggregate_episodic(all_features)
        absence_flags = detect_absences(all_features)

        # Channel 3: episode clustering (action-level) + behavioral clustering (trace-level)
        episode_clusters = self._cluster_episodes(engrams)
        behavioral_clusters, cluster_centroids = self._cluster_engrams(engrams)
        llm_deviation_analysis = self._llm_verify_deviations(engrams, deviation_flags)

        return MemoryStore(
            profile_id=profile_id,
            engrams=engrams,
            # Channel 1
            procedural_aggregate=procedural_aggregate,
            behavioral_patterns=behavioral_patterns,
            dimension_classifications=dimension_classifications,
            # Channel 2
            representative_samples=representative_samples,
            all_filenames=all_filenames,
            dir_structure_union=dir_structure_union,
            llm_narratives=llm_narratives,
            content_chunks=content_chunks,
            content_profile=content_profile,
            # Channel 3
            centroid=centroid,
            per_session_distances=distances,
            deviation_flags=deviation_flags,
            deviation_details=deviation_details,
            consistency_flags=consistency_flags,
            absence_flags=absence_flags,
            episode_clusters=episode_clusters,
            behavioral_clusters=behavioral_clusters,
            cluster_centroids=cluster_centroids,
            llm_deviation_analysis=llm_deviation_analysis,
        )

    def _weight_features(
        self,
        engrams: list[Engram],
        deviation_flags: dict[str, bool],
    ) -> list[dict[str, Any]]:
        """Return procedural features, duplicating non-deviant entries for weighting.

        Simple approach: include non-deviant engrams twice, deviant once.
        This dilutes the noise from perturbed trajectories in aggregation.
        """
        features = []
        for eng in engrams:
            features.append(eng.procedural)
            if not deviation_flags.get(eng.trajectory_id, False):
                # Non-deviant: add again for extra weight
                features.append(eng.procedural)
        return features

    def _merge_filenames(self, engrams: list[Engram]) -> list[str]:
        """Merge and deduplicate filenames across all engrams."""
        seen: set[str] = set()
        result: list[str] = []
        for eng in engrams:
            for fn in eng.semantic.created_filenames:
                if fn not in seen:
                    seen.add(fn)
                    result.append(fn)
        return result

    def _merge_dir_structure(self, engrams: list[Engram]) -> list[str]:
        """Merge and deduplicate directory structure diffs."""
        seen: set[str] = set()
        result: list[str] = []
        for eng in engrams:
            for d in eng.semantic.dir_structure_diff:
                if d not in seen:
                    seen.add(d)
                    result.append(d)
        return result

    # --- Channel 2: Episode clustering ---

    def _cluster_episodes(self, engrams: list[Engram]) -> list[dict[str, Any]]:
        """Collect all episodes, embed summaries via Cohere, cluster by similarity.

        Returns:
            [{cluster_id, theme, episode_ids, size}]
        """
        # Collect all episodes across engrams
        all_episodes = []
        for eng in engrams:
            for ep in eng.semantic.episodes:
                all_episodes.append(ep)

        if not all_episodes:
            return []

        # Embed episode content (use content narrative if available, fall back to summary)
        embedder = TextEmbedder()
        texts = [ep.content if ep.content else ep.summary for ep in all_episodes]
        embeddings = embedder.embed(texts)

        # Store embeddings back on episodes
        for ep, emb in zip(all_episodes, embeddings):
            ep.embedding = emb

        # Agglomerative clustering by cosine distance
        n = len(all_episodes)
        if n <= 1:
            return [{
                "cluster_id": 0,
                "theme": all_episodes[0].summary if all_episodes else "",
                "episode_ids": [ep.episode_id for ep in all_episodes],
                "trajectory_ids": list({ep.trajectory_id for ep in all_episodes}),
                "size": n,
            }]

        # Average-linkage agglomerative clustering
        sim_threshold = EPISODE_CLUSTER_THRESHOLD

        # Pre-compute pairwise cosine similarity matrix
        sim_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                s = embedder.cosine_similarity(embeddings[i], embeddings[j])
                sim_matrix[i][j] = s
                sim_matrix[j][i] = s

        # Each item starts in its own cluster
        clusters_map: dict[int, list[int]] = {i: [i] for i in range(n)}

        while len(clusters_map) > 1:
            # Find best pair to merge (highest average similarity)
            best_sim = -1.0
            best_pair: tuple[int, int] | None = None
            cids = list(clusters_map.keys())
            for ii in range(len(cids)):
                for jj in range(ii + 1, len(cids)):
                    ci, cj = cids[ii], cids[jj]
                    total = sum(
                        sim_matrix[a][b]
                        for a in clusters_map[ci]
                        for b in clusters_map[cj]
                    )
                    count = len(clusters_map[ci]) * len(clusters_map[cj])
                    avg = total / count if count > 0 else 0.0
                    if avg > best_sim:
                        best_sim = avg
                        best_pair = (ci, cj)

            if best_sim < sim_threshold or best_pair is None:
                break

            # Merge
            ci, cj = best_pair
            clusters_map[ci].extend(clusters_map[cj])
            del clusters_map[cj]

        result = []
        for cluster_id, (_, indices) in enumerate(sorted(clusters_map.items())):
            eps = [all_episodes[i] for i in indices]
            # Use the title of the largest episode as theme
            largest = max(eps, key=lambda e: e.event_count)
            result.append({
                "cluster_id": cluster_id,
                "theme": largest.title or largest.summary,
                "episode_ids": [ep.episode_id for ep in eps],
                "trajectory_ids": sorted(set(ep.trajectory_id for ep in eps)),
                "size": len(eps),
            })

        return sorted(result, key=lambda c: -c["size"])

    # --- Channel 2 (continued): LLM content profile synthesis ---

    def _llm_synthesize_content_profile(
        self,
        llm_narratives: dict[str, dict[str, Any]],
    ) -> str:
        """Merge per-trajectory behavioral JSONs into a unified content profile.

        Takes the N individual LLM narratives (each describing writing style,
        content structure, detail level, etc. for one trajectory) and produces
        a single coherent natural-language profile of the user's content habits.
        """
        if not llm_narratives or not self._llm_fn:
            return ""

        narrative_summaries: list[str] = []
        for tid, narrative in llm_narratives.items():
            parts = []
            for key in ("writing_style", "content_structure", "detail_level",
                        "work_approach", "notable_habits"):
                val = narrative.get(key, "")
                if isinstance(val, list):
                    val = "; ".join(str(v) for v in val)
                if val:
                    parts.append(f"{key}: {val}")
            if parts:
                narrative_summaries.append(f"[{tid}] {' | '.join(parts)}")

        if not narrative_summaries:
            return ""

        prompt = (
            "Below are per-session behavioral narratives extracted from a single user's "
            "file-system work trajectories. Each entry describes the user's writing style, "
            "content structure, detail level, and work approach in one task.\n\n"
            + "\n".join(narrative_summaries)
            + "\n\n## Instructions\n"
            "Synthesize these into a SINGLE unified content profile (3-6 sentences) that describes:\n"
            "1. The user's consistent writing style and preferences across sessions\n"
            "2. Any recurring content structure patterns (headings, tables, lists)\n"
            "3. Their typical level of detail and thoroughness in output\n"
            "4. Notable habits or distinctive content behaviors\n\n"
            "Focus on PATTERNS that appear across multiple sessions. "
            "Ignore one-off behaviors unless they are striking. "
            "Write in third person ('This user...' or 'The user...')."
        )

        try:
            response = self._llm_fn(prompt)
            return response.strip()
        except Exception as e:
            print(f"  [filegramos] content profile synthesis failed: {e}")
            return ""

    # --- Channel 2 (continued): Content chunking + embedding ---

    def _chunk_and_embed_content(
        self,
        engrams: list[Engram],
        deviation_flags: dict[str, bool],
    ) -> list[ContentChunk]:
        """Chunk user-created content and edit diffs, then embed via Cohere.

        Prioritizes non-deviant trajectories. Each chunk is stored with its
        embedding for query-time cosine-similarity retrieval.
        """
        raw_chunks: list[ContentChunk] = []

        sorted_engrams = sorted(
            engrams,
            key=lambda e: (deviation_flags.get(e.trajectory_id, False), -e.importance_score),
        )

        for eng in sorted_engrams:
            tid = eng.trajectory_id
            for cs in eng.semantic.created_files:
                text = cs.full_content or cs.content_preview
                if not text or len(text.strip()) < 50:
                    continue
                for i in range(0, len(text), CONTENT_CHUNK_SIZE):
                    chunk_text = text[i:i + CONTENT_CHUNK_SIZE].strip()
                    if len(chunk_text) < 50:
                        continue
                    raw_chunks.append(ContentChunk(
                        text=chunk_text,
                        source_path=cs.path,
                        trajectory_id=tid,
                        chunk_type="created_file",
                    ))

            for ec in eng.semantic.edit_chains:
                text = ec.full_diff or ec.diff_preview
                if not text or len(text.strip()) < 50:
                    continue
                raw_chunks.append(ContentChunk(
                    text=text[:CONTENT_CHUNK_SIZE].strip(),
                    source_path=ec.path,
                    trajectory_id=tid,
                    chunk_type="edit_diff",
                ))

        raw_chunks = raw_chunks[:CONTENT_CHUNK_MAX]

        if not raw_chunks:
            return []

        embedder = TextEmbedder()
        texts = [c.text for c in raw_chunks]
        try:
            vectors = embedder.embed(texts)
            for chunk, vec in zip(raw_chunks, vectors):
                chunk.embedding = vec
            print(f"  [filegramos] embedded {len(raw_chunks)} content chunks")
        except Exception as e:
            print(f"  [filegramos] content embedding failed: {e}")

        return raw_chunks

    # --- Channel 3: Behavioral clustering + LLM verification ---

    def _cluster_engrams(
        self,
        engrams: list[Engram],
    ) -> tuple[list[dict[str, Any]], list[list[float]]]:
        """Cluster engrams by fingerprint similarity (hierarchical, Ward-like).

        Returns:
            (clusters, centroids)
            clusters: [{cluster_id, trajectory_ids, centroid_idx, size}]
            centroids: list of centroid vectors
        """
        if len(engrams) <= 1:
            return (
                [{
                    "cluster_id": 0,
                    "trajectory_ids": [e.trajectory_id for e in engrams],
                    "size": len(engrams),
                }],
                [engrams[0].fingerprint] if engrams else [],
            )

        n = len(engrams)
        fps = [e.fingerprint for e in engrams]
        max_clusters = min(BEHAVIORAL_CLUSTER_MAX, n)

        # Simple k-means-style clustering with fingerprints
        # Start: each engram in its own cluster, then merge closest pairs
        # until we reach max_clusters
        cluster_assignments = list(range(n))

        def _centroid(indices: list[int]) -> list[float]:
            dim = len(fps[0])
            return [
                statistics.mean(fps[idx][d] for idx in indices)
                for d in range(dim)
            ]

        def _dist(a: list[float], b: list[float]) -> float:
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

        # Merge until max_clusters
        while True:
            active: dict[int, list[int]] = {}
            for idx, cid in enumerate(cluster_assignments):
                active.setdefault(cid, []).append(idx)
            if len(active) <= max_clusters:
                break

            # Find closest pair of clusters
            cluster_ids = sorted(active.keys())
            centroids_tmp = {cid: _centroid(active[cid]) for cid in cluster_ids}
            best_dist = float("inf")
            best_pair = (cluster_ids[0], cluster_ids[1])
            for i_idx in range(len(cluster_ids)):
                for j_idx in range(i_idx + 1, len(cluster_ids)):
                    ci, cj = cluster_ids[i_idx], cluster_ids[j_idx]
                    d = _dist(centroids_tmp[ci], centroids_tmp[cj])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (ci, cj)

            # Merge
            merge_from, merge_to = best_pair[1], best_pair[0]
            for idx in range(n):
                if cluster_assignments[idx] == merge_from:
                    cluster_assignments[idx] = merge_to

        # Build final clusters
        final: dict[int, list[int]] = {}
        for idx, cid in enumerate(cluster_assignments):
            final.setdefault(cid, []).append(idx)

        result_clusters = []
        result_centroids = []
        for cluster_id, (_, indices) in enumerate(sorted(final.items(), key=lambda x: -len(x[1]))):
            cent = _centroid(indices)
            result_centroids.append(cent)
            result_clusters.append({
                "cluster_id": cluster_id,
                "trajectory_ids": [engrams[i].trajectory_id for i in sorted(indices)],
                "task_ids": sorted(set(engrams[i].task_id for i in indices)),
                "size": len(indices),
            })

        return result_clusters, result_centroids

    def _llm_verify_deviations(
        self,
        engrams: list[Engram],
        deviation_flags: dict[str, bool],
    ) -> dict[str, str]:
        """For deviant engrams, LLM distinguishes:
        - 'task_dependent': different task type → different behavior (expected)
        - 'genuine_perturbation': unexpected shift
        - 'uncertain'

        Uses task_id + sequence_features as context.
        """
        deviant = [e for e in engrams if deviation_flags.get(e.trajectory_id, False)]
        if not deviant or not self._llm_fn:
            return {}

        # Build context about non-deviant engrams for comparison
        non_deviant_tasks = sorted(set(
            e.task_id for e in engrams
            if not deviation_flags.get(e.trajectory_id, False)
        ))

        result: dict[str, str] = {}
        for eng in deviant:
            # Build a compact description
            phase_pattern = eng.sequence_features.get("phase_pattern", "unknown")
            dominant = eng.sequence_features.get("dominant_transition", "unknown")

            prompt = (
                "A behavioral trajectory is flagged as an anomaly compared to the user's baseline. "
                "Determine if this deviation is TASK-DEPENDENT (the task type naturally elicits different behavior) "
                "or a GENUINE perturbation (unexpected behavioral shift).\n\n"
                f"## Deviant trajectory\n"
                f"- Trajectory: {eng.trajectory_id}\n"
                f"- Task: {eng.task_id}\n"
                f"- Phase pattern: {phase_pattern}\n"
                f"- Dominant transition: {dominant}\n"
                f"- Event count: {eng.behavioral_event_count}\n\n"
                f"## Baseline tasks (non-deviant)\n"
                f"- Tasks: {', '.join(non_deviant_tasks)}\n\n"
                "## Instructions\n"
                "Respond with ONLY one of: task_dependent | genuine_perturbation | uncertain\n"
                "If the task type (e.g., organize vs understand) naturally explains the different behavior, "
                "say 'task_dependent'. If the shift is unexpected given the task, say 'genuine_perturbation'."
            )

            try:
                resp = self._llm_fn(prompt, system_prompt=None)
                text = resp.strip().lower()
                if "task_dependent" in text:
                    result[eng.trajectory_id] = "task_dependent"
                elif "genuine_perturbation" in text:
                    result[eng.trajectory_id] = "genuine_perturbation"
                else:
                    result[eng.trajectory_id] = "uncertain"
            except Exception:
                result[eng.trajectory_id] = "uncertain"

        return result

    # --- LLM-based classification ---

    def _llm_classify(
        self,
        procedural_aggregate: dict[str, dict[str, Any]],
    ) -> tuple[list[str] | None, list[str] | None]:
        """Use LLM to classify dimensions and detect behavioral patterns.

        Returns (classifications, patterns) or (None, None) on failure.
        """
        stats_text = self._format_stats_for_llm(procedural_aggregate)
        n = next(iter(procedural_aggregate.values()), {}).get("_n_trajectories", "?")

        prompt = build_classification_prompt(
            n_trajectories=n,
            stats_text=stats_text,
        )

        try:
            response = self._llm_fn(prompt)
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                print("  [filegramos] LLM classification: no JSON found, falling back")
                return None, None

            data = json.loads(text[start : end + 1])

            # Convert classifications to standard string format:
            #   "attribute → classification [CONFIDENCE]: evidence"
            classifications = []
            for c in data.get("classifications", []):
                attr = c.get("attribute", "?")
                cls = c.get("classification", "?")
                conf = c.get("confidence", "LOW")
                evidence = c.get("evidence", "")
                classifications.append(f"{attr} → {cls} [{conf}]: {evidence}")

            patterns = data.get("patterns", [])

            if classifications:
                print(f"  [filegramos] LLM classification: {len(classifications)} dims, {len(patterns)} patterns")
                return classifications, patterns

            return None, None

        except Exception as e:
            print(f"  [filegramos] LLM classification failed: {e}, falling back")
            return None, None

    @staticmethod
    def _format_stats_for_llm(
        procedural_aggregate: dict[str, dict[str, Any]],
    ) -> str:
        """Format procedural aggregate statistics as readable text for LLM."""
        lines: list[str] = []
        for attr_name, features in procedural_aggregate.items():
            lines.append(f"\n### {attr_name}")
            n = features.get("_n_trajectories", "?")
            lines.append(f"  (based on {n} trajectories)")
            for key, value in sorted(features.items()):
                if key.startswith("_"):
                    continue
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.3f}")
                elif isinstance(value, dict):
                    lines.append(f"  {key}: {value}")
                else:
                    lines.append(f"  {key}: {value}")
        return "\n".join(lines)
