"""EverMemOS adapter: faithful reproduction of the real EverMemOS pipeline.

EverMemOS (https://github.com/EverMind-AI/EverMemOS) is a self-organizing
memory OS using an engram-inspired pipeline. This adapter faithfully
reproduces the real architecture, adapted for file-system behavioral
trajectories instead of conversations:

  Stage 0: MemCell Boundary Detection — LLM-based activity segmentation
           (real: ConvMemCellExtractor with CONV_BOUNDARY_DETECTION_PROMPT)
  Stage 1: Multi-type Memory Extraction
           1a. Episode Memory — narrative title + content + summary
               (real: EpisodeMemoryExtractor with EPISODE_GENERATION_PROMPT)
           1b. Profile Memory — multi-part behavioral extraction
               (real: ProfileMemoryExtractor with Part1 + Part2 prompts)
  Stage 2: Embedding-based Clustering — cosine similarity + centroid update
           (real: ClusterManager with vectorize service)
  Stage 3: Profile Consolidation — merge patterns, resolve contradictions
           (real: ProfileMemoryMerger with highest-level strategy)

Without llm_fn, falls back to simulated context_switch segmentation.

Registered as "evermemos" for benchmark comparison.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from .base import BaseAdapter
from .registry import register_adapter

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Stage 0: MemCell Boundary Detection Prompt
# Adapted from real EverMemOS CONV_BOUNDARY_DETECTION_PROMPT
# (src/memory_layer/prompts/en/conv_prompts.py)
# ---------------------------------------------------------------------------

BOUNDARY_DETECTION_PROMPT = """\
You are a behavioral episode boundary detection expert. You need to determine \
if the newly added file-system events should end the current behavioral \
episode and start a new one.

Current event history in this episode:
{event_history}

Newly added events:
{new_events}

Please carefully analyze the following aspects to determine if a new episode \
should begin:

1. **Activity Type Change** (Highest Priority):
   - Have the new events shifted to a fundamentally different activity? \
(e.g., from reading/exploring files to creating new content, or from \
writing to reorganizing)
   - Has the user moved from one distinct work phase to another?

2. **File Context Transition**:
   - Are the new events operating on a completely different set of files?
   - Has the user switched from one functional area to another?

3. **Behavioral Pattern Shift**:
   - Has the operation pattern changed significantly? (e.g., from many \
small reads to bulk writes, or from sequential processing to search-based)

4. **Structural Signals**:
   - Are there context_switch events indicating deliberate transitions?
   - Is there a shift from consumption (read/browse/search) to production \
(write/edit/create)?

**Rules**:
- Keep related operations together (e.g., reading files THEN writing a \
summary based on them = ONE episode)
- Split when the user starts a genuinely different activity
- Aim for episodes with 5-25 meaningful events each
- When in doubt, keep events together rather than over-splitting

Return JSON:
{{
  "reasoning": "One sentence summary of your reasoning",
  "should_end": true/false,
  "confidence": 0.0-1.0,
  "topic_summary": "If should_end=true, summarize the current episode's \
core activity, otherwise empty"
}}"""

# ---------------------------------------------------------------------------
# Stage 1a: Episode Memory Extraction Prompt
# Adapted from real EverMemOS EPISODE_GENERATION_PROMPT
# (src/memory_layer/prompts/en/episode_mem_prompts.py)
# ---------------------------------------------------------------------------

EPISODE_EXTRACTION_PROMPT = """\
You are converting file-system behavioral events into an episodic memory. \
Create a concise factual record of what the user did in this behavioral \
episode.

Task: {task_id}
Events in this episode:
{episode_events}

Follow these principles:
1. Each episode should be a complete, independent behavioral unit
2. Preserve all important information: file names, operations, patterns
3. Use declarative language describing what the user did, not event format
4. Highlight key behavioral choices (what they read first, how they organized, \
what they created)

Return JSON:
{{
  "title": "A concise descriptive title summarizing the activity (10-20 words)",
  "content": "A factual third-person narrative of the user's behavior in this \
episode. Include specific file names, operation sequences, and behavioral \
patterns observed. 3-8 sentences.",
  "summary": "One-sentence summary of the episode"
}}"""

# ---------------------------------------------------------------------------
# Stage 1b: Profile Memory Extraction Prompts (Multi-part)
# Adapted from real EverMemOS Part1 + Part2 prompts
# (src/memory_layer/prompts/en/profile_mem_part1_prompts.py)
# (src/memory_layer/prompts/en/profile_mem_part2_prompts.py)
# ---------------------------------------------------------------------------

PROFILE_PART1_PROMPT = """\
You are extracting behavioral profile traits from file-system events. \
Analyze the following episode to extract explicit behavioral evidence.

Task: {task_id}
Episode events:
{episode_events}

Extract the following dimensions (ONLY if explicitly evidenced):

1. **consumption_pattern**: How the user explores/reads files \
(sequential deep reading, targeted search, breadth-first browsing)
2. **production_style**: Output characteristics \
(comprehensive/detailed, balanced, minimal/concise)
3. **organization_preference**: File/directory management approach \
(deeply nested, adaptive, flat)
4. **iteration_strategy**: Edit behavior \
(incremental small edits, balanced, bulk rewrites)

For each dimension found, provide:
- The observed pattern
- Specific evidence (file names, operation counts, sequences)

Return JSON:
{{
  "traits": [
    {{
      "dimension": "<dimension_name>",
      "pattern": "<observed behavioral pattern>",
      "evidence": "<specific operations/files that demonstrate this>"
    }}
  ]
}}

Only extract traits with clear evidence. Quality over quantity."""

PROFILE_PART2_PROMPT = """\
You are extracting higher-level work habit traits from file-system events. \
Focus on workflow patterns and cross-cutting behavioral characteristics.

Task: {task_id}
Episode events:
{episode_events}

Extract the following dimensions (ONLY if explicitly evidenced):

1. **work_rhythm**: Workspace curation pattern \
(selective pruning, preservative accumulation, pragmatic moderate cleanup)
2. **cross_modal_behavior**: Use of visual/structured materials \
(tables, charts, images, structured data files)
3. **naming_convention**: File naming patterns observed \
(descriptive long, short abbreviation, date prefixed, etc.)
4. **version_strategy**: How the user handles file versions \
(keeps history/backups, archives old, overwrites directly)

For each dimension found, provide:
- The observed pattern
- Specific evidence

Return JSON:
{{
  "traits": [
    {{
      "dimension": "<dimension_name>",
      "pattern": "<observed behavioral pattern>",
      "evidence": "<specific operations/files that demonstrate this>"
    }}
  ]
}}

Only extract traits with clear evidence. Quality over quantity."""

# ---------------------------------------------------------------------------
# Stage 3: Inference Prompt
# ---------------------------------------------------------------------------

INFERENCE_PROMPT = """\
You are analyzing consolidated memories from EverMemOS, built from a user's \
file-system behavioral trajectories. EverMemOS organizes memories through a \
multi-stage pipeline: MemCell segmentation, episode extraction, profile \
extraction, embedding-based clustering, and profile consolidation.

## Clustered Episodes
{episodes}

## Consolidated Profile Patterns
{patterns}

## Contradiction Resolutions
{contradictions}

Based on these consolidated memory traces, infer the user's work habit \
profile for:
{attributes_list}

Respond in JSON format:
{{
  "inferred_profile": {{
    "<attribute_name>": {{
      "value": "<inferred value>",
      "justification": "<brief reasoning citing specific episode/pattern evidence>"
    }},
    ...
  }}
}}"""

FALLBACK_INFERENCE_PROMPT = """\
You are analyzing engram-like memory traces extracted by EverMemOS from a \
user's file-system behavioral trajectories. EverMemOS organizes memories \
through event boundaries and multi-level consolidation.

Engram traces (organized by event segments):
{engram_traces}

Based on these memory traces, infer the user's work habit profile for:
{attributes_list}

Respond in JSON format:
{{
  "inferred_profile": {{
    "<attribute_name>": {{
      "value": "<inferred value>",
      "justification": "<brief reasoning>"
    }},
    ...
  }}
}}"""


# ---------------------------------------------------------------------------
# Embedding utilities (Cohere, matching real EverMemOS vectorize service)
# ---------------------------------------------------------------------------


def _call_embedding_api(texts: list[str], input_type: str = "search_document") -> list[list[float]] | None:
    """Call Cohere Embed API for episode clustering.

    Mirrors the real EverMemOS vectorize service which uses embedding vectors
    for semantic clustering in ClusterManager.
    """
    import httpx

    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        return None

    url = "https://api.cohere.com/v2/embed"
    all_embeddings: list[list[float]] = []
    batch_size = 96

    for i in range(0, len(texts), batch_size):
        batch = [t[:4096] for t in texts[i : i + batch_size]]
        for attempt in range(3):
            try:
                resp = httpx.post(
                    url,
                    json={
                        "texts": batch,
                        "model": "embed-v4.0",
                        "input_type": input_type,
                        "embedding_types": ["float"],
                    },
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=60.0,
                )
                if resp.status_code == 429:
                    wait = 10 * (attempt + 1)
                    print(f"    [evermemos] Cohere rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    print(f"    [evermemos] Cohere API error {resp.status_code}: {resp.text[:200]}")
                    return None
                data = resp.json()
                all_embeddings.extend(data["embeddings"]["float"])
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    [evermemos] Cohere embedding failed: {e}")
                    return None
                time.sleep(5)

    return all_embeddings


# ---------------------------------------------------------------------------
# Helper: parse LLM JSON response
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> dict | None:
    """Parse JSON from LLM response, handling markdown fences and trailing commas."""
    text = text.strip()
    if "```" in text:
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.strip().startswith("```"))
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    text = text[start : end + 1]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_adapter("evermemos")
class EverMemOSAdapter(BaseAdapter):
    """Faithful reproduction of the EverMemOS memory pipeline.

    Real mode (llm_fn set):
      Stage 0: LLM-based MemCell boundary detection (event segmentation)
      Stage 1a: Episode memory extraction (title + content + summary)
      Stage 1b: Multi-part profile extraction (Part1: behavioral + Part2: workflow)
      Stage 2: Embedding-based episode clustering (Cohere + cosine similarity)
      Stage 3: Profile consolidation (merge patterns, resolve contradictions)

    Fallback mode (no llm_fn):
      Segment events by context_switch boundaries.
    """

    # Real EverMemOS force-split thresholds (from ConvMemCellExtractor)
    MAX_EVENTS_PER_MEMCELL = 30  # analogous to 50-message limit
    CLUSTER_SIMILARITY_THRESHOLD = 0.7  # cosine similarity for clustering

    def __init__(self, config: dict[str, Any] | None = None, llm_fn=None):
        super().__init__(config, llm_fn=llm_fn)
        self.name = "evermemos"
        # MemCells (Stage 0 output)
        self._memcells: list[dict[str, Any]] = []
        # Episode memories (Stage 1a output)
        self._episodes: list[dict[str, Any]] = []
        # Profile traits (Stage 1b output)
        self._profile_traits: list[dict[str, Any]] = []
        # Clusters (Stage 2 output)
        self._clusters: list[dict[str, Any]] = []
        # Consolidated (Stage 3 output)
        self._consolidated_patterns: list[dict[str, Any]] = []
        self._contradictions: list[dict[str, Any]] = []
        # Fallback state
        self._engram_traces: list[dict[str, Any]] = []

    @property
    def _use_real(self) -> bool:
        return self.llm_fn is not None

    # ==================================================================
    # Stage 0: MemCell Boundary Detection
    # Mirrors real ConvMemCellExtractor._detect_boundary()
    # ==================================================================

    def _detect_boundary(self, history_narrative: str, new_narrative: str) -> dict[str, Any]:
        """Use LLM to detect if new events should start a new MemCell.

        Returns dict with should_end, confidence, topic_summary.
        """
        prompt = BOUNDARY_DETECTION_PROMPT.format(
            event_history=history_narrative or "(start of task)",
            new_events=new_narrative,
        )
        try:
            response = self._call_llm(prompt, phase="ingest")
            data = _parse_json(response)
            if data:
                return data
        except Exception as e:
            print(f"    [evermemos] Boundary detection error: {e}")
        return {"should_end": False, "confidence": 0.0, "topic_summary": ""}

    def _create_memcells(self, events: list[dict[str, Any]], task_id: str) -> list[dict[str, Any]]:
        """Segment events into MemCells using LLM boundary detection.

        Mirrors real ConvMemCellExtractor with:
        - LLM-based boundary detection per event window
        - Force split at MAX_EVENTS_PER_MEMCELL (analogous to token/msg limits)
        - Smart windowing: evaluate boundaries every 5 events
        """
        if not events:
            return []

        memcells = []
        current_events: list[dict[str, Any]] = []
        eval_window = 5  # evaluate boundary every N events

        for i in range(0, len(events), eval_window):
            window = events[i : i + eval_window]
            current_events.extend(window)

            # Force split check (analogous to real 8192-token / 50-msg limit)
            force_split = len(current_events) >= self.MAX_EVENTS_PER_MEMCELL

            if force_split:
                topic = f"Activity segment in {task_id} ({len(current_events)} events)"
                memcells.append(
                    {
                        "task_id": task_id,
                        "events": list(current_events),
                        "topic_summary": topic,
                        "boundary_type": "force_split",
                    }
                )
                current_events = []
                continue

            # LLM boundary detection (only if enough history)
            if len(current_events) > eval_window:
                history_events = current_events[: -len(window)]
                history_narrative = self.events_to_narrative(history_events, max_content_chars=200)
                new_narrative = self.events_to_narrative(window, max_content_chars=200)

                result = self._detect_boundary(history_narrative, new_narrative)

                if result.get("should_end") and result.get("confidence", 0) >= 0.6:
                    # Split: current episode is everything before the window
                    memcells.append(
                        {
                            "task_id": task_id,
                            "events": list(history_events),
                            "topic_summary": result.get("topic_summary", ""),
                            "boundary_type": "llm_detected",
                        }
                    )
                    current_events = list(window)

        # Remaining events become the last MemCell
        if current_events:
            memcells.append(
                {
                    "task_id": task_id,
                    "events": list(current_events),
                    "topic_summary": f"Final segment in {task_id}",
                    "boundary_type": "end_of_task",
                }
            )

        return memcells

    # ==================================================================
    # Stage 1a: Episode Memory Extraction
    # Mirrors real EpisodeMemoryExtractor
    # ==================================================================

    def _extract_episode(self, memcell: dict[str, Any]) -> dict[str, Any] | None:
        """Extract episode memory from a MemCell.

        Returns dict with title, content, summary, task_id.
        """
        events_narrative = self.events_to_narrative(memcell["events"], max_content_chars=300)
        prompt = EPISODE_EXTRACTION_PROMPT.format(
            task_id=memcell["task_id"],
            episode_events=events_narrative,
        )
        try:
            response = self._call_llm(prompt, phase="ingest")
            data = _parse_json(response)
            if data and data.get("title") and data.get("content"):
                data["task_id"] = memcell["task_id"]
                data["n_events"] = len(memcell["events"])
                if not data.get("summary"):
                    data["summary"] = data["content"][:200]
                return data
        except Exception as e:
            print(f"    [evermemos] Episode extraction error: {e}")
        return None

    # ==================================================================
    # Stage 1b: Multi-part Profile Extraction
    # Mirrors real ProfileMemoryExtractor (Part1 + Part2)
    # ==================================================================

    def _extract_profile(self, memcell: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract profile traits from a MemCell using 2-part prompts.

        Part 1: consumption, production, organization, iteration
        Part 2: work_rhythm, cross_modal, naming, version_strategy

        Returns list of trait dicts.
        """
        events_narrative = self.events_to_narrative(memcell["events"], max_content_chars=300)
        all_traits = []

        for prompt_template in (PROFILE_PART1_PROMPT, PROFILE_PART2_PROMPT):
            prompt = prompt_template.format(
                task_id=memcell["task_id"],
                episode_events=events_narrative,
            )
            try:
                response = self._call_llm(prompt, phase="ingest")
                data = _parse_json(response)
                if data and "traits" in data:
                    for trait in data["traits"]:
                        trait["task_id"] = memcell["task_id"]
                    all_traits.extend(data["traits"])
            except Exception as e:
                print(f"    [evermemos] Profile extraction error: {e}")

        return all_traits

    # ==================================================================
    # Stage 2: Embedding-based Clustering
    # Mirrors real ClusterManager with cosine similarity + centroids
    # ==================================================================

    def _cluster_episodes(self) -> None:
        """Cluster episodes by semantic similarity using embeddings.

        Mirrors real ClusterManager:
        - Embed episode content via Cohere (vectorize service)
        - Assign to clusters by cosine similarity against centroids
        - Create new cluster if best similarity < threshold
        - Update centroids incrementally
        """
        if not self._episodes:
            return

        texts = [ep.get("content", ep.get("title", "")) for ep in self._episodes]

        # Try real embeddings (Cohere)
        embeddings = None
        if HAS_NUMPY:
            embeddings = _call_embedding_api(texts, input_type="search_document")

        if embeddings and HAS_NUMPY:
            self._cluster_with_embeddings(np.array(embeddings))
        else:
            # Fallback: keyword-based clustering
            self._cluster_by_keywords()

    def _cluster_with_embeddings(self, embeddings: Any) -> None:
        """Cluster using cosine similarity (mirrors real ClusterManager)."""
        import numpy as np

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        clusters: list[dict[str, Any]] = []
        threshold = self.CLUSTER_SIMILARITY_THRESHOLD

        for idx, emb in enumerate(embeddings):
            best_cluster = -1
            best_sim = -1.0

            for ci, cluster in enumerate(clusters):
                sim = float(np.dot(emb, cluster["centroid"]))
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = ci

            if best_sim >= threshold and best_cluster >= 0:
                # Add to existing cluster, update centroid incrementally
                cluster = clusters[best_cluster]
                n = len(cluster["episode_indices"])
                cluster["centroid"] = (cluster["centroid"] * n + emb) / (n + 1)
                # Re-normalize centroid
                norm = np.linalg.norm(cluster["centroid"])
                if norm > 0:
                    cluster["centroid"] /= norm
                cluster["episode_indices"].append(idx)
            else:
                # Create new cluster
                clusters.append(
                    {
                        "centroid": emb.copy(),
                        "episode_indices": [idx],
                    }
                )

        # Convert to serializable format
        self._clusters = []
        for ci, cluster in enumerate(clusters):
            episode_indices = cluster["episode_indices"]
            self._clusters.append(
                {
                    "cluster_id": f"cluster_{ci}",
                    "episode_indices": episode_indices,
                    "episodes": [self._episodes[i] for i in episode_indices],
                    "size": len(episode_indices),
                }
            )

        print(
            f"    [evermemos] Clustered {len(self._episodes)} episodes "
            f"into {len(self._clusters)} clusters (embedding-based)"
        )

    def _cluster_by_keywords(self) -> None:
        """Fallback clustering when embeddings are unavailable."""
        # Group by task_id as a simple proxy
        task_groups: dict[str, list[int]] = {}
        for idx, ep in enumerate(self._episodes):
            tid = ep.get("task_id", "unknown")
            task_groups.setdefault(tid, []).append(idx)

        self._clusters = []
        for ci, (tid, indices) in enumerate(task_groups.items()):
            self._clusters.append(
                {
                    "cluster_id": f"cluster_{ci}",
                    "episode_indices": indices,
                    "episodes": [self._episodes[i] for i in indices],
                    "size": len(indices),
                }
            )

        print(
            f"    [evermemos] Clustered {len(self._episodes)} episodes "
            f"into {len(self._clusters)} clusters (keyword fallback)"
        )

    # ==================================================================
    # Stage 3: Profile Consolidation
    # Mirrors real ProfileMemoryMerger
    # ==================================================================

    def _consolidate_profiles(self) -> None:
        """Consolidate profile traits across all MemCells.

        Mirrors real ProfileMemoryMerger:
        - Group traits by dimension
        - Merge evidence across tasks
        - Resolve contradictions (prefer patterns with more evidence)
        - Compute confidence from evidence count
        """
        if not self._profile_traits:
            return

        # Group by dimension
        dim_groups: dict[str, list[dict]] = {}
        for trait in self._profile_traits:
            dim = trait.get("dimension", "unknown")
            dim_groups.setdefault(dim, []).append(trait)

        patterns = []
        contradictions = []

        for dim, traits in dim_groups.items():
            # Check for contradictions within dimension
            pattern_texts = [t.get("pattern", "") for t in traits]
            unique_patterns: dict[str, list[dict]] = {}
            for trait in traits:
                pattern = trait.get("pattern", "")
                # Simple dedup: group similar patterns
                matched = False
                for existing_key in unique_patterns:
                    # Word overlap > 40% → merge (mirrors real MemOS similarity)
                    words_a = set(existing_key.lower().split())
                    words_b = set(pattern.lower().split())
                    if words_a and words_b:
                        overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
                        if overlap > 0.4:
                            unique_patterns[existing_key].append(trait)
                            matched = True
                            break
                if not matched:
                    unique_patterns[pattern] = [trait]

            if len(unique_patterns) > 1:
                # Multiple distinct patterns → potential contradiction
                sorted_groups = sorted(unique_patterns.items(), key=lambda x: -len(x[1]))
                primary = sorted_groups[0]
                for secondary in sorted_groups[1:]:
                    contradictions.append(
                        {
                            "dimension": dim,
                            "description": (
                                f"'{primary[0]}' (evidence: {len(primary[1])}) vs "
                                f"'{secondary[0]}' (evidence: {len(secondary[1])})"
                            ),
                            "resolution": (
                                f"Preferred '{primary[0]}' with {len(primary[1])} "
                                f"supporting observations across tasks: "
                                + ", ".join(set(t.get("task_id", "?") for t in primary[1]))
                            ),
                        }
                    )

            # Build consolidated pattern from the most-evidenced group
            if unique_patterns:
                best_pattern, best_traits = max(unique_patterns.items(), key=lambda x: len(x[1]))
                evidence_tasks = sorted(set(t.get("task_id", "?") for t in best_traits))
                evidence_details = [t.get("evidence", "") for t in best_traits if t.get("evidence")]
                patterns.append(
                    {
                        "dimension": dim,
                        "pattern": best_pattern,
                        "evidence_count": len(best_traits),
                        "evidence_tasks": evidence_tasks,
                        "evidence_details": evidence_details[:5],
                        "confidence": (
                            "high" if len(best_traits) >= 3 else "medium" if len(best_traits) >= 2 else "low"
                        ),
                    }
                )

        self._consolidated_patterns = patterns
        self._contradictions = contradictions

        print(f"    [evermemos] Consolidated: {len(patterns)} patterns, {len(contradictions)} contradictions")

    # ==================================================================
    # Fallback: simulated context_switch segmentation (no LLM)
    # ==================================================================

    def _segment_events(self, events: list[dict[str, Any]], task_id: str) -> list[dict[str, Any]]:
        """Segment events into engram-like traces using context switches."""
        segments = []
        current_segment: list[str] = []
        segment_count = 0

        for event in events:
            et = event.get("event_type", "")

            if et == "context_switch" and current_segment:
                segment_count += 1
                segments.append(
                    {
                        "segment_id": f"{task_id}_seg{segment_count}",
                        "events": list(current_segment),
                        "boundary": (
                            f"context_switch: "
                            f"{self._anonymize_path(event.get('from_file', ''))} -> "
                            f"{self._anonymize_path(event.get('to_file', ''))}"
                        ),
                    }
                )
                current_segment = []
                continue

            narrative = self.events_to_narrative([event])
            if narrative.strip():
                current_segment.append(narrative)

        if current_segment:
            segment_count += 1
            segments.append(
                {
                    "segment_id": f"{task_id}_seg{segment_count}",
                    "events": list(current_segment),
                    "boundary": "end_of_task",
                }
            )

        return segments

    # ==================================================================
    # Public API
    # ==================================================================

    def ingest(self, trajectories: list[dict[str, Any]]) -> None:
        """Full EverMemOS pipeline.

        Real mode (llm_fn set):
          Stage 0: LLM-based MemCell boundary detection
          Stage 1a: Episode memory extraction per MemCell
          Stage 1b: Multi-part profile extraction per MemCell
          Stage 2: Embedding-based episode clustering
          Stage 3: Profile consolidation with contradiction resolution

        Fallback mode (no llm_fn):
          Segment events by context_switch boundaries.
        """
        self._reset_state()

        if self._use_real:
            import threading
            from concurrent.futures import ThreadPoolExecutor

            INNER_PARALLEL = 8

            # Stage 0: MemCell Boundary Detection (parallel by trajectory)
            print(f"    [evermemos] Stage 0: MemCell boundary detection ({INNER_PARALLEL} threads)...")
            lock = threading.Lock()

            def _stage0_one(traj):
                events = self.filter_behavioral_events(traj["events"])
                task_id = traj["task_id"]
                memcells = self._create_memcells(events, task_id)
                with lock:
                    self._memcells.extend(memcells)
                    print(f"      [S0] {task_id}: {len(memcells)} memcells")
                self._on_trajectory_done(task_id)

            with ThreadPoolExecutor(max_workers=INNER_PARALLEL) as pool:
                list(pool.map(_stage0_one, trajectories))

            print(f"    [evermemos] Created {len(self._memcells)} MemCells from {len(trajectories)} trajectories")

            # Stage 1: Memory Extraction — parallel by memcell
            print(f"    [evermemos] Stage 1: Memory extraction ({INNER_PARALLEL} threads)...")
            episodes_lock = threading.Lock()
            done_mc = [0]
            total_mc = len(self._memcells)

            def _stage1_one(mc):
                episode = self._extract_episode(mc)
                traits = self._extract_profile(mc)
                with episodes_lock:
                    if episode:
                        self._episodes.append(episode)
                    self._profile_traits.extend(traits)
                    done_mc[0] += 1
                    if done_mc[0] % 5 == 0 or done_mc[0] == total_mc:
                        print(f"      [S1] {done_mc[0]}/{total_mc} memcells processed")

            with ThreadPoolExecutor(max_workers=INNER_PARALLEL) as pool:
                list(pool.map(_stage1_one, self._memcells))

            print(
                f"    [evermemos] Extracted {len(self._episodes)} episodes, {len(self._profile_traits)} profile traits"
            )

            # Stage 2: Embedding Clustering
            print("    [evermemos] Stage 2: Episode clustering...")
            self._cluster_episodes()

            # Stage 3: Profile Consolidation
            print("    [evermemos] Stage 3: Profile consolidation...")
            self._consolidate_profiles()
        else:
            # Fallback: simulated context_switch segmentation
            for traj in trajectories:
                events = self.filter_behavioral_events(traj["events"])
                task_id = traj["task_id"]
                segments = self._segment_events(events, task_id)
                self._engram_traces.extend(segments)

    def infer_profile(self, profile_attributes: list[str]) -> dict[str, str]:
        """Build inference prompt from consolidated EverMemOS memory."""
        attributes_list = "\n".join(f"- {attr}" for attr in profile_attributes)

        if self._use_real and (self._episodes or self._consolidated_patterns or self._profile_traits):
            # Format clustered episodes
            episode_lines = []
            if self._clusters:
                for cluster in self._clusters:
                    episode_lines.append(f"  ### {cluster['cluster_id']} ({cluster['size']} episodes)")
                    for ep in cluster["episodes"]:
                        episode_lines.append(f"    [{ep.get('task_id', '?')}] {ep.get('title', '?')}")
                        episode_lines.append(f"      {ep.get('content', ep.get('summary', ''))}")
            else:
                # No clustering — list episodes directly
                for ep in self._episodes:
                    episode_lines.append(f"  [{ep.get('task_id', '?')}] {ep.get('title', '?')}")
                    episode_lines.append(f"    {ep.get('content', ep.get('summary', ''))}")
            episodes_text = "\n".join(episode_lines) if episode_lines else "(no episodes extracted)"

            # Format consolidated patterns
            pattern_lines = []
            for pat in self._consolidated_patterns:
                dim = pat.get("dimension", "?")
                desc = pat.get("pattern", "")
                conf = pat.get("confidence", "?")
                count = pat.get("evidence_count", 0)
                tasks = ", ".join(pat.get("evidence_tasks", []))
                pattern_lines.append(f"  [{dim}] {desc} (confidence: {conf}, evidence: {count}, tasks: {tasks})")
                for detail in pat.get("evidence_details", [])[:2]:
                    pattern_lines.append(f"    evidence: {detail}")

            # If consolidation didn't run but we have raw traits, format those
            if not pattern_lines and self._profile_traits:
                for trait in self._profile_traits[:50]:
                    dim = trait.get("dimension", "?")
                    pattern = trait.get("pattern", "")
                    pattern_lines.append(f"  [{dim}] {pattern}")

            patterns_text = "\n".join(pattern_lines) if pattern_lines else "(no patterns extracted)"

            # Format contradictions
            contradiction_lines = []
            for c in self._contradictions:
                contradiction_lines.append(
                    f"  - [{c.get('dimension', '?')}] "
                    f"{c.get('description', '?')}\n"
                    f"    Resolution: {c.get('resolution', '?')}"
                )
            contradictions_text = "\n".join(contradiction_lines) if contradiction_lines else "(none)"

            prompt = INFERENCE_PROMPT.format(
                episodes=episodes_text,
                patterns=patterns_text,
                contradictions=contradictions_text,
                attributes_list=attributes_list,
            )
        else:
            # Fallback: simulated segmentation
            traces_text_parts = []
            for seg in self._engram_traces[:30]:
                events_text = "\n    ".join(seg["events"][:10])
                traces_text_parts.append(
                    f"  Segment {seg['segment_id']} (boundary: {seg['boundary']}):\n    {events_text}"
                )
            engram_text = "\n".join(traces_text_parts)
            prompt = FALLBACK_INFERENCE_PROMPT.format(
                engram_traces=engram_text,
                attributes_list=attributes_list,
            )

        return {"_prompt": prompt, "_method": "evermemos"}

    # ==================================================================
    # Caching
    # ==================================================================

    def _get_ingest_state(self):
        # Don't pickle numpy arrays from clusters — strip centroids
        serializable_clusters = []
        for c in self._clusters:
            serializable_clusters.append(
                {
                    "cluster_id": c["cluster_id"],
                    "episode_indices": c["episode_indices"],
                    "episodes": c["episodes"],
                    "size": c["size"],
                }
            )

        return {
            "memcells": self._memcells,
            "episodes": self._episodes,
            "profile_traits": self._profile_traits,
            "clusters": serializable_clusters,
            "consolidated_patterns": self._consolidated_patterns,
            "contradictions": self._contradictions,
            "engram_traces": self._engram_traces,
        }

    def _set_ingest_state(self, state):
        self._memcells = state.get("memcells", [])
        self._episodes = state.get("episodes", [])
        self._profile_traits = state.get("profile_traits", [])
        self._clusters = state.get("clusters", [])
        self._consolidated_patterns = state.get("consolidated_patterns", [])
        self._contradictions = state.get("contradictions", [])
        self._engram_traces = state.get("engram_traces", [])

    def _reset_state(self):
        """Clear all pipeline state."""
        self._memcells = []
        self._episodes = []
        self._profile_traits = []
        self._clusters = []
        self._consolidated_patterns = []
        self._contradictions = []
        self._engram_traces = []

    def reset(self) -> None:
        super().reset()
        self._reset_state()
