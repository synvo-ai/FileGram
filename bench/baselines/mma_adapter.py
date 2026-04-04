"""MMA (Multimodal Memory Agent) baseline adapter.

Implements MMA's confidence-scored memory retrieval: embedding-based search
reranked by dynamic reliability scores combining source credibility,
temporal decay, and conflict-aware consensus.

Reference: arxiv:2602.16493 — "MMA: Multimodal Memory Agent"

Architecture:
    Ingest:  events → narrative chunks → embed (Cohere) → memory items
             Each item: {text, embedding, source_tag, created_order, task_id}
             Post-ingest: compute neighbor links + confidence scores

    Retrieve: query → embed → cosine similarity top-K → rerank by confidence
              → format context string

Key difference from Naive RAG: two-stage retrieval with confidence reranking
based on source credibility, recency, and consensus among neighbors.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

from .base import BaseAdapter
from .registry import register_adapter

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ── Cohere embedding (shared with naive_rag) ────────────────

def _call_embedding_api(texts: list[str], input_type: str = "search_document") -> list[list[float]] | None:
    """Call Cohere Embed API. Returns list of embedding vectors."""
    import httpx

    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        print("    [mma] COHERE_API_KEY not set")
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
                    print(f"    [mma] Cohere rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    print(f"    [mma] Cohere API error {resp.status_code}: {resp.text[:200]}")
                    return None
                data = resp.json()
                all_embeddings.extend(data["embeddings"]["float"])
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    [mma] Cohere embedding failed: {e}")
                    return None
                time.sleep(5)

    return all_embeddings


# ── Confidence scoring (MMA core innovation) ────────────────

# Source credibility scores (from confidence_v2.yaml)
SOURCE_SCORES = {
    "file_operation": 0.95,    # Direct behavioral observation — highest
    "content_detail": 0.85,    # File content (write/edit)
    "search_pattern": 0.80,    # Search/browse behavior
    "structural": 0.75,        # Directory operations
    "summary": 0.60,           # Aggregated summary
}

# Confidence formula weights
W_SOURCE = 0.40
W_TIME = 0.30
W_CONSENSUS = 0.30

# Neighbor link parameters
NEIGHBOR_TOP_K = 5
CONSENSUS_TOP_K = 5
HALF_LIFE_ITEMS = 200  # temporal decay in item-order (not real time)


def _compute_source_score(source_tag: str) -> float:
    """Source credibility factor s ∈ [0, 1]."""
    return SOURCE_SCORES.get(source_tag, 0.6)


def _compute_time_score(created_order: int, total_items: int) -> float:
    """Temporal decay factor t ∈ [0, 1]. More recent = higher.

    Uses item creation order as proxy for time (all items from same ingest).
    Exponential decay: t = 0.5^(age / half_life).
    """
    if total_items <= 1:
        return 1.0
    age = total_items - 1 - created_order  # 0 = most recent
    return 0.5 ** (age / HALF_LIFE_ITEMS)


def _compute_consensus(
    item_idx: int,
    embeddings: Any,  # numpy array (N, D)
    confidences: list[float],
    neighbor_indices: list[list[int]],
) -> float:
    """Conflict-aware consensus factor c ∈ [-1, 1].

    For each neighbor, compute: link_weight × neighbor_confidence × support_factor.
    support_factor = cosine_similarity(item, neighbor).
    """
    neighbors = neighbor_indices[item_idx]
    if not neighbors:
        return 0.0

    item_emb = embeddings[item_idx]
    total_weight = 0.0
    consensus = 0.0

    for n_idx in neighbors[:CONSENSUS_TOP_K]:
        n_emb = embeddings[n_idx]
        # Cosine similarity (already normalized)
        support = float(np.dot(item_emb, n_emb))
        link_weight = max(0.0, support)  # non-negative link weight
        n_conf = confidences[n_idx]

        total_weight += link_weight
        consensus += link_weight * n_conf * support

    if total_weight > 0:
        consensus /= total_weight

    return max(-1.0, min(1.0, consensus))


def _compute_confidence_tri(s: float, t: float, c: float) -> float:
    """Tri-factor confidence: (w_s·s + w_t·t + w_c·c) / (w_s + w_t + w_c)."""
    raw = (W_SOURCE * s + W_TIME * t + W_CONSENSUS * c) / (W_SOURCE + W_TIME + W_CONSENSUS)
    return max(0.0, min(1.0, raw))


# ── Attribute queries (same as naive_rag for consistency) ────

ATTRIBUTE_QUERIES = {
    "name": "user name identity self-reference personal",
    "role": "profession role job occupation domain expertise",
    "language": "language used in content writing Chinese English",
    "reading_strategy": "file read sequence browsing order search before read sequential vs targeted access pattern",
    "output_detail": "file write content length output structure sections formatting detail level verbosity",
    "output_structure": "output file organization heading hierarchy document sections structure",
    "directory_style": "directory creation nesting depth folder structure organization pattern mkdir",
    "naming": "file naming convention rename operations filename length and patterns",
    "edit_strategy": "file edit frequency lines changed per edit incremental vs bulk modifications",
    "version_strategy": "file copy backup file delete version keeping archive behavior overwrite",
    "tone": "content writing style professional vs casual language formality register",
    "working_style": "work methodology approach systematic exploratory pragmatic phased",
    "thoroughness": "thoroughness exhaustiveness depth completeness reading everything",
    "documentation": "documentation README creation help files index auxiliary",
    "error_handling": "error handling recovery strategy defensive balanced",
    "cross_modal_behavior": "visual images tables charts diagrams figures cross-modal media",
}


INFERENCE_PROMPT = """\
You are analyzing file-system behavioral traces to infer a user's work habit \
profile. Below are the most relevant behavioral memories retrieved for each \
attribute, ranked by confidence score.

{retrieved_context}

Based on the retrieved behavioral memories above (higher confidence = more \
reliable), infer this user's profile for the following attributes. For each \
attribute, provide your best inference and a brief justification.

Attributes to infer:
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
}}
"""


# ── MMA Adapter ─────────────────────────────────────────────

@register_adapter("mma")
class MMAAdapter(BaseAdapter):
    """MMA: confidence-scored embedding retrieval with source/time/consensus."""

    def __init__(self, config: dict[str, Any] | None = None, llm_fn=None):
        super().__init__(config, llm_fn=llm_fn)
        self.name = "mma"
        self.chunk_size = self.config.get("chunk_size", 20)
        self.top_k = self.config.get("top_k", 8)

        # Memory store
        self._items: list[dict[str, Any]] = []  # {text, task_id, source_tag, created_order, confidence}
        self._embeddings: Any = None  # numpy (N, D), L2-normalized
        self._neighbor_indices: list[list[int]] = []  # per-item top-K neighbor indices
        self._confidences: list[float] = []

    def _classify_source(self, events: list[dict]) -> str:
        """Determine source tag for a chunk based on dominant event types."""
        type_counts: dict[str, int] = {}
        for e in events:
            et = e.get("event_type", "unknown")
            type_counts[et] = type_counts.get(et, 0) + 1

        # Priority order for source classification
        if type_counts.get("file_write", 0) + type_counts.get("file_edit", 0) > 0:
            # Check if content is included
            has_content = any(
                e.get("_resolved_content") or e.get("_resolved_diff")
                for e in events
            )
            return "content_detail" if has_content else "file_operation"
        if type_counts.get("file_search", 0) + type_counts.get("file_browse", 0) > 0:
            return "search_pattern"
        if type_counts.get("dir_create", 0) + type_counts.get("file_move", 0) + type_counts.get("file_rename", 0) > 0:
            return "structural"
        return "file_operation"

    def ingest(self, trajectories: list[dict[str, Any]]) -> None:
        """Chunk events → embed → compute neighbor links → confidence scores."""
        self._items = []
        self._embeddings = None
        self._neighbor_indices = []
        self._confidences = []

        # Step 1: Chunk events into memory items
        order = 0
        for traj in trajectories:
            events = self.filter_behavioral_events(traj["events"])
            task_id = traj["task_id"]

            for i in range(0, len(events), self.chunk_size):
                chunk_events = events[i : i + self.chunk_size]
                chunk_text = self.events_to_narrative(chunk_events)
                source_tag = self._classify_source(chunk_events)

                self._items.append({
                    "text": chunk_text,
                    "task_id": task_id,
                    "source_tag": source_tag,
                    "created_order": order,
                    "chunk_index": i // self.chunk_size,
                })
                order += 1

            self._on_trajectory_done(task_id)

        if not self._items:
            return

        # Step 2: Embed all items
        if HAS_NUMPY:
            print(f"    [mma] Embedding {len(self._items)} memory items...")
            texts = [it["text"] for it in self._items]
            embeddings = _call_embedding_api(texts)
            if embeddings and len(embeddings) == len(self._items):
                self._embeddings = np.array(embeddings, dtype=np.float32)
                norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                self._embeddings = self._embeddings / norms
                self.llm_calls_ingest += len(self._items)
                print(f"    [mma] Embedded {len(self._items)} items (dim={self._embeddings.shape[1]})")
            else:
                print("    [mma] Embedding failed, confidence will use source+time only")

        # Step 3: Compute neighbor links
        n = len(self._items)
        self._neighbor_indices = [[] for _ in range(n)]
        if self._embeddings is not None:
            # Similarity matrix
            sim_matrix = self._embeddings @ self._embeddings.T
            for i in range(n):
                sim_matrix[i, i] = -1  # exclude self
                top_k_idx = np.argsort(sim_matrix[i])[-NEIGHBOR_TOP_K:][::-1].tolist()
                self._neighbor_indices[i] = [j for j in top_k_idx if sim_matrix[i, j] > 0.3]

        # Step 4: Compute confidence scores (iterative: 2 passes for consensus convergence)
        self._confidences = [0.5] * n  # initial uniform

        for _pass in range(2):
            new_confidences = []
            for i in range(n):
                s = _compute_source_score(self._items[i]["source_tag"])
                t = _compute_time_score(self._items[i]["created_order"], n)

                if self._embeddings is not None:
                    c = _compute_consensus(i, self._embeddings, self._confidences, self._neighbor_indices)
                else:
                    c = 0.0

                conf = _compute_confidence_tri(s, t, c)
                new_confidences.append(conf)
            self._confidences = new_confidences

        # Store confidence in items
        for i, item in enumerate(self._items):
            item["confidence"] = self._confidences[i]

        avg_conf = sum(self._confidences) / len(self._confidences) if self._confidences else 0
        print(f"    [mma] Confidence computed: avg={avg_conf:.3f}, "
              f"min={min(self._confidences):.3f}, max={max(self._confidences):.3f}")

    def _retrieve_items(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Two-stage retrieval: embedding similarity → confidence reranking."""
        if self._embeddings is None:
            return self._retrieve_keyword(query, top_k)

        # Stage 1: Embedding similarity (retrieve 3x candidates)
        query_emb = _call_embedding_api([query], input_type="search_query")
        if not query_emb:
            return self._retrieve_keyword(query, top_k)

        query_vec = np.array(query_emb[0], dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec /= q_norm

        similarities = self._embeddings @ query_vec
        candidate_k = min(top_k * 3, len(self._items))
        candidate_indices = np.argsort(similarities)[-candidate_k:][::-1]
        self.llm_calls_infer += 1

        # Stage 2: Rerank by combined score = similarity × confidence
        scored = []
        for idx in candidate_indices:
            sim = float(similarities[idx])
            conf = self._confidences[idx]
            combined = sim * 0.6 + conf * 0.4  # weighted combination
            scored.append((combined, conf, sim, idx))

        scored.sort(key=lambda x: -x[0])
        return [
            {**self._items[idx], "_similarity": sim, "_combined": combined}
            for combined, conf, sim, idx in scored[:top_k]
        ]

    def _retrieve_keyword(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Fallback: keyword matching weighted by confidence."""
        keywords = set(query.lower().split())
        scored = []
        for i, item in enumerate(self._items):
            text_lower = item["text"].lower()
            kw_score = sum(1 for kw in keywords if kw in text_lower)
            conf = self._confidences[i] if i < len(self._confidences) else 0.5
            combined = kw_score * 0.6 + conf * 0.4
            scored.append((combined, conf, kw_score, i))

        scored.sort(key=lambda x: -x[0])
        return [
            {**self._items[idx], "_similarity": kw, "_combined": combined}
            for combined, conf, kw, idx in scored[:top_k]
        ]

    def infer_profile(self, profile_attributes: list[str]) -> dict[str, str]:
        """Retrieve relevant items per attribute with confidence scores, build prompt."""
        retrieved_sections = []

        for attr in profile_attributes:
            query = ATTRIBUTE_QUERIES.get(attr, attr)
            items = self._retrieve_items(query, self.top_k)
            item_texts = []
            for it in items:
                conf = it.get("confidence", 0.5)
                sim = it.get("_similarity", 0)
                text_preview = it["text"][:500]
                item_texts.append(
                    f"  [Task {it['task_id']}, conf={conf:.2f}]: {text_preview}"
                )
            retrieved_sections.append(f"### For attribute '{attr}':\n" + "\n".join(item_texts))

        retrieved_context = "\n\n".join(retrieved_sections)
        attributes_list = "\n".join(f"- {attr}" for attr in profile_attributes)

        prompt = INFERENCE_PROMPT.format(
            retrieved_context=retrieved_context,
            attributes_list=attributes_list,
        )
        return {"_prompt": prompt, "_method": "mma"}

    def _get_ingest_state(self):
        return {
            "items": [{k: v for k, v in it.items() if k != "_similarity" and k != "_combined"}
                      for it in self._items],
            "embeddings": self._embeddings,
            "neighbor_indices": self._neighbor_indices,
            "confidences": self._confidences,
        }

    def _set_ingest_state(self, state):
        self._items = state["items"]
        self._embeddings = state.get("embeddings")
        self._neighbor_indices = state.get("neighbor_indices", [])
        self._confidences = state.get("confidences", [0.5] * len(self._items))

    def reset(self) -> None:
        super().reset()
        self._items = []
        self._embeddings = None
        self._neighbor_indices = []
        self._confidences = []
