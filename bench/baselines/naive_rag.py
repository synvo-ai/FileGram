"""Naive RAG baseline: chunk trajectories, embed, retrieve relevant chunks.

Chunks behavioral events into fixed-size windows, embeds them using
Cohere embed-v4.0, and retrieves the most relevant chunks per attribute
query via cosine similarity.

Falls back to keyword matching if numpy is not available or embedding fails.
"""

from __future__ import annotations

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

INFERENCE_PROMPT = """\
You are analyzing file-system behavioral traces to infer a user's work habit \
profile. Below are the most relevant behavioral segments retrieved for each \
attribute you need to infer.

{retrieved_context}

Based on the retrieved behavioral segments above, infer this user's profile \
for the following attributes. For each attribute, provide your best inference \
and a brief justification.

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

# Attribute-specific retrieval queries
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


def _call_embedding_api(texts: list[str], input_type: str = "search_document") -> list[list[float]] | None:
    """Call Cohere Embed API. Returns list of embedding vectors.

    Args:
        texts: Texts to embed.
        input_type: "search_document" for corpus, "search_query" for queries.
    """
    import httpx

    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        print("    [naive_rag] COHERE_API_KEY not set")
        return None

    url = "https://api.cohere.com/v2/embed"

    all_embeddings: list[list[float]] = []
    batch_size = 96  # Cohere supports up to 96 texts per call
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Truncate each text to avoid token limits
        batch = [t[:4096] for t in batch]

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
                    print(f"    [naive_rag] Cohere rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    print(f"    [naive_rag] Cohere API error {resp.status_code}: {resp.text[:200]}")
                    return None
                data = resp.json()
                batch_embeddings = data["embeddings"]["float"]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    [naive_rag] Cohere embedding failed: {e}")
                    return None
                time.sleep(5)

    return all_embeddings


@register_adapter("naive_rag")
class NaiveRAGAdapter(BaseAdapter):
    """Baseline: chunk events, embed, retrieve per-attribute via cosine similarity."""

    def __init__(self, config: dict[str, Any] | None = None, llm_fn=None):
        super().__init__(config, llm_fn=llm_fn)
        self.name = "naive_rag"
        self.chunk_size = self.config.get("chunk_size", 20)  # events per chunk
        self.top_k = self.config.get("top_k", 5)  # chunks to retrieve
        self._chunks: list[dict[str, Any]] = []  # {text, events, task_id}
        self._embeddings: Any = None  # numpy array or None
        self._use_real = HAS_NUMPY

    def ingest(self, trajectories: list[dict[str, Any]]) -> None:
        """Chunk behavioral events and embed them."""
        self._chunks = []
        self._embeddings = None
        for traj in trajectories:
            events = self.filter_behavioral_events(traj["events"])
            task_id = traj["task_id"]

            for i in range(0, len(events), self.chunk_size):
                chunk_events = events[i : i + self.chunk_size]
                chunk_text = self.events_to_narrative(chunk_events)
                self._chunks.append(
                    {
                        "text": chunk_text,
                        "events": chunk_events,
                        "task_id": task_id,
                        "chunk_index": i // self.chunk_size,
                    }
                )
            self._on_trajectory_done(task_id)

        # Try real embedding
        if self._use_real and self._chunks:
            print(f"    [naive_rag] Embedding {len(self._chunks)} chunks...")
            chunk_texts = [c["text"] for c in self._chunks]
            embeddings = _call_embedding_api(chunk_texts)
            if embeddings and len(embeddings) == len(self._chunks):
                self._embeddings = np.array(embeddings, dtype=np.float32)
                # Normalize for cosine similarity
                norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                self._embeddings = self._embeddings / norms
                self.llm_calls_ingest += len(self._chunks)  # count embedding calls
                print(f"    [naive_rag] Embedded {len(self._chunks)} chunks (dim={self._embeddings.shape[1]})")
            else:
                print("    [naive_rag] Embedding failed, falling back to keyword matching")
                self._embeddings = None

    def _retrieve_chunks_embedding(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve chunks using real embedding cosine similarity."""
        query_emb = _call_embedding_api([query], input_type="search_query")
        if not query_emb:
            return self._retrieve_chunks_keyword(query, top_k)

        query_vec = np.array(query_emb[0], dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        similarities = self._embeddings @ query_vec
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        self.llm_calls_infer += 1  # count query embedding call
        return [self._chunks[i] for i in top_indices]

    def _retrieve_chunks_keyword(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Fallback: keyword-based retrieval."""
        keywords = set(query.lower().split(", "))

        scored = []
        for chunk in self._chunks:
            text_lower = chunk["text"].lower()
            score = sum(1 for kw in keywords if kw.strip() in text_lower)
            scored.append((score, chunk))

        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:top_k]]

    def _retrieve_chunks(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve most relevant chunks for a query."""
        if self._embeddings is not None:
            return self._retrieve_chunks_embedding(query, top_k)
        return self._retrieve_chunks_keyword(query, top_k)

    def infer_profile(self, profile_attributes: list[str]) -> dict[str, str]:
        """Retrieve relevant chunks per attribute, build inference prompt."""
        retrieved_sections = []

        for attr in profile_attributes:
            query = ATTRIBUTE_QUERIES.get(attr, attr)
            chunks = self._retrieve_chunks(query, self.top_k)
            chunk_texts = "\n".join(
                f"  [Task {c['task_id']}, chunk {c['chunk_index']}]: {c['text'][:500]}" for c in chunks
            )
            retrieved_sections.append(f"### For attribute '{attr}':\n{chunk_texts}")

        retrieved_context = "\n\n".join(retrieved_sections)
        attributes_list = "\n".join(f"- {attr}" for attr in profile_attributes)

        prompt = INFERENCE_PROMPT.format(
            retrieved_context=retrieved_context,
            attributes_list=attributes_list,
        )

        return {"_prompt": prompt, "_method": "naive_rag"}

    def _get_ingest_state(self):
        return {
            "chunks": [{k: v for k, v in c.items() if k != "events"} for c in self._chunks],
            "embeddings": self._embeddings,  # numpy array, pickle handles natively
            "use_real": self._use_real,
        }

    def _set_ingest_state(self, state):
        self._chunks = state["chunks"]
        self._embeddings = state.get("embeddings")
        self._use_real = state.get("use_real", self._embeddings is not None)

    def reset(self) -> None:
        super().reset()
        self._chunks = []
        self._embeddings = None
