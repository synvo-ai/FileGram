"""Thin Cohere embedding wrapper for episode clustering (Channel 2).

Reuses existing COHERE_API_KEY from .env. Graceful degradation:
if no API key available, returns zero vectors.
"""

from __future__ import annotations

import math
import os
from typing import Any

_BATCH_SIZE = 96  # Cohere API max texts per call


class TextEmbedder:
    """Embed text via Cohere API for clustering.

    Falls back to zero vectors if Cohere is unavailable.
    """

    def __init__(self, model: str = "embed-english-v3.0"):
        self._model = model
        self._client: Any = None
        self._available = False
        self._dim = 1024  # Cohere embed-english-v3.0 dimension

        api_key = os.environ.get("COHERE_API_KEY", "")
        if api_key:
            try:
                import cohere
                self._client = cohere.ClientV2(api_key=api_key)
                self._available = True
            except Exception as e:
                print(f"  [embedder] Cohere init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def embed(self, texts: list[str], input_type: str = "search_document") -> list[list[float]]:
        """Embed a list of texts, batching to stay within API limits.

        Args:
            texts: List of strings to embed.
            input_type: Cohere input type — "search_document" for stored docs,
                "search_query" for queries.

        Returns:
            List of embedding vectors. Zero vectors if Cohere unavailable.
        """
        if not texts:
            return []

        if not self._available or not self._client:
            return [[0.0] * self._dim for _ in texts]

        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            try:
                response = self._client.embed(
                    texts=batch,
                    model=self._model,
                    input_type=input_type,
                    embedding_types=["float"],
                )
                all_embeddings.extend(list(e) for e in response.embeddings.float_)
            except Exception as e:
                print(f"  [embedder] Cohere embed failed (batch {start}–{start+len(batch)}): {e}")
                all_embeddings.extend([0.0] * self._dim for _ in batch)

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for retrieval.

        Uses input_type="search_query" for asymmetric search.
        """
        vecs = self.embed([query], input_type="search_query")
        return vecs[0] if vecs else [0.0] * self._dim

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
