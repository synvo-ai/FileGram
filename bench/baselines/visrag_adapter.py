"""VisRAG baseline: visual document retrieval-augmented generation.

Inspired by VisRAG (OpenBMB), which processes documents as images rather
than text.  For FileGramBench, the key design is:

  - Behavioral events (events.json) → text chunks → text embedding  (same as Naive RAG)
  - Output documents (agent-created files) → rendered as page images → image embedding

This creates a hybrid text+visual index.  At query time, text queries
are embedded and matched against both text and visual embeddings.

Uses Cohere embed-v4.0 which supports text+image in the same vector space.
"""

from __future__ import annotations

import base64
import io
import os
import textwrap
import time
from typing import Any

from .base import BaseAdapter
from .registry import register_adapter

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ── Rendering ───────────────────────────────────────────────────────

def render_text_to_image(text: str, width: int = 800, font_size: int = 13,
                         padding: int = 16, max_lines: int = 80) -> bytes:
    """Render text content to a PNG image (page-style), return as bytes."""
    lines: list[str] = []
    for line in text.split("\n")[:max_lines * 2]:
        if len(line) > 105:
            lines.extend(textwrap.wrap(line, width=105))
        else:
            lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... (truncated)"]

    line_height = font_size + 4
    img_height = padding * 2 + line_height * len(lines)
    img = Image.new("RGB", (width, max(img_height, 40)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
    except Exception:
        font = ImageFont.load_default()

    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(0, 0, 0), font=font)
        y += line_height

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ── Cohere API helpers ──────────────────────────────────────────────

def _call_text_embedding(texts: list[str], input_type: str = "search_document") -> list[list[float]] | None:
    """Embed texts via Cohere embed-v4.0."""
    import httpx
    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        return None

    url = "https://api.cohere.com/v2/embed"
    all_embs: list[list[float]] = []
    batch_size = 96

    for i in range(0, len(texts), batch_size):
        batch = [t[:4096] for t in texts[i:i + batch_size]]
        for attempt in range(3):
            try:
                resp = httpx.post(url, json={
                    "texts": batch, "model": "embed-v4.0",
                    "input_type": input_type, "embedding_types": ["float"],
                }, headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }, timeout=60.0)
                if resp.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                if resp.status_code != 200:
                    print(f"    [visrag] text embed error {resp.status_code}: {resp.text[:200]}")
                    return None
                all_embs.extend(resp.json()["embeddings"]["float"])
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    [visrag] text embed failed: {e}")
                    return None
                time.sleep(5)
    return all_embs


def _call_image_embedding(image_bytes_list: list[bytes]) -> list[list[float]] | None:
    """Embed images via Cohere embed-v4.0. Max ~20 images per call for size reasons."""
    import httpx
    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        return None

    url = "https://api.cohere.com/v2/embed"
    all_embs: list[list[float]] = []
    batch_size = 10  # conservative for image payloads

    for i in range(0, len(image_bytes_list), batch_size):
        batch = image_bytes_list[i:i + batch_size]
        images_b64 = [
            f"data:image/png;base64,{base64.b64encode(b).decode()}"
            for b in batch
        ]
        for attempt in range(3):
            try:
                resp = httpx.post(url, json={
                    "images": images_b64, "model": "embed-v4.0",
                    "input_type": "image", "embedding_types": ["float"],
                }, headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }, timeout=120.0)
                if resp.status_code == 429:
                    wait = 15 * (attempt + 1)
                    print(f"    [visrag] image embed rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    print(f"    [visrag] image embed error {resp.status_code}: {resp.text[:300]}")
                    return None
                all_embs.extend(resp.json()["embeddings"]["float"])
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    [visrag] image embed failed: {e}")
                    return None
                time.sleep(5)
    return all_embs


# ── Attribute queries (same as Naive RAG) ───────────────────────────

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
profile. Below are the most relevant behavioral segments retrieved for each \
attribute you need to infer. Some segments are from TEXT behavioral logs, \
others are VISUAL renderings of the user's output documents (described below).

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


@register_adapter("visrag")
class VisRAGAdapter(BaseAdapter):
    """VisRAG-style: text chunks (events) + visual embeddings (output docs)."""

    def __init__(self, config: dict[str, Any] | None = None, llm_fn=None):
        super().__init__(config, llm_fn=llm_fn)
        self.name = "visrag"
        self.chunk_size = self.config.get("chunk_size", 20)
        self.top_k = self.config.get("top_k", 5)
        # Text items (event chunks)
        self._text_items: list[dict[str, Any]] = []
        self._text_embeddings: Any = None
        # Visual items (rendered output docs)
        self._visual_items: list[dict[str, Any]] = []
        self._visual_embeddings: Any = None

    def ingest(self, trajectories: list[dict[str, Any]]) -> None:
        self._text_items = []
        self._visual_items = []
        self._text_embeddings = None
        self._visual_embeddings = None

        for traj in trajectories:
            events = self.filter_behavioral_events(traj["events"])
            task_id = traj["task_id"]

            # ── Text chunks from behavioral events (same as Naive RAG) ──
            for i in range(0, len(events), self.chunk_size):
                chunk_events = events[i:i + self.chunk_size]
                chunk_text = self.events_to_narrative(chunk_events)
                self._text_items.append({
                    "text": chunk_text,
                    "task_id": task_id,
                    "chunk_index": i // self.chunk_size,
                    "source": "text",
                })

            # ── Visual items from output documents ──
            for ev in events:
                content = ev.get("_resolved_content", "")
                if not content or len(content) < 100:
                    continue
                # Only file_write creates (new documents)
                if ev.get("event_type") != "file_write":
                    continue
                if ev.get("operation") != "create":
                    continue

                file_path = ev.get("file_path", "unknown")
                self._visual_items.append({
                    "text": content,  # keep text for display fallback
                    "file_path": file_path,
                    "task_id": task_id,
                    "content_length": len(content),
                    "source": "visual",
                })

            self._on_trajectory_done(task_id)

        # ── Embed text items ──
        if self._text_items:
            print(f"    [visrag] Embedding {len(self._text_items)} text chunks...")
            texts = [it["text"] for it in self._text_items]
            embs = _call_text_embedding(texts)
            if embs and len(embs) == len(self._text_items):
                self._text_embeddings = np.array(embs, dtype=np.float32)
                norms = np.linalg.norm(self._text_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                self._text_embeddings /= norms
                print(f"    [visrag] Text: {len(self._text_items)} chunks embedded")

        # ── Render + embed visual items ──
        if self._visual_items and HAS_PIL:
            print(f"    [visrag] Rendering {len(self._visual_items)} output docs as images...")
            image_bytes_list = []
            for item in self._visual_items:
                img = render_text_to_image(item["text"])
                image_bytes_list.append(img)
                item["_img_size_kb"] = len(img) / 1024

            print(f"    [visrag] Embedding {len(image_bytes_list)} images via Cohere...")
            embs = _call_image_embedding(image_bytes_list)
            if embs and len(embs) == len(self._visual_items):
                self._visual_embeddings = np.array(embs, dtype=np.float32)
                norms = np.linalg.norm(self._visual_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                self._visual_embeddings /= norms
                print(f"    [visrag] Visual: {len(self._visual_items)} docs embedded")
            else:
                print(f"    [visrag] Image embedding failed, visual channel disabled")
        else:
            if not HAS_PIL:
                print("    [visrag] PIL not available, visual channel disabled")

        self.llm_calls_ingest += len(self._text_items) + len(self._visual_items)

    def _retrieve(self, query: str, top_k: int) -> list[dict]:
        """Retrieve top-K items from combined text+visual index."""
        query_emb = _call_text_embedding([query], input_type="search_query")
        if not query_emb:
            return self._text_items[:top_k]

        qvec = np.array(query_emb[0], dtype=np.float32)
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            qvec /= qnorm

        candidates: list[tuple[float, dict]] = []

        # Score text items
        if self._text_embeddings is not None:
            sims = self._text_embeddings @ qvec
            for i, s in enumerate(sims):
                candidates.append((float(s), self._text_items[i]))

        # Score visual items
        if self._visual_embeddings is not None:
            sims = self._visual_embeddings @ qvec
            for i, s in enumerate(sims):
                candidates.append((float(s), self._visual_items[i]))

        candidates.sort(key=lambda x: -x[0])
        self.llm_calls_infer += 1
        return [item for _, item in candidates[:top_k]]

    def infer_profile(self, profile_attributes: list[str]) -> dict[str, str]:
        retrieved_sections = []
        for attr in profile_attributes:
            query = ATTRIBUTE_QUERIES.get(attr, attr)
            items = self._retrieve(query, self.top_k)
            parts = []
            for it in items:
                if it["source"] == "text":
                    parts.append(f"  [TEXT, Task {it['task_id']}, chunk {it.get('chunk_index',0)}]: {it['text'][:500]}")
                else:
                    # Visual item — show file path + text excerpt as fallback
                    parts.append(f"  [VISUAL DOC, Task {it['task_id']}, {it.get('file_path','?')}]: {it['text'][:500]}")
            retrieved_sections.append(f"### For attribute '{attr}':\n" + "\n".join(parts))

        retrieved_context = "\n\n".join(retrieved_sections)
        attributes_list = "\n".join(f"- {attr}" for attr in profile_attributes)
        prompt = INFERENCE_PROMPT.format(
            retrieved_context=retrieved_context,
            attributes_list=attributes_list,
        )
        return {"_prompt": prompt, "_method": "visrag"}

    def _get_ingest_state(self) -> dict:
        return {
            "text_items": [{k: v for k, v in it.items() if k != "_img_size_kb"} for it in self._text_items],
            "text_embeddings": self._text_embeddings,
            "visual_items": [{k: v for k, v in it.items() if k != "_img_size_kb"} for it in self._visual_items],
            "visual_embeddings": self._visual_embeddings,
        }

    def _set_ingest_state(self, state: dict) -> None:
        self._text_items = state["text_items"]
        self._text_embeddings = state.get("text_embeddings")
        self._visual_items = state["visual_items"]
        self._visual_embeddings = state.get("visual_embeddings")

    def reset(self) -> None:
        super().reset()
        self._text_items = []
        self._text_embeddings = None
        self._visual_items = []
        self._visual_embeddings = None
