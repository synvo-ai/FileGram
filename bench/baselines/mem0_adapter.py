"""Mem0 adapter: interaction-driven memory framework.

Mem0 (https://github.com/mem0ai/mem0) is designed for conversation memory.
We adapt it by converting file-system behavioral events into pseudo-messages
that Mem0 can ingest, then querying for profile reconstruction.

When mem0ai is installed, uses the REAL Mem0 API (Memory.from_config()).
Otherwise falls back to simulated mode.

EXPECTED: Poor performance on file-system trajectories because Mem0's memory
extraction is optimized for dialogue patterns, not file operation sequences.
"""

from __future__ import annotations

import os
from typing import Any

from .base import BaseAdapter
from .registry import register_adapter

# Try importing real mem0
try:
    from mem0 import Memory as Mem0Memory

    HAS_MEM0 = True
except ImportError:
    HAS_MEM0 = False

INFERENCE_PROMPT = """\
You are analyzing memories extracted by Mem0 from a user's file-system \
behavioral trajectories. These memories were originally file operations \
(reads, writes, edits, directory operations) converted into interaction format.

Extracted memories:
{memories}

Based on these memories, infer the user's work habit profile for:
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

# Attribute-specific search queries for Mem0
ATTR_QUERIES = {
    "name": "user name identity",
    "role": "user profession job role occupation",
    "language": "language used writing communication",
    "tone": "writing tone style formality professional casual",
    "output_detail": "output detail level verbosity length thoroughness",
    "working_style": "working style methodology approach systematic exploratory",
    "thoroughness": "thoroughness exhaustiveness depth completeness",
    "documentation": "documentation README creation help files",
    "error_handling": "error handling recovery strategy response",
    "reading_strategy": "reading strategy file access browsing search sequential",
    "output_structure": "output structure headings sections organization formatting",
    "directory_style": "directory structure folder organization nesting depth",
    "naming": "file naming convention pattern style",
    "edit_strategy": "edit strategy modification incremental bulk rewrite",
    "version_strategy": "version strategy backup archive overwrite history",
    "cross_modal_behavior": "cross modal visual images tables charts diagrams",
}


@register_adapter("mem0")
class Mem0Adapter(BaseAdapter):
    """Adapter for Mem0 memory framework.

    Converts file-system events into pseudo-conversation messages,
    feeds them through Mem0's memory extraction, then queries.
    """

    def __init__(self, config: dict[str, Any] | None = None, llm_fn=None):
        super().__init__(config, llm_fn=llm_fn)
        self.name = "mem0"
        self._messages: list[dict[str, str]] = []
        self._memories: list[str] = []
        self._mem0: Any = None
        self._profile_id: str = "default_user"
        self._use_real = HAS_MEM0

    def _init_mem0(self):
        """Initialize real Mem0 client if available.

        Uses Azure OpenAI for LLM, Cohere embeddings (via OpenAI-compat API),
        and in-memory Qdrant (no external server needed).
        """
        if not HAS_MEM0:
            return
        try:
            api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://haku-chat.openai.azure.com")
            deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
            cohere_key = os.environ.get("COHERE_API_KEY", "")

            config = {
                "llm": {
                    "provider": "azure_openai",
                    "config": {
                        "azure_kwargs": {
                            "api_key": api_key,
                            "azure_deployment": deployment,
                            "azure_endpoint": endpoint,
                            "api_version": api_version,
                        },
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "api_key": cohere_key,
                        "openai_base_url": "https://api.cohere.com/compatibility/v1",
                        "model": "embed-v4.0",
                        "embedding_dims": 1536,
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "filegram_mem0",
                        "embedding_model_dims": 1536,
                        "path": ":memory:",
                    },
                },
            }
            self._mem0 = Mem0Memory.from_config(config_dict=config)
            self._use_real = True
            print("    [mem0] Real Mem0 initialized (Azure LLM + Cohere embeddings + in-memory Qdrant)")
        except Exception as e:
            print(f"    [mem0] Failed to init real Mem0 ({e}), using simulated mode")
            self._use_real = False

    def _events_to_messages(self, events: list[dict[str, Any]], task_id: str) -> list[dict[str, str]]:
        """Convert behavioral events into pseudo-conversation messages."""
        _anon = self._anonymize_path
        messages = []
        for event in events:
            et = event.get("event_type", "")

            if et == "file_read":
                messages.append(
                    {
                        "role": "user",
                        "content": f"I'm reading {_anon(event.get('file_path', ''))} "
                        f"(view #{event.get('view_count', 1)}, "
                        f"{event.get('content_length', 0)} chars, "
                        f"lines {event.get('view_range', [])})",
                    }
                )
            elif et == "file_write":
                op = event.get("operation", "write")
                msg = f"I {op}d the file {_anon(event.get('file_path', ''))} ({event.get('content_length', 0)} chars)"
                content = event.get("_resolved_content")
                if content:
                    preview = self.truncate_content(content, 1000)
                    msg += f"\nContent:\n{preview}"
                messages.append({"role": "assistant", "content": msg})
            elif et == "file_edit":
                msg = (
                    f"I edited {_anon(event.get('file_path', ''))} "
                    f"(+{event.get('lines_added', 0)}/-{event.get('lines_deleted', 0)} lines)"
                )
                content = event.get("_resolved_content")
                if content:
                    preview = self.truncate_content(content, 1000)
                    msg += f"\nDiff:\n{preview}"
                messages.append({"role": "assistant", "content": msg})
            elif et == "file_search":
                messages.append(
                    {
                        "role": "user",
                        "content": f"Search ({event.get('search_type', '')}): "
                        f"'{event.get('query', '')}' -> {event.get('files_matched', 0)} matches",
                    }
                )
            elif et == "file_browse":
                messages.append(
                    {
                        "role": "user",
                        "content": f"Browsing directory {_anon(event.get('directory_path', ''))} "
                        f"({event.get('files_listed', 0)} files)",
                    }
                )
            elif et == "dir_create":
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Created directory {_anon(event.get('dir_path', ''))} "
                        f"(depth={event.get('depth', 0)})",
                    }
                )
            elif et == "file_rename":
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Renamed {_anon(event.get('old_path', ''))} -> {_anon(event.get('new_path', ''))}",
                    }
                )
            elif et == "file_move":
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Moved {_anon(event.get('old_path', ''))} -> {_anon(event.get('new_path', ''))}",
                    }
                )
            elif et == "file_copy":
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Copied {_anon(event.get('source_path', ''))} -> "
                        f"{_anon(event.get('dest_path', ''))} (backup={event.get('is_backup', False)})",
                    }
                )
            elif et == "file_delete":
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Deleted {_anon(event.get('file_path', ''))}",
                    }
                )

        return messages

    def ingest(self, trajectories: list[dict[str, Any]]) -> None:
        """Convert trajectories to pseudo-messages and ingest via Mem0."""
        self._messages = []
        self._memories = []

        # Initialize Mem0 on first use
        if self._use_real and self._mem0 is None:
            self._init_mem0()

        total = len(trajectories)
        for ti, traj in enumerate(trajectories, 1):
            events = self.filter_behavioral_events(traj["events"])
            task_id = traj["task_id"]
            messages = self._events_to_messages(events, task_id)
            self._messages.extend(messages)
            self._on_trajectory_done(task_id)

        if self._use_real and self._mem0 is not None:
            # Real Mem0: add messages in batches
            total_batches = (len(self._messages) + 19) // 20
            print(f"    [mem0] Adding {len(self._messages)} messages via real Mem0 API ({total_batches} batches)...")
            batch_size = 20
            for i in range(0, len(self._messages), batch_size):
                batch_num = i // batch_size + 1
                print(f"    [mem0] Batch {batch_num}/{total_batches}...")
                batch = self._messages[i : i + batch_size]
                try:
                    self._mem0.add(batch, user_id=self._profile_id)
                    self.llm_calls_ingest += 1  # Mem0 makes internal LLM calls
                except Exception as e:
                    print(f"    [mem0] Error adding batch {i // batch_size}: {e}")
            print("    [mem0] Ingestion complete")
        else:
            # Simulated fallback
            self._memories = [m["content"] for m in self._messages]

    def infer_profile(self, profile_attributes: list[str]) -> dict[str, str]:
        """Query Mem0 for profile reconstruction."""
        if self._use_real and self._mem0 is not None:
            # Live Mem0: search per attribute via Mem0 API
            all_memories = []
            seen = set()
            for attr in profile_attributes:
                query = ATTR_QUERIES.get(attr, attr)
                try:
                    results = self._mem0.search(query, user_id=self._profile_id, limit=10)
                    self.llm_calls_infer += 1
                    for r in results.get("results", results) if isinstance(results, dict) else results:
                        mem_text = r.get("memory", r.get("text", str(r)))
                        if mem_text not in seen:
                            seen.add(mem_text)
                            all_memories.append(f"[{attr}] {mem_text}")
                except Exception as e:
                    print(f"    [mem0] Search error for {attr}: {e}")

            memories_text = "\n".join(f"- {m}" for m in all_memories[:150])
            print(f"    [mem0] Retrieved {len(all_memories)} unique memories via live search")
        elif self._use_real and self._memories:
            # Cached real Mem0: use pre-extracted memories (no live Mem0 instance)
            memories_text = "\n".join(f"- {m}" for m in self._memories[:150])
            print(f"    [mem0] Using {len(self._memories)} cached Mem0-extracted memories")
        else:
            # Simulated fallback: raw messages as memories
            memories_text = "\n".join(f"- {m}" for m in self._memories[:100])
            print(f"    [mem0] Using {len(self._memories)} raw message memories (simulated)")

        attributes_list = "\n".join(f"- {attr}" for attr in profile_attributes)

        prompt = INFERENCE_PROMPT.format(
            memories=memories_text,
            attributes_list=attributes_list,
        )

        return {"_prompt": prompt, "_method": "mem0"}

    def _get_ingest_state(self):
        # For real mode, extract memories from Mem0 to cache them
        cached_memories = self._memories
        if self._use_real and self._mem0 is not None:
            try:
                all_mems = self._mem0.get_all(user_id=self._profile_id)
                if isinstance(all_mems, dict):
                    all_mems = all_mems.get("results", [])
                cached_memories = [m.get("memory", m.get("text", str(m))) for m in all_mems]
            except Exception:
                pass
        return {
            "messages": self._messages,
            "memories": cached_memories,
            "use_real": self._use_real,
        }

    def _set_ingest_state(self, state):
        self._messages = state["messages"]
        self._memories = state["memories"]
        # Restore real/simulated flag from cache.
        # If cache was built in real mode, memories contain Mem0-extracted data,
        # and we can use them directly for inference (Mem0 search is not needed
        # because we already have the extracted memories).
        self._use_real = state.get("use_real", False)

    def reset(self) -> None:
        super().reset()
        self._messages = []
        self._memories = []
        if self._use_real and self._mem0 is not None:
            try:
                self._mem0.delete_all(user_id=self._profile_id)
            except Exception:
                pass
