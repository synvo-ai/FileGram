"""MemOS adapter: structured memory OS with multi-level management.

MemOS (https://github.com/MemTensor/MemOS) provides a structured memory
operating system with hierarchical memory organization. We adapt it by
using LLM to extract structured memory cells from file-system events,
then consolidating them into a graph-like memory store with typed
relationships.

EXPECTED: Moderate performance — MemOS's structured approach may partially
capture organization patterns, but its memory extraction is conversation-oriented.
"""

from __future__ import annotations

import json
from typing import Any

from .base import BaseAdapter
from .registry import register_adapter

EXTRACTION_PROMPT = """\
You are a memory extraction module for MemOS, a structured memory operating system.
Your task is to extract structured memory cells from file-system behavioral events.

Events:
{narrative}

Extract memory cells that capture the user's behavioral patterns. Each memory cell \
has a type and structured content. Output JSON:
{{
  "memories": [
    {{
      "type": "fact",
      "content": "<concrete factual observation about user behavior>",
      "confidence": <0.0-1.0>,
      "related_memories": ["<reference to other memory content if related>"]
    }},
    {{
      "type": "preference",
      "content": "<user preference inferred from behavior>",
      "confidence": <0.0-1.0>,
      "related_memories": []
    }},
    {{
      "type": "behavioral_pattern",
      "content": "<recurring behavioral pattern>",
      "confidence": <0.0-1.0>,
      "related_memories": []
    }}
  ]
}}

Memory types:
- "fact": Concrete observed actions (e.g., "User read all files sequentially before writing")
- "preference": Inferred preferences (e.g., "User prefers deeply nested directory structures")
- "behavioral_pattern": Recurring patterns across events (e.g., "User consistently backs up files before editing")

Focus on:
- Information consumption patterns (reading order, search vs browse, depth)
- Production style (output length, structure, auxiliary files)
- Organization habits (directory depth, naming conventions, version strategy)
- Iteration strategy (edit size, frequency, backup behavior)
- Work rhythm (phased vs bursty, context switching frequency)
- Cross-modal behavior (images, tables, visual elements)

Extract 8-15 memories per batch. Be specific and behavioral, not vague."""

INFERENCE_PROMPT = """\
You are analyzing structured memory cells extracted by MemOS from a user's \
file-system behavioral trajectories. MemOS organizes memories into typed cells \
(facts, preferences, behavioral patterns) with confidence scores and relationships.

Memory cells by type:

{memory_cells}

Memory statistics:
{memory_stats}

Based on these structured memories, infer the user's work habit profile for:
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


@register_adapter("memos")
class MemOSAdapter(BaseAdapter):
    """Adapter for MemOS structured memory system.

    Uses LLM to extract typed memory cells (fact, preference, behavioral_pattern),
    then consolidates them into a graph-like memory store with relationships.
    Falls back to rule-based event-to-cell conversion when no llm_fn is provided.
    """

    def __init__(self, config: dict[str, Any] | None = None, llm_fn=None):
        super().__init__(config, llm_fn=llm_fn)
        self.name = "memos"
        # LLM-extracted structured memories: list of {type, content, confidence, related_memories, task_id}
        self._memories: list[dict[str, Any]] = []
        # Fallback: simple event-derived cells (used when no llm_fn)
        self._memory_cells: list[dict[str, Any]] = []
        self._use_real = llm_fn is not None

    def _extract_memories_via_llm(self, narrative: str, task_id: str) -> list[dict[str, Any]]:
        """Use LLM to extract structured memory cells from event narrative."""
        prompt = EXTRACTION_PROMPT.format(narrative=narrative)
        try:
            response = self._call_llm(prompt, phase="ingest")
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(text[start : end + 1])
                memories = data.get("memories", [])
                for m in memories:
                    m["task_id"] = task_id
                    # Ensure required fields
                    m.setdefault("type", "fact")
                    m.setdefault("content", "")
                    m.setdefault("confidence", 0.5)
                    m.setdefault("related_memories", [])
                return memories
        except Exception as e:
            print(f"    [memos] Memory extraction error: {e}")
        return []

    def _consolidate_memories(self, new_memories: list[dict[str, Any]]) -> None:
        """Consolidate new memories into the existing memory store.

        For each new memory, check if it updates or supplements an existing one.
        If a new memory is sufficiently similar to an existing one (same type and
        overlapping content), merge by keeping the higher-confidence version and
        linking them. Otherwise, add as a new memory.
        """
        for new_mem in new_memories:
            merged = False
            for existing in self._memories:
                if existing["type"] != new_mem["type"]:
                    continue
                # Simple overlap check: if >40% of words overlap, consolidate
                existing_words = set(existing["content"].lower().split())
                new_words = set(new_mem["content"].lower().split())
                if not existing_words or not new_words:
                    continue
                overlap = len(existing_words & new_words) / min(len(existing_words), len(new_words))
                if overlap > 0.4:
                    # Keep higher confidence version, link the other
                    if new_mem.get("confidence", 0) > existing.get("confidence", 0):
                        old_content = existing["content"]
                        existing["content"] = new_mem["content"]
                        existing["confidence"] = new_mem["confidence"]
                        existing.setdefault("consolidated_from", []).append(old_content)
                    else:
                        existing.setdefault("consolidated_from", []).append(new_mem["content"])
                    # Merge related_memories
                    existing_related = set(existing.get("related_memories", []))
                    existing_related.update(new_mem.get("related_memories", []))
                    existing["related_memories"] = list(existing_related)
                    merged = True
                    break
            if not merged:
                self._memories.append(new_mem)

    def _events_to_memory_cells(self, events: list[dict[str, Any]], task_id: str) -> list[dict[str, Any]]:
        """Fallback: convert events into MemOS memory cells without LLM.

        MemOS memory cells have: content, metadata, level, relationships.
        """
        _anon = self._anonymize_path
        cells = []
        for event in events:
            et = event.get("event_type", "")

            cell = {
                "content": "",
                "level": "episodic",
                "task_id": task_id,
                "event_type": et,
            }

            if et == "file_read":
                cell["content"] = (
                    f"Read {_anon(event.get('file_path', ''))} "
                    f"(view #{event.get('view_count', 1)}, {event.get('content_length', 0)} chars)"
                )
                cell["level"] = "episodic"
            elif et == "file_write":
                msg = (
                    f"{event.get('operation', 'write').capitalize()} "
                    f"{_anon(event.get('file_path', ''))} ({event.get('content_length', 0)} chars)"
                )
                content = event.get("_resolved_content")
                if content:
                    preview = self.truncate_content(content, 1500)
                    msg += f"\nContent:\n{preview}"
                cell["content"] = msg
                cell["level"] = "semantic"
            elif et == "file_edit":
                msg = (
                    f"Edit {_anon(event.get('file_path', ''))} "
                    f"(+{event.get('lines_added', 0)}/-{event.get('lines_deleted', 0)})"
                )
                content = event.get("_resolved_content")
                if content:
                    preview = self.truncate_content(content, 1500)
                    msg += f"\nDiff:\n{preview}"
                cell["content"] = msg
                cell["level"] = "semantic"
            elif et == "dir_create":
                cell["content"] = f"Create dir {_anon(event.get('dir_path', ''))} (depth={event.get('depth', 0)})"
                cell["level"] = "procedural"
            elif et in ("file_rename", "file_move", "file_copy", "file_delete"):
                cell["content"] = self.events_to_narrative([event])
                cell["level"] = "procedural"
            elif et == "file_browse":
                cell["content"] = (
                    f"Browse {_anon(event.get('directory_path', ''))} ({event.get('files_listed', 0)} files)"
                )
                cell["level"] = "episodic"
            elif et == "file_search":
                cell["content"] = (
                    f"Search ({event.get('search_type', '')}) "
                    f"'{event.get('query', '')}' -> {event.get('files_matched', 0)} matches"
                )
                cell["level"] = "episodic"
            elif et == "cross_file_reference":
                cell["content"] = (
                    f"Cross-ref: {_anon(event.get('source_file', ''))} -> "
                    f"{_anon(event.get('target_file', ''))} ({event.get('reference_type', '')})"
                )
                cell["level"] = "semantic"
            else:
                continue

            cells.append(cell)

        return cells

    def ingest(self, trajectories: list[dict[str, Any]]) -> None:
        """Ingest trajectories into MemOS memory cells.

        Real mode (llm_fn available): Use LLM to extract structured memories
        (facts, preferences, behavioral patterns) with consolidation.
        Fallback mode: Convert events directly to typed cells.
        """
        self._memories = []
        self._memory_cells = []

        if self._use_real:
            # Real mode: LLM-based structured memory extraction
            total = len(trajectories)
            for ti, traj in enumerate(trajectories, 1):
                events = self.filter_behavioral_events(traj["events"])
                task_id = traj["task_id"]
                narrative = self.events_to_narrative(events)

                # Batch narrative to avoid oversized prompts
                batch_size = 3000  # chars
                for i in range(0, len(narrative), batch_size):
                    batch_text = narrative[i : i + batch_size]
                    new_memories = self._extract_memories_via_llm(batch_text, task_id)
                    self._consolidate_memories(new_memories)
                self._on_trajectory_done(task_id)

            print(
                f"    [memos] Extracted {len(self._memories)} memories "
                f"(facts: {sum(1 for m in self._memories if m['type'] == 'fact')}, "
                f"preferences: {sum(1 for m in self._memories if m['type'] == 'preference')}, "
                f"patterns: {sum(1 for m in self._memories if m['type'] == 'behavioral_pattern')})"
            )
        else:
            # Fallback: rule-based event-to-cell conversion
            for traj in trajectories:
                events = self.filter_behavioral_events(traj["events"])
                task_id = traj["task_id"]
                cells = self._events_to_memory_cells(events, task_id)
                self._memory_cells.extend(cells)

    def _format_memories_for_prompt(self) -> tuple[str, str]:
        """Format LLM-extracted memories grouped by type for inference prompt."""
        by_type: dict[str, list[dict[str, Any]]] = {}
        for m in self._memories:
            by_type.setdefault(m["type"], []).append(m)

        sections = []
        for mem_type in ["fact", "preference", "behavioral_pattern"]:
            mems = by_type.get(mem_type, [])
            if not mems:
                continue
            # Sort by confidence descending
            mems_sorted = sorted(mems, key=lambda x: -x.get("confidence", 0))
            lines = []
            for m in mems_sorted[:50]:  # cap per type
                conf = m.get("confidence", 0)
                task = m.get("task_id", "")
                line = f"  - [{conf:.1f}] {m['content']}"
                if task:
                    line += f" (from {task})"
                consolidated = m.get("consolidated_from", [])
                if consolidated:
                    line += f" [consolidated from {len(consolidated)} similar memories]"
                lines.append(line)
            sections.append(f"### {mem_type.replace('_', ' ').title()} ({len(mems)} total)\n" + "\n".join(lines))

        cells_text = "\n\n".join(sections)

        # Statistics
        total = len(self._memories)
        stats_lines = [
            f"- Total memory cells: {total}",
            f"- Facts: {len(by_type.get('fact', []))}",
            f"- Preferences: {len(by_type.get('preference', []))}",
            f"- Behavioral patterns: {len(by_type.get('behavioral_pattern', []))}",
        ]
        # Count consolidated
        consolidated_count = sum(1 for m in self._memories if m.get("consolidated_from"))
        if consolidated_count:
            stats_lines.append(f"- Consolidated memories: {consolidated_count}")

        # Confidence distribution
        if self._memories:
            confs = [m.get("confidence", 0) for m in self._memories]
            avg_conf = sum(confs) / len(confs)
            stats_lines.append(f"- Average confidence: {avg_conf:.2f}")

        return cells_text, "\n".join(stats_lines)

    def infer_profile(self, profile_attributes: list[str]) -> dict[str, str]:
        """Query MemOS memories for profile reconstruction.

        Real mode: builds prompt from LLM-extracted memories grouped by type.
        Fallback mode: builds prompt from rule-based memory cells.
        """
        if self._use_real and self._memories:
            cells_text, memory_stats = self._format_memories_for_prompt()
        else:
            # Fallback: use simple cell list
            cells_text = "\n".join(f"- [{c['level']}] {c['content']}" for c in self._memory_cells[:120])
            memory_stats = f"- Total cells: {len(self._memory_cells)}"

        attributes_list = "\n".join(f"- {attr}" for attr in profile_attributes)

        prompt = INFERENCE_PROMPT.format(
            memory_cells=cells_text,
            memory_stats=memory_stats,
            attributes_list=attributes_list,
        )

        return {"_prompt": prompt, "_method": "memos"}

    def _get_ingest_state(self):
        return {"memories": self._memories, "memory_cells": self._memory_cells, "use_real": self._use_real}

    def _set_ingest_state(self, state):
        self._memories = state.get("memories", [])
        self._memory_cells = state.get("memory_cells", [])
        self._use_real = state.get("use_real", bool(self._memories))

    def reset(self) -> None:
        super().reset()
        self._memories = []
        self._memory_cells = []
