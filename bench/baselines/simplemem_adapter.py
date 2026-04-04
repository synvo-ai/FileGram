"""SimpleMem adapter: semantic lossless compression with sliding-window extraction.

SimpleMem (https://github.com/aiming-lab/SimpleMem) is a three-stage memory
pipeline designed for long-term personalization from dialogues:

  Stage 1: Semantic Lossless Compression — sliding-window extraction of
           structured MemoryEntry objects with coreference resolution and
           temporal anchoring.
  Stage 2: Online Semantic Synthesis — cross-window deduplication, merge
           similar entries, confidence decay for stale memories, prune
           low-confidence entries.
  Stage 3: Intent-aware Multi-view Retrieval — query analysis → targeted
           retrieval across semantic (embedding), lexical (keyword), and
           structured (entity/topic) views.

Adapted for file-system behavioral trajectories: events are converted to
narrative text, then processed through the sliding-window extraction pipeline.

Registered as "simplemem" for benchmark comparison.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import BaseAdapter
from .registry import register_adapter

# ── Stage 1: Sliding-Window Extraction Prompt ─────────────────

EXTRACTION_PROMPT = """\
You are a memory extraction module for a long-term personalization system.
Your task is to extract structured memory entries from a user's file-system
behavioral events. Each memory captures a distinct behavioral observation
in a LOSSLESS manner — preserve all specific details (file names, counts,
patterns) without vague summarization.

Events (window {window_idx}):
{narrative}

{previous_context}

Extract memory entries. Each entry captures ONE distinct behavioral observation.
Output JSON:
{{
  "entries": [
    {{
      "lossless_restatement": "<precise, detail-preserving description of a behavioral observation>",
      "keywords": ["<key1>", "<key2>"],
      "topic": "<behavioral category: reading|writing|editing|organizing|searching|cross-modal|workflow>",
      "entities": ["<specific file names, tools, or patterns referenced>"],
      "confidence": <0.0-1.0, higher for directly observed, lower for inferred>
    }}
  ]
}}

Rules:
- Extract 5-12 entries per window.
- Each entry must be SELF-CONTAINED — understandable without reading the events.
- Resolve coreferences: use full file names, not "the file" or "it".
- Preserve quantities: "read 7 files" not "read several files".
- If you see a pattern that was already captured in previous entries, \
SKIP it (deduplication). Only add genuinely NEW observations.
- Focus on behavioral patterns: HOW the user works, not just WHAT files exist.
- Categories of observations to look for:
  * Reading habits: order, depth, revisit patterns, search vs browse
  * Writing style: content length, structure, heading depth, auxiliary files
  * Organization: directory creation, naming conventions, file placement
  * Editing: edit frequency, size of edits, overwrite vs incremental
  * Curation: deletion patterns, backup behavior, workspace cleanup
  * Cross-modal: tables, images, visual elements in output"""

# ── Stage 2: Synthesis Prompt (merge + deduplicate) ───────────

SYNTHESIS_PROMPT = """\
You are a memory consolidation module. You have a set of memory entries \
extracted from multiple windows of a user's behavioral events. Some entries \
may be redundant, contradictory, or refinable.

Current memory entries:
{entries_text}

Consolidate these entries:
1. MERGE entries that describe the same pattern (keep the more specific version).
2. RESOLVE contradictions by keeping the entry with more supporting evidence.
3. INCREASE confidence for patterns observed multiple times.
4. DECREASE confidence for patterns observed only once or in unusual contexts.
5. PRUNE entries with confidence < 0.2 after adjustment.

Output JSON:
{{
  "consolidated": [
    {{
      "lossless_restatement": "<consolidated description>",
      "keywords": ["<key1>", "<key2>"],
      "topic": "<behavioral category>",
      "entities": ["<entities>"],
      "confidence": <adjusted 0.0-1.0>,
      "observation_count": <how many original entries supported this>
    }}
  ]
}}

Keep 15-40 entries. Prioritize behavioral patterns over individual actions."""

# ── Stage 3: Inference Prompt ──────────────────────────────────

INFERENCE_PROMPT = """\
You are analyzing memory entries extracted from a user's file-system \
behavioral trajectories by SimpleMem, a semantic lossless compression system.

Memory entries (sorted by confidence):
{memory_text}

Memory statistics:
{memory_stats}

Based on these memories, infer the user's work habit profile for:
{attributes_list}

Respond in JSON format:
{{
  "inferred_profile": {{
    "<attribute_name>": {{
      "value": "<inferred value>",
      "justification": "<brief reasoning from memory entries>"
    }},
    ...
  }}
}}"""

# ── Configuration ──────────────────────────────────────────────

WINDOW_SIZE = 30  # events per window
WINDOW_OVERLAP = 3  # overlap between consecutive windows
MAX_ENTRIES = 80  # max entries before triggering synthesis
SYNTHESIS_THRESHOLD = 60  # trigger synthesis when entries exceed this


@register_adapter("simplemem")
class SimpleMemAdapter(BaseAdapter):
    """Adapter for SimpleMem semantic lossless compression pipeline.

    Three-stage pipeline adapted for file-system behavioral events:
    1. Sliding-window extraction: LLM extracts structured MemoryEntry objects
    2. Online synthesis: merge similar entries, resolve contradictions
    3. Intent-aware retrieval: format memories for profile inference
    """

    def __init__(self, config: dict[str, Any] | None = None, llm_fn=None):
        super().__init__(config, llm_fn=llm_fn)
        self.name = "simplemem"
        self._entries: list[dict[str, Any]] = []
        self._use_real = llm_fn is not None

    # ── Stage 1: Sliding-Window Extraction ─────────────────────

    def _extract_window(
        self,
        narrative: str,
        window_idx: int,
        previous_entries: list[dict[str, Any]],
        task_id: str,
    ) -> list[dict[str, Any]]:
        """Extract memory entries from a single window of events."""
        # Build previous context for deduplication
        prev_context = ""
        if previous_entries:
            prev_lines = [
                f"- {e['lossless_restatement']}" for e in previous_entries[-15:]
            ]
            prev_context = (
                "Previously extracted entries (DO NOT duplicate these):\n"
                + "\n".join(prev_lines)
            )

        prompt = EXTRACTION_PROMPT.format(
            narrative=narrative,
            window_idx=window_idx,
            previous_context=prev_context,
        )

        try:
            response = self._call_llm(prompt, phase="ingest")
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(text[start : end + 1])
                entries = data.get("entries", [])
                for e in entries:
                    e["task_id"] = task_id
                    e["window_idx"] = window_idx
                    e.setdefault("lossless_restatement", "")
                    e.setdefault("keywords", [])
                    e.setdefault("topic", "workflow")
                    e.setdefault("entities", [])
                    e.setdefault("confidence", 0.5)
                return [e for e in entries if e["lossless_restatement"]]
        except Exception as exc:
            print(f"    [simplemem] Window {window_idx} extraction error: {exc}")
        return []

    def _sliding_window_extract(
        self,
        events: list[dict[str, Any]],
        task_id: str,
    ) -> list[dict[str, Any]]:
        """Process events through sliding windows, extracting entries."""
        if not events:
            return []

        all_entries: list[dict[str, Any]] = []
        n = len(events)
        window_idx = 0
        pos = 0

        while pos < n:
            window_end = min(pos + WINDOW_SIZE, n)
            window_events = events[pos:window_end]
            narrative = self.events_to_narrative(window_events)

            new_entries = self._extract_window(
                narrative, window_idx, all_entries, task_id
            )
            all_entries.extend(new_entries)

            pos += WINDOW_SIZE - WINDOW_OVERLAP
            window_idx += 1

        return all_entries

    # ── Stage 2: Online Semantic Synthesis ─────────────────────

    def _simple_dedup(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fast rule-based deduplication by keyword/topic overlap."""
        if len(entries) <= 10:
            return entries

        kept: list[dict[str, Any]] = []
        for entry in entries:
            is_dup = False
            e_words = set(entry.get("lossless_restatement", "").lower().split())
            for existing in kept:
                ex_words = set(
                    existing.get("lossless_restatement", "").lower().split()
                )
                if not e_words or not ex_words:
                    continue
                overlap = len(e_words & ex_words) / min(len(e_words), len(ex_words))
                if overlap > 0.5:
                    # Merge: boost confidence of existing
                    existing["confidence"] = min(
                        1.0,
                        existing.get("confidence", 0.5) + 0.1,
                    )
                    existing.setdefault("observation_count", 1)
                    existing["observation_count"] += 1
                    # Merge keywords and entities
                    existing["keywords"] = list(
                        set(existing.get("keywords", []))
                        | set(entry.get("keywords", []))
                    )
                    existing["entities"] = list(
                        set(existing.get("entities", []))
                        | set(entry.get("entities", []))
                    )
                    is_dup = True
                    break
            if not is_dup:
                entry.setdefault("observation_count", 1)
                kept.append(entry)
        return kept

    def _llm_synthesis(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """LLM-based consolidation when entry count is high."""
        entries_lines = []
        for i, e in enumerate(entries, 1):
            conf = e.get("confidence", 0.5)
            topic = e.get("topic", "?")
            obs = e.get("observation_count", 1)
            line = f"[{i}] ({topic}, conf={conf:.1f}, obs={obs}) {e['lossless_restatement']}"
            entries_lines.append(line)

        prompt = SYNTHESIS_PROMPT.format(entries_text="\n".join(entries_lines))

        try:
            response = self._call_llm(prompt, phase="ingest")
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(text[start : end + 1])
                consolidated = data.get("consolidated", [])
                for e in consolidated:
                    e.setdefault("lossless_restatement", "")
                    e.setdefault("keywords", [])
                    e.setdefault("topic", "workflow")
                    e.setdefault("entities", [])
                    e.setdefault("confidence", 0.5)
                    e.setdefault("observation_count", 1)
                return [e for e in consolidated if e["lossless_restatement"]]
        except Exception as exc:
            print(f"    [simplemem] Synthesis error: {exc}")

        # Fallback: just return deduplicated entries
        return entries

    def _synthesize(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply synthesis pipeline: dedup → optional LLM consolidation."""
        # Step 1: fast rule-based dedup
        deduped = self._simple_dedup(entries)

        # Step 2: LLM synthesis if still too many entries
        if self._use_real and len(deduped) > SYNTHESIS_THRESHOLD:
            deduped = self._llm_synthesis(deduped)
            deduped = self._simple_dedup(deduped)  # final dedup pass

        # Step 3: cap and sort by confidence
        deduped.sort(key=lambda e: -e.get("confidence", 0))
        return deduped[:MAX_ENTRIES]

    # ── Fallback: Rule-based extraction (no LLM) ──────────────

    def _extract_fallback(
        self,
        events: list[dict[str, Any]],
        task_id: str,
    ) -> list[dict[str, Any]]:
        """Extract memory entries without LLM, using rule-based heuristics."""
        _anon = self._anonymize_path
        entries: list[dict[str, Any]] = []

        for event in events:
            et = event.get("event_type", "")
            entry: dict[str, Any] | None = None

            if et == "file_read":
                entry = {
                    "lossless_restatement": (
                        f"Read {_anon(event.get('file_path', '?'))} "
                        f"(view #{event.get('view_count', 1)}, "
                        f"{event.get('content_length', 0)} chars)"
                    ),
                    "keywords": ["read", "file_access"],
                    "topic": "reading",
                    "entities": [_anon(event.get("file_path", ""))],
                    "confidence": 0.8,
                }
            elif et == "file_write":
                op = event.get("operation", "write")
                entry = {
                    "lossless_restatement": (
                        f"{op.capitalize()} {_anon(event.get('file_path', '?'))} "
                        f"({event.get('content_length', 0)} chars)"
                    ),
                    "keywords": [op, "production"],
                    "topic": "writing",
                    "entities": [_anon(event.get("file_path", ""))],
                    "confidence": 0.8,
                }
            elif et == "file_edit":
                entry = {
                    "lossless_restatement": (
                        f"Edit {_anon(event.get('file_path', '?'))} "
                        f"(+{event.get('lines_added', 0)}/-{event.get('lines_deleted', 0)} lines)"
                    ),
                    "keywords": ["edit", "iteration"],
                    "topic": "editing",
                    "entities": [_anon(event.get("file_path", ""))],
                    "confidence": 0.8,
                }
            elif et == "dir_create":
                entry = {
                    "lossless_restatement": (
                        f"Create directory {_anon(event.get('dir_path', '?'))} "
                        f"(depth={event.get('depth', 0)})"
                    ),
                    "keywords": ["directory", "organization"],
                    "topic": "organizing",
                    "entities": [_anon(event.get("dir_path", ""))],
                    "confidence": 0.8,
                }
            elif et == "file_search":
                entry = {
                    "lossless_restatement": (
                        f"Search ({event.get('search_type', '?')}): "
                        f"'{event.get('query', '')}' → {event.get('files_matched', 0)} matches"
                    ),
                    "keywords": ["search", "information_seeking"],
                    "topic": "searching",
                    "entities": [],
                    "confidence": 0.8,
                }
            elif et == "file_delete":
                entry = {
                    "lossless_restatement": f"Delete {_anon(event.get('file_path', '?'))}",
                    "keywords": ["delete", "curation"],
                    "topic": "organizing",
                    "entities": [_anon(event.get("file_path", ""))],
                    "confidence": 0.8,
                }
            elif et in ("file_rename", "file_move", "file_copy"):
                src = event.get("old_path") or event.get("source_path", "?")
                dst = event.get("new_path") or event.get("dest_path", "?")
                entry = {
                    "lossless_restatement": f"{et}: {_anon(src)} → {_anon(dst)}",
                    "keywords": [et.split("_")[1], "organization"],
                    "topic": "organizing",
                    "entities": [_anon(src), _anon(dst)],
                    "confidence": 0.8,
                }

            if entry:
                entry["task_id"] = task_id
                entry["observation_count"] = 1
                entries.append(entry)

        return entries

    # ── Main Pipeline ──────────────────────────────────────────

    def ingest(self, trajectories: list[dict[str, Any]]) -> None:
        """Ingest trajectories through the SimpleMem pipeline.

        Real mode: sliding-window LLM extraction → synthesis.
        Fallback mode: rule-based extraction → dedup.
        """
        self._entries = []
        all_raw: list[dict[str, Any]] = []

        for traj in trajectories:
            events = self.filter_behavioral_events(traj["events"])
            task_id = traj["task_id"]

            if self._use_real:
                raw = self._sliding_window_extract(events, task_id)
            else:
                raw = self._extract_fallback(events, task_id)

            all_raw.extend(raw)
            self._on_trajectory_done(task_id)

        # Stage 2: synthesis (dedup + optional LLM consolidation)
        self._entries = self._synthesize(all_raw)

        # Stats
        topics = {}
        for e in self._entries:
            t = e.get("topic", "other")
            topics[t] = topics.get(t, 0) + 1
        topic_str = ", ".join(f"{k}:{v}" for k, v in sorted(topics.items()))
        print(
            f"    [simplemem] {len(all_raw)} raw → {len(self._entries)} entries "
            f"(topics: {topic_str})"
        )

    # ── Stage 3: Retrieval / Inference ─────────────────────────

    def infer_profile(self, profile_attributes: list[str]) -> dict[str, str]:
        """Build inference prompt from stored memory entries."""
        # Sort by confidence descending
        sorted_entries = sorted(
            self._entries, key=lambda e: -e.get("confidence", 0)
        )

        # Format entries
        lines = []
        for i, e in enumerate(sorted_entries[:60], 1):
            conf = e.get("confidence", 0.5)
            topic = e.get("topic", "?")
            obs = e.get("observation_count", 1)
            kw = ", ".join(e.get("keywords", [])[:5])
            line = (
                f"[{i}] (topic={topic}, conf={conf:.2f}, obs={obs}) "
                f"{e['lossless_restatement']}"
            )
            if kw:
                line += f" [kw: {kw}]"
            lines.append(line)

        memory_text = "\n".join(lines)

        # Statistics
        topics = {}
        for e in self._entries:
            t = e.get("topic", "other")
            topics[t] = topics.get(t, 0) + 1
        avg_conf = (
            sum(e.get("confidence", 0) for e in self._entries) / len(self._entries)
            if self._entries
            else 0
        )
        stats_lines = [
            f"- Total memory entries: {len(self._entries)}",
            f"- Average confidence: {avg_conf:.2f}",
            f"- Topics: {', '.join(f'{k} ({v})' for k, v in sorted(topics.items()))}",
        ]
        multi_obs = sum(1 for e in self._entries if e.get("observation_count", 1) > 1)
        if multi_obs:
            stats_lines.append(f"- Multi-observation entries: {multi_obs}")

        attributes_list = "\n".join(f"- {attr}" for attr in profile_attributes)

        prompt = INFERENCE_PROMPT.format(
            memory_text=memory_text,
            memory_stats="\n".join(stats_lines),
            attributes_list=attributes_list,
        )

        return {"_prompt": prompt, "_method": "simplemem"}

    # ── Pickle Cache ───────────────────────────────────────────

    def _get_ingest_state(self) -> dict[str, Any]:
        return {"entries": self._entries, "use_real": self._use_real}

    def _set_ingest_state(self, state: dict[str, Any]) -> None:
        self._entries = state.get("entries", [])
        self._use_real = state.get("use_real", bool(self._entries))

    def reset(self) -> None:
        super().reset()
        self._entries = []
