"""MemU adapter: hierarchical memory-unit system with LLM-based extraction.

MemU organizes memories into a three-level hierarchy:
  Resources -> Items -> Categories

- Resources: raw behavioral observations from file-system events
- Items: higher-level facts/patterns extracted via LLM from resource batches
- Categories: behavioral groupings of items (e.g., reading_behavior, writing_style)

When llm_fn is provided, real LLM extraction is used for Items and Categories.
Otherwise, falls back to simulated importance-weighted memory units.

EXPECTED: Similar to MemOS — structured but not designed for file-system signals.
"""

from __future__ import annotations

import json
from typing import Any

from .base import BaseAdapter
from .registry import register_adapter

ITEM_EXTRACTION_PROMPT = """\
You are extracting higher-level behavioral patterns from raw file-system observations.

Raw observations (resources):
{resources}

Analyze these observations and extract discrete behavioral facts/patterns. Output JSON:
{{
  "items": [
    {{
      "type": "<pattern_type: reading_habit|writing_style|organization|iteration|workflow|cross_modal>",
      "content": "<concise description of the behavioral pattern>",
      "importance": <0.0-1.0 float>,
      "source_resources": [<indices of source observations, 0-based>]
    }},
    ...
  ]
}}

Focus on:
- Reading patterns: sequential vs search-first, depth vs breadth, revisit behavior
- Writing patterns: output length, structure depth, auxiliary files, formatting style
- Organization patterns: directory nesting, naming conventions, backup behavior
- Iteration patterns: edit frequency, edit size, rewrite vs incremental
- Workflow patterns: phased vs interleaved, context switching frequency
- Cross-modal patterns: image/chart creation, table usage, text-only preference

Extract 5-15 items. Each item should be a distinct behavioral pattern, not a raw event."""

CATEGORY_EXTRACTION_PROMPT = """\
You are grouping behavioral pattern items into higher-level categories for a \
memory system analyzing file-system behavior.

Extracted behavioral items:
{items}

Group these items into behavioral categories. Output JSON:
{{
  "categories": [
    {{
      "name": "<category_name>",
      "summary": "<one-sentence summary of the behavioral tendency in this category>",
      "member_items": [<indices of member items, 0-based>],
      "confidence": <0.0-1.0 float>
    }},
    ...
  ]
}}

Use these category names where applicable:
- reading_behavior: how the user explores and consumes information
- writing_style: output format, detail level, structure preferences
- organization_habits: directory structure, naming, file management
- iteration_patterns: how the user revises and refines work
- workflow_rhythm: task phases, context switching, work cadence
- cross_modal_usage: visual materials, tables, charts usage

You may add other categories if needed. Each item should belong to exactly one category."""

INFERENCE_PROMPT_HIERARCHICAL = """\
You are analyzing hierarchical memories extracted by MemU from a user's \
file-system behavioral trajectories. Memories are organized in three levels:

## Categories (high-level behavioral groupings)
{categories_text}

## Items (specific behavioral patterns)
{items_text}

## Resource statistics
- Total raw observations: {resource_count}
- Extracted items: {item_count}
- Categories: {category_count}

Based on these hierarchical memories, infer the user's work habit profile for:
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

INFERENCE_PROMPT_FALLBACK = """\
You are analyzing memory units extracted by MemU from a user's file-system \
behavioral trajectories. Each memory unit captures a discrete behavioral \
observation with type and context.

Memory units:
{memory_units}

Based on these memory units, infer the user's work habit profile for:
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


@register_adapter("memu")
class MemUAdapter(BaseAdapter):
    """Adapter for MemU hierarchical memory-unit system.

    When llm_fn is available, uses real LLM extraction to build a three-level
    hierarchy: Resources -> Items -> Categories.

    When llm_fn is not available, falls back to simulated importance-weighted
    memory units (same as the original implementation).
    """

    def __init__(self, config: dict[str, Any] | None = None, llm_fn=None):
        super().__init__(config, llm_fn=llm_fn)
        self.name = "memu"
        # Fallback storage (always populated)
        self._units: list[dict[str, Any]] = []
        # Hierarchical storage (populated only with LLM)
        self._resources: list[dict[str, Any]] = []
        self._items: list[dict[str, Any]] = []
        self._categories: list[dict[str, Any]] = []
        self._use_real = llm_fn is not None

    def _events_to_resources(self, events: list[dict[str, Any]], task_id: str) -> list[dict[str, Any]]:
        """Convert events into MemU resource-level memory units (Layer 0).

        Resources are raw behavioral observations, one per event.
        """
        _anon = self._anonymize_path
        resources = []
        for event in events:
            et = event.get("event_type", "")

            resource = {
                "type": "observation",
                "content": "",
                "context": {"task_id": task_id},
                "importance": 0.5,
            }

            if et == "file_read":
                resource["type"] = "access"
                resource["content"] = (
                    f"Accessed {_anon(event.get('file_path', ''))} (view #{event.get('view_count', 1)})"
                )
                resource["importance"] = 0.3 + 0.1 * min(event.get("view_count", 1), 5)
            elif et == "file_write":
                resource["type"] = "creation"
                msg = f"Created/wrote {_anon(event.get('file_path', ''))} ({event.get('content_length', 0)} chars)"
                content = event.get("_resolved_content")
                if content:
                    preview = self.truncate_content(content, 1500)
                    msg += f"\nContent:\n{preview}"
                resource["content"] = msg
                resource["importance"] = 0.7
            elif et == "file_edit":
                resource["type"] = "modification"
                msg = (
                    f"Modified {_anon(event.get('file_path', ''))} "
                    f"(+{event.get('lines_added', 0)}/-{event.get('lines_deleted', 0)})"
                )
                content = event.get("_resolved_content")
                if content:
                    preview = self.truncate_content(content, 1500)
                    msg += f"\nDiff:\n{preview}"
                resource["content"] = msg
                resource["importance"] = 0.6
            elif et == "dir_create":
                resource["type"] = "organization"
                resource["content"] = (
                    f"Organized: created {_anon(event.get('dir_path', ''))} (depth={event.get('depth', 0)})"
                )
                resource["importance"] = 0.5
            elif et in ("file_rename", "file_move"):
                resource["type"] = "organization"
                resource["content"] = self.events_to_narrative([event])
                resource["importance"] = 0.5
            elif et == "file_copy":
                resource["type"] = "versioning"
                resource["content"] = (
                    f"Copied {_anon(event.get('source_path', ''))} -> "
                    f"{_anon(event.get('dest_path', ''))} "
                    f"(backup={event.get('is_backup', False)})"
                )
                resource["importance"] = 0.6
            elif et == "file_delete":
                resource["type"] = "cleanup"
                resource["content"] = f"Deleted {_anon(event.get('file_path', ''))}"
                resource["importance"] = 0.4
            elif et == "file_search":
                resource["type"] = "search"
                resource["content"] = (
                    f"Searched ({event.get('search_type', '')}): "
                    f"'{event.get('query', '')}' -> {event.get('files_matched', 0)} matches"
                )
                resource["importance"] = 0.4
            elif et == "file_browse":
                resource["type"] = "exploration"
                resource["content"] = (
                    f"Explored {_anon(event.get('directory_path', ''))} ({event.get('files_listed', 0)} items)"
                )
                resource["importance"] = 0.3
            else:
                continue

            resources.append(resource)

        return resources

    def _extract_items_via_llm(self, resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Use LLM to extract higher-level items from resource batches (Layer 1)."""
        if not resources:
            return []

        # Format resources for the prompt
        resource_lines = []
        for i, r in enumerate(resources):
            resource_lines.append(f"[{i}] [{r['type']}] (imp={r['importance']:.1f}) {r['content']}")
        resources_text = "\n".join(resource_lines)

        prompt = ITEM_EXTRACTION_PROMPT.format(resources=resources_text)
        try:
            response = self._call_llm(prompt, phase="ingest")
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(text[start : end + 1])
                items = data.get("items", [])
                # Validate and normalize items
                validated = []
                for item in items:
                    validated.append(
                        {
                            "type": item.get("type", "unknown"),
                            "content": item.get("content", ""),
                            "importance": float(item.get("importance", 0.5)),
                            "source_resources": item.get("source_resources", []),
                        }
                    )
                return validated
        except Exception as e:
            print(f"    [memu] Item extraction error: {e}")
        return []

    def _extract_categories_via_llm(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Use LLM to group items into behavioral categories (Layer 2)."""
        if not items:
            return []

        # Format items for the prompt
        item_lines = []
        for i, item in enumerate(items):
            item_lines.append(f"[{i}] [{item['type']}] (imp={item['importance']:.1f}) {item['content']}")
        items_text = "\n".join(item_lines)

        prompt = CATEGORY_EXTRACTION_PROMPT.format(items=items_text)
        try:
            response = self._call_llm(prompt, phase="ingest")
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(text[start : end + 1])
                categories = data.get("categories", [])
                # Validate and normalize categories
                validated = []
                for cat in categories:
                    validated.append(
                        {
                            "name": cat.get("name", "unknown"),
                            "summary": cat.get("summary", ""),
                            "member_items": cat.get("member_items", []),
                            "confidence": float(cat.get("confidence", 0.5)),
                        }
                    )
                return validated
        except Exception as e:
            print(f"    [memu] Category extraction error: {e}")
        return []

    def ingest(self, trajectories: list[dict[str, Any]]) -> None:
        """Convert trajectories into hierarchical MemU memory.

        With llm_fn: Resources -> (LLM) -> Items -> (LLM) -> Categories
        Without llm_fn: Resources only (fallback mode)
        """
        self._resources = []
        self._items = []
        self._categories = []
        self._units = []

        # Step 1: Build resources from all trajectories
        total = len(trajectories)
        for ti, traj in enumerate(trajectories, 1):
            events = self.filter_behavioral_events(traj["events"])
            task_id = traj["task_id"]
            resources = self._events_to_resources(events, task_id)
            self._resources.extend(resources)
            self._on_trajectory_done(task_id)

        # Also populate _units for fallback mode
        self._units = list(self._resources)

        if not self._use_real:
            return

        # Step 2: Extract items from resource batches via LLM
        batch_size = 50  # resources per batch
        all_items = []
        for i in range(0, len(self._resources), batch_size):
            batch = self._resources[i : i + batch_size]
            items = self._extract_items_via_llm(batch)
            # Offset source_resources indices to global
            for item in items:
                item["source_resources"] = [idx + i for idx in item["source_resources"] if isinstance(idx, int)]
            all_items.extend(items)

        self._items = all_items
        print(f"    [memu] Extracted {len(self._items)} items from {len(self._resources)} resources")

        # Step 3: Extract categories from items via LLM
        if self._items:
            self._categories = self._extract_categories_via_llm(self._items)
            print(f"    [memu] Grouped into {len(self._categories)} categories")

    def infer_profile(self, profile_attributes: list[str]) -> dict[str, str]:
        """Build inference prompt from hierarchical or flat memories."""
        attributes_list = "\n".join(f"- {attr}" for attr in profile_attributes)

        if self._use_real and self._items:
            # Hierarchical prompt: categories -> items
            categories_text = ""
            if self._categories:
                cat_lines = []
                for cat in self._categories:
                    members = cat.get("member_items", [])
                    cat_lines.append(
                        f"### {cat['name']} (confidence={cat['confidence']:.1f}, "
                        f"{len(members)} items)\n{cat['summary']}"
                    )
                categories_text = "\n\n".join(cat_lines)
            else:
                categories_text = "(no categories extracted)"

            item_lines = []
            for i, item in enumerate(self._items):
                item_lines.append(f"[{i}] [{item['type']}] (imp={item['importance']:.1f}) {item['content']}")
            items_text = "\n".join(item_lines)

            prompt = INFERENCE_PROMPT_HIERARCHICAL.format(
                categories_text=categories_text,
                items_text=items_text,
                resource_count=len(self._resources),
                item_count=len(self._items),
                category_count=len(self._categories),
                attributes_list=attributes_list,
            )
        else:
            # Fallback: importance-weighted flat units
            sorted_units = sorted(self._units, key=lambda u: -u["importance"])
            units_text = "\n".join(
                f"- [{u['type']}] (imp={u['importance']:.1f}) {u['content']}" for u in sorted_units[:120]
            )
            prompt = INFERENCE_PROMPT_FALLBACK.format(
                memory_units=units_text,
                attributes_list=attributes_list,
            )

        return {"_prompt": prompt, "_method": "memu"}

    def _get_ingest_state(self):
        return {
            "units": self._units,
            "resources": self._resources,
            "items": self._items,
            "categories": self._categories,
            "use_real": self._use_real,
        }

    def _set_ingest_state(self, state):
        self._units = state.get("units", [])
        self._resources = state.get("resources", [])
        self._items = state.get("items", [])
        self._categories = state.get("categories", [])
        self._use_real = state.get("use_real", bool(self._items))

    def reset(self) -> None:
        super().reset()
        self._units = []
        self._resources = []
        self._items = []
        self._categories = []
