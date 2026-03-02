"""Zep adapter: graph-based memory with temporal/relational reasoning.

Zep (https://github.com/getzep/zep) builds a knowledge graph from
conversations. We implement its core Graphiti algorithm locally:
1. LLM extracts entity-relationship triples from events
2. networkx stores the knowledge graph with temporal metadata
3. Graph traversal + time filtering for retrieval

EXPECTED: Better than Mem0 on temporal patterns (work rhythm)
but still weaker than FileGramOS on file-system-specific signals.
"""

from __future__ import annotations

import json
from typing import Any

from .base import BaseAdapter
from .registry import register_adapter

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

EXTRACTION_PROMPT = """\
You are extracting entity-relationship triples from file-system behavioral events.

Events:
{narrative}

Extract structured triples that capture the user's behavioral patterns. Output JSON:
{{
  "triples": [
    {{"subject": "<entity>", "relation": "<action/relationship>", "object": "<entity>", "properties": {{"pattern": "<behavioral pattern if any>"}}}},
    ...
  ]
}}

Focus on:
- File access patterns (User READ/WROTE/EDITED specific files)
- Organization patterns (User CREATED_DIR with nesting, MOVED files)
- Naming patterns (User NAMED files with specific conventions)
- Work patterns (User SEARCHED_BEFORE_READING, User REVISED_INCREMENTALLY)
- Content patterns (User PRODUCED comprehensive/minimal output)

Keep triples factual and behavioral. Extract 10-20 triples."""

INFERENCE_PROMPT = """\
You are analyzing a knowledge graph built by Zep from a user's file-system \
behavioral trajectories. The graph captures entities (files, directories, \
the User) and relationships (read, wrote, edited, moved, etc.) with temporal \
ordering across tasks.

Knowledge graph facts:
{graph_facts}

Graph statistics:
{graph_stats}

Based on these graph-structured memories, infer the user's work habit \
profile for:
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


@register_adapter("zep")
class ZepAdapter(BaseAdapter):
    """Adapter for Zep knowledge graph memory system.

    Uses LLM to extract entity-relationship triples, builds a networkx graph.
    """

    def __init__(self, config: dict[str, Any] | None = None, llm_fn=None):
        super().__init__(config, llm_fn=llm_fn)
        self.name = "zep"
        self._graph_facts: list[str] = []
        self._graph: Any = None  # networkx graph
        self._use_real = HAS_NETWORKX and llm_fn is not None

    def _extract_triples_via_llm(self, narrative: str, task_id: str) -> list[dict]:
        """Use LLM to extract entity-relationship triples from event narrative."""
        prompt = EXTRACTION_PROMPT.format(narrative=narrative)
        try:
            response = self._call_llm(prompt, phase="ingest")
            # Parse JSON from response
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(text[start : end + 1])
                triples = data.get("triples", [])
                # Add task_id as temporal metadata
                for t in triples:
                    t["task_id"] = task_id
                return triples
        except Exception as e:
            print(f"    [zep] Triple extraction error: {e}")
        return []

    def _build_graph(self, all_triples: list[dict]):
        """Build networkx graph from extracted triples."""
        self._graph = nx.MultiDiGraph()
        for triple in all_triples:
            subj = str(triple.get("subject", "Unknown"))
            obj = str(triple.get("object", "Unknown"))
            rel = str(triple.get("relation", "RELATED_TO"))
            props = triple.get("properties", {})
            task_id = triple.get("task_id", "")
            # Sanitize props: convert unhashable types to strings
            safe_props = {}
            if isinstance(props, dict):
                for k, v in props.items():
                    safe_props[str(k)] = str(v) if isinstance(v, (list, dict, set)) else v
            self._graph.add_edge(
                subj,
                obj,
                relation=rel,
                task_id=task_id,
                **safe_props,
            )

    def _graph_to_facts(self) -> tuple[str, str]:
        """Convert graph to readable facts and statistics."""
        facts = []
        for u, v, data in self._graph.edges(data=True):
            rel = data.get("relation", "RELATED_TO")
            task = data.get("task_id", "")
            pattern = data.get("pattern", "")
            fact = f"({u}) -[{rel}]-> ({v})"
            if task:
                fact += f" [task:{task}]"
            if pattern:
                fact += f" {{pattern: {pattern}}}"
            facts.append(fact)

        stats_lines = [
            f"- Nodes: {self._graph.number_of_nodes()}",
            f"- Edges: {self._graph.number_of_edges()}",
            f"- Unique relations: {len(set(d.get('relation', '') for _, _, d in self._graph.edges(data=True)))}",
        ]

        # Node degree analysis
        if self._graph.number_of_nodes() > 0:
            degrees = sorted(
                ((n, self._graph.degree(n)) for n in self._graph.nodes()),
                key=lambda x: -x[1],
            )
            stats_lines.append(f"- Most connected nodes: {', '.join(f'{n}({d})' for n, d in degrees[:10])}")

        return "\n".join(f"- {f}" for f in facts[:200]), "\n".join(stats_lines)

    def _events_to_graph_facts(self, events: list[dict[str, Any]], task_id: str) -> list[str]:
        """Fallback: convert events into text-based graph facts."""
        _anon = self._anonymize_path
        facts = []
        for event in events:
            et = event.get("event_type", "")
            if et == "file_read":
                facts.append(
                    f"User READ file '{_anon(event.get('file_path', ''))}' "
                    f"(view #{event.get('view_count', 1)}, "
                    f"{event.get('content_length', 0)} chars) [task:{task_id}]"
                )
            elif et == "file_write":
                op = event.get("operation", "write")
                fact = (
                    f"User {op.upper()} file '{_anon(event.get('file_path', ''))}' "
                    f"({event.get('content_length', 0)} chars) [task:{task_id}]"
                )
                content = event.get("_resolved_content")
                if content:
                    preview = self.truncate_content(content, 1000)
                    fact += f"\nContent:\n{preview}"
                facts.append(fact)
            elif et == "file_edit":
                fact = (
                    f"User EDITED file '{_anon(event.get('file_path', ''))}' "
                    f"(+{event.get('lines_added', 0)}/-{event.get('lines_deleted', 0)}) "
                    f"[task:{task_id}]"
                )
                content = event.get("_resolved_content")
                if content:
                    preview = self.truncate_content(content, 1000)
                    fact += f"\nDiff:\n{preview}"
                facts.append(fact)
            elif et == "dir_create":
                facts.append(
                    f"User CREATED directory '{_anon(event.get('dir_path', ''))}' "
                    f"(depth={event.get('depth', 0)}) [task:{task_id}]"
                )
            elif et == "file_move":
                facts.append(
                    f"User MOVED '{_anon(event.get('old_path', ''))}' -> "
                    f"'{_anon(event.get('new_path', ''))}' [task:{task_id}]"
                )
            elif et == "file_rename":
                facts.append(
                    f"User RENAMED '{_anon(event.get('old_path', ''))}' -> "
                    f"'{_anon(event.get('new_path', ''))}' [task:{task_id}]"
                )
            elif et == "file_copy":
                facts.append(
                    f"User COPIED '{_anon(event.get('source_path', ''))}' -> "
                    f"'{_anon(event.get('dest_path', ''))}' "
                    f"(backup={event.get('is_backup', False)}) [task:{task_id}]"
                )
            elif et == "file_delete":
                facts.append(f"User DELETED '{_anon(event.get('file_path', ''))}' [task:{task_id}]")
            elif et == "cross_file_reference":
                facts.append(
                    f"File '{_anon(event.get('source_file', ''))}' REFERENCED "
                    f"'{_anon(event.get('target_file', ''))}' "
                    f"({event.get('reference_type', '')}) [task:{task_id}]"
                )
            elif et == "file_search":
                facts.append(
                    f"User SEARCHED ({event.get('search_type', '')}): "
                    f"'{event.get('query', '')}' -> {event.get('files_matched', 0)} matches [task:{task_id}]"
                )
        return facts

    def ingest(self, trajectories: list[dict[str, Any]]) -> None:
        """Extract knowledge graph from trajectories."""
        self._graph_facts = []

        if self._use_real:
            # Real mode: LLM-based triple extraction + networkx graph
            import threading
            import time as _time
            from concurrent.futures import ThreadPoolExecutor, as_completed

            total = len(trajectories)
            all_triples = []
            lock = threading.Lock()
            done_count = [0]  # mutable counter

            def _process_trajectory(ti, traj):
                events = self.filter_behavioral_events(traj["events"])
                task_id = traj["task_id"]
                narrative = self.events_to_narrative(events)
                batch_size = 30000
                num_batches = max(1, (len(narrative) + batch_size - 1) // batch_size)
                print(
                    f"    [zep] Trajectory {ti}/{total}: {task_id} | "
                    f"{len(events)} events, {len(narrative)} chars, "
                    f"{num_batches} batch(es)"
                )

                traj_triples = []
                for bi, i in enumerate(range(0, len(narrative), batch_size), 1):
                    batch_text = narrative[i : i + batch_size]
                    t0 = _time.time()
                    triples = self._extract_triples_via_llm(batch_text, task_id)
                    elapsed = _time.time() - t0
                    traj_triples.extend(triples)
                    print(f"      [{task_id}] batch {bi}/{num_batches}: {len(triples)} triples ({elapsed:.1f}s)")

                with lock:
                    all_triples.extend(traj_triples)
                    done_count[0] += 1
                    print(
                        f"    [zep] ✓ {task_id}: {len(traj_triples)} triples | "
                        f"done {done_count[0]}/{total}, total triples: {len(all_triples)}"
                    )
                self._on_trajectory_done(task_id)
                return traj_triples

            INNER_PARALLEL = 4
            print(f"    [zep] Processing {total} trajectories with {INNER_PARALLEL} threads")
            with ThreadPoolExecutor(max_workers=INNER_PARALLEL) as pool:
                futures = {pool.submit(_process_trajectory, ti, traj): ti for ti, traj in enumerate(trajectories, 1)}
                for f in as_completed(futures):
                    f.result()  # raise if error

            print(f"    [zep] Extracted {len(all_triples)} triples")
            self._build_graph(all_triples)
        else:
            # Fallback: simulated text-based facts
            for traj in trajectories:
                events = self.filter_behavioral_events(traj["events"])
                task_id = traj["task_id"]
                facts = self._events_to_graph_facts(events, task_id)
                self._graph_facts.extend(facts)

    def infer_profile(self, profile_attributes: list[str]) -> dict[str, str]:
        """Query knowledge graph for profile reconstruction."""
        if self._use_real and self._graph is not None:
            graph_text, graph_stats = self._graph_to_facts()
        else:
            graph_text = "\n".join(f"- {f}" for f in self._graph_facts[:150])
            graph_stats = f"- Total facts: {len(self._graph_facts)}"

        attributes_list = "\n".join(f"- {attr}" for attr in profile_attributes)

        prompt = INFERENCE_PROMPT.format(
            graph_facts=graph_text,
            graph_stats=graph_stats,
            attributes_list=attributes_list,
        )

        return {"_prompt": prompt, "_method": "zep"}

    def _get_ingest_state(self):
        state = {"graph_facts": self._graph_facts, "use_real": self._use_real}
        if self._graph is not None:
            # Serialize networkx graph as edge list
            state["graph_edges"] = [{"u": u, "v": v, **data} for u, v, data in self._graph.edges(data=True)]
        return state

    def _set_ingest_state(self, state):
        self._graph_facts = state.get("graph_facts", [])
        edges = state.get("graph_edges")
        if edges and HAS_NETWORKX:
            import networkx as nx

            self._graph = nx.MultiDiGraph()
            for edge in edges:
                u, v = edge.pop("u"), edge.pop("v")
                self._graph.add_edge(u, v, **edge)
            self._use_real = True
        else:
            self._graph = None
            self._use_real = False

    def reset(self) -> None:
        super().reset()
        self._graph_facts = []
        self._graph = None
