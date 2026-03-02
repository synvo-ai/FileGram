"""Eager summarization baseline: summarize each trajectory, then infer profile.

Two-stage approach:
  1. For each trajectory, summarize the behavioral events into a natural language summary.
  2. Concatenate summaries, then ask LLM to infer the user profile.

This tests whether eager abstraction (summarize first) loses signal vs.
full-context (raw events) or dimension-aware extraction (FileGramOS).
"""

from __future__ import annotations

from typing import Any

from .base import BaseAdapter
from .registry import register_adapter

SUMMARIZE_PROMPT = """\
You are analyzing a file-system behavioral trajectory from a user performing \
a task on their computer. Summarize this user's behavioral patterns in this \
task session.

Focus on:
- How they explored and read files (order, depth, strategy)
- What they created/wrote (file types, detail level, structure)
- How they organized files (directory structure, naming conventions)
- How they edited/revised work (incremental vs bulk, frequency)
- Version management behavior (backups, overwrites, deletions)
- Cross-modal behavior (use of tables, images, structured data)
- Overall work rhythm and style

Task: {task_id} ({n_events} behavioral events)

Trajectory:
{narrative}

Provide a concise behavioral summary (200-400 words) focusing on observable \
patterns, NOT the task content itself."""

INFERENCE_PROMPT = """\
You are analyzing behavioral summaries from a single user across multiple \
task sessions. Your goal is to infer this user's work habit profile from \
the patterns described in these summaries.

Below are summaries of {n_sessions} task sessions performed by the same user:

{summaries}

Based on ALL the summaries above, infer this user's profile for the \
following attributes. For each attribute, provide your best inference of the \
value and a brief justification.

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


@register_adapter("eager_summarization")
class EagerSummarizationAdapter(BaseAdapter):
    """Baseline: summarize each trajectory eagerly, then infer from summaries."""

    def __init__(self, config: dict[str, Any] | None = None, llm_fn=None):
        super().__init__(config, llm_fn=llm_fn)
        self.name = "eager_summarization"
        self._summarize_prompts: list[dict[str, str]] = []
        self._summaries: list[str] = []

    def ingest(self, trajectories: list[dict[str, Any]]) -> None:
        """Build per-trajectory summarization prompts.

        Does NOT call LLM — the test runner handles that in a pre-pass.
        After the test runner calls set_summaries(), infer_profile() works.
        """
        self._summarize_prompts = []
        self._summaries = []
        for traj in trajectories:
            events = self.filter_behavioral_events(traj["events"])
            task_id = traj["task_id"]
            narrative = self.events_to_narrative(events)
            prompt = SUMMARIZE_PROMPT.format(
                task_id=task_id,
                n_events=len(events),
                narrative=narrative,
            )
            self._summarize_prompts.append(
                {
                    "task_id": task_id,
                    "prompt": prompt,
                }
            )
            self._on_trajectory_done(task_id)

    def get_summarize_prompts(self) -> list[dict[str, str]]:
        """Return the per-trajectory summarization prompts.

        The test runner calls LLM with these, then feeds results back
        via set_summaries().
        """
        return self._summarize_prompts

    def set_summaries(self, summaries: list[str]) -> None:
        """Set LLM-generated summaries (one per trajectory).

        Called by the test runner after the summarization pre-pass.
        """
        self._summaries = summaries

    def infer_profile(self, profile_attributes: list[str]) -> dict[str, str]:
        """Build inference prompt from summaries.

        If summaries haven't been set yet, returns just the summarize prompts
        so the caller knows a pre-pass is needed.
        """
        if not self._summaries:
            return {
                "_prompt": None,
                "_method": "eager_summarization",
                "_needs_summarization": True,
                "_summarize_prompts": [p["prompt"] for p in self._summarize_prompts],
            }

        summaries_text = "\n\n".join(f"=== Session {i + 1} ===\n{s}" for i, s in enumerate(self._summaries))
        attributes_list = "\n".join(f"- {attr}" for attr in profile_attributes)

        prompt = INFERENCE_PROMPT.format(
            n_sessions=len(self._summaries),
            summaries=summaries_text,
            attributes_list=attributes_list,
        )

        return {"_prompt": prompt, "_method": "eager_summarization"}

    def _get_ingest_state(self):
        return {"summarize_prompts": self._summarize_prompts, "summaries": self._summaries}

    def _set_ingest_state(self, state):
        self._summarize_prompts = state["summarize_prompts"]
        self._summaries = state.get("summaries", [])

    def reset(self) -> None:
        super().reset()
        self._summarize_prompts = []
        self._summaries = []
