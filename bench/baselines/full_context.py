"""Full-context baseline: feed ALL trajectory data directly into LLM context.

This is the lower-bound baseline. No memory processing — just dump everything
into the context window and ask the LLM to infer the profile.
"""

from __future__ import annotations

from typing import Any

from .base import BaseAdapter
from .registry import register_adapter

INFERENCE_PROMPT = """\
You are analyzing file-system behavioral trajectories from a single user \
performing various tasks on their computer. Your goal is to infer this user's \
work habit profile from their behavioral patterns.

Below are the user's behavioral trajectories across {n_tasks} tasks. Each \
trajectory records the sequence of file operations (reads, writes, edits, \
searches, directory operations, etc.) the user performed.

{trajectories}

Based on ALL the trajectories above, infer this user's profile for the \
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


@register_adapter("full_context")
class FullContextAdapter(BaseAdapter):
    """Baseline: concatenate all trajectories into a single LLM prompt."""

    def __init__(self, config: dict[str, Any] | None = None, llm_fn=None):
        super().__init__(config, llm_fn=llm_fn)
        self.name = "full_context"
        self._narratives: list[str] = []

    def ingest(self, trajectories: list[dict[str, Any]]) -> None:
        """Convert trajectories to narratives and store."""
        self._narratives = []
        for traj in trajectories:
            events = self.filter_behavioral_events(traj["events"])
            task_id = traj["task_id"]
            narrative = self.events_to_narrative(events)
            self._narratives.append(f"=== Task: {task_id} ({len(events)} events) ===\n{narrative}")
            self._on_trajectory_done(task_id)

    # Max chars for trajectory content to stay within TPM limits.
    # 160K chars ≈ 40K tokens; with prompt template + max_tokens=4096 ≈ 45K total.
    MAX_PROMPT_CHARS = 160_000

    def infer_profile(self, profile_attributes: list[str]) -> dict[str, str]:
        """Build the full-context prompt for LLM inference.

        Returns the prompt dict — actual LLM call is done by run_eval.py.
        Truncates trajectory content to fit within TPM limits if needed.
        """
        trajectories_text = "\n\n".join(self._narratives)

        if len(trajectories_text) > self.MAX_PROMPT_CHARS:
            # Truncate evenly: keep as many full trajectories as possible,
            # then truncate the last one that fits
            truncated = []
            remaining = self.MAX_PROMPT_CHARS
            for narr in self._narratives:
                if remaining <= 0:
                    break
                if len(narr) <= remaining:
                    truncated.append(narr)
                    remaining -= len(narr) + 2  # account for "\n\n"
                else:
                    truncated.append(narr[:remaining] + "\n... [truncated]")
                    remaining = 0
            trajectories_text = "\n\n".join(truncated)
            print(
                f"    [full_context] Truncated: {len(self._narratives)} trajectories "
                f"→ {len(truncated)} kept, {len(trajectories_text)} chars"
            )

        attributes_list = "\n".join(f"- {attr}" for attr in profile_attributes)

        prompt = INFERENCE_PROMPT.format(
            n_tasks=len(self._narratives),
            trajectories=trajectories_text,
            attributes_list=attributes_list,
        )

        return {"_prompt": prompt, "_method": "full_context"}

    def _get_ingest_state(self):
        return {"narratives": self._narratives}

    def _set_ingest_state(self, state):
        self._narratives = state["narratives"]

    def reset(self) -> None:
        super().reset()
        self._narratives = []
