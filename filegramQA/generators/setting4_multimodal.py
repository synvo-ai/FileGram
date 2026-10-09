"""Setting 4: Multimodal — File Grounding MCQ & Visual Grounding Open-ended.

Track 4 tests whether memory systems degrade when output content is rendered as
PDF/images instead of Markdown text.  Two sub-tracks:

  File Grounding MCQ
    Re-uses content-dependent questions from Tracks 1-3.  During evaluation the
    Markdown output blobs are converted to PDF/images before memory ingestion;
    behavioural event logs remain unchanged.  Systems relying on structured traces
    (e.g. FileGramOS) should be modality-invariant; content-dependent baselines
    will degrade.

    Selection criteria (principled, reproducible):
      HIGH  — answer REQUIRES content analysis:
              • semantic_attribute (all)
              • dimension_classification where target dim ∈ {B, F}
      MEDIUM — closest behavioural distractor distinguishable only via content:
              • behavioral_fingerprint where min event-dim distance to
                closest distractor ≤ 1 (i.e. A/C/D/E nearly identical,
                B/F is the primary differentiator)
              • counterfactual on create / synthesize tasks (production
                output is the main signal)

  Visual Grounding Open-ended
    Template questions with placeholder ``video_path`` fields.  To be populated
    once screen-recording data is collected.  Two question types:
      • Video-to-Profile Reconstruction — infer L/M/R profile from recording
      • Proactive Prediction — predict next actions from partial recording
"""

from __future__ import annotations

import copy
import random
from typing import Any

from .base import (
    DIMENSIONS, DIM_NAMES, TIER_LABELS, TASK_TYPES, TASK_NAMES,
    ProfileMatrix, TrajectoryStore,
    SEMANTIC_DIM_DESCS,
)

# Event-based dimensions (content-independent)
_EVENT_DIMS = ("A", "C", "D", "E")

# Task types where production output / cross-modal is the main signal
_CONTENT_TASK_TYPES = ("create", "synthesize")


# ── helpers ──────────────────────────────────────────────────

def _min_event_dim_distance(
    profile_id: str, distractors: list[str], matrix: ProfileMatrix,
) -> int:
    """Min number of event-based dims (A/C/D/E) that differ with closest distractor."""
    best = len(_EVENT_DIMS)
    for dp in distractors:
        if dp not in matrix.profiles or profile_id not in matrix.profiles:
            continue
        diff = sum(
            1 for d in _EVENT_DIMS
            if matrix.get_tier(profile_id, d) != matrix.get_tier(dp, d)
        )
        best = min(best, diff)
    return best


def _content_dependency(q: dict, matrix: ProfileMatrix) -> tuple[str, str]:
    """Return (level, reason) for content dependency.

    Levels: "high", "medium", "none".
    """
    sc = q.get("sub_class", "")

    # HIGH: answer directly requires content analysis
    if sc == "semantic_attribute":
        attr = q.get("metadata", {}).get("attribute", "")
        return "high", f"semantic_attribute:{attr}"

    if sc == "dimension_classification":
        dim = q.get("metadata", {}).get("target_dimension", "")
        if dim in ("B", "F"):
            return "high", f"dim_cls_{dim}"
        return "none", ""

    # MEDIUM: content helps differentiate
    if sc == "behavioral_fingerprint":
        pid = q.get("profile_id", "")
        distractors = q.get("metadata", {}).get("distractor_profiles", [])
        min_ev = _min_event_dim_distance(pid, distractors, matrix)
        if min_ev <= 1:
            return "medium", f"fingerprint_event_diff={min_ev}"
        return "none", ""

    if sc == "counterfactual":
        task_type = q.get("metadata", {}).get("task_type", "")
        if task_type in _CONTENT_TASK_TYPES:
            return "medium", f"counterfactual_{task_type}"
        return "none", ""

    return "none", ""


# ── Setting 4 Generator ─────────────────────────────────────

class Setting4Generator:
    """Generate Track 4 multimodal questions.

    Unlike Settings 1-3 which generate from trajectories, Setting 4
    selects content-dependent questions from existing T1-T3 output and
    creates visual grounding templates.

    Parameters
    ----------
    store : TrajectoryStore
        Loaded trajectory data (used for visual grounding profile list).
    matrix : ProfileMatrix
        Profile matrix for content-dependency scoring.
    source_questions : list[dict]
        Already-generated T1-T3 questions to filter for file grounding.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        store: TrajectoryStore,
        matrix: ProfileMatrix,
        source_questions: list[dict],
        seed: int = 42,
        max_file_grounding: int = 550,
    ):
        self.store = store
        self.matrix = matrix
        self.source_questions = source_questions
        self.max_file_grounding = max_file_grounding
        self.rng = random.Random(seed)

    def generate_all(self) -> list[dict]:
        questions: list[dict] = []
        questions.extend(self._select_file_grounding())
        questions.extend(self._gen_visual_grounding())
        return questions

    # ── File Grounding MCQ ───────────────────────────────────

    def _select_file_grounding(self) -> list[dict]:
        """Select content-dependent T1-T3 questions for file grounding eval.

        Priority order (all HIGH kept, MEDIUM filled by event-diff closeness):
          1. HIGH  — sem_attr, dim_cls B/F  (always included)
          2. MEDIUM priority-1 — fingerprint event-diff=0  (B/F sole differentiator)
          3. MEDIUM priority-2 — counterfactual create/synthesize
          4. MEDIUM priority-3 — fingerprint event-diff=1  (sampled to fill cap)
        """
        # Bucket candidates by priority
        buckets: dict[int, list[tuple[dict, str, str]]] = {1: [], 2: [], 3: [], 4: []}

        for q in self.source_questions:
            level, reason = _content_dependency(q, self.matrix)
            if level == "none":
                continue

            if level == "high":
                buckets[1].append((q, level, reason))
            elif q.get("sub_class", "") == "behavioral_fingerprint":
                # Split fingerprint by event-diff distance
                pid = q.get("profile_id", "")
                distractors = q.get("metadata", {}).get("distractor_profiles", [])
                min_ev = _min_event_dim_distance(pid, distractors, self.matrix)
                if min_ev == 0:
                    buckets[2].append((q, level, reason))
                else:
                    buckets[4].append((q, level, reason))
            else:
                # counterfactual create/synthesize
                buckets[3].append((q, level, reason))

        # Fill in priority order up to cap
        selected_raw: list[tuple[dict, str, str]] = []
        remaining = self.max_file_grounding

        for pri in (1, 2, 3, 4):
            bucket = buckets[pri]
            if len(bucket) <= remaining:
                selected_raw.extend(bucket)
                remaining -= len(bucket)
            else:
                selected_raw.extend(self.rng.sample(bucket, remaining))
                remaining = 0
            if remaining == 0:
                break

        # Build output questions
        selected = []
        for q, level, reason in selected_raw:
            fq = copy.deepcopy(q)

            old_id = fq.get("question_id", "")
            fq["question_id"] = f"s4_fg_{old_id}"
            fq["sub_type"] = "file_grounding"

            fq.setdefault("metadata", {})
            fq["metadata"]["original_track"] = q.get("track", "")
            fq["metadata"]["original_sub_track"] = q.get("sub_track", "")
            fq["metadata"]["original_sub_class"] = q.get("sub_class", "")
            fq["metadata"]["content_dependency"] = level
            fq["metadata"]["content_dependency_reason"] = reason
            fq["metadata"]["rendering"] = "multimodal"

            for key in ("track", "sub_track", "sub_class", "channel"):
                fq.pop(key, None)

            selected.append(fq)

        return selected

    # ── Visual Grounding Open-ended ──────────────────────────

    def _gen_visual_grounding(self) -> list[dict]:
        """Generate visual grounding template questions with placeholder video_path.

        Creates 5 questions per profile (3 reconstruction + 2 prediction) = 100 total.
        """
        questions: list[dict] = []
        profiles = sorted(self.store.all_profiles)

        for pid in profiles:
            trajs = self.store.get_profile_trajectories(pid)
            if len(trajs) < 3:
                continue

            # Pick 5 tasks for this profile (deterministic)
            tasks = sorted([t["task_id"] for t in trajs])
            self.rng.shuffle(tasks)
            selected_tasks = tasks[:5]

            # 3 Video-to-Profile Reconstruction
            for i, task_id in enumerate(selected_tasks[:3]):
                task_name = TASK_NAMES.get(task_id, task_id)
                q = {
                    "question_id": f"s4_vg_recon_{pid}_{task_id}",
                    "sub_type": "visual_grounding",
                    "question_type": "open_ended",
                    "question": (
                        f"You are watching a screen recording of a user working on "
                        f'the task "{task_name}" ({task_id}). Based on their file '
                        f"operations, reading patterns, output creation, directory "
                        f"management, editing behavior, and workspace curation "
                        f"observed in the video, reconstruct the user's behavioral "
                        f"profile across the following six dimensions:\n\n"
                        + "\n".join(
                            f"  {d}. {DIM_NAMES[d]}: "
                            f"{TIER_LABELS[d]['L']} (L) / "
                            f"{TIER_LABELS[d]['M']} (M) / "
                            f"{TIER_LABELS[d]['R']} (R)"
                            for d in DIMENSIONS
                        )
                        + "\n\nFor each dimension, select L, M, or R and provide "
                        "a brief justification based on specific behaviors observed "
                        "in the recording."
                    ),
                    "video_path": "__PLACEHOLDER__",
                    "profile_id": pid,
                    "metadata": {
                        "task_id": task_id,
                        "task_type": TASK_TYPES.get(task_id, ""),
                        "vg_type": "reconstruction",
                        "ground_truth": {
                            d: self.matrix.get_tier(pid, d)
                            for d in DIMENSIONS
                        },
                    },
                }
                questions.append(q)

            # 2 Proactive Prediction
            for i, task_id in enumerate(selected_tasks[3:5]):
                task_name = TASK_NAMES.get(task_id, task_id)
                task_type = TASK_TYPES.get(task_id, "")

                # Build ground-truth behavioral expectations
                gt_dims = {d: self.matrix.get_tier(pid, d) for d in DIMENSIONS}

                q = {
                    "question_id": f"s4_vg_pred_{pid}_{task_id}",
                    "sub_type": "visual_grounding",
                    "question_type": "open_ended",
                    "question": (
                        f"You are watching the FIRST HALF of a screen recording "
                        f"where a user is working on the task "
                        f'"{task_name}" ({task_id}, type: {task_type}). '
                        f"The recording stops midway through the task.\n\n"
                        f"Based on the behavioral patterns observed so far, predict:\n"
                        f"1. What file operations will the user perform next?\n"
                        f"2. How will the user organize any new output files?\n"
                        f"3. What will the user's editing strategy be for refining output?\n"
                        f"4. Will the user create any visual materials (charts, diagrams)?\n"
                        f"5. Will the user clean up intermediate files at the end?\n\n"
                        f"Justify each prediction based on specific behavioral patterns "
                        f"observed in the first half of the recording."
                    ),
                    "video_path": "__PLACEHOLDER__",
                    "profile_id": pid,
                    "metadata": {
                        "task_id": task_id,
                        "task_type": task_type,
                        "vg_type": "prediction",
                        "ground_truth": gt_dims,
                        "ground_truth_descriptions": {
                            d: TIER_LABELS[d][gt_dims[d]]
                            for d in DIMENSIONS
                        },
                    },
                }
                questions.append(q)

        return questions
