"""Setting 3: Behavioral Perturbation — Episodic Consistency.

Tests whether memory systems can maintain per-session behavioral awareness
and detect when a user's behavior deviates from their established baseline.

Sub-types:
  - perturbation_detection: given N trajectories (mix of standard + perturbed),
    identify which session deviates from the user's baseline.
  - shifted_dimension_id: given a perturbed trajectory, identify which
    behavioral dimension shifted. 6-choice MCQ (A-F).
  - shift_direction: given the shifted dimension, identify the direction
    of change (e.g., L->M). 3-choice MCQ.
  - behavioral_change_description: which natural-language before→after
    description correctly captures what changed? 4-choice MCQ.
"""

from __future__ import annotations

import random
from typing import Any

from .base import (
    DIMENSIONS, DIM_NAMES, TIER_LABELS, TASK_NAMES, TASK_TYPES,
    _DIM_PORTRAIT_FRAGMENTS,
    ProfileMatrix, TrajectoryStore,
    build_choices,
)

MIN_STANDARD = 4  # need at least this many standard trajectories


class Setting3Generator:
    """Generate Setting 3 (Perturbation / Episodic) questions."""

    def __init__(self, store: TrajectoryStore, matrix: ProfileMatrix, seed: int = 42):
        self.store = store
        self.matrix = matrix
        self.rng = random.Random(seed)

    def generate_all(self) -> list[dict[str, Any]]:
        questions: list[dict] = []
        questions.extend(self._gen_perturbation_detection())
        questions.extend(self._gen_shifted_dimension_id())
        questions.extend(self._gen_shift_direction())
        questions.extend(self._gen_behavioral_change_description())
        return questions

    # ------------------------------------------------------------------
    # Helper: build session label for a trajectory
    # ------------------------------------------------------------------
    @staticmethod
    def _session_label(traj: dict, label: str) -> str:
        tid = traj["task_id"]
        return (
            f"Session {label}: {TASK_NAMES.get(tid, tid)} "
            f"({tid}, {TASK_TYPES.get(tid, '?')})"
        )

    # ------------------------------------------------------------------
    # 3.1  Perturbation Detection  (~60 questions)
    # ------------------------------------------------------------------
    def _gen_perturbation_detection(self) -> list[dict]:
        """Given N trajectories (4 standard + 1 perturbed), find the deviant.

        The memory system must compare individual sessions against the
        aggregate profile to flag the outlier.
        """
        questions = []

        for pid in self.store.all_profiles:
            standard = self.store.get_standard_trajectories(pid)
            perturbed = self.store.get_perturbed_trajectories(pid)

            if len(standard) < MIN_STANDARD or not perturbed:
                continue

            # Use ALL available perturbed trajectories (uncapped)
            selected_perturbed = list(perturbed)

            for qi, ptraj in enumerate(selected_perturbed):
                # Pick 4 standard trajectories as baseline
                std_sample = self.rng.sample(
                    standard, min(4, len(standard))
                )

                all_five = list(std_sample) + [ptraj]
                self.rng.shuffle(all_five)
                perturbed_idx = next(
                    i for i, t in enumerate(all_five) if t is ptraj
                )

                labels = ["A", "B", "C", "D", "E"]
                correct_letter = labels[perturbed_idx]

                session_lines = [
                    self._session_label(t, labels[i])
                    for i, t in enumerate(all_five)
                ]

                # Extract perturbation info for metadata
                pinfo = ptraj["perturbation"]
                shifted_dims = list(pinfo.get("shifted_dims", {}).keys())

                questions.append({
                    "question_id": f"s3_det_{pid}_{qi}",
                    "setting": "perturbation",
                    "sub_type": "perturbation_detection",
                    "profile_id": pid,
                    "input_trajectories": [t["dir_name"] for t in all_five],
                    "question": (
                        "Five behavioral trajectories from the same user are provided. "
                        "Four sessions follow the user's typical behavioral patterns, "
                        "while one session shows a deviation from their established "
                        "baseline. Based on the behavioral patterns in the memory "
                        "context, which session deviates from the user's normal "
                        "behavior?\n\n"
                        + "\n".join(f"  {line}" for line in session_lines)
                    ),
                    "choices": {l: f"Session {l}" for l in labels},
                    "correct": correct_letter,
                    "metadata": {
                        "perturbed_task": ptraj["task_id"],
                        "shifted_dims": shifted_dims,
                        "perturbation": pinfo.get("shifted_dims", {}),
                    },
                })

        return questions

    # ------------------------------------------------------------------
    # 3.2  Shifted Dimension Identification  (~60 questions)
    # ------------------------------------------------------------------
    def _gen_shifted_dimension_id(self) -> list[dict]:
        """Given a perturbed trajectory, which dimension shifted?

        6-choice MCQ (one per dimension A-F).
        """
        questions = []

        for pid in self.store.all_profiles:
            perturbed = self.store.get_perturbed_trajectories(pid)
            standard = self.store.get_standard_trajectories(pid)

            if len(standard) < MIN_STANDARD or not perturbed:
                continue

            for qi, ptraj in enumerate(perturbed):
                pinfo = ptraj["perturbation"]
                shifted_dims = list(pinfo.get("shifted_dims", {}).keys())
                if not shifted_dims:
                    continue

                # For single-dim shift, the correct answer is clear
                correct_dim = shifted_dims[0]
                correct_text = f"{correct_dim} ({DIM_NAMES[correct_dim]})"

                # All 6 dimensions as choices
                all_dim_choices = [
                    f"{d} ({DIM_NAMES[d]})" for d in DIMENSIONS
                ]
                # Build choices dict directly (6-choice)
                dim_labels = ["A", "B", "C", "D", "E", "F"]
                choices = {
                    dim_labels[i]: text
                    for i, text in enumerate(all_dim_choices)
                }
                # Find the correct letter
                correct_letter = next(
                    l for l, v in choices.items() if v == correct_text
                )

                # Provide standard trajectories as baseline context
                std_sample = self.rng.sample(
                    standard, min(5, len(standard))
                )

                questions.append({
                    "question_id": f"s3_dimid_{pid}_{qi}",
                    "setting": "perturbation",
                    "sub_type": "shifted_dimension_id",
                    "profile_id": pid,
                    "input_trajectories": (
                        [t["dir_name"] for t in std_sample]
                        + [ptraj["dir_name"]]
                    ),
                    "question": (
                        f"A user's typical behavioral patterns are shown across "
                        f"{len(std_sample)} baseline sessions. One additional session "
                        f"({TASK_NAMES.get(ptraj['task_id'], ptraj['task_id'])}, "
                        f"{ptraj['task_id']}) shows a behavioral deviation from "
                        f"the baseline. Which behavioral dimension shifted in "
                        f"the deviant session?"
                    ),
                    "choices": choices,
                    "correct": correct_letter,
                    "metadata": {
                        "perturbed_task": ptraj["task_id"],
                        "correct_dim": correct_dim,
                        "shift_detail": pinfo.get("shifted_dims", {}),
                    },
                })

        return questions

    # ------------------------------------------------------------------
    # 3.3  Shift Direction  (~40 questions)
    # ------------------------------------------------------------------
    def _gen_shift_direction(self) -> list[dict]:
        """Given the shifted dimension, identify the direction of change.

        3-choice MCQ: the correct transition plus 2 plausible distractors.
        """
        questions = []

        for pid in self.store.all_profiles:
            perturbed = self.store.get_perturbed_trajectories(pid)
            standard = self.store.get_standard_trajectories(pid)

            if len(standard) < MIN_STANDARD or not perturbed:
                continue

            for qi, ptraj in enumerate(perturbed):
                pinfo = ptraj["perturbation"]
                shifted_dims = pinfo.get("shifted_dims", {})
                if not shifted_dims:
                    continue

                for dim_key, shift_info in shifted_dims.items():
                    orig = shift_info["original"]
                    pert = shift_info["perturbed"]
                    orig_name = TIER_LABELS[dim_key][orig]
                    pert_name = TIER_LABELS[dim_key][pert]

                    correct_text = (
                        f"{orig_name} ({orig}) \u2192 {pert_name} ({pert})"
                    )

                    # Build distractors: other plausible tier transitions
                    all_tiers = ["L", "M", "R"]
                    wrong_transitions = []
                    for from_t in all_tiers:
                        for to_t in all_tiers:
                            if from_t == to_t:
                                continue
                            if from_t == orig and to_t == pert:
                                continue
                            wrong_transitions.append(
                                f"{TIER_LABELS[dim_key][from_t]} ({from_t}) \u2192 "
                                f"{TIER_LABELS[dim_key][to_t]} ({to_t})"
                            )
                    self.rng.shuffle(wrong_transitions)

                    if len(wrong_transitions) < 2:
                        continue

                    choices, correct_letter = build_choices(
                        correct_text, wrong_transitions[:2], self.rng
                    )

                    std_sample = self.rng.sample(
                        standard, min(5, len(standard))
                    )

                    questions.append({
                        "question_id": f"s3_dir_{pid}_{qi}_{dim_key}",
                        "setting": "perturbation",
                        "sub_type": "shift_direction",
                        "profile_id": pid,
                        "input_trajectories": (
                            [t["dir_name"] for t in std_sample]
                            + [ptraj["dir_name"]]
                        ),
                        "question": (
                            f"A behavioral deviation was detected in this user's "
                            f"session on {TASK_NAMES.get(ptraj['task_id'], ptraj['task_id'])} "
                            f"({ptraj['task_id']}). The shifted dimension is "
                            f"{DIM_NAMES[dim_key]} (Dimension {dim_key}). "
                            f"In which direction did the behavior change?"
                        ),
                        "choices": choices,
                        "correct": correct_letter,
                        "metadata": {
                            "perturbed_task": ptraj["task_id"],
                            "dimension": dim_key,
                            "original_tier": orig,
                            "perturbed_tier": pert,
                        },
                    })

        return questions

    # ------------------------------------------------------------------
    # 3.4  Behavioral Change Description  (~96 questions)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_change_desc(dim: str, from_tier: str, to_tier: str) -> str:
        """Build a before→after change description without dimension labels."""
        from_desc = _DIM_PORTRAIT_FRAGMENTS[dim][from_tier]
        to_desc = _DIM_PORTRAIT_FRAGMENTS[dim][to_tier]
        return f"Changed from {from_desc} to {to_desc}"

    def _gen_behavioral_change_description(self) -> list[dict]:
        """Which natural-language description captures the behavioral change?

        4-choice MCQ: each choice describes a concrete before→after
        behavioral change on a DIFFERENT dimension (no dimension labels).
        The evaluator must understand both WHICH aspect changed and HOW,
        combining dimension identification and direction into a single
        harder question that requires semantic understanding.
        """
        questions = []

        for pid in self.store.all_profiles:
            standard = self.store.get_standard_trajectories(pid)
            perturbed = self.store.get_perturbed_trajectories(pid)

            if len(standard) < MIN_STANDARD or not perturbed:
                continue

            profile_dims = self.matrix.profiles[pid]["dimensions"]

            for qi, ptraj in enumerate(perturbed):
                pinfo = ptraj["perturbation"]
                shifted_dims = pinfo.get("shifted_dims", {})
                if not shifted_dims:
                    continue

                # Build correct answer from the actual perturbation
                dim_key = list(shifted_dims.keys())[0]
                orig_tier = shifted_dims[dim_key]["original"]
                pert_tier = shifted_dims[dim_key]["perturbed"]

                correct_text = self._build_change_desc(
                    dim_key, orig_tier, pert_tier
                )

                # Distractors: plausible changes on OTHER dimensions
                other_dims = [d for d in DIMENSIONS if d != dim_key]
                self.rng.shuffle(other_dims)

                distractors = []
                for od in other_dims:
                    od_orig = profile_dims.get(od, "M")
                    possible_pert = [t for t in ["L", "M", "R"] if t != od_orig]
                    od_pert = self.rng.choice(possible_pert)
                    distractors.append(
                        self._build_change_desc(od, od_orig, od_pert)
                    )
                    if len(distractors) >= 3:
                        break

                choices, correct_letter = build_choices(
                    correct_text, distractors, self.rng
                )

                std_sample = self.rng.sample(
                    standard, min(5, len(standard))
                )

                questions.append({
                    "question_id": f"s3_chg_{pid}_{qi}",
                    "setting": "perturbation",
                    "sub_type": "behavioral_change_description",
                    "profile_id": pid,
                    "input_trajectories": (
                        [t["dir_name"] for t in std_sample]
                        + [ptraj["dir_name"]]
                    ),
                    "question": (
                        f"A user's typical behavioral patterns are shown across "
                        f"{len(std_sample)} baseline sessions. One additional session "
                        f"({TASK_NAMES.get(ptraj['task_id'], ptraj['task_id'])}, "
                        f"{ptraj['task_id']}) shows a behavioral deviation from "
                        f"the baseline. Which description best characterizes the "
                        f"behavioral change observed in the deviant session?"
                    ),
                    "choices": choices,
                    "correct": correct_letter,
                    "metadata": {
                        "perturbed_task": ptraj["task_id"],
                        "dimension": dim_key,
                        "original_tier": orig_tier,
                        "perturbed_tier": pert_tier,
                    },
                })

        return questions
