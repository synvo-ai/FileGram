"""Trajectory MCQ (Multiple Choice Questions) generator.

Auto-generates questions from trajectory data to test whether a memory
system can recall specific behavioral details. Ground truth comes from
the trajectory itself, NOT from profile YAML.

Question categories map to profile attributes:
- Reading behavior -> reading_strategy
- Output characteristics -> output_detail, verbosity
- Organization patterns -> directory_style, naming
- Edit patterns -> edit_strategy
- Version management -> version_strategy
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class MCQGenerator:
    """Generate multiple-choice questions from trajectory events."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_from_trajectory(
        self,
        events: list[dict[str, Any]],
        task_id: str,
        profile_id: str,
        n_questions: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate MCQs from a single trajectory.

        Returns list of question dicts:
            {
                "question_id": str,
                "profile_id": str,
                "task_id": str,
                "category": str (maps to attribute),
                "question": str,
                "choices": {"A": str, "B": str, "C": str, "D": str},
                "correct": str ("A"/"B"/"C"/"D"),
                "source_event_types": list[str],
            }
        """
        behavioral = [
            e
            for e in events
            if e.get("event_type")
            not in {
                "tool_call",
                "llm_response",
                "compaction_triggered",
                "session_start",
                "session_end",
                "iteration_start",
                "iteration_end",
            }
        ]

        generators = [
            self._q_files_read_count,
            self._q_first_action_type,
            self._q_dirs_created_count,
            self._q_max_dir_depth,
            self._q_files_created_count,
            self._q_total_edits,
            self._q_edit_magnitude,
            self._q_search_vs_browse,
            self._q_file_operations_order,
            self._q_backup_behavior,
        ]

        questions = []
        for gen in generators:
            q = gen(behavioral, task_id, profile_id)
            if q:
                questions.append(q)

        # Shuffle and limit
        self.rng.shuffle(questions)
        return questions[:n_questions]

    def generate_from_profile(
        self,
        all_trajectories: list[dict[str, Any]],
        profile_id: str,
        n_questions: int = 10,
    ) -> list[dict[str, Any]]:
        """Generate MCQs from all trajectories for a profile.

        Includes both per-trajectory and cross-trajectory questions.
        """
        questions = []

        # Per-trajectory questions
        for traj in all_trajectories:
            qs = self.generate_from_trajectory(traj["events"], traj["task_id"], profile_id, n_questions=3)
            questions.extend(qs)

        # Cross-trajectory questions
        cross_qs = self._cross_trajectory_questions(all_trajectories, profile_id)
        questions.extend(cross_qs)

        self.rng.shuffle(questions)
        return questions[:n_questions]

    # ---- Individual question generators ----

    def _q_files_read_count(self, events: list[dict], task_id: str, profile_id: str) -> dict[str, Any] | None:
        reads = [e for e in events if e.get("event_type") == "file_read"]
        unique = len(set(e.get("file_path", "") for e in reads))
        if unique < 2:
            return None

        correct = str(unique)
        distractors = self._numeric_distractors(unique, min_val=1)

        choices, correct_letter = self._build_choices(correct, distractors)
        return {
            "question_id": f"{profile_id}_{task_id}_reads",
            "profile_id": profile_id,
            "task_id": task_id,
            "category": "reading_strategy",
            "question": f"In task {task_id}, how many unique files did this user read?",
            "choices": choices,
            "correct": correct_letter,
            "source_event_types": ["file_read"],
        }

    def _q_first_action_type(self, events: list[dict], task_id: str, profile_id: str) -> dict[str, Any] | None:
        if not events:
            return None

        first = events[0]
        et = first.get("event_type", "")
        action_labels = {
            "file_read": "Read a file",
            "file_browse": "Browse a directory",
            "file_search": "Search for files",
            "file_write": "Create a file",
            "dir_create": "Create a directory",
            "file_edit": "Edit a file",
        }

        if et not in action_labels:
            return None

        correct = action_labels[et]
        distractors = [v for k, v in action_labels.items() if k != et]
        self.rng.shuffle(distractors)
        distractors = distractors[:3]

        choices, correct_letter = self._build_choices(correct, distractors)
        return {
            "question_id": f"{profile_id}_{task_id}_first_action",
            "profile_id": profile_id,
            "task_id": task_id,
            "category": "reading_strategy",
            "question": f"In task {task_id}, what was the user's FIRST file operation?",
            "choices": choices,
            "correct": correct_letter,
            "source_event_types": [et],
        }

    def _q_dirs_created_count(self, events: list[dict], task_id: str, profile_id: str) -> dict[str, Any] | None:
        dirs = [e for e in events if e.get("event_type") == "dir_create"]
        count = len(dirs)
        if count < 1:
            return None

        correct = str(count)
        distractors = self._numeric_distractors(count, min_val=0)

        choices, correct_letter = self._build_choices(correct, distractors)
        return {
            "question_id": f"{profile_id}_{task_id}_dirs",
            "profile_id": profile_id,
            "task_id": task_id,
            "category": "directory_style",
            "question": f"In task {task_id}, how many directories did this user create?",
            "choices": choices,
            "correct": correct_letter,
            "source_event_types": ["dir_create"],
        }

    def _q_max_dir_depth(self, events: list[dict], task_id: str, profile_id: str) -> dict[str, Any] | None:
        dirs = [e for e in events if e.get("event_type") == "dir_create"]
        if not dirs:
            return None

        max_depth = max(e.get("depth", 0) for e in dirs)
        correct = str(max_depth)
        distractors = self._numeric_distractors(max_depth, min_val=0)

        choices, correct_letter = self._build_choices(correct, distractors)
        return {
            "question_id": f"{profile_id}_{task_id}_depth",
            "profile_id": profile_id,
            "task_id": task_id,
            "category": "directory_style",
            "question": f"In task {task_id}, what was the maximum directory depth created?",
            "choices": choices,
            "correct": correct_letter,
            "source_event_types": ["dir_create"],
        }

    def _q_files_created_count(self, events: list[dict], task_id: str, profile_id: str) -> dict[str, Any] | None:
        creates = [e for e in events if e.get("event_type") == "file_write" and e.get("operation") == "create"]
        count = len(creates)
        if count < 1:
            return None

        correct = str(count)
        distractors = self._numeric_distractors(count, min_val=0)

        choices, correct_letter = self._build_choices(correct, distractors)
        return {
            "question_id": f"{profile_id}_{task_id}_creates",
            "profile_id": profile_id,
            "task_id": task_id,
            "category": "output_detail",
            "question": f"In task {task_id}, how many new files did this user create?",
            "choices": choices,
            "correct": correct_letter,
            "source_event_types": ["file_write"],
        }

    def _q_total_edits(self, events: list[dict], task_id: str, profile_id: str) -> dict[str, Any] | None:
        edits = [e for e in events if e.get("event_type") == "file_edit"]
        count = len(edits)
        if count < 1:
            return None

        correct = str(count)
        distractors = self._numeric_distractors(count, min_val=0)

        choices, correct_letter = self._build_choices(correct, distractors)
        return {
            "question_id": f"{profile_id}_{task_id}_edits",
            "profile_id": profile_id,
            "task_id": task_id,
            "category": "edit_strategy",
            "question": f"In task {task_id}, how many file edits did this user make?",
            "choices": choices,
            "correct": correct_letter,
            "source_event_types": ["file_edit"],
        }

    def _q_edit_magnitude(self, events: list[dict], task_id: str, profile_id: str) -> dict[str, Any] | None:
        edits = [e for e in events if e.get("event_type") == "file_edit"]
        if len(edits) < 2:
            return None

        avg_lines = sum(e.get("lines_added", 0) + e.get("lines_deleted", 0) for e in edits) / len(edits)

        if avg_lines < 5:
            correct = "Small (under 5 lines per edit)"
        elif avg_lines < 20:
            correct = "Medium (5-20 lines per edit)"
        else:
            correct = "Large (over 20 lines per edit)"

        all_options = [
            "Small (under 5 lines per edit)",
            "Medium (5-20 lines per edit)",
            "Large (over 20 lines per edit)",
            "Very large (over 50 lines per edit)",
        ]
        distractors = [o for o in all_options if o != correct][:3]

        choices, correct_letter = self._build_choices(correct, distractors)
        return {
            "question_id": f"{profile_id}_{task_id}_edit_mag",
            "profile_id": profile_id,
            "task_id": task_id,
            "category": "edit_strategy",
            "question": f"In task {task_id}, what was the typical edit magnitude?",
            "choices": choices,
            "correct": correct_letter,
            "source_event_types": ["file_edit"],
        }

    def _q_search_vs_browse(self, events: list[dict], task_id: str, profile_id: str) -> dict[str, Any] | None:
        searches = len([e for e in events if e.get("event_type") == "file_search"])
        browses = len([e for e in events if e.get("event_type") == "file_browse"])

        if searches == 0 and browses == 0:
            return None

        if searches > browses:
            correct = "More searches than directory browses"
        elif browses > searches:
            correct = "More directory browses than searches"
        else:
            correct = "Equal searches and browses"

        all_options = [
            "More searches than directory browses",
            "More directory browses than searches",
            "Equal searches and browses",
            "No searches or browses at all",
        ]
        distractors = [o for o in all_options if o != correct][:3]

        choices, correct_letter = self._build_choices(correct, distractors)
        return {
            "question_id": f"{profile_id}_{task_id}_search_browse",
            "profile_id": profile_id,
            "task_id": task_id,
            "category": "reading_strategy",
            "question": f"In task {task_id}, did the user use more searches or directory browses?",
            "choices": choices,
            "correct": correct_letter,
            "source_event_types": ["file_search", "file_browse"],
        }

    def _q_file_operations_order(self, events: list[dict], task_id: str, profile_id: str) -> dict[str, Any] | None:
        reads = [e for e in events if e.get("event_type") == "file_read"]
        writes = [e for e in events if e.get("event_type") == "file_write" and e.get("operation") == "create"]

        if not reads or not writes:
            return None

        first_read_ts = min(e.get("timestamp", float("inf")) for e in reads)
        first_write_ts = min(e.get("timestamp", float("inf")) for e in writes)

        if first_read_ts < first_write_ts:
            correct = "Read files before creating new ones"
        else:
            correct = "Created files before reading existing ones"

        all_options = [
            "Read files before creating new ones",
            "Created files before reading existing ones",
            "Read and created files simultaneously",
            "Only read files, never created any",
        ]
        distractors = [o for o in all_options if o != correct][:3]

        choices, correct_letter = self._build_choices(correct, distractors)
        return {
            "question_id": f"{profile_id}_{task_id}_order",
            "profile_id": profile_id,
            "task_id": task_id,
            "category": "reading_strategy",
            "question": f"In task {task_id}, did the user read existing files before or after creating new ones?",
            "choices": choices,
            "correct": correct_letter,
            "source_event_types": ["file_read", "file_write"],
        }

    def _q_backup_behavior(self, events: list[dict], task_id: str, profile_id: str) -> dict[str, Any] | None:
        copies = [e for e in events if e.get("event_type") == "file_copy"]
        backups = [e for e in copies if e.get("is_backup", False)]
        deletes = [e for e in events if e.get("event_type") == "file_delete"]

        if not copies and not deletes:
            return None

        if len(backups) > 0:
            correct = "Made backup copies of files"
        elif len(copies) > 0:
            correct = "Copied files but not as backups"
        elif len(deletes) > 0:
            correct = "Deleted files without making backups"
        else:
            correct = "No file copies or deletions"

        all_options = [
            "Made backup copies of files",
            "Copied files but not as backups",
            "Deleted files without making backups",
            "No file copies or deletions",
        ]
        distractors = [o for o in all_options if o != correct][:3]

        choices, correct_letter = self._build_choices(correct, distractors)
        return {
            "question_id": f"{profile_id}_{task_id}_backup",
            "profile_id": profile_id,
            "task_id": task_id,
            "category": "version_strategy",
            "question": f"In task {task_id}, what was the user's file versioning behavior?",
            "choices": choices,
            "correct": correct_letter,
            "source_event_types": ["file_copy", "file_delete"],
        }

    # ---- Cross-trajectory questions ----

    def _cross_trajectory_questions(
        self, all_trajectories: list[dict[str, Any]], profile_id: str
    ) -> list[dict[str, Any]]:
        """Generate questions that span multiple trajectories."""
        questions = []

        # Q: Across all tasks, what was the most common first action?
        first_actions = []
        for traj in all_trajectories:
            behavioral = [
                e
                for e in traj["events"]
                if e.get("event_type")
                not in {
                    "tool_call",
                    "llm_response",
                    "compaction_triggered",
                    "session_start",
                    "session_end",
                    "iteration_start",
                    "iteration_end",
                    "fs_snapshot",
                }
            ]
            if behavioral:
                first_actions.append(behavioral[0].get("event_type", ""))

        if first_actions:
            from collections import Counter

            most_common = Counter(first_actions).most_common(1)[0][0]
            action_labels = {
                "file_read": "Reading a file",
                "file_browse": "Browsing a directory",
                "file_search": "Searching for files",
                "file_write": "Creating a file",
                "dir_create": "Creating a directory",
            }
            correct = action_labels.get(most_common, most_common)
            distractors = [v for k, v in action_labels.items() if k != most_common]
            self.rng.shuffle(distractors)
            distractors = distractors[:3]

            choices, correct_letter = self._build_choices(correct, distractors)
            questions.append(
                {
                    "question_id": f"{profile_id}_cross_first_action",
                    "profile_id": profile_id,
                    "task_id": "cross",
                    "category": "reading_strategy",
                    "question": "Across all tasks, what was this user's most common FIRST action?",
                    "choices": choices,
                    "correct": correct_letter,
                    "source_event_types": ["cross_trajectory"],
                }
            )

        # Q: Average number of directories created per task
        dirs_per_task = []
        for traj in all_trajectories:
            dirs = [e for e in traj["events"] if e.get("event_type") == "dir_create"]
            dirs_per_task.append(len(dirs))

        if any(d > 0 for d in dirs_per_task):
            avg = sum(dirs_per_task) / len(dirs_per_task)
            if avg < 1:
                correct = "Less than 1 on average"
            elif avg < 3:
                correct = "1-3 on average"
            elif avg < 6:
                correct = "3-6 on average"
            else:
                correct = "More than 6 on average"

            all_options = [
                "Less than 1 on average",
                "1-3 on average",
                "3-6 on average",
                "More than 6 on average",
            ]
            distractors = [o for o in all_options if o != correct][:3]

            choices, correct_letter = self._build_choices(correct, distractors)
            questions.append(
                {
                    "question_id": f"{profile_id}_cross_dirs",
                    "profile_id": profile_id,
                    "task_id": "cross",
                    "category": "directory_style",
                    "question": "Across all tasks, how many directories did this user typically create per task?",
                    "choices": choices,
                    "correct": correct_letter,
                    "source_event_types": ["cross_trajectory"],
                }
            )

        return questions

    # ---- Utility methods ----

    def _numeric_distractors(self, correct: int, min_val: int = 0) -> list[str]:
        """Generate plausible numeric distractors."""
        candidates = set()
        for delta in [-3, -2, -1, 1, 2, 3, 5]:
            val = correct + delta
            if val >= min_val and val != correct:
                candidates.add(val)

        candidates_list = sorted(candidates)
        self.rng.shuffle(candidates_list)
        return [str(c) for c in candidates_list[:3]]

    def _build_choices(self, correct: str, distractors: list[str]) -> tuple[dict[str, str], str]:
        """Shuffle correct answer and distractors into labeled choices."""
        options = [correct] + distractors[:3]
        self.rng.shuffle(options)

        labels = ["A", "B", "C", "D"]
        choices = {}
        correct_letter = "A"
        for i, opt in enumerate(options[:4]):
            choices[labels[i]] = opt
            if opt == correct:
                correct_letter = labels[i]

        return choices, correct_letter

    def save_mcqs(
        self,
        questions: list[dict[str, Any]],
        output_dir: Path,
        profile_id: str,
    ) -> Path:
        """Save generated MCQs to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{profile_id}_mcqs.json"
        output_file.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_file
