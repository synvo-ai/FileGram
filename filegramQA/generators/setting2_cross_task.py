"""Setting 2: Cross-Task — Multitask Trace Disentanglement.

Sub-types:
  - multi_user_attribution: 2 users × same task mixed, identify behavioral differences
  - single_user_task_separation: 1 user × 2 tasks mixed, identify task boundaries
  - impersonation_detection: 5 real + 1 impostor, find the fake
"""

from __future__ import annotations

import random
from typing import Any

from .base import (
    DIMENSIONS, DIM_NAMES, TIER_LABELS, TASK_TYPES, TASK_NAMES,
    FINE_GRAINED_PAIRS, COARSE_PAIRS,
    ProfileMatrix, TrajectoryStore,
    compute_dim_stats, build_choices,
)

MIN_TRAJS = 3


class Setting2Generator:
    """Generate Setting 2 (Cross-Task) questions."""

    def __init__(self, store: TrajectoryStore, matrix: ProfileMatrix, seed: int = 42):
        self.store = store
        self.matrix = matrix
        self.rng = random.Random(seed)

    def generate_all(self) -> list[dict[str, Any]]:
        questions: list[dict] = []
        questions.extend(self._gen_multi_user_attribution())
        questions.extend(self._gen_single_user_task_separation())
        questions.extend(self._gen_impersonation_detection())
        return questions

    # ------------------------------------------------------------------
    # Helper: semantic behavioral difference descriptions (no raw numbers)
    # ------------------------------------------------------------------
    def _describe_pair_difference(
        self, stats_a: dict, stats_b: dict, pid_a: str, pid_b: str,
    ) -> list[dict]:
        """Return a list of semantic behavioral difference descriptions."""
        diffs = []
        dims_a = self.matrix.profiles[pid_a]["dimensions"]
        dims_b = self.matrix.profiles[pid_b]["dimensions"]

        # A: consumption — semantic difference
        if dims_a.get("A") != dims_b.get("A"):
            a_desc = {
                "L": "reads files sequentially and thoroughly",
                "M": "uses a search-first targeted approach",
                "R": "scans files broadly without deep reading",
            }
            diffs.append({
                "dim": "A",
                "desc": (
                    f"User A {a_desc[dims_a.get('A', 'M')]} "
                    f"while User B {a_desc[dims_b.get('A', 'M')]}"
                ),
            })
        elif abs(stats_a["reads"] - stats_b["reads"]) > 3 or abs(stats_a["searches"] - stats_b["searches"]) > 1:
            if stats_a["searches"] > stats_b["searches"]:
                diffs.append({
                    "dim": "A",
                    "desc": "User A uses a search-first strategy while User B reads more sequentially",
                })
            elif stats_b["searches"] > stats_a["searches"]:
                diffs.append({
                    "dim": "A",
                    "desc": "User A reads more sequentially while User B uses a search-first strategy",
                })

        # B: production
        if dims_a.get("B") != dims_b.get("B"):
            b_desc = {
                "L": "produces comprehensive, detailed output with auxiliary files",
                "M": "creates moderately detailed output",
                "R": "writes concise, minimal output",
            }
            diffs.append({
                "dim": "B",
                "desc": (
                    f"User A {b_desc[dims_a.get('B', 'M')]} "
                    f"while User B {b_desc[dims_b.get('B', 'M')]}"
                ),
            })
        elif abs(stats_a["mean_content_length"] - stats_b["mean_content_length"]) > 1000:
            more = "A" if stats_a["mean_content_length"] > stats_b["mean_content_length"] else "B"
            less = "B" if more == "A" else "A"
            diffs.append({
                "dim": "B",
                "desc": f"User {more} produces significantly longer, more detailed output while User {less} is more concise",
            })

        # C: organization
        if dims_a.get("C") != dims_b.get("C"):
            c_desc = {
                "L": "creates deeply nested directory structures with systematic naming",
                "M": "organizes files adaptively as needed",
                "R": "keeps files flat in the root directory",
            }
            diffs.append({
                "dim": "C",
                "desc": (
                    f"User A {c_desc[dims_a.get('C', 'M')]} "
                    f"while User B {c_desc[dims_b.get('C', 'M')]}"
                ),
            })
        elif abs(stats_a["dir_creates"] - stats_b["dir_creates"]) > 1:
            more = "A" if stats_a["dir_creates"] > stats_b["dir_creates"] else "B"
            less = "B" if more == "A" else "A"
            diffs.append({
                "dim": "C",
                "desc": f"User {more} creates more directory structure while User {less} keeps files flatter",
            })

        # D: iteration
        if dims_a.get("D") != dims_b.get("D"):
            d_desc = {
                "L": "makes many small incremental edits",
                "M": "edits with moderate scope",
                "R": "rewrites files in large decisive passes",
            }
            diffs.append({
                "dim": "D",
                "desc": (
                    f"User A {d_desc[dims_a.get('D', 'M')]} "
                    f"while User B {d_desc[dims_b.get('D', 'M')]}"
                ),
            })
        elif abs(stats_a["edits"] - stats_b["edits"]) > 1:
            more = "A" if stats_a["edits"] > stats_b["edits"] else "B"
            less = "B" if more == "A" else "A"
            diffs.append({
                "dim": "D",
                "desc": f"User {more} edits files more frequently with smaller changes while User {less} makes fewer, larger modifications",
            })

        # E: curation
        if dims_a.get("E") != dims_b.get("E"):
            e_desc = {
                "L": "actively curates the workspace by deleting unnecessary files",
                "M": "preserves nearly all files, rarely deleting anything",
                "R": "occasionally removes unneeded files but mostly accumulates",
            }
            diffs.append({
                "dim": "E",
                "desc": (
                    f"User A {e_desc[dims_a.get('E', 'M')]} "
                    f"while User B {e_desc[dims_b.get('E', 'M')]}"
                ),
            })
        elif abs(stats_a.get("delete_count", 0) - stats_b.get("delete_count", 0)) > 2:
            more_del = "A" if stats_a.get("delete_count", 0) > stats_b.get("delete_count", 0) else "B"
            less_del = "B" if more_del == "A" else "A"
            diffs.append({
                "dim": "E",
                "desc": f"User {more_del} actively deletes files during work while User {less_del} keeps almost everything",
            })

        # F: cross-modal
        if dims_a.get("F") != dims_b.get("F"):
            f_desc = {
                "L": "actively creates visual materials and media files",
                "M": "uses structured formatting like tables",
                "R": "produces pure text without visual elements",
            }
            diffs.append({
                "dim": "F",
                "desc": (
                    f"User A {f_desc[dims_a.get('F', 'M')]} "
                    f"while User B {f_desc[dims_b.get('F', 'M')]}"
                ),
            })

        return diffs

    # ------------------------------------------------------------------
    # 2.1  Multi-User Attribution  (~100 questions)
    # ------------------------------------------------------------------
    def _gen_multi_user_attribution(self) -> list[dict]:
        """Mix 2 users doing same task. Ask about differences and attribution."""
        questions = []

        # Build pairs: all fine-grained + some coarse + some random
        pairs_to_test = []
        for p1, p2, diff_dims, diff_count in FINE_GRAINED_PAIRS + COARSE_PAIRS:
            common = self.store.get_common_tasks(p1, p2)
            if len(common) >= 1:
                pairs_to_test.append((p1, p2, diff_dims, common))

        # Add random pairs to increase coverage
        all_p = self.store.all_profiles
        seen_pairs = {frozenset([a, b]) for a, b, _, _ in pairs_to_test}
        for _ in range(80):  # try more to get ~40 unique random pairs
            if len(pairs_to_test) >= 60:
                break
            p1, p2 = self.rng.sample(all_p, 2)
            pair_key = frozenset([p1, p2])
            if pair_key in seen_pairs:
                continue
            common = self.store.get_common_tasks(p1, p2)
            if len(common) >= 1:
                diff_dims = self.matrix.differing_dims(p1, p2)
                pairs_to_test.append((p1, p2, diff_dims, common))
                seen_pairs.add(pair_key)

        for pair_idx, (p1, p2, diff_dims, common_tasks) in enumerate(pairs_to_test):
            # Pick up to 3 shared tasks
            tasks_to_use = self.rng.sample(common_tasks, min(3, len(common_tasks)))

            for task_id in tasks_to_use:
                traj_a = self.store.get_trajectory(p1, task_id)
                traj_b = self.store.get_trajectory(p2, task_id)
                if not traj_a or not traj_b:
                    continue

                stats_a = compute_dim_stats(traj_a["events"])
                stats_b = compute_dim_stats(traj_b["events"])
                diffs = self._describe_pair_difference(stats_a, stats_b, p1, p2)

                group_id = f"s2_multi_{p1}_{p2}_{task_id}"
                mixing_spec = {
                    "user_a": {"profile_id": p1, "task_id": task_id, "dir_name": traj_a["dir_name"]},
                    "user_b": {"profile_id": p2, "task_id": task_id, "dir_name": traj_b["dir_name"]},
                }

                # Q1: What is the primary behavioral difference?
                if diffs:
                    correct_diff = diffs[0]
                    wrong_diffs = [
                        "Both users have nearly identical reading and writing patterns with no meaningful behavioral differences",
                        "User A and User B differ primarily in the language they write in, not in their behavioral approach",
                        "The users show opposite documentation habits — one always creates README files while the other never does",
                    ]
                    self.rng.shuffle(wrong_diffs)
                    choices, cl = build_choices(correct_diff["desc"], wrong_diffs[:3], self.rng)
                    questions.append({
                        "question_id": f"{group_id}_diff",
                        "setting": "cross_task",
                        "sub_type": "multi_user_attribution",
                        "group_id": group_id,
                        "mixing_spec": mixing_spec,
                        "question": (
                            "Two users (User A and User B) performed the same task. "
                            "Their event streams have been interleaved. "
                            "What is the primary behavioral difference between them?"
                        ),
                        "choices": choices,
                        "correct": cl,
                        "metadata": {
                            "profiles": [p1, p2],
                            "task_id": task_id,
                            "diff_dims": diff_dims,
                        },
                    })

                # Q2: Who reads more extensively?
                if abs(stats_a["reads"] - stats_b["reads"]) >= 3:
                    more_reader = "User A" if stats_a["reads"] > stats_b["reads"] else "User B"
                    choices2, cl2 = build_choices(
                        f"{more_reader} reads significantly more files",
                        [
                            f"{'User B' if more_reader == 'User A' else 'User A'} reads significantly more files",
                            "Both users read approximately the same number of files",
                            "Neither user reads input files extensively",
                        ],
                        self.rng,
                    )
                    questions.append({
                        "question_id": f"{group_id}_reads",
                        "setting": "cross_task",
                        "sub_type": "multi_user_attribution",
                        "group_id": group_id,
                        "mixing_spec": mixing_spec,
                        "question": "In the mixed event stream, how do the two users' file reading patterns compare?",
                        "choices": choices2,
                        "correct": cl2,
                        "metadata": {
                            "profiles": [p1, p2],
                            "task_id": task_id,
                        },
                    })

                # Q3: Who produces more output?
                if abs(stats_a["total_content_length"] - stats_b["total_content_length"]) > 2000:
                    more_out = "User A" if stats_a["total_content_length"] > stats_b["total_content_length"] else "User B"
                    choices3, cl3 = build_choices(
                        f"{more_out} produces substantially more output content",
                        [
                            f"{'User B' if more_out == 'User A' else 'User A'} produces substantially more output content",
                            "Both users produce similar amounts of output",
                            "Neither user creates significant output files",
                        ],
                        self.rng,
                    )
                    questions.append({
                        "question_id": f"{group_id}_output",
                        "setting": "cross_task",
                        "sub_type": "multi_user_attribution",
                        "group_id": group_id,
                        "mixing_spec": mixing_spec,
                        "question": "In the mixed event stream, which user produces more output content?",
                        "choices": choices3,
                        "correct": cl3,
                        "metadata": {
                            "profiles": [p1, p2],
                            "task_id": task_id,
                        },
                    })

                # Q4: Which dimension do they differ on? (if known)
                if diff_dims:
                    correct_dim_desc = ", ".join(
                        f"{d} ({DIM_NAMES[d]})" for d in diff_dims[:2]
                    )
                    # Pick wrong dimensions
                    wrong_dims = [d for d in DIMENSIONS if d not in diff_dims]
                    self.rng.shuffle(wrong_dims)
                    wrong_descs = [
                        ", ".join(f"{d} ({DIM_NAMES[d]})" for d in wrong_dims[i:i+len(diff_dims[:2])])
                        for i in range(0, min(6, len(wrong_dims)), len(diff_dims[:2]))
                    ]
                    wrong_descs = [w for w in wrong_descs if w][:3]
                    if wrong_descs:
                        choices4, cl4 = build_choices(correct_dim_desc, wrong_descs, self.rng)
                        questions.append({
                            "question_id": f"{group_id}_dims",
                            "setting": "cross_task",
                            "sub_type": "multi_user_attribution",
                            "group_id": group_id,
                            "mixing_spec": mixing_spec,
                            "question": (
                                "On which behavioral dimension(s) do User A and User B primarily differ?"
                            ),
                            "choices": choices4,
                            "correct": cl4,
                            "metadata": {
                                "profiles": [p1, p2],
                                "task_id": task_id,
                                "true_diff_dims": diff_dims,
                            },
                        })

                # Q5: Who creates more directory structure?
                if abs(stats_a["dir_creates"] - stats_b["dir_creates"]) >= 2:
                    more_org = "User A" if stats_a["dir_creates"] > stats_b["dir_creates"] else "User B"
                    choices5, cl5 = build_choices(
                        f"{more_org} creates significantly more directory structure to organize files",
                        [
                            f"{'User B' if more_org == 'User A' else 'User A'} creates significantly more directory structure to organize files",
                            "Both users organize files with similar directory structures",
                            "Neither user creates any directory structure",
                        ],
                        self.rng,
                    )
                    questions.append({
                        "question_id": f"{group_id}_org",
                        "setting": "cross_task",
                        "sub_type": "multi_user_attribution",
                        "group_id": group_id,
                        "mixing_spec": mixing_spec,
                        "question": "In the mixed event stream, how do the two users' file organization patterns compare?",
                        "choices": choices5,
                        "correct": cl5,
                        "metadata": {
                            "profiles": [p1, p2],
                            "task_id": task_id,
                        },
                    })

        return questions

    # ------------------------------------------------------------------
    # Helpers for task_sep Q2 behavioral descriptions
    # ------------------------------------------------------------------
    @staticmethod
    def _describe_task_pair_behavior(
        stats1: dict, stats2: dict, t1: str, t2: str,
    ) -> str:
        """Build a behavioral pattern description for two tasks (no type names)."""
        parts = []

        # Reading vs writing ratio
        r1, w1 = stats1["reads"], stats1["writes"] + stats1["edits"]
        r2, w2 = stats2["reads"], stats2["writes"] + stats2["edits"]
        if r1 > w1 * 2 and w2 > r2:
            parts.append(
                "One group involves extensive file reading with minimal output"
            )
            parts.append(
                "the other focuses on creating or modifying files"
            )
        elif r2 > w2 * 2 and w1 > r1:
            parts.append(
                "One group focuses on creating or modifying files"
            )
            parts.append(
                "the other involves extensive file reading with minimal output"
            )
        else:
            # Fallback: compare on dominant operation
            if stats1["dir_creates"] > stats2["dir_creates"] + 1:
                parts.append(
                    "One group involves directory restructuring and file movement"
                )
                parts.append(
                    "the other focuses on content reading and summary production"
                )
            elif stats2["dir_creates"] > stats1["dir_creates"] + 1:
                parts.append(
                    "One group focuses on content reading and summary production"
                )
                parts.append(
                    "the other involves directory restructuring and file movement"
                )
            elif stats1["edits"] > stats2["edits"] + 2:
                parts.append(
                    "One group involves iterative editing and revision of existing files"
                )
                parts.append(
                    "the other focuses on reading and producing new content"
                )
            elif stats2["edits"] > stats1["edits"] + 2:
                parts.append(
                    "One group focuses on reading and producing new content"
                )
                parts.append(
                    "the other involves iterative editing and revision of existing files"
                )
            else:
                parts.append(
                    "One group produces longer, more detailed output files"
                )
                parts.append(
                    "the other produces shorter, more concise results"
                )

        return "; ".join(parts)

    def _build_task_behavior_distractors(
        self, stats1: dict, stats2: dict, t1: str, t2: str,
        type1: str, type2: str,
    ) -> list[str]:
        """Build plausible but wrong behavioral pattern descriptions."""
        pool = [
            "Both groups show identical reading and writing patterns with no distinguishable behavioral differences",
            "One group primarily deletes and cleans up files; the other creates backup copies of every document",
            "One group uses search tools extensively before reading; the other only browses directory listings",
            "Both groups involve equal amounts of file creation and file editing with no clear separation",
            "One group works exclusively with image and media files; the other only processes text documents",
            "One group actively deletes and curates files during work; the other preserves everything without cleanup",
        ]
        self.rng.shuffle(pool)
        return pool[:3]

    # ------------------------------------------------------------------
    # 2.2  Single-User Task Separation  (~60 questions)
    # ------------------------------------------------------------------
    def _gen_single_user_task_separation(self) -> list[dict]:
        """Mix same user's 2 tasks. Ask about task boundaries and patterns."""
        questions = []

        for pid in self.store.all_profiles:
            trajs = self.store.get_profile_trajectories(pid)
            if len(trajs) < 4:
                continue

            # Pick 2 task pairs with different types
            task_ids = [t["task_id"] for t in trajs]
            tried = set()
            pairs_made = 0

            for _ in range(100):  # more attempts to find 10 valid pairs
                if pairs_made >= 10:
                    break
                t1, t2 = self.rng.sample(task_ids, 2)
                key = tuple(sorted([t1, t2]))
                if key in tried:
                    continue
                tried.add(key)

                type1 = TASK_TYPES.get(t1, "unknown")
                type2 = TASK_TYPES.get(t2, "unknown")
                if type1 == type2:
                    continue  # same type is too easy to separate

                traj1 = self.store.get_trajectory(pid, t1)
                traj2 = self.store.get_trajectory(pid, t2)
                if not traj1 or not traj2:
                    continue

                stats1 = compute_dim_stats(traj1["events"])
                stats2 = compute_dim_stats(traj2["events"])
                pairs_made += 1

                group_id = f"s2_sep_{pid}_{t1}_{t2}"
                mixing_spec = {
                    "task_a": {"profile_id": pid, "task_id": t1, "dir_name": traj1["dir_name"]},
                    "task_b": {"profile_id": pid, "task_id": t2, "dir_name": traj2["dir_name"]},
                }

                # Q1: How many tasks in this stream?
                choices_count, cl_count = build_choices(
                    "2 tasks",
                    ["1 task", "3 tasks", "4 tasks"],
                    self.rng,
                )
                questions.append({
                    "question_id": f"{group_id}_count",
                    "setting": "cross_task",
                    "sub_type": "single_user_task_separation",
                    "group_id": group_id,
                    "mixing_spec": mixing_spec,
                    "question": (
                        "This event stream is from a single user but may contain multiple tasks "
                        "interleaved. How many distinct tasks are represented?"
                    ),
                    "choices": choices_count,
                    "correct": cl_count,
                    "metadata": {"profile_id": pid, "tasks": [t1, t2]},
                })

                # Q2: What behavioral patterns distinguish the two task groups?
                correct_pattern = self._describe_task_pair_behavior(stats1, stats2, t1, t2)
                wrong_patterns = self._build_task_behavior_distractors(
                    stats1, stats2, t1, t2, type1, type2
                )
                choices_types, cl_types = build_choices(correct_pattern, wrong_patterns[:3], self.rng)
                questions.append({
                    "question_id": f"{group_id}_types",
                    "setting": "cross_task",
                    "sub_type": "single_user_task_separation",
                    "group_id": group_id,
                    "mixing_spec": mixing_spec,
                    "question": (
                        "Two tasks from the same user are interleaved in this event stream. "
                        "Which description best characterizes the behavioral patterns that "
                        "distinguish the two task groups?"
                    ),
                    "choices": choices_types,
                    "correct": cl_types,
                    "metadata": {"profile_id": pid, "task_types": [type1, type2]},
                })

                # Q3: Which task produced the directory creation events?
                if stats1["dir_creates"] != stats2["dir_creates"] and \
                   (stats1["dir_creates"] > 0 or stats2["dir_creates"] > 0):
                    dir_task = t1 if stats1["dir_creates"] > stats2["dir_creates"] else t2
                    other_task = t2 if dir_task == t1 else t1
                    correct_text = f"{TASK_NAMES.get(dir_task, dir_task)} ({dir_task})"
                    distractor_text = f"{TASK_NAMES.get(other_task, other_task)} ({other_task})"
                    choices_dir, cl_dir = build_choices(
                        correct_text,
                        [distractor_text, "Both tasks equally", "Neither task"],
                        self.rng,
                    )
                    questions.append({
                        "question_id": f"{group_id}_dirs",
                        "setting": "cross_task",
                        "sub_type": "single_user_task_separation",
                        "group_id": group_id,
                        "mixing_spec": mixing_spec,
                        "question": "Which task produced the directory creation events?",
                        "choices": choices_dir,
                        "correct": cl_dir,
                        "metadata": {"profile_id": pid},
                    })

        return questions

    # ------------------------------------------------------------------
    # 2.3  Impersonation Detection  (~45 questions)
    # ------------------------------------------------------------------
    def _gen_impersonation_detection(self) -> list[dict]:
        """5 real sessions + 1 impostor from fine-grained pair. Find the fake.

        Session descriptions are anonymous (task ID only, no stats).
        """
        questions = []

        for pid in self.store.all_profiles:
            trajs = self.store.get_profile_trajectories(pid)
            if len(trajs) < 5:
                continue

            # Find fine-grained neighbours
            neighbours = []
            for p1, p2, diff_dims, diff_count in FINE_GRAINED_PAIRS:
                if pid == p1:
                    neighbours.append((p2, diff_dims, diff_count))
                elif pid == p2:
                    neighbours.append((p1, diff_dims, diff_count))

            # Also add close profiles (dim_distance <= 4)
            for p in self.store.all_profiles:
                if p == pid or any(p == n[0] for n in neighbours):
                    continue
                dist = self.matrix.dim_distance(pid, p)
                if dist <= 4 and len(self.store.get_profile_trajectories(p)) >= 1:
                    diff = self.matrix.differing_dims(pid, p)
                    neighbours.append((p, diff, dist))

            if not neighbours:
                continue

            # Generate up to 12 impersonation questions per profile
            for qi in range(min(12, len(neighbours))):
                imp_profile, diff_dims, diff_count = neighbours[qi]
                imp_trajs = self.store.get_profile_trajectories(imp_profile)
                if not imp_trajs:
                    continue

                # Pick 5 real + 1 impostor
                real_sample = self.rng.sample(trajs, min(5, len(trajs)))
                imp_traj = self.rng.choice(imp_trajs)

                all_six = list(real_sample) + [imp_traj]
                self.rng.shuffle(all_six)
                imp_idx = next(i for i, t in enumerate(all_six) if t is imp_traj)

                labels = ["A", "B", "C", "D", "E", "F"][:len(all_six)]
                correct_letter = labels[imp_idx]

                # Session reference lines with task names
                session_lines = [
                    f"Session {labels[i]}: {TASK_NAMES.get(t['task_id'], t['task_id'])} ({t['task_id']})"
                    for i, t in enumerate(all_six)
                ]

                questions.append({
                    "question_id": f"s2_imp_{pid}_{qi}",
                    "setting": "cross_task",
                    "sub_type": "impersonation_detection",
                    "profile_id": pid,
                    "input_trajectories": [t["dir_name"] for t in all_six],
                    "question": (
                        "Six behavioral trajectories are provided. Five belong to the same "
                        "user and one was produced by a different user with a similar behavioral "
                        "profile. Based on the behavioral patterns in the memory context, "
                        "which session is the impostor?\n\n"
                        + "\n".join(f"  {line}" for line in session_lines)
                    ),
                    "choices": {l: f"Session {l}" for l in labels},
                    "correct": correct_letter,
                    "metadata": {
                        "impostor_profile": imp_profile,
                        "diff_dims": diff_dims,
                        "diff_count": diff_count,
                    },
                })

        return questions
