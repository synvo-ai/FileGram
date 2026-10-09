"""Setting 1: Basic — Profile Reconstruction & Behavioral QA.

Sub-types:
  - dimension_classification: identify user's L/M/R for a dimension (3-choice)
  - behavioral_fingerprint: identify which profile produced a trajectory (4-choice)
  - counterfactual: predict user behavior on held-out task (4-choice)
  - anomaly_detection: find the trajectory not from this user (5-choice)
  - semantic_attribute: identify user's language, tone, or documentation habits (3/4-choice)
"""

from __future__ import annotations

import random
from typing import Any

from .base import (
    DIMENSIONS, DIM_NAMES, TIER_LABELS, TASK_TYPES, TASK_NAMES,
    FINE_GRAINED_PAIRS,
    ProfileMatrix, TrajectoryStore, ProfileYAMLLoader,
    compute_dim_stats, aggregate_stats, profile_dim_activation_rate,
    render_description, build_choices, build_profile_portrait,
    build_counterfactual_prediction, DIM_ACTIVATION_THRESHOLDS,
)

MIN_TRAJS_FOR_PROFILE = 3  # need at least this many trajectories

# Semantic attribute definitions sourced from YAML profile fields
_YAML_ATTR_DEFS = {
    "tone": {
        "question": "Based on the writing style in this user's output files, which tone best characterizes their communication?",
        "descriptions": {
            "professional": "Professional and formal tone with structured, business-appropriate language",
            "friendly": "Friendly and approachable tone with warm, conversational language",
            "casual": "Casual and informal tone with relaxed, everyday language",
        },
    },
    "verbosity": {
        "question": "Based on this user's output documents, how would you characterize their level of detail?",
        "descriptions": {
            "detailed": "Detailed and thorough — provides extensive explanations, examples, and supporting information",
            "balanced": "Balanced — includes necessary detail without being excessive or sparse",
            "concise": "Concise and brief — communicates key points efficiently with minimal elaboration",
        },
    },
    "working_style": {
        "question": "Based on the behavioral patterns in this user's trajectories, which working style best describes their approach?",
        "descriptions": {
            "methodical": "Methodical — follows a structured, step-by-step approach with careful planning before execution",
            "exploratory": "Exploratory — experiments freely, trying different approaches and adapting as they discover more",
            "pragmatic": "Pragmatic — focuses on getting results efficiently, choosing the most practical path to completion",
        },
    },
    "thoroughness": {
        "question": "How thorough is this user in their approach to consuming and processing information?",
        "descriptions": {
            "exhaustive": "Exhaustive — reads every available file, checks every detail, and ensures complete coverage",
            "balanced": "Balanced — covers important material adequately without trying to read everything",
            "minimal": "Minimal — focuses on the essentials only, quickly identifying and processing key information",
        },
    },
    "naming_convention": {
        "question": "Based on the file names this user creates, which naming convention do they prefer?",
        "descriptions": {
            "descriptive_long": "Descriptive long names — uses full descriptive names with prefixes, dates, or categories",
            "short_abbreviation": "Short abbreviations — uses brief, abbreviated names for efficiency",
            "date_prefixed": "Date-prefixed — consistently uses date prefixes to organize files chronologically",
        },
    },
    "version_strategy": {
        "question": "How does this user handle file versions and backups during their work?",
        "descriptions": {
            "keep_history": "Keeps history — creates backups before modifications and preserves all previous versions",
            "archive_old": "Archives old — moves superseded files to an archive location rather than deleting",
            "overwrite": "Overwrites directly — replaces files in place without creating backups or version copies",
        },
    },
    "error_handling": {
        "question": "When encountering errors or issues, how does this user typically respond?",
        "descriptions": {
            "defensive": "Defensive — anticipates potential issues, creates safeguards, and handles errors carefully",
            "balanced": "Balanced — addresses errors when they occur with reasonable attention but without excessive caution",
            "optimistic": "Optimistic — moves forward quickly, dealing with errors only when they become blocking issues",
        },
    },
    "organization_style": {
        "question": "How does this user organize their file system and workspace?",
        "descriptions": {
            "deeply_nested": "Deeply nested — creates multi-level directory hierarchies with systematic categorization",
            "flat": "Flat — keeps most files in the root or minimal directory structure",
            "by_type": "By type — organizes files primarily by their type or format",
            "by_project": "By project — groups files by project or task context",
            "by_date": "By date — organizes files chronologically with date-based structures",
        },
    },
}


class Setting1Generator:
    """Generate Setting 1 (Basic) questions."""

    def __init__(self, store: TrajectoryStore, matrix: ProfileMatrix, seed: int = 42):
        self.store = store
        self.matrix = matrix
        self.rng = random.Random(seed)
        self.yaml_loader = ProfileYAMLLoader()

    def generate_all(self) -> list[dict[str, Any]]:
        questions: list[dict] = []
        questions.extend(self._gen_dimension_classification())
        questions.extend(self._gen_behavioral_fingerprint())
        questions.extend(self._gen_counterfactual())
        questions.extend(self._gen_anomaly_detection())
        questions.extend(self._gen_semantic_attribute())
        return questions

    # ------------------------------------------------------------------
    # 1.1  Dimension Classification  (6 per profile, ~100 total)
    # ------------------------------------------------------------------
    def _gen_dimension_classification(self) -> list[dict]:
        """For each profile × dimension: which L/M/R description matches?

        Choices are pure semantic behavioral descriptions — no raw statistics.
        """
        questions = []

        for pid in self.store.all_profiles:
            trajs = self.store.get_profile_trajectories(pid)
            if len(trajs) < MIN_TRAJS_FOR_PROFILE:
                continue

            for dim in DIMENSIONS:
                # Skip if dimension not well-activated for this profile
                rate = profile_dim_activation_rate(trajs, dim)
                if rate < 0.2:
                    continue

                gt_tier = self.matrix.get_tier(pid, dim)

                # Build 3 semantic descriptions: one per tier (L/M/R)
                descs = {tier: render_description(dim, tier) for tier in ("L", "M", "R")}

                correct_desc = descs[gt_tier]
                distractors = [descs[t] for t in ("L", "M", "R") if t != gt_tier]
                choices, correct_letter = build_choices(correct_desc, distractors, self.rng)

                questions.append({
                    "question_id": f"s1_dimcls_{pid}_{dim}",
                    "setting": "basic",
                    "sub_type": "dimension_classification",
                    "profile_id": pid,
                    "input_trajectories": [t["dir_name"] for t in trajs],
                    "question": (
                        f"Based on the behavioral trajectories provided, which description "
                        f"best matches this user's {DIM_NAMES[dim]} (Dimension {dim})?"
                    ),
                    "choices": choices,
                    "correct": correct_letter,
                    "metadata": {
                        "target_dimension": dim,
                        "ground_truth_tier": gt_tier,
                        "ground_truth_label": TIER_LABELS[dim][gt_tier],
                        "n_trajectories": len(trajs),
                        "activation_rate": round(rate, 2),
                    },
                })

        return questions

    # ------------------------------------------------------------------
    # 1.2  Behavioral Fingerprint  (5 per profile, ~80 total)
    # ------------------------------------------------------------------
    def _gen_behavioral_fingerprint(self) -> list[dict]:
        """Give 1 anonymous trajectory, identify from 4 candidate profiles.

        Choices are multi-dimension behavioral portraits (not profile IDs).
        """
        questions = []

        # Build a quick lookup of fine-grained neighbours
        fg_neighbours: dict[str, list[str]] = {}
        for p1, p2, _, _ in FINE_GRAINED_PAIRS:
            fg_neighbours.setdefault(p1, []).append(p2)
            fg_neighbours.setdefault(p2, []).append(p1)

        for pid in self.store.all_profiles:
            trajs = self.store.get_profile_trajectories(pid)
            if len(trajs) < 5:
                continue

            # Pick up to 28 trajectories as test cases
            test_trajs = self.rng.sample(trajs, min(28, len(trajs)))

            # Build portrait for this profile
            correct_dims = self.matrix.profiles[pid]["dimensions"]
            correct_portrait = build_profile_portrait(correct_dims)

            for qi, traj in enumerate(test_trajs):
                task_id = traj["task_id"]

                # Pick 3 distractors: stratified — 1 fine-grained + 1 medium + 1 coarse
                distractor_pool = [
                    p for p in self.store.all_profiles
                    if p != pid and len(self.store.get_profile_trajectories(p)) >= MIN_TRAJS_FOR_PROFILE
                ]
                fg_cands = [p for p in fg_neighbours.get(pid, []) if p in distractor_pool]

                # Stratify by distance: fine (1-2), medium (3-4), coarse (5-6)
                by_dist: dict[str, list[str]] = {"fine": [], "medium": [], "coarse": []}
                for p in distractor_pool:
                    dist = self.matrix.dim_distance(pid, p)
                    if dist <= 2:
                        by_dist["fine"].append(p)
                    elif dist <= 4:
                        by_dist["medium"].append(p)
                    else:
                        by_dist["coarse"].append(p)

                picked: list[str] = []
                # 1 fine-grained (prefer known fg pairs)
                if fg_cands:
                    picked.append(self.rng.choice(fg_cands))
                elif by_dist["fine"]:
                    picked.append(self.rng.choice(by_dist["fine"]))
                # 1 medium distance
                med_pool = [p for p in by_dist["medium"] if p not in picked]
                if med_pool:
                    picked.append(self.rng.choice(med_pool))
                # 1 coarse distance
                coarse_pool = [p for p in by_dist["coarse"] if p not in picked]
                if coarse_pool:
                    picked.append(self.rng.choice(coarse_pool))
                # Fill remaining from any pool
                remaining = [p for p in distractor_pool if p not in picked]
                self.rng.shuffle(remaining)
                picked.extend(remaining[:3 - len(picked)])

                if len(picked) < 3:
                    continue

                # Choices: behavioral portraits (not profile IDs)
                distractor_portraits = [
                    build_profile_portrait(self.matrix.profiles[p]["dimensions"])
                    for p in picked
                ]
                choices, correct_letter = build_choices(
                    correct_portrait, distractor_portraits, self.rng,
                )

                questions.append({
                    "question_id": f"s1_fp_{pid}_{qi}",
                    "setting": "basic",
                    "sub_type": "behavioral_fingerprint",
                    "profile_id": pid,
                    "input_trajectories": [traj["dir_name"]],
                    "question": (
                        "Based on the behavioral trajectory provided, which user profile "
                        "description best matches the observed behavior?"
                    ),
                    "choices": choices,
                    "correct": correct_letter,
                    "metadata": {
                        "task_id": task_id,
                        "task_type": TASK_TYPES.get(task_id, "unknown"),
                        "distractor_profiles": picked,
                        "has_fine_grained_distractor": bool(fg_cands),
                    },
                })

        return questions

    # ------------------------------------------------------------------
    # 1.3  Counterfactual Reasoning  (4 per profile, ~70 total)
    # ------------------------------------------------------------------
    def _gen_counterfactual(self) -> list[dict]:
        """Given N-1 trajectories, predict behavior on held-out task.

        Choices are semantic behavioral predictions (not raw statistics).
        """
        questions = []

        for pid in self.store.all_profiles:
            trajs = self.store.get_profile_trajectories(pid)
            if len(trajs) < 5:
                continue

            correct_dims = self.matrix.profiles[pid]["dimensions"]

            # Pick up to 28 tasks as held-out targets
            task_ids = [t["task_id"] for t in trajs]
            holdout_tasks = self.rng.sample(task_ids, min(28, len(task_ids)))

            for held_task in holdout_tasks:
                held_traj = self.store.get_trajectory(pid, held_task)
                if held_traj is None:
                    continue
                remaining = [t for t in trajs if t["task_id"] != held_task]
                if len(remaining) < 3:
                    continue

                task_type = TASK_TYPES.get(held_task, "unknown")

                # Build semantic prediction for the correct profile
                gt_desc = build_counterfactual_prediction(correct_dims, task_type)

                # Distractors: predictions from 3 other profiles
                other_trajs = [
                    t for t in self.store.get_task_trajectories(held_task)
                    if t["profile_id"] != pid
                ]
                if len(other_trajs) < 3:
                    continue
                distractor_trajs = self.rng.sample(other_trajs, 3)
                distractor_descs = []
                distractor_profiles = []
                for dt in distractor_trajs:
                    dp = dt["profile_id"]
                    distractor_profiles.append(dp)
                    dp_dims = self.matrix.profiles[dp]["dimensions"]
                    distractor_descs.append(
                        build_counterfactual_prediction(dp_dims, task_type)
                    )

                # Deduplicate — if two profiles have identical predictions, skip
                all_descs = [gt_desc] + distractor_descs
                if len(set(all_descs)) < 4:
                    continue

                choices, correct_letter = build_choices(gt_desc, distractor_descs, self.rng)

                questions.append({
                    "question_id": f"s1_cf_{pid}_{held_task}",
                    "setting": "basic",
                    "sub_type": "counterfactual",
                    "profile_id": pid,
                    "input_trajectories": [t["dir_name"] for t in remaining],
                    "question": (
                        f"Based on this user's behavior across {len(remaining)} other tasks, "
                        f"predict how they would approach a {task_type} task ({held_task}). "
                        f"Which behavioral prediction is most consistent with this user?"
                    ),
                    "choices": choices,
                    "correct": correct_letter,
                    "metadata": {
                        "held_out_task": held_task,
                        "task_type": task_type,
                        "distractor_profiles": distractor_profiles,
                    },
                })

        return questions

    # ------------------------------------------------------------------
    # 1.4  Anomaly Detection  (3 per profile, ~50 total)
    # ------------------------------------------------------------------
    def _gen_anomaly_detection(self) -> list[dict]:
        """4 trajectories from same profile + 1 from different. Find the outlier.

        Question text references sessions by anonymous label only —
        the memory system must identify the outlier from ingested data.
        """
        questions = []

        for pid in self.store.all_profiles:
            trajs = self.store.get_profile_trajectories(pid)
            if len(trajs) < 5:
                continue

            used_impostors: set[str] = set()  # track used impostor profiles

            # Generate up to 25 anomaly questions per profile
            for qi in range(min(25, len(trajs) - 4)):
                seed_offset = hash(f"{pid}_{qi}") % 10000
                local_rng = random.Random(self.rng.randint(0, 99999) + seed_offset)

                # Pick 4 from this profile
                real_trajs = local_rng.sample(trajs, 4)
                real_task_ids = {t["task_id"] for t in real_trajs}

                # Pick 1 impostor: prefer fine-grained pair on a shared task
                impostor_candidates = []
                for p in self.store.all_profiles:
                    if p == pid:
                        continue
                    p_trajs = self.store.get_profile_trajectories(p)
                    # Prefer same task as one of our real ones (harder)
                    for pt in p_trajs:
                        if pt["task_id"] in real_task_ids:
                            impostor_candidates.append((p, pt, self.matrix.dim_distance(pid, p)))
                            break

                if not impostor_candidates:
                    continue

                # Prefer close profiles (fine-grained) for harder questions
                # Deprioritize already-used impostors
                impostor_candidates.sort(key=lambda x: (x[0] in used_impostors, x[2]))
                impostor_profile, impostor_traj, dist = impostor_candidates[
                    local_rng.randint(0, min(2, len(impostor_candidates) - 1))
                ]
                used_impostors.add(impostor_profile)

                # Build 5 sessions — referenced by label, NO stats in question text
                all_five = real_trajs + [impostor_traj]
                local_rng.shuffle(all_five)
                impostor_idx = next(i for i, t in enumerate(all_five) if t is impostor_traj)

                labels = ["A", "B", "C", "D", "E"]
                correct_letter = labels[impostor_idx]

                # Session reference lines with task names
                session_lines = [
                    f"Session {labels[i]}: {TASK_NAMES.get(t['task_id'], t['task_id'])} ({t['task_id']})"
                    for i, t in enumerate(all_five)
                ]

                questions.append({
                    "question_id": f"s1_anom_{pid}_{qi}",
                    "setting": "basic",
                    "sub_type": "anomaly_detection",
                    "profile_id": pid,
                    "input_trajectories": [t["dir_name"] for t in all_five],
                    "question": (
                        "Five behavioral trajectories are provided. Four belong to the same "
                        "user and one was produced by a different user. Based on the "
                        "behavioral patterns in the memory context, which session is the "
                        "outlier?\n\n"
                        + "\n".join(f"  {line}" for line in session_lines)
                    ),
                    "choices": {l: f"Session {l}" for l in labels},
                    "correct": correct_letter,
                    "metadata": {
                        "impostor_profile": impostor_profile,
                        "impostor_task": impostor_traj["task_id"],
                        "dim_distance": dist,
                    },
                })

        return questions

    # ------------------------------------------------------------------
    # 1.5  Semantic Attribute  (NEW — tests semantic memory channel)
    # ------------------------------------------------------------------
    def _gen_semantic_attribute(self) -> list[dict]:
        """Questions about user identity and style attributes.

        Tests whether the memory system captures semantic signals from
        produced content: writing language, profession/role, documentation habits.
        """
        questions = []

        for pid in self.store.all_profiles:
            trajs = self.store.get_profile_trajectories(pid)
            if len(trajs) < MIN_TRAJS_FOR_PROFILE:
                continue

            pdata = self.matrix.profiles[pid]
            gt_lang = pdata.get("language", "")
            gt_role = pdata.get("role", "")

            # Q1: Language — what language does this user primarily write in?
            if gt_lang:
                other_langs = ["English", "Chinese"]
                if gt_lang in other_langs:
                    wrong_lang = [l for l in other_langs if l != gt_lang][0]
                    choices, cl = build_choices(
                        gt_lang,
                        [wrong_lang, "Both equally", "Unable to determine from behavioral data"],
                        self.rng,
                    )
                    questions.append({
                        "question_id": f"s1_sem_lang_{pid}",
                        "setting": "basic",
                        "sub_type": "semantic_attribute",
                        "profile_id": pid,
                        "input_trajectories": [t["dir_name"] for t in trajs],
                        "question": (
                            "Based on the user's output files and behavioral traces, "
                            "what language does this user primarily write in?"
                        ),
                        "choices": choices,
                        "correct": cl,
                        "metadata": {
                            "attribute": "language",
                            "ground_truth": gt_lang,
                        },
                    })

            # Q2: Role — what profession does this user appear to have?
            if gt_role:
                all_roles = list({
                    self.matrix.profiles[p].get("role", "")
                    for p in self.store.all_profiles
                    if self.matrix.profiles[p].get("role", "") and p != pid
                })
                self.rng.shuffle(all_roles)
                wrong_roles = all_roles[:3]
                if len(wrong_roles) >= 3:
                    choices, cl = build_choices(gt_role, wrong_roles, self.rng)
                    questions.append({
                        "question_id": f"s1_sem_role_{pid}",
                        "setting": "basic",
                        "sub_type": "semantic_attribute",
                        "profile_id": pid,
                        "input_trajectories": [t["dir_name"] for t in trajs],
                        "question": (
                            "Based on the content and style of this user's output documents, "
                            "which professional role best matches this user?"
                        ),
                        "choices": choices,
                        "correct": cl,
                        "metadata": {
                            "attribute": "role",
                            "ground_truth": gt_role,
                        },
                    })

            # Q3: Documentation habit — does user create auxiliary docs?
            dims = pdata.get("dimensions", {})
            b_tier = dims.get("B", "M")
            doc_descs = {
                "L": "Consistently creates auxiliary documentation (README, index, appendix files) alongside main output",
                "M": "Occasionally creates supporting documents when the task warrants it",
                "R": "Rarely or never creates supplementary documentation — produces only the primary output file",
            }
            correct_doc = doc_descs[b_tier]
            wrong_docs = [doc_descs[t] for t in ("L", "M", "R") if t != b_tier]
            choices, cl = build_choices(correct_doc, wrong_docs, self.rng)
            questions.append({
                "question_id": f"s1_sem_doc_{pid}",
                "setting": "basic",
                "sub_type": "semantic_attribute",
                "profile_id": pid,
                "input_trajectories": [t["dir_name"] for t in trajs],
                "question": (
                    "Based on this user's behavioral traces, which description best "
                    "characterizes their documentation habits?"
                ),
                "choices": choices,
                "correct": cl,
                "metadata": {
                    "attribute": "documentation",
                    "ground_truth_tier": b_tier,
                },
            })

            # Q4+: YAML-sourced attribute questions (tone, verbosity, etc.)
            for attr, adef in _YAML_ATTR_DEFS.items():
                gt_val = self.yaml_loader.get_attr(pid, attr)
                if not gt_val or gt_val not in adef["descriptions"]:
                    continue

                correct_desc = adef["descriptions"][gt_val]
                wrong_descs = [
                    adef["descriptions"][v]
                    for v in adef["descriptions"]
                    if v != gt_val
                ]
                if len(wrong_descs) < 2:
                    continue

                self.rng.shuffle(wrong_descs)
                choices, cl = build_choices(correct_desc, wrong_descs[:3], self.rng)
                questions.append({
                    "question_id": f"s1_sem_{attr}_{pid}",
                    "setting": "basic",
                    "sub_type": "semantic_attribute",
                    "profile_id": pid,
                    "input_trajectories": [t["dir_name"] for t in trajs],
                    "question": adef["question"],
                    "choices": choices,
                    "correct": cl,
                    "metadata": {
                        "attribute": attr,
                        "ground_truth": gt_val,
                    },
                })

        return questions
