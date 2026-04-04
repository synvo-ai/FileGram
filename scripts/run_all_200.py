#!/usr/bin/env python3
"""
Run all 640 trajectories: 20 profiles × 32 tasks (20 original + 12 multimodal).
Usage:
    python run_all_200.py                    # sequential
    python run_all_200.py --parallel 3       # 3 concurrent
    python run_all_200.py --parallel 3 --dry # preview only
    python run_all_200.py --profiles p1,p3   # specific profiles
    python run_all_200.py --tasks t01,t11    # specific tasks
    python run_all_200.py --signal-dir signal/ablation/perturbation_levels  # override output dir
    python run_all_200.py --perturb-dims 0   # force no perturbation
    python run_all_200.py --perturb-dims 1   # force exactly 1 dimension perturbed
    python run_all_200.py --perturb-dims 2   # force exactly 2 dimensions perturbed
"""

import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TASKS_DIR = PROJECT_ROOT / "tasks"
PROFILES_DIR = PROJECT_ROOT / "profiles"
SIGNAL_DIR = PROJECT_ROOT / "signal"
LOG_DIR = PROJECT_ROOT / "logs"
SUFFIX = os.environ.get("TRAJECTORY_SUFFIX", "")

ALL_PROFILES = [f"p{i}" for i in range(1, 21)]
ALL_TASKS = [f"t{i:02d}" for i in range(1, 21)] + [
    "t21", "t22", "t23", "t24", "t25", "t26", "t27", "t28", "t29", "t30", "t31", "t32",
]

# --- Perturbation design (Track 3: Profile Shift) ---
# 8 candidate tasks, each profile randomly picks 5 → 100 perturbed trajectories total.
# Selection criteria: ablation-validated dims (C>D>B>A), natural "twin" standard tasks,
# dimension coverage across all 6 dims.
PERTURBATION_CANDIDATES = [
    "t12",  # iterate   — twins: t11, t13  — dims: C, D, B (ablation best)
    "t18",  # maintain  — twin:  t10       — dims: D, C, E
    "t22",  # understand— twins: t01, t02  — dims: A, B, F (best F candidate)
    "t24",  # synthesize— twins: t06, t15  — dims: A, B, E
    "t26",  # organize  — twins: t05, t14  — dims: C, A, F
    "t29",  # organize  — twins: t05, t14  — dims: C, D, A
    "t31",  # create    — twins: t04, t08  — dims: B, C, F
    "t16",  # understand— twins: t01, t02  — dims: A, B, E
]
PERTURB_PER_PROFILE = 5  # each profile gets 5 out of 8 candidates perturbed

# Per-task active dimensions (only perturb dims the task can express)
# Derived from task design + ablation validation (2026-02-26).
TASK_ACTIVE_DIMS = {
    "t12": ["C", "D", "B"],       # format standardization → org, iteration, production
    "t16": ["A", "B", "E"],        # time-constrained triage → consumption, production, curation
    "t18": ["D", "C", "E"],        # KB maintenance → iteration, organization, curation (multi-round)
    "t22": ["A", "B", "F"],        # film collection understand → consumption, production, cross-modal
    "t24": ["A", "B", "E"],        # earnings synthesis → consumption, production, curation
    "t26": ["C", "A", "F"],        # digital assets organize → organization, consumption, cross-modal
    "t29": ["C", "D", "A"],        # company docs organize → organization, iteration, consumption
    "t31": ["B", "C", "F"],        # nature create → production, organization, cross-modal
}


def get_perturbed_tasks(profile_short: str) -> set[str]:
    """Return the set of tasks that should be perturbed for this profile.

    Deterministic: same profile always gets the same 5 tasks.
    Balanced: each candidate task is picked by ~12-13 out of 20 profiles.
    """
    seed_str = f"{profile_short}_perturbation_assignment"
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    candidates = list(PERTURBATION_CANDIDATES)
    rng.shuffle(candidates)
    return set(candidates[:PERTURB_PER_PROFILE])

# --- Profile dimension vectors (from profile_matrix.json) ---
PROFILE_DIMS = {
    "p1":  {"A": "L", "B": "L", "C": "L", "D": "L", "E": "L", "F": "M"},
    "p2":  {"A": "L", "B": "L", "C": "R", "D": "R", "E": "L", "F": "M"},
    "p3":  {"A": "M", "B": "R", "C": "R", "D": "R", "E": "R", "F": "R"},
    "p4":  {"A": "M", "B": "L", "C": "M", "D": "L", "E": "M", "F": "L"},
    "p5":  {"A": "R", "B": "M", "C": "M", "D": "M", "E": "M", "F": "M"},
    "p6":  {"A": "R", "B": "M", "C": "L", "D": "R", "E": "R", "F": "R"},
    "p7":  {"A": "L", "B": "M", "C": "M", "D": "M", "E": "R", "F": "L"},
    "p8":  {"A": "R", "B": "R", "C": "M", "D": "L", "E": "M", "F": "R"},
    "p9":  {"A": "M", "B": "M", "C": "L", "D": "M", "E": "L", "F": "L"},
    "p10": {"A": "L", "B": "R", "C": "R", "D": "R", "E": "M", "F": "M"},
    "p11": {"A": "M", "B": "L", "C": "L", "D": "L", "E": "L", "F": "R"},
    "p12": {"A": "R", "B": "L", "C": "R", "D": "L", "E": "R", "F": "M"},
    "p13": {"A": "L", "B": "M", "C": "L", "D": "M", "E": "L", "F": "L"},
    "p14": {"A": "M", "B": "R", "C": "L", "D": "L", "E": "R", "F": "R"},
    "p15": {"A": "R", "B": "L", "C": "M", "D": "M", "E": "M", "F": "M"},
    "p16": {"A": "M", "B": "M", "C": "R", "D": "M", "E": "L", "F": "R"},
    "p17": {"A": "L", "B": "L", "C": "L", "D": "L", "E": "R", "F": "L"},
    "p18": {"A": "R", "B": "R", "C": "R", "D": "R", "E": "L", "F": "M"},
    "p19": {"A": "M", "B": "M", "C": "R", "D": "M", "E": "R", "F": "R"},
    "p20": {"A": "L", "B": "R", "C": "M", "D": "R", "E": "M", "F": "L"},
}

# --- Perturbation override text per dimension × shift direction ---
# Each shift is one tier: L→M, M→L, M→R, R→M
# Keys: (dim, from_tier, to_tier) → {"zh": ..., "en": ...}
PERTURBATION_TEXT = {
    ("A", "L", "M"): {
        "zh": "【扰动-信息获取】今天时间有限，你决定用搜索来定位关键文件，而不是逐个阅读所有文件。先用grep/glob找到相关内容再针对性阅读。",
        "en": "[PERTURBATION-CONSUMPTION] Due to time pressure today, search for key files using keywords rather than reading everything sequentially. Use grep/glob first, then read targeted sections.",
    },
    ("A", "M", "L"): {
        "zh": "【扰动-信息获取】今天你比平时更仔细，决定从头到尾逐个阅读所有文件，不使用搜索工具，按顺序深入理解每个文件。",
        "en": "[PERTURBATION-CONSUMPTION] Today you're being extra thorough — read every file from beginning to end, in order, without using search. Understand each file deeply before moving on.",
    },
    ("A", "M", "R"): {
        "zh": "【扰动-信息获取】今天你想快速浏览所有文件，只看前10-20行就行，追求覆盖面而非深度。",
        "en": "[PERTURBATION-CONSUMPTION] Today you're in a hurry — skim each file by reading only the first 10-20 lines, prioritize breadth over depth.",
    },
    ("A", "R", "M"): {
        "zh": "【扰动-信息获取】今天你比平时更专注，用搜索精确定位你需要的内容，而不是广泛浏览。找到后针对性深入阅读。",
        "en": "[PERTURBATION-CONSUMPTION] Today you're more focused — use keyword search to locate exactly what you need, then read those sections in depth rather than browsing broadly.",
    },
    ("B", "L", "M"): {
        "zh": "【扰动-内容产出】今天你决定简洁一些，不写附录和辅助文件。一个主文件、两级标题就够了，控制在80-150行。",
        "en": "[PERTURBATION-PRODUCTION] Today keep it simpler — produce a single, well-organized document without appendices or auxiliary files. Two heading levels, 80-150 lines.",
    },
    ("B", "M", "L"): {
        "zh": "【扰动-内容产出】今天你特别想做到详尽，多写子标题(####)、数据表格、附加参考材料，并创建辅助文件(README、索引)。",
        "en": "[PERTURBATION-PRODUCTION] Today be extra detailed — add multi-level headings (####), data tables, supplementary reference files, and create auxiliary materials like README and index.",
    },
    ("B", "M", "R"): {
        "zh": "【扰动-内容产出】今天追求效率，输出要极简——只保留关键要点的bullet list，控制在60行以内，不要表格和额外格式。",
        "en": "[PERTURBATION-PRODUCTION] Today maximize efficiency — output only key bullet points, keep it under 60 lines, no tables or extra formatting.",
    },
    ("B", "R", "M"): {
        "zh": "【扰动-内容产出】今天你想做得更完整一些，加上适当的段落结构和分节(##/###)，比平时更有条理，80-150行左右。",
        "en": "[PERTURBATION-PRODUCTION] Today be more polished — add proper sections (##/###), headings, and a bit more detail than usual. Target 80-150 lines.",
    },
    ("C", "L", "M"): {
        "zh": "【扰动-文件组织】今天你决定简化目录结构，只在必要时建一层子目录，不追求深层嵌套。",
        "en": "[PERTURBATION-ORGANIZATION] Today simplify your organization — create subdirectories only when necessary (1-2 levels max), avoid deep nesting.",
    },
    ("C", "M", "L"): {
        "zh": "【扰动-文件组织】今天你想把文件组织得特别整齐，建立3级以上目录结构，每个层级都有清晰的分类，并创建索引文件。",
        "en": "[PERTURBATION-ORGANIZATION] Today organize meticulously — create 3+ levels of subdirectories with descriptive names and an index file at each level.",
    },
    ("C", "M", "R"): {
        "zh": "【扰动-文件组织】今天你懒得建目录，所有文件直接放在当前目录下，用简短文件名，不建子文件夹。",
        "en": "[PERTURBATION-ORGANIZATION] Today keep things flat — put all files in the root directory, no subdirectories, use short file names.",
    },
    ("C", "R", "M"): {
        "zh": "【扰动-文件组织】今天你想稍微整理一下，如果同类文件超过3个就建一个子目录归类。",
        "en": "[PERTURBATION-ORGANIZATION] Today organize a bit — create a subdirectory when you have 3+ related files of the same type.",
    },
    ("D", "L", "M"): {
        "zh": "【扰动-迭代策略】今天你不像平时那么小心翼翼，可以一次修改较大段内容(10-30行)，不需要每次只改几行。",
        "en": "[PERTURBATION-ITERATION] Today be more decisive with edits — make medium-sized changes (10-30 lines at once), no need for tiny incremental patches.",
    },
    ("D", "M", "L"): {
        "zh": "【扰动-迭代策略】今天你想特别谨慎，每次只用edit修改几行(≤10行)，改完就review，并在修改前用cp备份原文件。",
        "en": "[PERTURBATION-ITERATION] Today be extra careful — make tiny edits (≤10 lines each), review after each change, and always cp backup files before modifying.",
    },
    ("D", "M", "R"): {
        "zh": "【扰动-迭代策略】今天你想一步到位，如果需要修改，直接用write重写整个文件而不是反复小编辑。",
        "en": "[PERTURBATION-ITERATION] Today if revision is needed, rewrite the entire file from scratch with write rather than making incremental edits.",
    },
    ("D", "R", "M"): {
        "zh": "【扰动-迭代策略】今天你想更有条理地修改，用edit做中等规模的修改(10-30行)，而不是直接重写整个文件。",
        "en": "[PERTURBATION-ITERATION] Today be more measured — use focused edits (10-30 lines each) rather than rewriting entire files at once.",
    },
    ("E", "L", "M"): {
        "zh": "【扰动-工作节奏】今天你不严格分阶段了，可以边读边写，自然交替进行，保持稳定节奏。",
        "en": "[PERTURBATION-CURATION] Today don't separate reading and writing into strict phases — interleave them naturally at a steady pace.",
    },
    ("E", "M", "L"): {
        "zh": "【扰动-工作节奏】今天你想更有条理：先把所有材料读完(Phase 1)，然后计划(Phase 2)，最后再动手写(Phase 3)。完成一个阶段再进入下一个。",
        "en": "[PERTURBATION-CURATION] Today be more structured: Phase 1 — read ALL materials first. Phase 2 — plan. Phase 3 — write. Complete each phase before moving on.",
    },
    ("E", "M", "R"): {
        "zh": "【扰动-工作节奏】今天你特别急，读到什么就马上写，快速在文件间切换，不用按顺序来，哪个文件有灵感就先处理哪个。",
        "en": "[PERTURBATION-CURATION] Today you're in a rush — start writing immediately after reading, switch between files rapidly, work on whatever grabs your attention first.",
    },
    ("E", "R", "M"): {
        "zh": "【扰动-工作节奏】今天你想稍微沉稳一点，按逻辑顺序处理文件，不要频繁跳转，保持稳定的工作节奏。",
        "en": "[PERTURBATION-CURATION] Today work at a steadier pace — process files in a logical order without jumping around, maintain a consistent rhythm.",
    },
    ("F", "L", "M"): {
        "zh": "【扰动-跨模态】今天你不创建独立的图片/图表文件了，但可以在文档中使用Markdown表格来展示数据。",
        "en": "[PERTURBATION-CROSSMODAL] Today skip creating separate image/chart files — use markdown tables in your documents when data needs visualization.",
    },
    ("F", "M", "L"): {
        "zh": "【扰动-跨模态】今天你想用更多视觉材料——为数据创建图表或Mermaid图，在文档中引用Figure编号，建一个figures/目录。",
        "en": "[PERTURBATION-CROSSMODAL] Today add more visual materials — create charts/Mermaid diagrams for data, reference figures in documents, maintain a figures/ directory.",
    },
    ("F", "M", "R"): {
        "zh": "【扰动-跨模态】今天你只用纯文本，不用表格，不用图表，所有数据用文字段落和bullet list描述。",
        "en": "[PERTURBATION-CROSSMODAL] Today use pure text only — no tables, no charts, no images. Describe all data in prose paragraphs and bullet lists.",
    },
    ("F", "R", "M"): {
        "zh": "【扰动-跨模态】今天你可以用Markdown表格来展示对比数据和摘要信息，让输出更结构化。",
        "en": "[PERTURBATION-CROSSMODAL] Today include markdown tables where comparisons or structured data need to be presented.",
    },
}

# Profile name → profile_id mapping (filename stems)
PROFILE_MAP = {
    "p1": "p1_methodical",
    "p2": "p2_thorough_reviser",
    "p3": "p3_efficient_executor",
    "p4": "p4_structured_analyst",
    "p5": "p5_balanced_organizer",
    "p6": "p6_quick_curator",
    "p7": "p7_visual_reader",
    "p8": "p8_minimal_editor",
    "p9": "p9_visual_organizer",
    "p10": "p10_silent_auditor",
    "p11": "p11_meticulous_planner",
    "p12": "p12_prolific_scanner",
    "p13": "p13_visual_architect",
    "p14": "p14_concise_organizer",
    "p15": "p15_thorough_surveyor",
    "p16": "p16_phased_minimalist",
    "p17": "p17_creative_archivist",
    "p18": "p18_decisive_scanner",
    "p19": "p19_agile_pragmatist",
    "p20": "p20_visual_auditor",
}

# Chinese profiles use prompt_zh, English profiles use prompt_en
ENGLISH_PROFILES = {"p3", "p6", "p8", "p10", "p11", "p14", "p15", "p16", "p18", "p20"}

# Tier name mapping for metadata readability
TIER_NAMES = {
    "A": {"L": "sequential", "M": "targeted", "R": "breadth_first"},
    "B": {"L": "comprehensive", "M": "balanced", "R": "minimal"},
    "C": {"L": "deeply_nested", "M": "adaptive", "R": "flat"},
    "D": {"L": "incremental", "M": "balanced", "R": "rewrite"},
    "E": {"L": "phased", "M": "steady", "R": "bursty"},
    "F": {"L": "visual_heavy", "M": "balanced", "R": "text_only"},
}


def _shift_tier(tier: str, rng: random.Random) -> str:
    """Shift a tier by one step. L→M, R→M, M→L or M→R (random)."""
    if tier == "L":
        return "M"
    elif tier == "R":
        return "M"
    else:  # M
        return rng.choice(["L", "R"])


def generate_perturbation(profile_short: str, task_short: str, force_dims: int | None = None) -> dict | None:
    """Generate behavioral perturbation for a trajectory.

    Args:
        profile_short: e.g., "p1"
        task_short: e.g., "t12"
        force_dims: if set, force exactly N dimensions to perturb (0=skip, 1, 2).
                    If None, uses random 1-2 for PERTURBATION_TASKS, skip for others.

    Returns dict with keys: shifted_dims, effective_vector, prompt_text, seed
    or None if perturbation is disabled / not applicable.
    """
    # force_dims=0 means explicitly no perturbation
    if force_dims == 0:
        return None

    # If not forcing, only perturb tasks assigned to this profile
    if force_dims is None:
        perturbed_for_profile = get_perturbed_tasks(profile_short)
        if task_short not in perturbed_for_profile:
            return None

    dims = PROFILE_DIMS.get(profile_short)
    if not dims:
        return None

    # Deterministic seed from profile + task + force_dims
    seed_str = f"{profile_short}_{task_short}_perturbation"
    if force_dims is not None:
        seed_str += f"_force{force_dims}"
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # Select dimensions to perturb (from task-active dims only)
    num_shifts = force_dims if force_dims is not None else 1  # default: 1 dim (cleaner signal per ablation)
    active = TASK_ACTIVE_DIMS.get(task_short, list("ABCDEF"))
    if isinstance(active, (set, list)):
        active = list(active)
    rng.shuffle(active)
    selected_dims = active[:num_shifts]
    if not selected_dims:
        return None

    # Compute shifts
    shifted_dims = {}
    effective_vector = dict(dims)  # copy original
    lang = "en" if profile_short in ENGLISH_PROFILES else "zh"
    prompt_parts = []

    for dim in sorted(selected_dims):
        original_tier = dims[dim]
        new_tier = _shift_tier(original_tier, rng)
        shifted_dims[dim] = {
            "original": original_tier,
            "perturbed": new_tier,
            "original_name": TIER_NAMES[dim][original_tier],
            "perturbed_name": TIER_NAMES[dim][new_tier],
        }
        effective_vector[dim] = new_tier

        # Get override text
        key = (dim, original_tier, new_tier)
        text_entry = PERTURBATION_TEXT.get(key)
        if text_entry:
            prompt_parts.append(text_entry[lang])

    if not prompt_parts:
        return None

    # Build prompt override block
    if lang == "zh":
        header = "\n\n---\n⚠️ 今日行为调整（请在本次任务中遵循以下临时变化）：\n"
    else:
        header = "\n\n---\nBEHAVIORAL ADJUSTMENT (follow these temporary changes for this task):\n"
    prompt_text = header + "\n".join(prompt_parts)

    return {
        "seed": seed,
        "num_shifts": num_shifts,
        "shifted_dims": shifted_dims,
        "original_vector": dims,
        "effective_vector": effective_vector,
        "prompt_text": prompt_text,
    }


def write_perturbation_metadata(profile_id: str, task_id: str, perturbation: dict,
                                signal_base: Path | None = None):
    """Write perturbation ground truth to signal directory."""
    base = signal_base if signal_base else SIGNAL_DIR
    sig_dir = base / f"{profile_id}_{task_id}{SUFFIX}"
    if not sig_dir.exists():
        return
    meta = {
        "applied": True,
        "seed": perturbation["seed"],
        "num_shifts": perturbation["num_shifts"],
        "shifted_dims": perturbation["shifted_dims"],
        "original_vector": perturbation["original_vector"],
        "effective_vector": perturbation["effective_vector"],
    }
    meta_path = sig_dir / "perturbation.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load_task(task_short: str) -> dict:
    """Load task JSON. task_short like 't01'."""
    task_file = TASKS_DIR / f"{task_short}.json"
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    with open(task_file) as f:
        return json.load(f)


def get_prompt(task_data: dict, profile_short: str) -> str:
    """Get the appropriate prompt (zh or en) for a profile."""
    if profile_short in ENGLISH_PROFILES:
        return task_data.get("prompt_en", task_data.get("prompt_zh", ""))
    return task_data.get("prompt_zh", task_data.get("prompt_en", ""))


def is_completed(profile_id: str, task_id: str) -> bool:
    """Check if trajectory completed successfully.

    A trajectory is complete if and only if its events.json contains a
    session_end event. The agent only emits session_end on successful
    completion — error/timeout exits skip it.
    """
    sig_dir = SIGNAL_DIR / f"{profile_id}_{task_id}{SUFFIX}"
    events_path = sig_dir / "events.json"
    if not events_path.exists():
        return False
    try:
        with open(events_path) as f:
            events = json.load(f)
        if not events:
            return False
        # Check if session_end event is present (normally the last event)
        return any(e.get("event_type") == "session_end" for e in events)
    except (json.JSONDecodeError, OSError):
        return False


def run_single(profile_short: str, task_short: str, dry: bool = False,
               force_perturb_dims: int | None = None, signal_base: Path | None = None,
               log_base: Path | None = None) -> dict:
    """Run a single trajectory. Returns result dict."""
    profile_id = PROFILE_MAP[profile_short]
    task_data = load_task(task_short)
    task_id = task_data["task_id"]  # e.g., "T-01"
    prompt = get_prompt(task_data, profile_short)

    # Use override signal dir if provided
    effective_signal_dir = signal_base if signal_base else SIGNAL_DIR

    result = {
        "profile": profile_id,
        "task": task_id,
        "status": "unknown",
        "events": 0,
        "duration_s": 0,
    }

    # Skip if already completed
    sig_dir = effective_signal_dir / f"{profile_id}_{task_id}{SUFFIX}"
    events_path = sig_dir / "events.json"
    if events_path.exists():
        try:
            with open(events_path) as f:
                events = json.load(f)
            if events and any(e.get("event_type") == "session_end" for e in events):
                result["status"] = "skipped"
                return result
        except (json.JSONDecodeError, OSError):
            pass

    # Generate perturbation
    perturbation = generate_perturbation(profile_short, task_short, force_dims=force_perturb_dims)
    if perturbation:
        prompt = prompt + perturbation["prompt_text"]
        result["perturbation"] = {
            "applied": True,
            "shifted_dims": list(perturbation["shifted_dims"].keys()),
        }

    if dry:
        result["status"] = "dry_run"
        perturb_info = ""
        if perturbation:
            shifts = ", ".join(
                f"{d}:{v['original']}→{v['perturbed']}"
                for d, v in perturbation["shifted_dims"].items()
            )
            perturb_info = f" [PERTURB: {shifts}]"
        print(f"  [DRY] {profile_id} × {task_id}{perturb_info}: {prompt[:60]}...")
        return result

    print(f"  [START] {profile_id} × {task_id}", end="")
    if perturbation:
        shifts = ", ".join(
            f"{d}:{v['original']}→{v['perturbed']}"
            for d, v in perturbation["shifted_dims"].items()
        )
        print(f" [PERTURB: {shifts}]", end="")
    print()
    t0 = time.time()

    try:
        script = str(PROJECT_ROOT / "run_trajectory.sh")
        # Log file for each trajectory
        effective_log_dir = log_base if log_base else LOG_DIR
        effective_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = effective_log_dir / f"{profile_id}_{task_id}{SUFFIX}.log"
        log_file = open(log_path, "w")

        # Pass SIGNAL_BASE and LOG_BASE env vars if overridden
        env = os.environ.copy()
        if signal_base:
            env["SIGNAL_BASE"] = str(signal_base)
        if log_base:
            env["LOG_BASE"] = str(log_base)

        # Use Popen with process group so we can kill ALL children on timeout
        proc = subprocess.Popen(
            ["bash", script, profile_id, task_id, prompt],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
            start_new_session=True,  # new process group for clean kill
        )

        try:
            proc.wait(timeout=None)  # no timeout
        except subprocess.TimeoutExpired:
            # Kill entire process group (bash + filegram + all children)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            log_file.close()
            result["status"] = "timeout"
            result["duration_s"] = 1200
            print(f"  [TIMEOUT] {profile_id} × {task_id} (log: {log_path})")
            return result

        log_file.close()
        duration = time.time() - t0
        result["duration_s"] = round(duration, 1)

        if proc.returncode == 0:
            # Count events
            signal_path = effective_signal_dir / f"{profile_id}_{task_id}{SUFFIX}" / "events.json"
            if signal_path.exists():
                with open(signal_path) as f:
                    events = json.load(f)
                result["events"] = len(events)
            result["status"] = "success"
            # Write perturbation metadata to signal directory
            if perturbation:
                write_perturbation_metadata(profile_id, task_id, perturbation, effective_signal_dir)
            print(f"  [DONE] {profile_id} × {task_id}: {result['events']} events in {duration:.0f}s")
        else:
            # Read error from log
            log_content = log_path.read_text()[-500:] if log_path.exists() else "no log"
            result["status"] = "error"
            result["error"] = log_content
            print(f"  [FAIL] {profile_id} × {task_id}: {log_content[:100]}")

    except Exception as e:
        if 'log_file' in locals() and not log_file.closed:
            log_file.close()
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  [ERROR] {profile_id} × {task_id}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run 200 trajectories")
    parser.add_argument("--parallel", type=int, default=1, help="Number of concurrent runs")
    parser.add_argument("--dry", action="store_true", help="Preview without running")
    parser.add_argument("--profiles", type=str, help="Comma-separated profile shorts (e.g., p1,p3)")
    parser.add_argument("--tasks", type=str, help="Comma-separated task shorts (e.g., t01,t11)")
    parser.add_argument("--signal-dir", type=str, help="Override signal output directory (default: signal/)")
    parser.add_argument("--log-dir", type=str, help="Override log output directory (default: logs/)")
    parser.add_argument("--perturb-dims", type=int, choices=[0, 1, 2], default=None,
                        help="Force perturbation dimension count: 0=none, 1=one dim, 2=two dims (default: random 1-2 for perturbation tasks)")
    args = parser.parse_args()

    signal_base = Path(args.signal_dir) if args.signal_dir else None
    if signal_base and not signal_base.is_absolute():
        signal_base = PROJECT_ROOT / signal_base

    log_base = Path(args.log_dir) if args.log_dir else None
    if log_base and not log_base.is_absolute():
        log_base = PROJECT_ROOT / log_base

    profiles = args.profiles.split(",") if args.profiles else ALL_PROFILES
    tasks = args.tasks.split(",") if args.tasks else ALL_TASKS

    # Validate
    for p in profiles:
        if p not in PROFILE_MAP:
            print(f"ERROR: Unknown profile '{p}'. Valid: {list(PROFILE_MAP.keys())}")
            sys.exit(1)
    for t in tasks:
        task_file = TASKS_DIR / f"{t}.json"
        if not task_file.exists():
            print(f"ERROR: Task file not found: {task_file}")
            sys.exit(1)

    # Build job list
    jobs = [(p, t) for p in profiles for t in tasks]
    total = len(jobs)

    # Count already completed (check in the target signal dir)
    def _check_completed(profile_id, task_id):
        base = signal_base if signal_base else SIGNAL_DIR
        ep = base / f"{profile_id}_{task_id}{SUFFIX}" / "events.json"
        if not ep.exists():
            return False
        try:
            with open(ep) as f:
                evts = json.load(f)
            return bool(evts) and any(e.get("event_type") == "session_end" for e in evts)
        except (json.JSONDecodeError, OSError):
            return False

    completed_count = sum(
        1 for p, t in jobs
        if _check_completed(PROFILE_MAP[p], load_task(t)["task_id"])
    )

    print(f"=== FileGram Batch Run ===")
    print(f"Profiles: {len(profiles)} ({', '.join(profiles)})")
    print(f"Tasks: {len(tasks)} ({', '.join(tasks)})")
    print(f"Total: {total} trajectories ({completed_count} already done, {total - completed_count} to run)")
    print(f"Parallelism: {args.parallel}")
    if signal_base:
        print(f"Signal dir: {signal_base}")
    if log_base:
        print(f"Log dir: {log_base}")
    if args.perturb_dims is not None:
        print(f"Perturbation: forced {args.perturb_dims} dims")
    print(f"{'DRY RUN' if args.dry else 'LIVE RUN'}")
    print()

    results = []
    t_start = time.time()

    if args.parallel <= 1:
        for i, (p, t) in enumerate(jobs):
            print(f"[{i+1}/{total}]", end="")
            r = run_single(p, t, dry=args.dry, force_perturb_dims=args.perturb_dims, signal_base=signal_base, log_base=log_base)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_single, p, t, args.dry, args.perturb_dims, signal_base, log_base): (p, t)
                for p, t in jobs
            }
            for i, future in enumerate(as_completed(futures)):
                r = future.result()
                results.append(r)

    elapsed = time.time() - t_start

    # Summary
    by_status = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r)

    print()
    print(f"=== Summary ({elapsed:.0f}s total) ===")
    for status, items in sorted(by_status.items()):
        print(f"  {status}: {len(items)}")

    success = by_status.get("success", [])
    if success:
        avg_events = sum(r["events"] for r in success) / len(success)
        avg_dur = sum(r["duration_s"] for r in success) / len(success)
        print(f"  avg events per trajectory: {avg_events:.0f}")
        print(f"  avg duration per trajectory: {avg_dur:.0f}s")

    # Save results log
    log_path = PROJECT_ROOT / "run_all_200_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "profiles": profiles,
            "tasks": tasks,
            "total": total,
            "elapsed_s": round(elapsed, 1),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nLog saved: {log_path}")


if __name__ == "__main__":
    main()
