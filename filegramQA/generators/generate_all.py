#!/usr/bin/env python3
"""FileGramQA — Generate all benchmark questions from trajectory data.

Usage:
    python -m filegramQA.generate_all                    # generate all
    python -m filegramQA.generate_all --settings 1 2     # specific settings
    python -m filegramQA.generate_all --dry               # preview counts only
    python -m filegramQA.generate_all --out custom_dir    # custom output dir

Output:
    filegramQA/questions/
    ├── setting1_basic.json
    ├── setting2_cross_task.json
    ├── setting3_profile_shift.json
    └── all_questions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .base import ProfileMatrix, TrajectoryStore, SIGNAL_DIR, MATRIX_PATH, enrich_taxonomy
from .setting1_basic import Setting1Generator
from .setting2_cross_task import Setting2Generator
from .setting3_profile_shift import Setting3Generator
from .setting4_multimodal import Setting4Generator


def main():
    parser = argparse.ArgumentParser(description="Generate FileGramQA benchmark questions")
    parser.add_argument(
        "--settings", nargs="*", type=int, default=[1, 2, 3, 4],
        help="Which settings to generate (1, 2, 3, 4). Default: all.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--signal-dir", type=str, default=None,
        help="Path to signal directory. Default: project_root/signal/",
    )
    parser.add_argument(
        "--matrix", type=str, default=None,
        help="Path to profile_matrix.json. Default: project_root/profiles/profile_matrix.json",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output directory. Default: filegramQA/questions/",
    )
    parser.add_argument(
        "--dry", action="store_true",
        help="Dry run — show counts without writing files.",
    )
    args = parser.parse_args()

    # Resolve paths
    signal_dir = Path(args.signal_dir) if args.signal_dir else SIGNAL_DIR
    matrix_path = Path(args.matrix) if args.matrix else MATRIX_PATH
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "questions"

    # Load data
    print(f"Loading trajectory data from {signal_dir} ...")
    store = TrajectoryStore(signal_dir)
    print(f"  {len(store.all_profiles)} profiles, {len(store.all_tasks)} tasks, "
          f"{sum(len(v) for v in store.by_profile.values())} trajectories")

    print(f"Loading profile matrix from {matrix_path} ...")
    matrix = ProfileMatrix(matrix_path)
    print(f"  {len(matrix.profiles)} profiles in matrix")

    # Generate questions per setting
    all_questions: list[dict] = []
    setting_results: dict[int, list[dict]] = {}

    generators = {
        1: ("Setting 1: Basic", Setting1Generator),
        2: ("Setting 2: Cross-Task", Setting2Generator),
        3: ("Setting 3: Profile Shift", Setting3Generator),
    }

    # Generate T1-T3 first (Setting 4 depends on their output)
    for setting_num in sorted(s for s in args.settings if s <= 3):
        if setting_num not in generators:
            print(f"  [SKIP] Unknown setting {setting_num}")
            continue

        name, gen_cls = generators[setting_num]
        print(f"\nGenerating {name} ...")
        gen = gen_cls(store, matrix, seed=args.seed)
        raw_questions = gen.generate_all()
        questions = [enrich_taxonomy(q) for q in raw_questions]
        setting_results[setting_num] = questions
        all_questions.extend(questions)

        # Print breakdown by sub_track → sub_class
        track_counts = Counter((q["sub_track"], q["sub_class"]) for q in questions)
        for (sub_track, sub_class), count in sorted(track_counts.items()):
            print(f"  {sub_track}/{sub_class}: {count}")
        print(f"  Total: {len(questions)}")

    # Setting 4: Multimodal (depends on T1-T3 questions)
    if 4 in args.settings:
        # If T1-T3 weren't generated in this run, load from existing files
        source_qs = list(all_questions)
        if not source_qs:
            for s in (1, 2, 3):
                for fname in (f"setting{s}_basic.json", f"setting{s}_cross_task.json",
                              f"setting{s}_profile_shift.json",
                              f"setting_basic.json", f"setting_cross_task.json",
                              f"setting_profile_shift.json"):
                    fpath = out_dir / fname
                    if fpath.exists():
                        source_qs.extend(json.loads(fpath.read_text(encoding="utf-8")))
                        break
            # Also try all_questions.json as fallback
            if not source_qs:
                combined = out_dir / "all_questions.json"
                if combined.exists():
                    source_qs = json.loads(combined.read_text(encoding="utf-8"))
            if source_qs:
                print(f"  Loaded {len(source_qs)} existing T1-T3 questions as source")
            else:
                print("  [WARN] No T1-T3 questions available for Setting 4 file grounding")

        print(f"\nGenerating Setting 4: Multimodal ...")
        gen4 = Setting4Generator(store, matrix, source_qs, seed=args.seed)
        raw_4 = gen4.generate_all()
        questions_4 = [enrich_taxonomy(q) for q in raw_4]
        setting_results[4] = questions_4
        # T4 questions are NOT added to all_questions (they reuse T1-T3 under
        # different eval conditions), but we write them to a separate file.

        track_counts = Counter((q["sub_track"], q["sub_class"]) for q in questions_4)
        for (sub_track, sub_class), count in sorted(track_counts.items()):
            print(f"  {sub_track}/{sub_class}: {count}")
        print(f"  Total: {len(questions_4)}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Total questions: {len(all_questions)}")
    track_counts = Counter(q["track"] for q in all_questions)
    for track, count in sorted(track_counts.items()):
        print(f"  {track}: {count}")

    # Choice distribution check
    correct_dist = Counter(q["correct"] for q in all_questions)
    print(f"\nCorrect answer distribution: {dict(sorted(correct_dist.items()))}")

    # Profile coverage
    profiles_used = set()
    for q in all_questions:
        if "profile_id" in q:
            profiles_used.add(q["profile_id"])
        if "metadata" in q and "profiles" in q["metadata"]:
            profiles_used.update(q["metadata"]["profiles"])
    print(f"Profiles involved: {len(profiles_used)}/{len(store.all_profiles)}")

    if args.dry:
        print("\n[DRY RUN] No files written.")
        return

    # Write output
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = {
        1: "setting1_basic.json",
        2: "setting2_cross_task.json",
        3: "setting3_profile_shift.json",
        4: "setting4_multimodal.json",
    }

    for setting_num, questions in setting_results.items():
        out_path = out_dir / filenames[setting_num]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {len(questions)} questions to {out_path}")

    # Combined file (T1-T3 only; T4 is a separate eval condition)
    if all_questions:
        combined_path = out_dir / "all_questions.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_questions, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(all_questions)} total questions to {combined_path}")
    else:
        print("Skipped all_questions.json (no T1-T3 questions generated in this run)")

    # Write summary stats
    stats = {
        "total_questions": len(all_questions),
        "by_track": dict(sorted(track_counts.items())),
        "by_sub_track": dict(sorted(Counter(q["sub_track"] for q in all_questions).items())),
        "by_sub_class": dict(sorted(Counter(q["sub_class"] for q in all_questions).items())),
        "by_channel": dict(sorted(Counter(q["channel"] for q in all_questions).items())),
        "correct_distribution": dict(sorted(correct_dist.items())),
        "profiles_used": len(profiles_used),
        "profiles_total": len(store.all_profiles),
        "trajectories_loaded": sum(len(v) for v in store.by_profile.values()),
        "seed": args.seed,
    }
    stats_path = out_dir / "generation_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Wrote generation stats to {stats_path}")


if __name__ == "__main__":
    main()
