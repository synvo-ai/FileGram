"""Main evaluation runner for FileGramBench pilot.

Usage:
    python -m bench.run_eval --signals-dir signal --profiles-dir profiles

Orchestrates:
1. Load trajectories for each profile
2. Run each baseline adapter (ingest + infer)
3. Run FileGramOS Simple (feature extraction + synthesis)
4. Score all inferred profiles via LLM-as-Judge
5. Generate and score MCQs
6. Output comparison tables
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure this script works when run directly (python bench/run_eval.py)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Import adapters to trigger registration
from baselines import get_adapter
from baselines.base import BaseAdapter
from evaluation.mcq_generator import MCQGenerator

# Profile attributes to infer (must match mini profile schema — 9 observable attributes)
PROFILE_ATTRIBUTES = [
    "reading_strategy",
    "output_detail",
    "output_structure",
    "directory_style",
    "naming",
    "edit_strategy",
    "version_strategy",
    "tone",
    "cross_modal_behavior",
]

# All methods to evaluate
METHOD_NAMES = [
    "full_context",
    "naive_rag",
    "eager_summarization",
    "mem0",
    "zep",
    "memos",
    "memu",
    "evermemos",
]


def load_ground_truth(profiles_dir: Path) -> dict[str, dict[str, Any]]:
    """Load ground truth mini profiles from YAML files."""
    import yaml

    profiles = {}
    for yaml_file in sorted(profiles_dir.glob("*.yaml")):
        profile_id = yaml_file.stem
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        profiles[profile_id] = data
    return profiles


def run_profile_evaluation(
    profile_id: str,
    trajectories: list[dict[str, Any]],
    method_name: str,
) -> dict[str, Any]:
    """Run a single method on a single profile's trajectories.

    Returns:
        Dict with method, profile_id, and the inference prompt/result.
    """
    adapter = get_adapter(method_name)
    adapter.ingest(trajectories)
    result = adapter.infer_profile(PROFILE_ATTRIBUTES)
    adapter.reset()

    return {
        "method": method_name,
        "profile_id": profile_id,
        "result": result,
    }


def generate_mcqs_for_profile(
    profile_id: str,
    trajectories: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Generate MCQs from a profile's trajectories."""
    generator = MCQGenerator(seed=42)
    questions = generator.generate_from_profile(trajectories, profile_id, n_questions=10)
    generator.save_mcqs(questions, output_dir, profile_id)
    return questions


def main():
    parser = argparse.ArgumentParser(description="Run FileGramBench pilot evaluation")
    parser.add_argument(
        "--signals-dir",
        type=Path,
        default=Path("signal"),
        help="Directory with trajectory data",
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=Path("profiles"),
        help="Directory with ground truth profile YAMLs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bench"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHOD_NAMES,
        help="Methods to evaluate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate prompts without calling LLM",
    )

    args = parser.parse_args()

    signals_dir = args.signals_dir
    profiles_dir = args.profiles_dir
    output_dir = args.output_dir
    scores_dir = output_dir / "scores"
    mcq_dir = output_dir / "mcq"

    scores_dir.mkdir(parents=True, exist_ok=True)
    mcq_dir.mkdir(parents=True, exist_ok=True)

    # Check if trajectory data exists
    if not signals_dir.exists():
        print(f"[!] Signals directory not found: {signals_dir}")
        print("    Run FileGram pipeline first to generate trajectories.")
        print("    Generating framework skeleton with empty results...")
        _generate_skeleton(output_dir, args.methods)
        return

    # Load ground truth profiles
    ground_truth = load_ground_truth(profiles_dir)
    if not ground_truth:
        print(f"[!] No profiles found in {profiles_dir}")
        return

    profile_ids = sorted(ground_truth.keys())
    print(f"Found {len(profile_ids)} profiles: {profile_ids}")

    # Phase B+C: Run all methods on all profiles
    all_results: dict[str, list[dict[str, Any]]] = {m: [] for m in args.methods}

    for profile_id in profile_ids:
        print(f"\n=== Profile: {profile_id} ===")

        # Load trajectories
        trajectories = BaseAdapter.load_trajectories(signals_dir, profile_id)
        if not trajectories:
            print(f"  [!] No trajectories found for {profile_id}, skipping")
            continue
        print(f"  Loaded {len(trajectories)} trajectories")

        # Generate MCQs
        questions = generate_mcqs_for_profile(profile_id, trajectories, mcq_dir)
        print(f"  Generated {len(questions)} MCQs")

        # Run each method
        for method in args.methods:
            print(f"  Running {method}...")
            result = run_profile_evaluation(profile_id, trajectories, method)
            all_results[method].append(result)

            if args.dry_run:
                # Save the prompt for inspection
                prompt_file = scores_dir / f"{method}_{profile_id}_prompt.json"
                prompt_file.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"    Saved prompt to {prompt_file}")

    # Save all results
    results_file = output_dir / "eval_results.json"
    results_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {results_file}")

    if not args.dry_run:
        print("\n[Next steps]")
        print("1. Feed generated prompts to LLM judge for scoring")
        print("2. Run MCQ evaluation with each memory system")
        print("3. Generate comparison table")


def _generate_skeleton(output_dir: Path, methods: list[str]):
    """Generate skeleton result files when no trajectories exist yet."""
    skeleton = {
        "status": "awaiting_trajectories",
        "methods": methods,
        "expected_structure": {
            "per_method": {
                "<method>": [
                    {
                        "profile_id": "<p1..p6>",
                        "result": {
                            "_prompt": "<inference prompt>",
                            "_method": "<method_name>",
                        },
                    }
                ]
            }
        },
        "evaluation_pipeline": [
            "1. Run FileGram to generate 60 trajectories (6 profiles x 10 tasks)",
            "2. Run this script again to generate inference prompts",
            "3. Feed prompts to judge LLM for per-attribute scoring",
            "4. Run MCQ evaluation",
            "5. Generate comparison table",
        ],
    }

    output_file = output_dir / "eval_skeleton.json"
    output_file.write_text(json.dumps(skeleton, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Skeleton saved to {output_file}")


if __name__ == "__main__":
    main()
