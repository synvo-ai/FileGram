#!/usr/bin/env python3
"""Learning curve evaluation: measure profile accuracy at N=1,2,3 trajectories.

Shows how each method scales with increasing data.
Tracks both score and prompt length (proxy for context burden).

Usage:
    python bench/test_curve.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from baselines import get_adapter
from evaluation.judge_scoring import JudgeScorer
from test_baselines import (
    PROFILE_ATTRIBUTES,
    PROFILES_DIR,
    call_llm,
    load_ground_truth,
    load_trajectories_filtered,
    parse_json_response,
)

RESULTS_DIR = _SCRIPT_DIR / "test_results"

METHODS = [
    "full_context",
    "naive_rag",
    "eager_summarization",
    "mem0",
    "zep",
    "memos",
    "memu",
    "evermemos",
    "filegramos_simple",
]

PROFILES = [
    "p1_methodical",
    "p2_thorough_reviser",
    "p3_efficient_executor",
    "p4_structured_analyst",
    "p5_balanced_organizer",
    "p6_quick_curator",
    "p7_visual_reader",
    "p8_minimal_editor",
    "p9_visual_organizer",
    "p10_silent_auditor",
]


def run_single(
    method_name: str,
    trajectories: list[dict[str, Any]],
    ground_truth: dict[str, str],
    profile_dir: Path,
) -> dict[str, Any]:
    """Run one method on given trajectories, return score + prompt_length."""
    adapter = get_adapter(method_name)
    adapter.ingest(trajectories)

    # Eager summarization pre-pass
    if method_name == "eager_summarization":
        prompts = adapter.get_summarize_prompts()
        summaries = []
        for sp in prompts:
            summaries.append(call_llm(sp["prompt"], temperature=0.3, max_tokens=1024))
        adapter.set_summaries(summaries)

    result = adapter.infer_profile(PROFILE_ATTRIBUTES)
    prompt = result.get("_prompt")
    if not prompt:
        adapter.reset()
        return {"avg_score": 0, "prompt_length": 0, "error": "no prompt"}

    # Call LLM for inference
    response = call_llm(prompt, temperature=0.3, max_tokens=2048)
    try:
        parsed = parse_json_response(response)
        inferred = parsed.get("inferred_profile", parsed)
        inferred_flat = {}
        for attr, detail in inferred.items():
            if isinstance(detail, dict):
                inferred_flat[attr] = detail.get("value", str(detail))
            else:
                inferred_flat[attr] = str(detail)
    except (json.JSONDecodeError, KeyError):
        inferred_flat = {}

    # Judge (per-attribute, with detailed rubrics)
    judge = JudgeScorer(PROFILES_DIR, profile_dir)
    parsed_scores = judge.judge_all_attributes(
        ground_truth=ground_truth,
        inferred_profile=inferred_flat,
        method_name=method_name,
        attributes=PROFILE_ATTRIBUTES,
        llm_fn=call_llm,
    )
    scores = parsed_scores.get("scores", {})
    avg = parsed_scores.get("overall_mean", 0)

    adapter.reset()

    # Channel breakdown
    CH_PROC = [
        "working_style",
        "thoroughness",
        "error_handling",
        "reading_strategy",
        "directory_style",
        "edit_strategy",
        "version_strategy",
        "output_detail",
    ]
    CH_SEM = ["name", "role", "language", "tone", "output_structure", "documentation"]

    proc_vals = [
        scores.get(a, {}).get("score", 0) for a in CH_PROC if isinstance(scores.get(a, {}).get("score"), (int, float))
    ]
    sem_vals = [
        scores.get(a, {}).get("score", 0) for a in CH_SEM if isinstance(scores.get(a, {}).get("score"), (int, float))
    ]

    return {
        "avg_score": round(avg, 3),
        "prompt_length": len(prompt),
        "n_events": sum(len(t["events"]) for t in trajectories),
        "proc_score": round(sum(proc_vals) / len(proc_vals), 3) if proc_vals else 0,
        "sem_score": round(sum(sem_vals) / len(sem_vals), 3) if sem_vals else 0,
        "judge_scores": parsed_scores,
    }


def main():
    print("=" * 70)
    print("FileGramBench Learning Curve Evaluation")
    # Target N values for data efficiency curve
    N_VALUES = [1, 3, 5, 10, 20]

    print(f"Methods: {len(METHODS)} | Profiles: {len(PROFILES)} | N: {N_VALUES}")
    print("=" * 70)

    # Load existing results to support incremental runs
    output_file = RESULTS_DIR / "curve_results.json"
    curve_results = {}
    if output_file.exists():
        try:
            curve_results = json.loads(output_file.read_text(encoding="utf-8"))
            print(f"Loaded existing curve_results.json ({len(curve_results)} profiles)")
        except (json.JSONDecodeError, OSError):
            pass

    for profile_id in PROFILES:
        print(f"\n{'#' * 60}")
        print(f"# {profile_id}")
        print(f"{'#' * 60}")

        all_trajectories = load_trajectories_filtered(profile_id)
        if not all_trajectories:
            print("  No trajectories, skipping")
            continue

        ground_truth = load_ground_truth(profile_id)
        profile_dir = RESULTS_DIR / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        if profile_id not in curve_results:
            curve_results[profile_id] = {}

        for method in METHODS:
            if method not in curve_results[profile_id]:
                curve_results[profile_id][method] = {}

            for n in N_VALUES:
                if n > len(all_trajectories):
                    continue
                # Skip if already computed
                if (
                    str(n) in curve_results[profile_id][method]
                    and curve_results[profile_id][method][str(n)].get("avg_score", 0) > 0
                ):
                    cached = curve_results[profile_id][method][str(n)]
                    print(f"  {method} N={n} SKIP (cached score={cached['avg_score']:.2f})")
                    continue
                subset = all_trajectories[:n]
                task_ids = [t["task_id"] for t in subset]
                print(f"  {method} N={n} ({task_ids})...", end=" ", flush=True)

                try:
                    result = run_single(method, subset, ground_truth, profile_dir)
                    print(
                        f"score={result['avg_score']:.2f} "
                        f"prompt={result['prompt_length']}chars "
                        f"proc={result['proc_score']:.2f} sem={result['sem_score']:.2f}"
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                    result = {"avg_score": 0, "prompt_length": 0, "error": str(e)}

                curve_results[profile_id][method][str(n)] = result

            # Incremental save after each method completes
            output_file.write_text(json.dumps(curve_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Final save
    output_file.write_text(json.dumps(curve_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("LEARNING CURVE SUMMARY")
    print(f"{'=' * 70}\n")

    # Use the target N values for summary
    n_values = [str(n) for n in N_VALUES]

    # Average across profiles
    for method in METHODS:
        scores_by_n = {}
        prompts_by_n = {}
        proc_by_n = {}
        sem_by_n = {}
        for n in n_values:
            s_list, p_list, pr_list, se_list = [], [], [], []
            for profile_id in PROFILES:
                r = curve_results.get(profile_id, {}).get(method, {}).get(n, {})
                if r.get("avg_score"):
                    s_list.append(r["avg_score"])
                    p_list.append(r["prompt_length"])
                    pr_list.append(r.get("proc_score", 0))
                    se_list.append(r.get("sem_score", 0))
            scores_by_n[n] = sum(s_list) / len(s_list) if s_list else 0
            prompts_by_n[n] = sum(p_list) / len(p_list) if p_list else 0
            proc_by_n[n] = sum(pr_list) / len(pr_list) if pr_list else 0
            sem_by_n[n] = sum(se_list) / len(se_list) if se_list else 0

        score_parts = "  ".join(f"N={n}: {scores_by_n[n]:.2f}" for n in n_values)
        prompt_parts = "  ".join(f"N={n}: {prompts_by_n[n]:.0f}" for n in n_values)
        proc_parts = "  ".join(f"N={n}: {proc_by_n[n]:.2f}" for n in n_values)
        sem_parts = "  ".join(f"N={n}: {sem_by_n[n]:.2f}" for n in n_values)
        print(f"{method}:")
        print(f"  Score:  {score_parts}")
        print(f"  Prompt: {prompt_parts}")
        print(f"  Proc:   {proc_parts}")
        print(f"  Sem:    {sem_parts}")
        print()

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
