#!/usr/bin/env python3
"""Quick learning curve: measure profile inference at N=1,3,5,10,15,20,30.

Uses pre-built ingest caches (no re-ingestion needed).
Only does inference + judge calls per (profile, method, N) triple.

Usage:
    python bench/run_curve_quick.py
    python bench/run_curve_quick.py --methods filegramos  # single method
    python bench/run_curve_quick.py --parallel 3
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
OUTPUT_FILE = RESULTS_DIR / "curve_results_v2.json"
CACHE_DIR = _SCRIPT_DIR / "ingest_cache_gemini_2.5_flash"

N_VALUES = [1, 3, 5, 10, 15, 20, 30]

DEFAULT_METHODS = ["filegramos", "full_context", "eager_summarization", "naive_rag"]

PROFILES = [
    "p1_methodical",
    "p2_thorough_reviser",
    "p3_efficient_executor",
    "p5_balanced_organizer",
    "p8_minimal_editor",
]

# Channel attribute mapping
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
CH_MIX = ["naming", "cross_modal_behavior"]

# Dimension attribute mapping
DIM_MAP = {
    "IS": ["reading_strategy", "thoroughness"],
    "CG": ["output_detail", "output_structure", "documentation"],
    "FO": ["directory_style", "naming", "version_strategy"],
    "ER": ["edit_strategy", "error_handling"],
    "WR": ["working_style"],
    "MM": ["cross_modal_behavior"],
}


def _slice_adapter_state(adapter, method_name, n, task_ids_first_n):
    """Slice the adapter's internal state to only include first N trajectories."""
    if method_name == "full_context":
        adapter._narratives = adapter._narratives[:n]
    elif method_name == "eager_summarization":
        adapter._summarize_prompts = adapter._summarize_prompts[:n]
        adapter._summaries = adapter._summaries[:n]
    elif method_name == "naive_rag":
        kept = set(task_ids_first_n)
        if hasattr(adapter, '_chunks') and adapter._chunks:
            # Find indices of chunks belonging to first N tasks
            keep_indices = [i for i, c in enumerate(adapter._chunks) if c.get("task_id") in kept]
            adapter._chunks = [adapter._chunks[i] for i in keep_indices]
            if adapter._embeddings is not None:
                import numpy as np
                adapter._embeddings = adapter._embeddings[keep_indices]


def _load_cached_adapter(method_name, profile_id):
    """Load adapter with cached ingest state. Returns (adapter, success)."""
    adapter = get_adapter(method_name)
    cache_path = CACHE_DIR / profile_id / f"{method_name}.pkl"
    if adapter.load_ingest_cache(cache_path):
        return adapter, True
    return adapter, False


def run_single_cached(method_name, profile_id, n, task_ids_first_n, ground_truth, profile_dir):
    """Run one method using cached ingest, sliced to N trajectories."""
    adapter, cache_hit = _load_cached_adapter(method_name, profile_id)
    if not cache_hit:
        return {"avg_score": 0, "error": "cache miss"}

    # Slice to first N trajectories
    _slice_adapter_state(adapter, method_name, n, task_ids_first_n)

    result = adapter.infer_profile(PROFILE_ATTRIBUTES)
    prompt = result.get("_prompt")
    if not prompt:
        adapter.reset()
        return {"avg_score": 0, "error": "no prompt"}

    # LLM inference
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

    # Judge scoring
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

    # Channel scores
    def chan_avg(attrs):
        vals = [
            scores.get(a, {}).get("score", 0) for a in attrs if isinstance(scores.get(a, {}).get("score"), (int, float))
        ]
        return round(sum(vals) / len(vals), 3) if vals else 0

    # Dimension scores
    dim_scores = {d: chan_avg(attrs) for d, attrs in DIM_MAP.items()}

    return {
        "avg_score": round(avg, 3),
        "prompt_length": len(prompt),
        "proc_score": chan_avg(CH_PROC),
        "sem_score": chan_avg(CH_SEM),
        "mix_score": chan_avg(CH_MIX),
        "dim_scores": dim_scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--profiles", nargs="+", default=PROFILES)
    args = parser.parse_args()

    methods = args.methods
    profiles = args.profiles

    print("=" * 70)
    print("FileGramBench Learning Curve (Quick — Cache Mode)")
    print(f"Methods: {methods}")
    print(f"Profiles: {len(profiles)} | N values: {N_VALUES}")
    print(f"Total runs: {len(methods) * len(profiles) * len(N_VALUES)}")
    print(f"Cache dir: {CACHE_DIR}")
    print("=" * 70)

    # Pre-load trajectory task_ids for each profile (needed for slicing)
    profile_task_ids = {}
    for profile_id in profiles:
        trajs = load_trajectories_filtered(profile_id)
        profile_task_ids[profile_id] = [t["task_id"] for t in trajs]
        print(f"  [{profile_id}] {len(trajs)} trajectories available")

    # Load existing results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    if OUTPUT_FILE.exists():
        try:
            results = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
            total_cached = sum(
                1
                for p in results.values()
                for m in p.values()
                for n_data in m.values()
                if isinstance(n_data, dict) and n_data.get("avg_score", 0) > 0
            )
            print(f"Loaded existing results ({total_cached} cached entries)")
        except (json.JSONDecodeError, OSError):
            pass

    def process_one(profile_id, method, n):
        """Process a single (profile, method, N) triple."""
        task_ids = profile_task_ids.get(profile_id, [])
        if n > len(task_ids):
            return profile_id, method, n, None

        task_ids_first_n = task_ids[:n]
        ground_truth = load_ground_truth(profile_id)
        profile_dir = RESULTS_DIR / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        result = run_single_cached(method, profile_id, n, task_ids_first_n, ground_truth, profile_dir)
        elapsed = time.time() - t0

        print(
            f"  [{profile_id}] {method} N={n:>2d} → "
            f"score={result['avg_score']:.2f} "
            f"proc={result['proc_score']:.2f} sem={result['sem_score']:.2f} "
            f"({elapsed:.0f}s)"
        )
        return profile_id, method, n, result

    # Build work queue (skip cached)
    work = []
    for profile_id in profiles:
        for method in methods:
            for n in N_VALUES:
                cached = results.get(profile_id, {}).get(method, {}).get(str(n), {})
                if cached.get("avg_score", 0) > 0:
                    print(f"  [{profile_id}] {method} N={n} SKIP (cached {cached['avg_score']:.2f})")
                    continue
                work.append((profile_id, method, n))

    print(f"\n{len(work)} runs to execute\n")

    if args.parallel > 1 and len(work) > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(process_one, p, m, n): (p, m, n) for p, m, n in work}
            for future in as_completed(futures):
                p, m, n = futures[future]
                try:
                    _, _, _, result = future.result()
                    if result:
                        results.setdefault(p, {}).setdefault(m, {})[str(n)] = result
                        OUTPUT_FILE.write_text(
                            json.dumps(results, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                except Exception as e:
                    print(f"  [{p}] {m} N={n} ERROR: {e}")
                    results.setdefault(p, {}).setdefault(m, {})[str(n)] = {"avg_score": 0, "error": str(e)}
    else:
        for p, m, n in work:
            try:
                _, _, _, result = process_one(p, m, n)
                if result:
                    results.setdefault(p, {}).setdefault(m, {})[str(n)] = result
                    OUTPUT_FILE.write_text(
                        json.dumps(results, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
            except Exception as e:
                print(f"  [{p}] {m} N={n} ERROR: {e}")
                results.setdefault(p, {}).setdefault(m, {})[str(n)] = {"avg_score": 0, "error": str(e)}

    # Summary
    print(f"\n{'=' * 70}")
    print("CONVERGENCE SUMMARY (cross-profile average)")
    print(f"{'=' * 70}\n")

    for method in methods:
        print(f"{method}:")
        line_score = []
        line_proc = []
        line_sem = []
        line_dims = {d: [] for d in DIM_MAP}
        for n in N_VALUES:
            scores_at_n = []
            proc_at_n = []
            sem_at_n = []
            dims_at_n = {d: [] for d in DIM_MAP}
            for profile_id in profiles:
                r = results.get(profile_id, {}).get(method, {}).get(str(n), {})
                if r.get("avg_score"):
                    scores_at_n.append(r["avg_score"])
                    proc_at_n.append(r.get("proc_score", 0))
                    sem_at_n.append(r.get("sem_score", 0))
                    for d in DIM_MAP:
                        dims_at_n[d].append(r.get("dim_scores", {}).get(d, 0))

            avg_s = sum(scores_at_n) / len(scores_at_n) if scores_at_n else 0
            avg_p = sum(proc_at_n) / len(proc_at_n) if proc_at_n else 0
            avg_sem = sum(sem_at_n) / len(sem_at_n) if sem_at_n else 0
            line_score.append(f"N={n:>2d}: {avg_s:.3f}")
            line_proc.append(f"{avg_p:.2f}")
            line_sem.append(f"{avg_sem:.2f}")
            for d in DIM_MAP:
                vals = dims_at_n[d]
                avg_d = sum(vals) / len(vals) if vals else 0
                line_dims[d].append(f"{avg_d:.2f}")

        print(f"  Overall: {' | '.join(line_score)}")
        print(f"  Proc:    {' | '.join(line_proc)}")
        print(f"  Sem:     {' | '.join(line_sem)}")
        for d in DIM_MAP:
            print(f"  {d:>2s}:      {' | '.join(line_dims[d])}")
        print()

    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
