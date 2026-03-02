#!/usr/bin/env python3
"""Ablation: rebuild filegramos + filegramos_simple caches, then run QA eval.

Avoids importing all adapters (qdrant_client breaks on py3.11).

Usage:
    python bench/run_ablation.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
_ROOT = _BENCH.parent
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from baselines.base import BaseAdapter
from baselines.filegramos_adapter import FileGramOSAdapter
from baselines.filegramos_simple import FileGramOSSimpleAdapter

SIGNALS_DIR = _ROOT / "signal"
CACHE_DIR = _BENCH / "ingest_cache_gemini_2.5_flash"

ALL_PROFILES = [
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
    "p11_meticulous_planner",
    "p12_prolific_scanner",
    "p13_visual_architect",
    "p14_concise_organizer",
    "p15_thorough_surveyor",
    "p16_phased_minimalist",
    "p17_creative_archivist",
    "p18_decisive_scanner",
    "p19_agile_pragmatist",
    "p20_visual_auditor",
]


def get_gemini_llm_fn():
    """Create Gemini LLM callable for filegramos adapter."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    print(f"LLM: gemini / {model}")

    def _call(prompt: str, system_prompt: str | None = None) -> str:
        config = types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        if system_prompt:
            config.system_instruction = system_prompt
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return (resp.text or "").strip()

    return _call


def rebuild_caches():
    """Rebuild filegramos_simple.pkl and filegramos.pkl for all profiles."""
    llm_fn = get_gemini_llm_fn()

    for profile_id in ALL_PROFILES:
        cache_dir = CACHE_DIR / profile_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Load trajectories
        trajectories = BaseAdapter.load_trajectories(SIGNALS_DIR, profile_id)
        if not trajectories:
            print(f"[{profile_id}] No trajectories, skipping")
            continue
        print(f"\n{'=' * 60}")
        print(f"[{profile_id}] {len(trajectories)} trajectories")

        # --- filegramos_simple ---
        simple_cache = cache_dir / "filegramos_simple.pkl"
        if simple_cache.exists():
            print("  filegramos_simple: CACHED (skipping)")
        else:
            print("  filegramos_simple: building...")
            t0 = time.time()
            adapter = FileGramOSSimpleAdapter()
            adapter.ingest(trajectories)
            adapter.save_ingest_cache(simple_cache)
            adapter.reset()
            print(f"  filegramos_simple: done ({time.time() - t0:.1f}s)")

        # --- filegramos (formal) ---
        formal_cache = cache_dir / "filegramos.pkl"
        if formal_cache.exists():
            print("  filegramos: CACHED (skipping)")
        else:
            print("  filegramos: building...")
            t0 = time.time()
            adapter = FileGramOSAdapter(llm_fn=llm_fn)
            adapter.ingest(trajectories)
            adapter.save_ingest_cache(formal_cache)
            adapter.reset()
            print(f"  filegramos: done ({time.time() - t0:.1f}s)")

    print(f"\n{'=' * 60}")
    print("All caches rebuilt.")


def run_qa():
    """Run QA eval for filegramos methods only."""
    print(f"\n{'=' * 60}")
    print("Running QA evaluation...")
    print(f"{'=' * 60}")
    ret = os.system(
        f"cd {_ROOT} && python -m filegramQA.run_qa_eval "
        f"--cache-dir gemini_2.5_flash "
        f"--methods filegramos_simple filegramos "
        f"--api gemini --parallel 10 --mode qa"
    )
    return ret


if __name__ == "__main__":
    rebuild_caches()
    run_qa()
