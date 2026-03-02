#!/usr/bin/env python3
"""Quick single-trajectory test: run all 9 methods on one trajectory, print scores.

Usage:
    python bench/test_single.py signal/p1_methodical_T-01_multimodal
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCH_DIR.parent

if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from baselines import get_adapter
from baselines.base import BaseAdapter
from baselines.eager_summarization import EagerSummarizationAdapter  # noqa: F401
from baselines.evermemos_adapter import EverMemOSAdapter  # noqa: F401
from baselines.filegramos_simple import FileGramOSSimpleAdapter  # noqa: F401
from baselines.full_context import FullContextAdapter  # noqa: F401
from baselines.mem0_adapter import Mem0Adapter  # noqa: F401
from baselines.memos_adapter import MemOSAdapter  # noqa: F401
from baselines.memu_adapter import MemUAdapter  # noqa: F401
from baselines.naive_rag import NaiveRAGAdapter  # noqa: F401
from baselines.zep_adapter import ZepAdapter  # noqa: F401
from evaluation.judge_scoring import JudgeScorer

PROFILES_DIR = _PROJECT_ROOT / "profiles"

METHODS = [
    "filegramos_simple",
    "full_context",
    "eager_summarization",
    "naive_rag",
    "mem0",
    "zep",
    "memos",
    "memu",
    "evermemos",
]

PROFILE_ATTRIBUTES = [
    "name",
    "role",
    "language",
    "tone",
    "output_detail",
    "working_style",
    "thoroughness",
    "documentation",
    "error_handling",
    "reading_strategy",
    "output_structure",
    "directory_style",
    "naming",
    "edit_strategy",
    "version_strategy",
    "cross_modal_behavior",
]


# ---------------------------------------------------------------------------
# Azure OpenAI caller (copied from test_baselines.py)
# ---------------------------------------------------------------------------
def call_llm(prompt, system_prompt=None, temperature=0.3, max_tokens=4096):
    import httpx

    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    endpoint = (
        os.environ.get("AZURE_OPENAI_ENDPOINT", "https://haku-chat.openai.azure.com").rstrip("/").split("/openai/")[0]
    )
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = httpx.post(
                url,
                json={"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                headers={"api-key": api_key, "Content-Type": "application/json"},
                timeout=180.0,
            )
        except (httpx.ReadTimeout, httpx.ConnectTimeout):
            wait = min(10 * attempt, 60)
            print(f"    Timeout, retry in {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code == 429:
            wait = 15
            try:
                import re

                m = re.search(
                    r"retry after (\d+) seconds", resp.json().get("error", {}).get("message", ""), re.IGNORECASE
                )
                if m:
                    wait = int(m.group(1)) + 2
            except Exception:
                pass
            print(f"    Rate limited, wait {wait}s (attempt {attempt})")
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            time.sleep(min(10 * attempt, 60))
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"Azure {resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]


def load_ground_truth(profile_id):
    import yaml

    with open(PROFILES_DIR / f"{profile_id}.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    basic, pers, wh = data.get("basic", {}), data.get("personality", {}), data.get("work_habits", {})
    return {
        "name": basic.get("name", ""),
        "role": basic.get("role", ""),
        "language": basic.get("language", ""),
        "tone": pers.get("tone", ""),
        "output_detail": pers.get("output_detail", pers.get("verbosity", "")),
        "working_style": wh.get("working_style", ""),
        "thoroughness": wh.get("thoroughness", ""),
        "documentation": wh.get("documentation", ""),
        "error_handling": wh.get("error_handling", ""),
        "reading_strategy": wh.get("reading_strategy", ""),
        "output_structure": wh.get("output_structure", ""),
        "directory_style": wh.get("directory_style", ""),
        "naming": wh.get("naming", ""),
        "edit_strategy": wh.get("edit_strategy", ""),
        "version_strategy": wh.get("version_strategy", ""),
        "cross_modal_behavior": wh.get("cross_modal_behavior", ""),
    }


def parse_json_response(text):
    text = text.strip()
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def main():
    if len(sys.argv) < 2:
        print("Usage: python bench/test_single.py <trajectory_dir>")
        print("Example: python bench/test_single.py signal/p1_methodical_T-01_multimodal")
        sys.exit(1)

    traj_dir = Path(sys.argv[1])
    if not traj_dir.is_absolute():
        traj_dir = _PROJECT_ROOT / traj_dir
    if not traj_dir.exists():
        print(f"ERROR: {traj_dir} not found")
        sys.exit(1)

    # Parse profile and task from dir name
    dir_name = traj_dir.name
    idx = dir_name.rfind("_T-")
    if idx == -1:
        print(f"ERROR: Can't parse profile/task from {dir_name}")
        sys.exit(1)
    profile_id = dir_name[:idx]
    task_id = dir_name[idx + 1 :]

    print(f"{'=' * 60}")
    print("Single Trajectory Test")
    print(f"  Profile: {profile_id}")
    print(f"  Task:    {task_id}")
    print(f"  Dir:     {traj_dir}")
    print(f"{'=' * 60}\n")

    # Load trajectory
    events_file = traj_dir / "events.json"
    summary_file = traj_dir / "summary.json"
    events = json.loads(events_file.read_text(encoding="utf-8")) if events_file.exists() else []
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
    BaseAdapter._resolve_media_refs(traj_dir, events)

    trajectories = [
        {
            "task_id": task_id,
            "events": events,
            "summary": summary,
            "path": str(traj_dir),
        }
    ]
    print(f"Loaded {len(events)} events\n")

    # Load ground truth
    gt = load_ground_truth(profile_id)

    # Temp dir for judge (needs profile dir)
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp())

    # Run each method
    results = {}
    for mi, method_name in enumerate(METHODS, 1):
        print(f"[{mi}/{len(METHODS)}] {method_name}")
        t0 = time.time()

        try:
            adapter = get_adapter(method_name, llm_fn=call_llm)
            adapter.ingest(trajectories)

            # Eager summarization needs extra LLM call
            if method_name == "eager_summarization":
                for sp in adapter.get_summarize_prompts():
                    s = call_llm(sp["prompt"], temperature=0.3, max_tokens=1024)
                    adapter.set_summaries([s])

            result = adapter.infer_profile(PROFILE_ATTRIBUTES)
            prompt = result.get("_prompt", "")
            response = call_llm(prompt, temperature=0.3, max_tokens=2048)

            try:
                parsed = parse_json_response(response)
                inferred = parsed.get("inferred_profile", parsed)
                inferred_flat = {}
                for attr, detail in inferred.items():
                    inferred_flat[attr] = detail.get("value", str(detail)) if isinstance(detail, dict) else str(detail)
            except (json.JSONDecodeError, KeyError):
                inferred_flat = {}

            # Judge
            judge = JudgeScorer(PROFILES_DIR, tmp_dir)
            scores = judge.judge_all_attributes(
                ground_truth=gt,
                inferred_profile=inferred_flat,
                method_name=method_name,
                attributes=PROFILE_ATTRIBUTES,
                llm_fn=call_llm,
                verbose=False,
                max_workers=4,
            )
            avg = scores.get("overall_mean", 0)
            dt = time.time() - t0

            results[method_name] = {
                "avg_score": round(avg, 3),
                "prompt_len": len(prompt),
                "time_s": round(dt, 1),
                "per_attr": {a: scores.get("scores", {}).get(a, {}).get("score", 0) for a in PROFILE_ATTRIBUTES},
            }
            print(f"  score={avg:.3f}  prompt={len(prompt)}chars  time={dt:.1f}s")
            adapter.reset()

        except Exception as e:
            dt = time.time() - t0
            print(f"  ERROR: {e.__class__.__name__}: {e}")
            results[method_name] = {"avg_score": 0, "error": str(e), "time_s": round(dt, 1)}

    # Cleanup
    import shutil

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Print summary table
    print(f"\n{'=' * 60}")
    print(f"RESULTS — {profile_id} / {task_id}")
    print(f"{'=' * 60}")
    print(f"{'Method':<25s} {'Score':>6s} {'Prompt':>8s} {'Time':>6s}")
    print("-" * 50)
    ranked = sorted(results.items(), key=lambda x: -x[1].get("avg_score", 0))
    for method, r in ranked:
        s = r.get("avg_score", 0)
        p = r.get("prompt_len", 0)
        t = r.get("time_s", 0)
        print(f"{method:<25s} {s:>6.3f} {p:>7d}c {t:>5.1f}s")

    # Per-attribute breakdown
    print(f"\n{'Attribute':<25s}", end="")
    for method, _ in ranked:
        print(f" {method[:8]:>8s}", end="")
    print()
    print("-" * (25 + 9 * len(ranked)))
    for attr in PROFILE_ATTRIBUTES:
        print(f"{attr:<25s}", end="")
        for method, r in ranked:
            s = r.get("per_attr", {}).get(attr, 0)
            print(f" {s:>8.1f}" if s else " {:>8s}".format("-"), end="")
        print()


if __name__ == "__main__":
    main()
