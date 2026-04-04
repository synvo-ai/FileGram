#!/usr/bin/env python3
"""Cross-backbone trace validation: same raw traces, different QA backbones.

Tests whether behavioral signal in traces is model-agnostic by running
profile reconstruction with different LLM backends on the SAME raw event
traces (full_context adapter style — no FileGramOS encoding).

Usage:
    python bench/run_cross_backbone.py
    python bench/run_cross_backbone.py --parallel 2
    python bench/run_cross_backbone.py --profiles p1_methodical p3_efficient_executor
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    load_ground_truth,
)

RESULTS_DIR = _SCRIPT_DIR / "test_results"
OUTPUT_FILE = RESULTS_DIR / "cross_backbone_results.json"
CACHE_DIR = _SCRIPT_DIR / "ingest_cache_gemini_2.5_flash"

ALL_PROFILES = [
    "p1_methodical", "p2_thorough_reviser", "p3_efficient_executor",
    "p4_structured_analyst", "p5_balanced_organizer", "p6_quick_curator",
    "p7_visual_reader", "p8_minimal_editor", "p9_visual_organizer",
    "p10_silent_auditor", "p11_meticulous_planner", "p12_prolific_scanner",
    "p13_visual_architect", "p14_concise_organizer", "p15_thorough_surveyor",
    "p16_phased_minimalist", "p17_creative_archivist", "p18_decisive_scanner",
    "p19_agile_pragmatist", "p20_visual_auditor",
]

# Channel attribute mapping
CH_PROC = [
    "working_style", "thoroughness", "error_handling", "reading_strategy",
    "directory_style", "edit_strategy", "version_strategy", "output_detail",
]
CH_SEM = ["name", "role", "language", "tone", "output_structure", "documentation"]


# ---------------------------------------------------------------------------
# Robust JSON parser
# ---------------------------------------------------------------------------
def robust_parse_json(text: str) -> dict:
    """Parse JSON from LLM response with aggressive error recovery."""
    text = text.strip()

    # Strip markdown code fences
    if "```" in text:
        lines = text.split("\n")
        filtered = []
        in_fence = False
        for line in lines:
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            filtered.append(line)
        text = "\n".join(filtered)

    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas: ,} or ,]
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fix missing commas between }\n" (common Claude issue)
    cleaned2 = re.sub(r'}\s*\n\s*"', '},\n"', cleaned)
    try:
        return json.loads(cleaned2)
    except json.JSONDecodeError:
        pass

    # Last resort: try to extract attribute values with regex
    result = {}
    for attr in PROFILE_ATTRIBUTES:
        pattern = rf'"{re.escape(attr)}"\s*:\s*\{{[^}}]*?"value"\s*:\s*"([^"]*)"'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            result[attr] = m.group(1)
    if result:
        return {"inferred_profile": {k: {"value": v} for k, v in result.items()}}

    raise json.JSONDecodeError("Could not parse JSON from response", text, 0)


# ---- LLM backends ----

def _call_gemini(prompt, system_prompt=None, temperature=0.3, max_tokens=4096):
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)

    contents = []
    if system_prompt:
        contents.append(f"[System]\n{system_prompt}\n\n[User]\n{prompt}")
    else:
        contents.append(prompt)

    for attempt in range(1, 20):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens + 4096,
                    thinking_config=types.ThinkingConfig(thinking_budget=2048),
                ),
            )
            return (resp.text or "").strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = min(15 * attempt, 60)
                print(f"    [gemini] Rate limited (attempt {attempt}), wait {wait}s")
                time.sleep(wait)
                continue
            if "500" in err_str or "503" in err_str:
                time.sleep(min(10 * attempt, 60))
                continue
            raise


def _call_azure(prompt, system_prompt=None, temperature=0.3, max_tokens=4096):
    import httpx

    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://haku-chat.openai.azure.com").rstrip("/").split("/openai/")[0]
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(1, 20):
        try:
            resp = httpx.post(
                url, json={"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                headers={"api-key": api_key, "Content-Type": "application/json"},
                timeout=300.0,
            )
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout):
            time.sleep(min(10 * attempt, 60))
            continue
        if resp.status_code == 429:
            wait = 15
            print(f"    [azure] Rate limited (attempt {attempt}), wait {wait}s")
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            time.sleep(min(10 * attempt, 60))
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"Azure returned {resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]


def _call_anthropic(prompt, system_prompt=None, temperature=0.3, max_tokens=4096):
    import httpx

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system_prompt:
        payload["system"] = system_prompt

    for attempt in range(1, 20):
        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                timeout=300.0,
            )
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout):
            time.sleep(min(10 * attempt, 60))
            continue
        if resp.status_code == 429:
            wait = 15
            print(f"    [anthropic] Rate limited (attempt {attempt}), wait {wait}s")
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            time.sleep(min(10 * attempt, 60))
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"Anthropic returned {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        return "".join(b.get("text", "") for b in data.get("content", []))


def _call_openai(prompt, system_prompt=None, temperature=0.3, max_tokens=4096):
    import httpx

    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(1, 20):
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=300.0,
            )
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout):
            time.sleep(min(10 * attempt, 60))
            continue
        if resp.status_code == 429:
            wait = 15
            print(f"    [openai] Rate limited (attempt {attempt}), wait {wait}s")
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            time.sleep(min(10 * attempt, 60))
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI returned {resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]


# Map backbone name → inference_fn
# Judge always uses gemini for consistency
BACKBONES = {
    "gemini-2.5-flash": _call_gemini,
    "gpt-4.1": _call_azure,
    "claude-sonnet-4": _call_anthropic,
}


def run_one(profile_id, backbone_name, backbone_fn):
    """Run profile reconstruction for one (profile, backbone) pair.

    Uses full_context cache (raw event traces as narratives) — no FileGramOS
    encoding, so we are testing the backbone's ability to read raw traces.
    """
    # Load full_context cache (raw traces as narratives)
    adapter = get_adapter("full_context")
    cache_path = CACHE_DIR / profile_id / "full_context.pkl"
    if not adapter.load_ingest_cache(cache_path):
        return {"avg_score": 0, "error": "cache miss"}

    # Build inference prompt from raw traces
    result = adapter.infer_profile(PROFILE_ATTRIBUTES)
    prompt = result.get("_prompt")
    if not prompt:
        return {"avg_score": 0, "error": "no prompt"}

    # Inference with target backbone
    response = backbone_fn(prompt, temperature=0.3, max_tokens=4096)
    if not response:
        return {"avg_score": 0, "error": "empty response"}

    try:
        parsed = robust_parse_json(response)
        inferred = parsed.get("inferred_profile", parsed)
        inferred_flat = {}
        for attr, detail in inferred.items():
            if isinstance(detail, dict):
                inferred_flat[attr] = detail.get("value", str(detail))
            else:
                inferred_flat[attr] = str(detail)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"    [{backbone_name}] {profile_id} PARSE ERROR: {e}")
        print(f"    Response preview: {response[:200]}...")
        inferred_flat = {}

    if not inferred_flat:
        return {"avg_score": 0, "error": "parse_failed"}

    # Judge scoring (always use gemini for consistency)
    ground_truth = load_ground_truth(profile_id)
    profile_dir = RESULTS_DIR / "cross_backbone" / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)

    judge = JudgeScorer(PROFILES_DIR, profile_dir)
    parsed_scores = judge.judge_all_attributes(
        ground_truth=ground_truth,
        inferred_profile=inferred_flat,
        method_name=f"raw_trace_{backbone_name}",
        attributes=PROFILE_ATTRIBUTES,
        llm_fn=_call_gemini,  # consistent judge
    )
    scores = parsed_scores.get("scores", {})
    avg = parsed_scores.get("overall_mean", 0)

    def chan_avg(attrs):
        vals = [scores.get(a, {}).get("score", 0) for a in attrs
                if isinstance(scores.get(a, {}).get("score"), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else 0

    return {
        "avg_score": round(avg, 3),
        "proc_score": chan_avg(CH_PROC),
        "sem_score": chan_avg(CH_SEM),
        "per_attr": {a: scores.get(a, {}).get("score", 0) for a in PROFILE_ATTRIBUTES},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--profiles", nargs="+", default=ALL_PROFILES)
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES.keys()))
    parser.add_argument("--clear", action="store_true", help="Clear existing results")
    args = parser.parse_args()

    profiles = args.profiles
    backbones = {k: v for k, v in BACKBONES.items() if k in args.backbones}

    print("=" * 70)
    print("Cross-Backbone Trace Validation (raw traces, no FileGramOS encoding)")
    print(f"Backbones: {list(backbones.keys())}")
    print(f"Profiles: {len(profiles)}")
    print(f"Total runs: {len(backbones) * len(profiles)}")
    print("=" * 70)

    # Load existing results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    if not args.clear and OUTPUT_FILE.exists():
        try:
            results = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
            cached = sum(1 for b in results.values() for p in b.values()
                         if isinstance(p, dict) and p.get("avg_score", 0) > 0)
            print(f"Loaded existing results ({cached} cached)")
        except (json.JSONDecodeError, OSError):
            pass

    if args.clear:
        results = {}
        print("Cleared existing results")

    # Build work queue
    work = []
    for backbone_name in backbones:
        for profile_id in profiles:
            cached = results.get(backbone_name, {}).get(profile_id, {})
            if cached.get("avg_score", 0) > 0:
                print(f"  [{backbone_name}] {profile_id} SKIP ({cached['avg_score']:.2f})")
                continue
            work.append((backbone_name, profile_id))

    print(f"\n{len(work)} runs to execute\n")

    def process_one(backbone_name, profile_id):
        t0 = time.time()
        result = run_one(profile_id, backbone_name, backbones[backbone_name])
        elapsed = time.time() - t0
        print(
            f"  [{backbone_name}] {profile_id} → "
            f"avg={result['avg_score']:.2f} proc={result.get('proc_score', 0):.2f} "
            f"sem={result.get('sem_score', 0):.2f} ({elapsed:.0f}s)"
        )
        return backbone_name, profile_id, result

    if args.parallel > 1 and len(work) > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(process_one, b, p): (b, p) for b, p in work}
            for future in as_completed(futures):
                b, p = futures[future]
                try:
                    _, _, result = future.result()
                    results.setdefault(b, {})[p] = result
                    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception as e:
                    print(f"  [{b}] {p} ERROR: {e}")
                    results.setdefault(b, {})[p] = {"avg_score": 0, "error": str(e)}
    else:
        for b, p in work:
            try:
                _, _, result = process_one(b, p)
                results.setdefault(b, {})[p] = result
                OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                print(f"  [{b}] {p} ERROR: {e}")
                results.setdefault(b, {})[p] = {"avg_score": 0, "error": str(e)}

    # Summary
    print(f"\n{'=' * 70}")
    print("CROSS-BACKBONE SUMMARY (raw traces → different backbones → Gemini judge)")
    print(f"{'=' * 70}\n")

    header = f"{'Backbone':<22s}  {'Avg':>5s}  {'Proc':>5s}  {'Sem':>5s}  {'n':>3s}"
    print(header)
    print("-" * 45)

    for backbone_name in backbones:
        bdata = results.get(backbone_name, {})
        avgs = [v["avg_score"] for v in bdata.values() if isinstance(v, dict) and v.get("avg_score", 0) > 0]
        procs = [v.get("proc_score", 0) for v in bdata.values() if isinstance(v, dict) and v.get("avg_score", 0) > 0]
        sems = [v.get("sem_score", 0) for v in bdata.values() if isinstance(v, dict) and v.get("avg_score", 0) > 0]
        n = len(avgs)
        avg = sum(avgs) / n if n else 0
        proc = sum(procs) / n if n else 0
        sem = sum(sems) / n if n else 0
        print(f"{backbone_name:<22s}  {avg:5.2f}  {proc:5.2f}  {sem:5.2f}  {n:3d}")

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
