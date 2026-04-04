#!/usr/bin/env python3
"""Standalone test: run 8 baselines + FileGramOS Simple on P1 trajectories.

Loads P1 trajectory data, runs each method's adapter to build prompts,
calls Azure OpenAI gpt-4.1 for inference, then runs LLM-as-Judge scoring.

Usage:
    cd bench
    python test_baselines.py

    # Or from project root:
    python bench/test_baselines.py

Requires:
    - AZURE_OPENAI_API_KEY set in environment or .env
    - Trajectory data in signal/p1_methodical_T-01/ and T-03/
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup: ensure imports work from any working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent  # bench/
_BENCH_DIR = _SCRIPT_DIR
_PROJECT_ROOT = _SCRIPT_DIR.parent  # FileGram/

# Add bench dir so "baselines", "evaluation", "filegramos" resolve
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
# Add project root so "filegram" package resolves (for .env loading)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env from project root
from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

# Now import adapters (triggers registration via decorators)
from baselines import get_adapter
from baselines.base import BaseAdapter
from baselines.eager_summarization import EagerSummarizationAdapter  # noqa: F401
from baselines.evermemos_adapter import EverMemOSAdapter  # noqa: F401
from baselines.filegramos_adapter import FileGramOSAdapter  # noqa: F401
from baselines.full_context import FullContextAdapter  # noqa: F401
try:
    from baselines.mem0_adapter import Mem0Adapter  # noqa: F401
except Exception:
    pass
from baselines.memos_adapter import MemOSAdapter  # noqa: F401
from baselines.memu_adapter import MemUAdapter  # noqa: F401
from baselines.naive_rag import NaiveRAGAdapter  # noqa: F401
from baselines.mma_adapter import MMAAdapter  # noqa: F401
from baselines.visrag_adapter import VisRAGAdapter  # noqa: F401
from baselines.simplemem_adapter import SimpleMemAdapter  # noqa: F401
from baselines.zep_adapter import ZepAdapter  # noqa: F401
from evaluation.judge_scoring import JudgeScorer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
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
SIGNALS_DIR = _PROJECT_ROOT / "signal"
PROFILES_DIR = _PROJECT_ROOT / "profiles"
RESULTS_DIR = _BENCH_DIR / "test_results"
CACHE_DIR = _BENCH_DIR / "ingest_cache"
SKIP_PERTURBED = False  # False = include perturbed (main table); True = standard only (ablation)
USE_FILTERED_EVENTS = False  # Legacy: events.json is now pre-cleaned (was events_filtered.json)
USE_INGEST_CACHE = True  # Load/save ingest cache to avoid re-running LLM extraction
STRIP_CONTENT = False  # True = strip _resolved_content from events (multimodal simulation)

METHODS = [
    "naive_rag",
    "eager_summarization",
    "mem0",
    "zep",
    "memos",
    "memu",
    "evermemos",
    "simplemem",
    "mma",
    "visrag",
    "filegramos",
    "full_context",  # last: sends huge prompts, often rate-limited
]
PARALLEL_JOBS = 3  # outer (profile×method) concurrency; inner judge = 4 threads each
MAX_TASKS = 0  # 0 = all tasks; >0 = limit trajectories per profile
LOGS_DIR = RESULTS_DIR / "logs"

PROFILE_ATTRIBUTES = [
    # basic
    "name",
    "role",
    "language",
    # personality
    "tone",
    "output_detail",
    # work_habits
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

INGEST_ONLY = False  # --ingest-only: skip inference, only build pkl caches


# ---------------------------------------------------------------------------
# Per-method logging: tee stdout to per-method log file
# ---------------------------------------------------------------------------
class _TeeWriter:
    """Write to both the original stream and a log file."""

    def __init__(self, original, log_file: io.TextIOWrapper):
        self._original = original
        self._log = log_file

    def write(self, text: str):
        self._original.write(text)
        try:
            self._log.write(text)
            self._log.flush()
        except Exception:
            pass

    def flush(self):
        self._original.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    # Forward other attributes to original
    def __getattr__(self, name):
        return getattr(self._original, name)


class MethodLogger:
    """Context manager: tee all stdout/stderr to a per-method log file.

    Log file path: {LOGS_DIR}/{method_name}.log
    Each entry is appended (not overwritten) so multiple profiles accumulate.
    """

    def __init__(self, method_name: str, profile_id: str):
        self.method_name = method_name
        self.profile_id = profile_id
        self._log_file = None
        self._old_stdout = None
        self._old_stderr = None

    def __enter__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / f"{self.method_name}.log"
        self._log_file = open(log_path, "a", encoding="utf-8")
        self._log_file.write(
            f"\n{'=' * 60}\n"
            f"[{self.profile_id}][{self.method_name}] "
            f"started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'=' * 60}\n"
        )
        self._log_file.flush()
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = _TeeWriter(self._old_stdout, self._log_file)
        sys.stderr = _TeeWriter(self._old_stderr, self._log_file)
        return self

    def __exit__(self, *exc_info):
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        if self._log_file:
            try:
                self._log_file.write(
                    f"\n--- [{self.profile_id}][{self.method_name}] "
                    f"finished at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
                )
                self._log_file.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# LLM caller (Azure OpenAI / Gemini)
# ---------------------------------------------------------------------------
LLM_API = "azure"  # "azure" or "gemini"


def _get_azure_config() -> dict[str, str]:
    """Read Azure OpenAI config from environment."""
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY not set. Set it in your environment or in .env")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://haku-chat.openai.azure.com")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    return {
        "api_key": api_key,
        "endpoint": endpoint.rstrip("/").split("/openai/")[0],  # normalize
        "deployment": deployment,
        "api_version": api_version,
    }


def _call_azure(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """Call Azure OpenAI chat completion and return the response text."""
    import httpx

    cfg = _get_azure_config()
    url = f"{cfg['endpoint']}/openai/deployments/{cfg['deployment']}/chat/completions?api-version={cfg['api_version']}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = httpx.post(
                url,
                json=payload,
                headers={
                    "api-key": cfg["api_key"],
                    "Content-Type": "application/json",
                },
                timeout=180.0,
            )
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout) as e:
            wait = min(10 * attempt, 60)
            print(f"    Timeout ({e.__class__.__name__}), retrying in {wait}s (attempt {attempt})...")
            time.sleep(wait)
            continue

        if resp.status_code == 429:
            # Rate limited — parse retry-after or default to 15s
            wait = 15
            try:
                err = resp.json()
                msg = err.get("error", {}).get("message", "")
                import re as _re

                m = _re.search(r"retry after (\d+) seconds", msg, _re.IGNORECASE)
                if m:
                    wait = int(m.group(1)) + 2
            except Exception:
                msg = resp.text[:200]
            prompt_chars = sum(len(m.get("content", "")) for m in messages)
            if attempt <= 3 or attempt % 10 == 0:
                print(
                    f"    Rate limited (attempt {attempt}), prompt≈{prompt_chars}chars, wait {wait}s. Err: {msg[:120]}"
                )
            else:
                print(f"    Rate limited, waiting {wait}s (attempt {attempt})...")
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            # Transient server errors (500/502/503): retry
            if resp.status_code >= 500:
                wait = min(10 * attempt, 60)
                print(f"    Server error {resp.status_code}, retrying in {wait}s (attempt {attempt})...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Azure OpenAI returned {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        return data["choices"][0]["message"]["content"]


_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _call_gemini(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """Call Gemini API and return the response text."""
    from google.genai import types

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    client = _get_gemini_client()

    contents = []
    if system_prompt:
        contents.append(f"[System]\n{system_prompt}\n\n[User]\n{prompt}")
    else:
        contents.append(prompt)

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens + 4096,  # extra room for thinking
                    thinking_config=types.ThinkingConfig(thinking_budget=2048),
                ),
            )
            return (resp.text or "").strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = min(15 * attempt, 60)
                if attempt <= 3 or attempt % 10 == 0:
                    print(f"    Gemini rate limited (attempt {attempt}), wait {wait}s: {err_str[:120]}")
                else:
                    print(f"    Gemini rate limited, waiting {wait}s (attempt {attempt})...")
                time.sleep(wait)
                continue
            if "500" in err_str or "503" in err_str or "UNAVAILABLE" in err_str:
                wait = min(10 * attempt, 60)
                print(f"    Gemini server error (attempt {attempt}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise


def call_llm(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """Route to the active LLM API (azure or gemini)."""
    if LLM_API == "gemini":
        return _call_gemini(prompt, system_prompt, temperature, max_tokens)
    return _call_azure(prompt, system_prompt, temperature, max_tokens)


# ---------------------------------------------------------------------------
# Ground truth extraction
# ---------------------------------------------------------------------------
def load_ground_truth(profile_id: str) -> dict[str, str]:
    """Load ground truth profile attributes from a profile YAML."""
    import yaml

    yaml_path = PROFILES_DIR / f"{profile_id}.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    basic = data.get("basic", {})
    personality = data.get("personality", {})
    wh = data.get("work_habits", {})

    gt = {
        # basic
        "name": basic.get("name", ""),
        "role": basic.get("role", ""),
        "language": basic.get("language", ""),
        # personality
        "tone": personality.get("tone", ""),
        "output_detail": personality.get("output_detail", personality.get("verbosity", "")),
        # work_habits
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
    return gt


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------
def parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response (handles markdown fences, trailing commas)."""
    import re as _re

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
            if in_fence or not any(line.strip().startswith("```") for _ in [0]):
                filtered.append(line)
        text = "\n".join(filtered)

    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas (common Gemini issue): ,\s*} or ,\s*]
    cleaned = _re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Trajectory loader (supports filtered events)
# ---------------------------------------------------------------------------
def load_trajectories_filtered(profile_id: str) -> list[dict[str, Any]]:
    """Load trajectories using events_filtered.json when available.

    Falls back to events.json if filtered version doesn't exist.
    Media refs are resolved in both cases.
    Skips perturbed trajectories when SKIP_PERTURBED is True.
    """
    trajectories = []
    skipped_perturbed = 0
    included_perturbed = 0
    for traj_dir in sorted(SIGNALS_DIR.iterdir()):
        if not traj_dir.is_dir():
            continue
        if not traj_dir.name.startswith(profile_id + "_"):
            continue
        task_id = traj_dir.name[len(profile_id) + 1 :]

        # Check perturbation status
        is_perturbed = False
        perturb_file = traj_dir / "perturbation.json"
        if perturb_file.exists():
            try:
                pdata = json.loads(perturb_file.read_text(encoding="utf-8"))
                if pdata.get("applied"):
                    is_perturbed = True
            except Exception:
                pass

        # Skip perturbed if requested (ablation mode)
        if SKIP_PERTURBED and is_perturbed:
            skipped_perturbed += 1
            continue

        # Prefer filtered events
        filtered_file = traj_dir / "events_filtered.json"
        raw_file = traj_dir / "events.json"
        summary_file = traj_dir / "summary.json"

        if USE_FILTERED_EVENTS and filtered_file.exists():
            events = json.loads(filtered_file.read_text(encoding="utf-8"))
            source = "filtered"
        elif raw_file.exists():
            events = json.loads(raw_file.read_text(encoding="utf-8"))
            source = "raw"
        else:
            continue

        # Skip incomplete trajectories
        summary = {}
        if summary_file.exists():
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            if not summary.get("completed", True):
                continue

        # Resolve media content
        BaseAdapter._resolve_media_refs(traj_dir, events)

        # Strip resolved content to simulate multimodal setting (PDF/image outputs)
        if STRIP_CONTENT:
            for event in events:
                for key in ("_resolved_content", "_resolved_diff",
                            "_resolved_content_old", "_resolved_content_new"):
                    event.pop(key, None)

        if is_perturbed:
            included_perturbed += 1

        trajectories.append(
            {
                "task_id": task_id,
                "events": events,
                "summary": summary,
                "path": str(traj_dir),
                "source": source,
                "is_perturbed": is_perturbed,
            }
        )

    n_standard = len(trajectories) - included_perturbed
    if skipped_perturbed:
        print(f"  [{profile_id}] Loaded {len(trajectories)} standard, skipped {skipped_perturbed} perturbed")
    elif included_perturbed:
        print(
            f"  [{profile_id}] Loaded {n_standard} standard + {included_perturbed} perturbed = {len(trajectories)} total"
        )
    return trajectories


# ---------------------------------------------------------------------------
# File-level lock for concurrent results writing
# ---------------------------------------------------------------------------
import threading

try:
    import filelock as _filelock_mod
except ImportError:
    # Fallback: use threading lock (works for in-process concurrency)
    class _FakeLock:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class _filelock_mod:
        @staticmethod
        def FileLock(*a, **kw):
            return _FakeLock()


_results_lock = threading.Lock()


def _save_method_result(profile_id: str, method_name: str, result: dict[str, Any]):
    """Thread-safe: merge one method result into profile results.json."""
    profile_dir = RESULTS_DIR / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    results_file = profile_dir / "results.json"
    lock_file = results_file.with_suffix(".json.lock")

    lock = _filelock_mod.FileLock(str(lock_file), timeout=30)
    with lock:
        existing = {}
        if results_file.exists():
            try:
                existing = json.loads(results_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        existing[method_name] = result
        results_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 1: Ingest + Infer (one method × one profile)
# ---------------------------------------------------------------------------
def run_ingest_infer(
    profile_id: str,
    method_name: str,
    trajectories: list[dict],
) -> tuple[str, dict[str, Any] | None]:
    """Ingest trajectories and infer profile. Save inferred profile JSON.

    Returns (profile_id, inferred_meta) or (profile_id, None) on skip/error.
    Output: test_results/{profile_id}/{method_name}_inferred.json
    """
    tag = f"[{profile_id}][{method_name}]"
    profile_dir = RESULTS_DIR / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    inferred_file = profile_dir / f"{method_name}_inferred.json"

    # Skip logic
    if INGEST_ONLY:
        cache_path = CACHE_DIR / profile_id / f"{method_name}.pkl"
        if cache_path.exists():
            print(f"  {tag} SKIP (cache exists)")
            return profile_id, None
    else:
        if inferred_file.exists():
            print(f"  {tag} SKIP (inferred exists)")
            return profile_id, None

    try:
        adapter = get_adapter(method_name, llm_fn=call_llm)

        # Try loading from ingest cache
        cache_path = CACHE_DIR / profile_id / f"{method_name}.pkl"
        cache_hit = False
        if USE_INGEST_CACHE:
            cache_hit = adapter.load_ingest_cache(cache_path)
            if cache_hit:
                print(f"  {tag} ingest cache hit")

        if not cache_hit:
            from tqdm import tqdm

            pbar = tqdm(
                total=len(trajectories),
                desc=f"  {tag} ingest",
                unit="traj",
                leave=True,
                ncols=90,
            )
            adapter._progress_callback = lambda _tid: pbar.update(1)
            adapter.ingest(trajectories)
            pbar.close()

            # Special handling for eager_summarization
            if method_name == "eager_summarization":
                summarize_prompts = adapter.get_summarize_prompts()
                summaries = []
                for i, sp in enumerate(summarize_prompts):
                    summary = call_llm(sp["prompt"], temperature=0.3, max_tokens=1024)
                    summaries.append(summary)
                adapter.set_summaries(summaries)

            # Save ingest cache
            if USE_INGEST_CACHE:
                adapter.save_ingest_cache(cache_path)

        # Ingest-only mode: skip inference
        if INGEST_ONLY:
            print(f"  {tag} ingest done (ingest-only mode)")
            adapter.reset()
            return profile_id, {"ingest_only": True}

        # Build base inference prompt (with attribute list)
        result = adapter.infer_profile(PROFILE_ATTRIBUTES)
        prompt = result.get("_prompt")
        if not prompt:
            print(f"  {tag} ERROR: No prompt generated")
            adapter.reset()
            return profile_id, None

        inferred_meta = {
            "llm_calls_ingest": adapter.llm_calls_ingest,
            "tokens_ingest": adapter.tokens_ingest,
        }

        # ── Structured inference (with attribute list) ──
        if not inferred_file.exists():
            prompt_file = profile_dir / f"{method_name}_prompt.txt"
            prompt_file.write_text(prompt, encoding="utf-8")

            t0 = time.time()
            response = call_llm(prompt, temperature=0.3, max_tokens=2048)
            dt = time.time() - t0

            try:
                parsed = parse_json_response(response)
                inferred = parsed.get("inferred_profile", parsed)
                inferred_flat = {}
                for attr, detail in inferred.items():
                    if isinstance(detail, dict):
                        inferred_flat[attr] = detail.get("value", str(detail))
                    else:
                        inferred_flat[attr] = str(detail)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  {tag} PARSE ERROR: {e}")
                raw_file = profile_dir / f"{method_name}_raw_response.txt"
                raw_file.write_text(response, encoding="utf-8")
                inferred_flat = {}

            inferred_meta.update(
                {
                    "inferred_profile": inferred_flat,
                    "prompt_length": len(prompt),
                    "inference_time_s": round(dt, 2),
                }
            )
            inferred_file.write_text(json.dumps(inferred_meta, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"  {tag} structured profile saved (prompt_len={len(prompt)}, infer={dt:.1f}s, attrs={len(inferred_flat)})"
            )

        adapter.reset()
        return profile_id, inferred_meta

    except Exception as e:
        print(f"  {tag} ERROR: {e.__class__.__name__}: {e}")
        return profile_id, None


# ---------------------------------------------------------------------------
# Phase 2: Judge (one method × one profile)
# ---------------------------------------------------------------------------
def run_judge_one(
    profile_id: str,
    method_name: str,
    ground_truth: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Load inferred profile and run LLM judge. Returns (profile_id, method_name, result)."""
    tag = f"[{profile_id}][{method_name}]"
    profile_dir = RESULTS_DIR / profile_id

    # Skip if already judged
    results_file = profile_dir / "results.json"
    if results_file.exists():
        try:
            existing = json.loads(results_file.read_text(encoding="utf-8"))
            if method_name in existing and "judge_scores" in existing[method_name]:
                avg = existing[method_name].get("avg_score", "?")
                print(f"  {tag} SKIP judge (already done, score={avg})")
                return profile_id, method_name, existing[method_name]
        except (json.JSONDecodeError, OSError):
            pass

    # Load inferred profile
    inferred_file = profile_dir / f"{method_name}_inferred.json"
    if not inferred_file.exists():
        print(f"  {tag} SKIP judge (no inferred profile)")
        return profile_id, method_name, {"error": "No inferred profile", "avg_score": 0}

    try:
        inferred_meta = json.loads(inferred_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  {tag} ERROR loading inferred: {e}")
        return profile_id, method_name, {"error": str(e), "avg_score": 0}

    inferred_flat = inferred_meta.get("inferred_profile", {})
    if not inferred_flat:
        print(f"  {tag} SKIP judge (empty inferred profile)")
        return profile_id, method_name, {"error": "Empty inferred profile", "avg_score": 0}

    try:
        judge = JudgeScorer(PROFILES_DIR, profile_dir)
        jt0 = time.time()
        parsed_scores = judge.judge_all_attributes(
            ground_truth=ground_truth,
            inferred_profile=inferred_flat,
            method_name=method_name,
            attributes=PROFILE_ATTRIBUTES,
            llm_fn=call_llm,
            verbose=False,
            max_workers=4,
        )
        jdt = time.time() - jt0

        avg = parsed_scores.get("overall_mean", 0)
        print(f"  {tag} score={avg:.2f} (judge={jdt:.1f}s)")

        method_result = {
            **inferred_meta,
            "ground_truth": ground_truth,
            "judge_scores": parsed_scores,
            "judge_time_s": round(jdt, 2),
            "avg_score": round(avg, 3),
        }

    except Exception as e:
        print(f"  {tag} JUDGE ERROR: {e.__class__.__name__}: {e}")
        method_result = {**inferred_meta, "error": f"{e.__class__.__name__}: {e}", "avg_score": 0}

    # Save immediately (thread-safe)
    _save_method_result(profile_id, method_name, method_result)
    return profile_id, method_name, method_result


# ---------------------------------------------------------------------------
# Main test runner: Phase 1 (ingest+infer) → Phase 2 (judge)
# ---------------------------------------------------------------------------
def run_test(phase: str = "all"):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load all trajectories and ground truths
    profile_data: dict[str, tuple[list, dict]] = {}
    for profile_id in ALL_PROFILES:
        trajectories = load_trajectories_filtered(profile_id)
        if not trajectories:
            print(f"  [!] {profile_id}: No trajectories, skipping")
            continue
        if MAX_TASKS > 0:
            trajectories = trajectories[:MAX_TASKS]
        ground_truth = load_ground_truth(profile_id)
        profile_data[profile_id] = (trajectories, ground_truth)
        (RESULTS_DIR / profile_id).mkdir(parents=True, exist_ok=True)

    total_infer = len(profile_data) * len(METHODS)
    total_judge = total_infer

    print("=" * 70)
    print("FileGramBench Two-Phase Evaluation")
    print(f"Profiles: {len(profile_data)} | Methods: {len(METHODS)}")
    print(f"Phase 1 (Ingest+Infer): {total_infer} jobs | Phase 2 (Judge): {total_judge} jobs")
    print(f"Profile concurrency: {PARALLEL_JOBS}")
    print(f"Logs: {LOGS_DIR}/")
    print("=" * 70)

    # ===================================================================
    # Phase 1: Ingest + Infer — method-sequential, profiles-parallel
    # Each method gets its own log file
    # ===================================================================
    if phase in ("all", "1"):
        print(f"\n{'=' * 70}")
        print("PHASE 1: Ingest + Infer (per-method parallel profiles)")
        print(f"{'=' * 70}")

        total_jobs = len(METHODS) * len(profile_data)
        print(f"  Total jobs: {total_jobs} | {len(METHODS)} methods × {len(profile_data)} profiles")
        print(f"  Per-method profile concurrency: {PARALLEL_JOBS}")

        done = 0
        lock = __import__("threading").Lock()

        def run_method_batch(method_name):
            """Run one method across all profiles with PARALLEL_JOBS concurrency."""
            nonlocal done
            results = []
            with ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as pool:
                futures = {}
                for pid, (trajs, _gt) in profile_data.items():
                    f = pool.submit(run_ingest_infer, pid, method_name, trajs)
                    futures[f] = pid

                for future in as_completed(futures):
                    pid = futures[future]
                    with lock:
                        done += 1
                        cur = done
                    try:
                        _, meta = future.result()
                        status = "OK" if meta else "skip/cached"
                    except Exception as e:
                        print(f"  [{pid}][{method_name}] FATAL: {e}")
                        status = "ERROR"
                    print(f"  [{cur}/{total_jobs}] {method_name}/{pid}: {status}")
                    results.append((pid, method_name, status))
            return results

        # All methods run concurrently; each method parallelizes profiles independently
        with ThreadPoolExecutor(max_workers=len(METHODS)) as method_pool:
            method_futures = {method_pool.submit(run_method_batch, m): m for m in METHODS}
            for future in as_completed(method_futures):
                method_name = method_futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  [METHOD-LEVEL ERROR] {method_name}: {e}")

        if phase == "1":
            print(f"\n{'=' * 70}")
            print("Phase 1 complete. Run with --phase 2 to judge.")
            print(f"{'=' * 70}")
            return

    # ===================================================================
    # Phase 2: Judge — all (profile × method) in parallel
    # ===================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 2: LLM Judge")
    print(f"{'=' * 70}")

    all_profile_results: dict[str, dict[str, Any]] = {pid: {} for pid in profile_data}
    judge_jobs = [(pid, m) for pid in profile_data for m in METHODS]

    done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as pool:
        futures = {}
        for pid, method in judge_jobs:
            _trajs, gt = profile_data[pid]
            f = pool.submit(run_judge_one, pid, method, gt)
            futures[f] = (pid, method)

        for future in as_completed(futures):
            pid, method = futures[future]
            try:
                _, _, result = future.result()
                all_profile_results[pid][method] = result
            except Exception as e:
                print(f"  [{pid}][{method}] JUDGE FATAL: {e}")
                all_profile_results[pid][method] = {"error": str(e), "avg_score": 0}
            done += 1
            if done % len(METHODS) == 0 or done == len(judge_jobs):
                print(f"  --- Judge progress: {done}/{len(judge_jobs)} ---")

    # ===================================================================
    # Save master results + comparison table
    # ===================================================================
    master_file = RESULTS_DIR / "all_results.json"
    existing_master = {}
    if master_file.exists():
        try:
            existing_master = json.loads(master_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    for pid, pres in all_profile_results.items():
        if pid not in existing_master:
            existing_master[pid] = {}
        existing_master[pid].update(pres)
    master_file.write_text(json.dumps(existing_master, indent=2, ensure_ascii=False), encoding="utf-8")

    # Generate cross-profile comparison table
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}\n")

    md_lines = ["# FileGramBench Baseline Comparison\n"]
    md_lines.append(f"**{len(ALL_PROFILES)} profiles × {len(METHODS)} methods**\n")

    # Table: Method (rows) × Profile (cols) + Overall
    md_lines.append("## Average Scores (1-5)\n")
    header = "| Method | " + " | ".join(p.replace("_", " ") for p in ALL_PROFILES) + " | **Overall** |"
    sep = "|" + "---|" * (len(ALL_PROFILES) + 2)
    md_lines.append(header)
    md_lines.append(sep)

    method_overalls = []
    for method_name in METHODS:
        cells = []
        scores = []
        for profile_id in ALL_PROFILES:
            pr = all_profile_results.get(profile_id, {}).get(method_name, {})
            avg = pr.get("avg_score", 0)
            cells.append(f"{avg:.2f}" if avg else "-")
            if avg:
                scores.append(avg)
        overall = sum(scores) / len(scores) if scores else 0
        method_overalls.append((method_name, overall))
        md_lines.append(f"| {method_name} | " + " | ".join(cells) + f" | **{overall:.2f}** |")

    # Per-attribute breakdown
    md_lines.append("\n## Per-Attribute Breakdown (averaged across profiles)\n")
    md_lines.append("| Method | " + " | ".join(PROFILE_ATTRIBUTES) + " |")
    md_lines.append("|" + "---|" * (len(PROFILE_ATTRIBUTES) + 1))

    for method_name in METHODS:
        attr_avgs = {a: [] for a in PROFILE_ATTRIBUTES}
        for profile_id in ALL_PROFILES:
            pr = all_profile_results.get(profile_id, {}).get(method_name, {})
            judge = pr.get("judge_scores", {}).get("scores", {})
            for attr in PROFILE_ATTRIBUTES:
                s = judge.get(attr, {}).get("score")
                if isinstance(s, (int, float)):
                    attr_avgs[attr].append(s)

        cells = []
        for attr in PROFILE_ATTRIBUTES:
            vals = attr_avgs[attr]
            avg = sum(vals) / len(vals) if vals else 0
            cells.append(f"{avg:.1f}" if vals else "-")
        md_lines.append(f"| {method_name} | " + " | ".join(cells) + " |")

    # Ranking
    md_lines.append("\n## Ranking\n")
    method_overalls.sort(key=lambda x: -x[1])
    for rank, (method, score) in enumerate(method_overalls, 1):
        md_lines.append(f"{rank}. **{method}**: {score:.3f}")

    # Cost comparison
    md_lines.append("\n## Cost Comparison (LLM calls during ingest)\n")
    md_lines.append("| Method | Ingest LLM Calls | Infer LLM Calls | Total |")
    md_lines.append("|---|---|---|---|")
    for method_name in METHODS:
        total_ingest_calls = 0
        total_infer_calls = 0
        for profile_id in ALL_PROFILES:
            pr = all_profile_results.get(profile_id, {}).get(method_name, {})
            total_ingest_calls += pr.get("llm_calls_ingest", 0)
            total_infer_calls += pr.get("llm_calls_infer", 0)
        md_lines.append(
            f"| {method_name} | {total_ingest_calls} | {total_infer_calls} | {total_ingest_calls + total_infer_calls} |"
        )

    comparison_md = RESULTS_DIR / "comparison.md"
    comparison_md.write_text("\n".join(md_lines), encoding="utf-8")

    # Print summary
    print(f"{'Method':<25s} {'Overall':>8s}")
    print("-" * 35)
    for method, score in method_overalls:
        print(f"{method:<25s} {score:>8.3f}")

    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Comparison: {comparison_md}")
    print(f"Per-method logs: {LOGS_DIR}/")
    print(f"\n{'=' * 70}")
    print("Done!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, help="Comma-separated methods to run")
    parser.add_argument("--profiles", type=str, help="Comma-separated profiles to run")
    parser.add_argument("--no-cache", action="store_true", help="Disable ingest cache")
    parser.add_argument("--parallel", type=int, default=3, help="Profile concurrency within each method (default 3)")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "all"],
        default="all",
        help="Run only phase 1 (ingest+infer), phase 2 (judge), or all (default)",
    )
    parser.add_argument(
        "--skip-perturbed",
        action="store_true",
        help="Exclude perturbed trajectories (ablation mode; default: include all)",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=0, help="Limit trajectories to first N tasks per profile (0=all)"
    )
    parser.add_argument(
        "--api",
        choices=["azure", "gemini"],
        default="gemini",
        help="LLM API: azure (GPT-4.1) or gemini (Gemini 2.5 Pro)",
    )
    parser.add_argument("--ingest-only", action="store_true", help="Only run ingest (build pkl caches), skip inference")
    parser.add_argument(
        "--strip-content",
        action="store_true",
        help="Strip _resolved_content from events (simulates multimodal output setting where files are PDF/images)",
    )
    args = parser.parse_args()
    if args.methods:
        METHODS[:] = [m.strip() for m in args.methods.split(",")]
    if args.profiles:
        ALL_PROFILES[:] = [p.strip() for p in args.profiles.split(",")]
    if args.no_cache:
        USE_INGEST_CACHE = False
    if args.skip_perturbed:
        SKIP_PERTURBED = True
    if args.ingest_only:
        INGEST_ONLY = True
    if args.strip_content:
        STRIP_CONTENT = True
    LLM_API = args.api
    # Resolve actual model name for dir naming
    if LLM_API == "azure":
        _model_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    else:
        _model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    # Clean model name for dir: "gpt-4.1" -> "gpt4.1", "gemini-2.5-pro" -> "gemini_2.5_pro"
    if _model_name.startswith("gpt"):
        _model_slug = _model_name.replace("-", "")
    else:
        _model_slug = _model_name.replace("-", "_")
    _suffix = f"_nocontent_{_model_slug}" if STRIP_CONTENT else f"_{_model_slug}"
    RESULTS_DIR = _BENCH_DIR / f"test_results{_suffix}"
    CACHE_DIR = _BENCH_DIR / f"ingest_cache{_suffix}"
    LOGS_DIR = RESULTS_DIR / "logs"
    PARALLEL_JOBS = args.parallel
    MAX_TASKS = args.max_tasks
    print(f"LLM: {LLM_API} / {_model_name}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Cache:   {CACHE_DIR}")
    if MAX_TASKS:
        print(f"Max tasks per profile: {MAX_TASKS}")
    run_test(phase=args.phase)
