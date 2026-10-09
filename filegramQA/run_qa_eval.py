#!/usr/bin/env python3
"""Run QA evaluation on MCQ questions using existing ingest caches.

Usage:
    python -m filegramQA.run_qa_eval --cache-dir gpt4.1                     # Azure, all (MCQ + open-ended)
    python -m filegramQA.run_qa_eval --cache-dir gpt4.1 --mode qa           # MCQ only
    python -m filegramQA.run_qa_eval --cache-dir gpt4.1 --mode openended    # free-form ProfRec only
    python -m filegramQA.run_qa_eval --cache-dir gpt4.1 --mode all          # both (default)
    python -m filegramQA.run_qa_eval --cache-dir gpt4.1 --api gemini        # Gemini Flash answering
    python -m filegramQA.run_qa_eval --cache-dir gpt4.1 --parallel 20
    python -m filegramQA.run_qa_eval --cache-dir gpt4.1 --resume            # resume from checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Paths ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# Ensure bench/ is in sys.path so pickle can deserialize filegramos dataclasses
_bench_dir = str(ROOT / "bench")
if _bench_dir not in sys.path:
    sys.path.insert(0, _bench_dir)
CACHE_DIR = ROOT / "bench" / "ingest_cache"  # overridden by --cache-dir
QUESTIONS_DIR = Path(__file__).resolve().parent / "questions"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SIGNAL_DIR = ROOT / "signal"

TASK_IDS = [f"T-{i:02d}" for i in range(1, 33)]

# ── Parameter sensitivity config (overridden via CLI) ─────
_PS_DISPLAY_CHARS: int | None = None  # None = use tuning.py default
_PS_PREVIEW_CHARS: int = 200          # per-task content preview chars
_PS_FILES_PER_TASK: int = 3           # max created files shown per task
_PS_INCLUDE_EDIT_DIFFS: bool = False  # include edit chain diffs in per-task
_PS_INCLUDE_CONTENT_PREVIEWS: bool = True  # include content previews in per-task
_PS_RESULT_SUFFIX: str = ""           # appended to result filename

METHODS = [
    "full_context", "filegramos", "eager_summarization",
    "naive_rag", "naive_rag_capped", "zep", "memos", "memu", "evermemos", "mem0",
    "mma", "visrag", "simplemem",
    "osworld_screen_buffer", "officebench_tool_history",
    "filegramos_no_proc", "filegramos_no_sem", "filegramos_no_epi",
]

# Methods that share another method's pkl cache
CACHE_ALIAS = {
    "naive_rag_capped": "naive_rag",
    "osworld_screen_buffer": "full_context",
    "officebench_tool_history": "full_context",
    "filegramos_no_proc": "filegramos",
    "filegramos_no_sem": "filegramos",
    "filegramos_no_epi": "filegramos",
}

# ── Track / Sub-track mapping (sub_class → paper tracks, for backward compat) ──
TRACK_MAP = {
    "dimension_classification":     ("T1", "Attribute Recognition"),
    "semantic_attribute":           ("T1", "Attribute Recognition"),
    "behavioral_fingerprint":       ("T1", "Behavioral Fingerprint"),
    "counterfactual":               ("T2", "Behavioral Inference"),
    "multi_user_attribution":       ("T2", "Trace Disentanglement"),
    "single_user_task_separation":  ("T2", "Trace Disentanglement"),
    "anomaly_detection":            ("T3", "Anomaly Detection"),
    "impersonation_detection":      ("T3", "Anomaly Detection"),
    "perturbation_detection":       ("T3", "Anomaly Detection"),
    "no_shift_control":             ("T3", "Anomaly Detection"),
    "shifted_dimension_id":         ("T3", "Shift Analysis"),
    "shift_direction":              ("T3", "Shift Analysis"),
    "behavioral_change_description": ("T3", "Shift Analysis"),
    "file_grounding":               ("T4", "File Grounding"),
    "visual_grounding":             ("T4", "Visual Grounding"),
}

CHANNEL_MAP = {
    "dimension_classification":     "procedural",
    "semantic_attribute":           "semantic",
    "behavioral_fingerprint":       "procedural",
    "counterfactual":               "procedural",
    "anomaly_detection":            "procedural",
    "multi_user_attribution":       "procedural",
    "single_user_task_separation":  "procedural",
    "impersonation_detection":      "episodic",
    "perturbation_detection":       "episodic",
    "no_shift_control":             "episodic",
    "shifted_dimension_id":         "episodic",
    "shift_direction":              "episodic",
    "behavioral_change_description": "episodic",
    "profile_reconstruction":       "semantic",
    "file_grounding":               "procedural",
    "visual_grounding":             "procedural",
}

TRACK_ORDER = ["T1", "T2", "T3", "T4"]
TRACK_LABELS = {
    "T1": "T1:Understanding", "T2": "T2:Reasoning",
    "T3": "T3:Detection", "T4": "T4:VisGrd",
}
TRACK_SUBTYPES = {
    "T1": ["Attribute Recognition", "Behavioral Fingerprint", "Profile Reconstruction"],
    "T2": ["Behavioral Inference", "Trace Disentanglement"],
    "T3": ["Anomaly Detection", "Shift Analysis"],
    "T4": ["File Grounding", "Visual Grounding"],
}
# Ordered flat list for column rendering
SUBTYPE_ORDER = [
    ("T1", "Attribute Recognition"), ("T1", "Behavioral Fingerprint"), ("T1", "Profile Reconstruction"),
    ("T2", "Behavioral Inference"), ("T2", "Trace Disentanglement"),
    ("T3", "Anomaly Detection"), ("T3", "Shift Analysis"),
    ("T4", "File Grounding"), ("T4", "Visual Grounding"),
]
# Short names for table column headers
SUBTYPE_SHORT = {
    "Attribute Recognition":  "AttrRec",
    "Profile Reconstruction": "ProfRec",
    "Behavioral Inference":   "BehavInf",
    "Trace Disentanglement":  "TraceDis",
    "Anomaly Detection":      "AnomDet",
    "Shift Analysis":         "ShiftAna",
    "Behavioral Fingerprint":  "BehavFP",
    "File Grounding":         "FileGrd",
    "Visual Grounding":       "VisGrd",
}
# Sub-class short names for component breakdown
SUB_CLASS_SHORT = {
    "dimension_classification": "dim_cls", "semantic_attribute": "sem_attr",
    "behavioral_fingerprint": "fp", "counterfactual": "cf",
    "anomaly_detection": "anomaly", "multi_user_attribution": "multi_usr",
    "single_user_task_separation": "task_sep", "impersonation_detection": "imperson",
    "perturbation_detection": "pert_det", "no_shift_control": "no_shift",
    "shifted_dimension_id": "dim_id", "shift_direction": "direction",
    "behavioral_change_description": "beh_chg",
    "file_grounding": "file_grd", "visual_grounding": "vis_grd",
}
CHANNEL_ORDER = ["procedural", "semantic", "episodic"]
CHANNEL_SHORT = {"procedural": "Proc", "semantic": "Sem", "episodic": "Epi"}

PROFILE_ATTRIBUTES = [
    "name", "role", "language", "tone", "output_detail", "working_style",
    "thoroughness", "documentation", "error_handling", "reading_strategy",
    "output_structure", "directory_style", "naming", "edit_strategy",
    "version_strategy", "cross_modal_behavior",
]

# ── LLM client (set up by init_llm_client) ───────────────────
_llm_fn = None   # callable(prompt, max_tokens) -> str
_llm_multimodal_fn = None  # callable(parts, max_tokens) -> str (for visrag)
_current_api = ""  # "azure" or "gemini"
MODEL = ""
_llm_lock = Lock()
_llm_calls = 0


def init_llm_client(api: str):
    """Initialize LLM client based on --api flag."""
    global _llm_fn, _llm_multimodal_fn, _current_api, MODEL
    _current_api = api

    if api == "azure":
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

        def _azure_call(prompt: str, max_tokens: int) -> str:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()

        _llm_fn = _azure_call

    elif api == "gemini":
        from google import genai
        from google.genai import types
        gclient = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        def _gemini_call(prompt: str, max_tokens: int) -> str:
            resp = gclient.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return (resp.text or "").strip()

        def _gemini_call_multimodal(parts: list, max_tokens: int) -> str:
            """Call Gemini with multimodal content (text + images)."""
            content_parts = []
            for part in parts:
                if isinstance(part, str):
                    content_parts.append(types.Part(text=part))
                elif isinstance(part, bytes):
                    content_parts.append(types.Part(
                        inline_data=types.Blob(mime_type="image/png", data=part)
                    ))
            resp = gclient.models.generate_content(
                model=MODEL,
                contents=types.Content(parts=content_parts),
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return (resp.text or "").strip()

        _llm_fn = _gemini_call
        _llm_multimodal_fn = _gemini_call_multimodal

    else:
        raise ValueError(f"Unknown API: {api}. Use 'azure' or 'gemini'.")

    print(f"LLM: {api} / {MODEL}")


def call_llm(prompt: str, max_tokens: int = 64, retries: int = 12) -> str:
    global _llm_calls
    with _llm_lock:
        _llm_calls += 1
    for attempt in range(retries):
        try:
            return _llm_fn(prompt, max_tokens)
        except Exception as e:
            if "429" in str(e) or "RateLimit" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = min(2 ** attempt * 5 + random.uniform(0, 3), 120)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Rate limited after {retries} retries")


def call_llm_multimodal(parts: list, max_tokens: int = 64, retries: int = 12) -> str:
    """Call LLM with multimodal parts (text + images). Gemini only."""
    global _llm_calls
    if _llm_multimodal_fn is None:
        raise RuntimeError("Multimodal LLM not available (requires --api gemini)")
    with _llm_lock:
        _llm_calls += 1
    for attempt in range(retries):
        try:
            return _llm_multimodal_fn(parts, max_tokens)
        except Exception as e:
            if "429" in str(e) or "RateLimit" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = min(2 ** attempt * 5 + random.uniform(0, 3), 120)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Rate limited after {retries} retries")


# ── Cache loading ────────────────────────────────────────────
_cache = {}  # (profile, method) -> state


def load_cache(profile: str, method: str) -> dict | None:
    key = (profile, method)
    if key in _cache:
        return _cache[key]
    cache_method = CACHE_ALIAS.get(method, method)
    path = CACHE_DIR / profile / f"{cache_method}.pkl"
    if not path.exists():
        return None
    try:
        state = pickle.load(open(path, "rb"))
    except Exception as e:
        print(f"  [cache] Failed to load {path.name}: {e}")
        _cache[key] = None
        return None
    _cache[key] = state
    return state


# ── Context extraction per method ────────────────────────────

def _extract_full_context(state: dict, task_ids: set[str]) -> str:
    filtered = []
    for narr in state["narratives"]:
        for tid in task_ids:
            if f"Task: {tid}" in narr:
                filtered.append(narr)
                break
    return "\n\n".join(filtered)


def _iter_task_narratives(state: dict, task_ids: set[str]) -> list[tuple[str, str]]:
    """Return (task_id, narrative) pairs from a full_context-style cache."""
    pairs = []
    for narr in state.get("narratives", []):
        for tid in TASK_IDS:
            if tid in task_ids and f"Task: {tid}" in narr:
                pairs.append((tid, narr))
                break
    return pairs


def _event_lines(narrative: str) -> list[str]:
    """Extract compact event lines from a task narrative."""
    lines = []
    for line in narrative.splitlines():
        text = line.strip()
        if re.match(r"^\[\d+\]\s+", text):
            lines.append(re.sub(r"\s+", " ", text))
    return lines


def _extract_osworld_screen_buffer(state: dict, task_ids: set[str]) -> str:
    """OSWorld-style recent observation/action buffer.

    GUI agents usually keep a short within-task history of screen observations
    and actions for task execution.  This baseline therefore preserves only
    recent chronological observations/actions and does not aggregate
    cross-session behavioral statistics.
    """
    task_pairs = _iter_task_narratives(state, task_ids)
    # Recency-oriented: last tasks and last action observations per task.
    chunks = []
    for tid, narr in task_pairs[-8:]:
        events = _event_lines(narr)
        recent = events[-18:]
        chunks.append(f"[Recent screen/action buffer: {tid}]\n" + "\n".join(recent))
    return "\n\n".join(chunks)[-14000:]


_OFFICE_OP_RE = re.compile(
    r"(Read file|Wrote|Write|Created|Edited|Moved|Renamed|Deleted|Copied|"
    r"Search|Searched|Directory|Created directory|Cross-file ref|Context switch)",
    re.IGNORECASE,
)


def _extract_officebench_tool_history(state: dict, task_ids: set[str]) -> str:
    """OfficeBench-style tool/file-operation history.

    Office automation agents typically retain explicit tool-call and file
    context history inside a workflow.  This baseline keeps chronological
    file-operation traces and lightweight task separators, but performs no
    FileGramOS-style procedural aggregation or query-time abstraction.
    """
    chunks = []
    budget = 18000
    used = 0
    for tid, narr in _iter_task_narratives(state, task_ids):
        op_lines = []
        for line in _event_lines(narr):
            if _OFFICE_OP_RE.search(line):
                # Drop long content previews; keep the operation chain.
                line = re.sub(r"\s*\|\s*content:.*$", "", line)
                op_lines.append(line[:260])
        if not op_lines:
            continue
        chunk = f"[Tool/file history: {tid}]\n" + "\n".join(op_lines[:80])
        if used + len(chunk) > budget:
            remaining = budget - used
            if remaining > 500:
                chunks.append(chunk[:remaining])
            break
        chunks.append(chunk)
        used += len(chunk)
    return "\n\n".join(chunks)


def _extract_filegramos(state: dict, task_ids: set[str]) -> str:
    lines = []
    for i, feat in enumerate(state["features"]):
        tid = TASK_IDS[i] if i < len(TASK_IDS) else f"T-{i+1:02d}"
        if tid in task_ids:
            parts = [f"[Task {tid}]"]
            for k, v in feat.items():
                parts.append(f"  {k}: {v}")
            lines.append("\n".join(parts))
    # Also include semantic if available
    for i, sem in enumerate(state.get("semantic", [])):
        tid = TASK_IDS[i] if i < len(TASK_IDS) else f"T-{i+1:02d}"
        if tid in task_ids and sem:
            sem_str = json.dumps(sem, ensure_ascii=False)[:500] if isinstance(sem, dict) else str(sem)[:500]
            lines.append(f"[Task {tid} semantic] {sem_str}")
    return "\n\n".join(lines)


def _extract_eager_sum(state: dict, task_ids: set[str]) -> str:
    filtered = []
    for i, summ in enumerate(state["summaries"]):
        tid = TASK_IDS[i] if i < len(TASK_IDS) else f"T-{i+1:02d}"
        if tid in task_ids:
            filtered.append(f"[Task {tid}]\n{summ}")
    return "\n\n".join(filtered)


def _extract_naive_rag(state: dict, task_ids: set[str]) -> str:
    filtered = [c["text"] for c in state["chunks"] if c.get("task_id") in task_ids]
    return "\n---\n".join(filtered[:60])


def _extract_naive_rag_capped(state: dict, task_ids: set[str]) -> str:
    """Naive RAG with context budget capped to ~12K chars (matching other methods)."""
    filtered = [c["text"] for c in state["chunks"] if c.get("task_id") in task_ids]
    budget = 12000
    parts, used = [], 0
    for chunk_text in filtered:
        remaining = budget - used
        if remaining <= 100:
            break
        if len(chunk_text) > remaining:
            parts.append(chunk_text[:remaining])
            break
        parts.append(chunk_text)
        used += len(chunk_text) + 5
    return "\n---\n".join(parts)


def _extract_zep(state: dict, task_ids: set[str]) -> str:
    edges = state.get("graph_edges", [])
    filtered = []
    for e in edges:
        if e.get("task_id") in task_ids:
            filtered.append(
                f"{e['u']} --[{e['relation']}]--> {e['v']}"
                + (f" ({e['pattern']})" if e.get("pattern") else "")
            )
    return "\n".join(filtered[:200])


def _extract_memos(state: dict, task_ids: set[str]) -> str:
    filtered = [m["content"] for m in state["memories"] if m.get("task_id") in task_ids]
    return "\n".join(f"- {m}" for m in filtered[:120])


def _extract_memu(state: dict, task_ids: set[str]) -> str:
    filtered = [
        u["content"] for u in state["units"]
        if u.get("context", {}).get("task_id") in task_ids
    ]
    # Also include high-level items (no task_id, cross-task patterns)
    items = state.get("items", [])
    item_strs = [it["content"] for it in items[:20] if isinstance(it, dict)]
    parts = ["\n".join(f"- {u}" for u in filtered[:80])]
    if item_strs:
        parts.append("Cross-task patterns:\n" + "\n".join(f"- {s}" for s in item_strs))
    # Top-level behavioral categories (highest aggregation layer)
    categories = state.get("categories", [])
    if categories:
        cat_lines = []
        for cat in categories:
            name = cat.get("name", "?")
            summary = cat.get("summary", "")[:300]
            conf = cat.get("confidence", "?")
            n_items = len(cat.get("member_items", []))
            cat_lines.append(f"  [{name}] (confidence: {conf}, {n_items} items): {summary}")
        parts.append("Behavioral categories:\n" + "\n".join(cat_lines))
    return "\n\n".join(parts)


def _extract_evermemos(state: dict, task_ids: set[str]) -> str:
    """Extract context from EverMemOS adapter.

    pkl keys: episodes, profile_traits, clusters, consolidated_patterns,
    contradictions, memcells, engram_traces.
    """
    parts: list[str] = []

    # 1. Episodes filtered by task_id — full content, not just summary
    episodes = state.get("episodes", [])
    filtered_eps = [e for e in episodes if e.get("task_id") in task_ids]
    if not filtered_eps:
        filtered_eps = episodes  # fallback: use all if no task_id match
    if filtered_eps:
        ep_lines = []
        for ep in filtered_eps[:40]:
            tid = ep.get("task_id", "?")
            title = ep.get("title", "")
            content = ep.get("content", ep.get("summary", ""))[:400]
            ep_lines.append(f"  [{tid}] {title}\n    {content}")
        parts.append("Episodes:\n" + "\n".join(ep_lines))

    # 2. Consolidated patterns (correct key, was 'profile_patterns')
    patterns = state.get("consolidated_patterns", [])
    if patterns:
        pat_lines = []
        for pat in patterns:
            dim = pat.get("dimension", "?")
            desc = pat.get("pattern", "")
            conf = pat.get("confidence", "?")
            count = pat.get("evidence_count", 0)
            tasks = ", ".join(pat.get("evidence_tasks", [])[:8])
            details = pat.get("evidence_details", [])
            pat_lines.append(
                f"  [{dim}] {desc} (confidence: {conf}, evidence: {count}, tasks: {tasks})"
            )
            for d in details[:2]:
                pat_lines.append(f"    evidence: {str(d)[:200]}")
        parts.append("Consolidated Patterns:\n" + "\n".join(pat_lines))

    # 3. Profile traits filtered by task_id
    traits = state.get("profile_traits", [])
    filtered_traits = [t for t in traits if t.get("task_id") in task_ids]
    if not filtered_traits:
        filtered_traits = traits  # fallback
    if filtered_traits:
        trait_lines = []
        for t in filtered_traits[:60]:
            dim = t.get("dimension", "?")
            pattern = t.get("pattern", "")
            evidence = t.get("evidence", "")[:150]
            trait_lines.append(f"  [{dim}] {pattern}: {evidence}")
        parts.append("Profile Traits:\n" + "\n".join(trait_lines))

    # 4. Contradictions
    contradictions = state.get("contradictions", [])
    if contradictions:
        contra_lines = []
        for c in contradictions[:15]:
            dim = c.get("dimension", "?")
            desc = c.get("description", "")[:200]
            resolution = c.get("resolution", "")[:200]
            contra_lines.append(f"  [{dim}] {desc}\n    Resolution: {resolution}")
        parts.append("Contradictions:\n" + "\n".join(contra_lines))

    return "\n\n".join(parts)


def _extract_mem0(state: dict, task_ids: set[str]) -> str:
    """mem0 has no task_id — use all memories."""
    memories = state.get("memories", [])
    if memories and isinstance(memories[0], str):
        return "\n".join(f"- {m}" for m in memories[:120])
    return "\n".join(
        f"- {m.get('content', str(m))}" for m in memories[:120]
    )


def _extract_filegramos_formal(state: dict, task_ids: set[str]) -> str:
    """Extract context from FileGramOS formal adapter (engram-based).

    Combines structured three-channel retriever output with per-task raw
    procedural features for maximum signal preservation.

    Respects module-level _PS_* config for parameter sensitivity experiments.
    """
    store = state.get("store")
    if store is None:
        return _extract_filegramos(state, task_ids)  # fallback to simple

    from filegramos.retriever import QueryAdaptiveRetriever

    retriever = QueryAdaptiveRetriever()

    # Filter store to relevant tasks
    filtered = store.filter_by_tasks(task_ids)

    # Structured three-channel output
    structured = retriever.retrieve(filtered, query_type="mcq")

    # Per-task raw procedural features
    per_task_lines = ["### Per-Task Raw Features"]
    for eng in filtered.engrams:
        tid = eng.task_id
        parts = [f"[Task {tid}]"]
        for k, v in eng.procedural.items():
            parts.append(f"  {k}: {v}")
        # Include semantic info per task
        sem = eng.semantic
        if sem.created_filenames:
            parts.append(f"  created_filenames: {sem.created_filenames}")
        if _PS_INCLUDE_CONTENT_PREVIEWS and sem.created_files:
            previews = []
            for cf in sem.created_files[:_PS_FILES_PER_TASK]:
                preview = cf.content_preview.replace("\n", " ").strip()[:_PS_PREVIEW_CHARS] if cf.content_preview else ""
                previews.append(f"{cf.path}({cf.content_length}chars): {preview}")
            parts.append(f"  content_previews: {previews}")
        if _PS_INCLUDE_EDIT_DIFFS and sem.edit_chains:
            diffs = []
            for ec in sem.edit_chains[:3]:
                diff_text = (ec.diff_preview or "")[:_PS_PREVIEW_CHARS]
                diffs.append(f"{ec.path} (+{ec.lines_added}/-{ec.lines_deleted}): {diff_text}")
            parts.append(f"  edit_diffs: {diffs}")
        per_task_lines.append("\n".join(parts))

    return structured + "\n\n" + "\n\n".join(per_task_lines)


def _extract_filegramos_ablation(state: dict, task_ids: set[str], disabled_channels: set[str]) -> str:
    """Extract FileGramOS context with one channel disabled (for ablation A1)."""
    store = state.get("store")
    if store is None:
        return _extract_filegramos(state, task_ids)

    from filegramos.retriever import QueryAdaptiveRetriever

    retriever = QueryAdaptiveRetriever()
    filtered = store.filter_by_tasks(task_ids)
    structured = retriever.retrieve(filtered, query_type="mcq", disabled_channels=disabled_channels)

    # Per-task raw features: skip if procedural OR episodic channel is disabled.
    # Per-task features are individual session observations (episodic in nature);
    # without the episodic channel, only cross-session aggregates remain.
    if "procedural" not in disabled_channels and "episodic" not in disabled_channels:
        per_task_lines = ["### Per-Task Raw Features"]
        for eng in filtered.engrams:
            tid = eng.task_id
            parts = [f"[Task {tid}]"]
            for k, v in eng.procedural.items():
                parts.append(f"  {k}: {v}")
            # Semantic fields only when semantic channel is active
            if "semantic" not in disabled_channels:
                sem = eng.semantic
                if sem.created_filenames:
                    parts.append(f"  created_filenames: {sem.created_filenames}")
                if sem.created_files:
                    previews = []
                    for cf in sem.created_files[:3]:
                        preview = cf.content_preview.replace("\n", " ").strip()[:200] if cf.content_preview else ""
                        previews.append(f"{cf.path}({cf.content_length}chars): {preview}")
                    parts.append(f"  content_previews: {previews}")
            per_task_lines.append("\n".join(parts))
        return structured + "\n\n" + "\n\n".join(per_task_lines)

    return structured


def _extract_filegramos_no_proc(state: dict, task_ids: set[str]) -> str:
    return _extract_filegramos_ablation(state, task_ids, {"procedural"})


def _extract_filegramos_no_sem(state: dict, task_ids: set[str]) -> str:
    return _extract_filegramos_ablation(state, task_ids, {"semantic"})


def _extract_filegramos_no_epi(state: dict, task_ids: set[str]) -> str:
    return _extract_filegramos_ablation(state, task_ids, {"episodic"})


def _extract_mma(state: dict, task_ids: set[str]) -> str:
    """MMA: confidence-scored memory items with task filtering."""
    items = state.get("items", [])
    confidences = state.get("confidences", [])
    filtered = []
    for i, it in enumerate(items):
        if it.get("task_id") in task_ids:
            conf = confidences[i] if i < len(confidences) else 0.5
            filtered.append((conf, it))
    # Sort by confidence descending
    filtered.sort(key=lambda x: -x[0])
    lines = []
    for conf, it in filtered[:80]:
        text = it.get("text", "")[:400]
        lines.append(f"(conf={conf:.2f}, {it.get('task_id', '?')}) {text}")
    return "\n".join(f"- {l}" for l in lines)


def _extract_visrag(state: dict, task_ids: set[str]) -> str:
    """VisRAG: text chunks + visual doc items with task filtering."""
    text_items = state.get("text_items", [])
    visual_items = state.get("visual_items", [])
    lines = []
    # Text items
    for it in text_items:
        if it.get("task_id") in task_ids:
            lines.append(f"[TEXT, {it.get('task_id', '?')}] {it.get('text', '')[:400]}")
    # Visual items
    for it in visual_items:
        if it.get("task_id") in task_ids:
            fp = it.get("file_path", "?")
            lines.append(f"[VISUAL, {it.get('task_id', '?')}, {fp}] {it.get('text', '')[:400]}")
    # Truncate to reasonable size
    return "\n".join(f"- {l}" for l in lines[:80])


def _extract_simplemem(state: dict, task_ids: set[str]) -> str:
    entries = state.get("entries", [])
    filtered = [e for e in entries if e.get("task_id") in task_ids]
    # If no task_id filtering possible (consolidated entries lose task_id), use all
    if not filtered:
        filtered = entries
    lines = []
    for e in sorted(filtered, key=lambda x: -x.get("confidence", 0))[:60]:
        conf = e.get("confidence", 0.5)
        topic = e.get("topic", "?")
        obs = e.get("observation_count", 1)
        lines.append(f"({topic}, conf={conf:.1f}, obs={obs}) {e.get('lossless_restatement', '')}")
    return "\n".join(f"- {l}" for l in lines)


EXTRACTORS = {
    "full_context": _extract_full_context,
    "filegramos": _extract_filegramos_formal,
    "eager_summarization": _extract_eager_sum,
    "naive_rag": _extract_naive_rag,
    "naive_rag_capped": _extract_naive_rag_capped,
    "zep": _extract_zep,
    "memos": _extract_memos,
    "memu": _extract_memu,
    "evermemos": _extract_evermemos,
    "mem0": _extract_mem0,
    "mma": _extract_mma,
    "visrag": _extract_visrag,
    "simplemem": _extract_simplemem,
    "osworld_screen_buffer": _extract_osworld_screen_buffer,
    "officebench_tool_history": _extract_officebench_tool_history,
    "filegramos_no_proc": _extract_filegramos_no_proc,
    "filegramos_no_sem": _extract_filegramos_no_sem,
    "filegramos_no_epi": _extract_filegramos_no_epi,
}


# ── VisRAG multimodal helpers (render output docs as images) ──

_MAX_VISUAL_ITEMS_PER_PROFILE = 15  # cap images to avoid huge requests
_render_fn = None  # lazy-loaded


def _get_render_fn():
    """Lazy-load render_text_to_image from visrag adapter."""
    global _render_fn
    if _render_fn is None:
        from bench.baselines.visrag_adapter import render_text_to_image
        _render_fn = render_text_to_image
    return _render_fn


def _build_visrag_multimodal_parts(
    pairs: list[tuple[str, set[str]]], method: str = "visrag"
) -> tuple[list, str]:
    """Build multimodal parts list (str | bytes) from VisRAG cache.

    Returns (parts_list, text_context_for_logging).
    Text items → text strings; visual items → rendered PNG bytes.
    """
    render = _get_render_fn()
    parts: list = []
    text_log_parts: list[str] = []

    for i, (profile, task_ids) in enumerate(pairs):
        state = load_cache(profile, method)
        if state is None:
            parts.append("(no cache for profile)\n")
            text_log_parts.append("(no cache)")
            continue

        if len(pairs) > 1:
            header = f"\n--- User {i+1} ---\n"
            parts.append(header)
            text_log_parts.append(header)

        # Text items (behavioral events) → text
        text_items = state.get("text_items", [])
        text_lines = []
        for it in text_items:
            if it.get("task_id") in task_ids:
                line = f"[TEXT, {it.get('task_id', '?')}] {it.get('text', '')[:400]}"
                text_lines.append(line)
        if text_lines:
            chunk = "\n".join(text_lines[:60]) + "\n"
            parts.append(chunk)
            text_log_parts.append(chunk[:200] + "...")

        # Visual items (output docs) → rendered PNG images
        visual_items = state.get("visual_items", [])
        vis_count = 0
        for it in visual_items:
            if it.get("task_id") in task_ids:
                if vis_count >= _MAX_VISUAL_ITEMS_PER_PROFILE:
                    break
                fp = it.get("file_path", "?")
                caption = f"\n[VISUAL DOC, {it.get('task_id', '?')}, {fp}]:\n"
                parts.append(caption)
                img_bytes = render(it.get("text", ""))
                parts.append(img_bytes)  # bytes → will be sent as image
                text_log_parts.append(f"[IMG: {fp}]")
                vis_count += 1

    text_context = " ".join(text_log_parts)[:500]
    return parts, text_context


# ── Free-form evaluation helpers ─────────────────────────────

def _load_ground_truth(profile_id: str) -> dict[str, str]:
    """Load ground truth from profile YAML."""
    import yaml
    for profiles_dir in [ROOT / "profiles", ROOT / "filegram" / "profile" / "profiles"]:
        path = profiles_dir / f"{profile_id}.yaml"
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            basic = data.get("basic", {})
            personality = data.get("personality", {})
            wh = data.get("work_habits", {})
            return {
                "name": basic.get("name", ""),
                "role": basic.get("role", ""),
                "language": basic.get("language", ""),
                "tone": personality.get("tone", ""),
                "output_detail": personality.get("output_detail", personality.get("verbosity", "")),
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
    return {}


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response (handles markdown fences)."""
    text = text.strip()
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
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


_FREEFORM_PER_ATTR_PROMPT = """\
You are a strict evaluator for user profiling systems. You will score ONE \
specific attribute based on a free-form behavioral profile response.

The system was given behavioral trajectories and asked to freely describe the \
user's work habits — WITHOUT being told what attributes to look for. You must \
evaluate whether the free-form response accurately captures this specific attribute.

## Attribute: {attr_name}
{attr_description}

## Possible values
{value_space}

## Scoring rubric (1-5)
{rubric}

## VAGUENESS PENALTY
{vagueness_penalty}

## CRITICAL RULE
If the response is vague, generic, or could describe ANY user without specific \
behavioral evidence for this attribute, the MAXIMUM score is 3. Scores of 4-5 \
REQUIRE concrete, distinguishing behavioral descriptions. If this attribute is \
NOT mentioned at all in the response, score 1.

=== GROUND TRUTH ===
Attribute: {attr_name}
Value: {gt_value}

=== SYSTEM'S FREE-FORM RESPONSE ===
{response}

Score this single attribute. Respond ONLY in JSON:
{{"score": <1-5>, "justification": "<1-2 sentences>"}}"""


_FREEFORM_QUESTION = """\
Based on ALL the behavioral data above, write a comprehensive profile of this \
user's work habits and behavioral patterns.

You are NOT given a predefined list of attributes. Describe what you genuinely \
observe from the data — how this user reads files, creates content, organizes \
their workspace, iterates on work, manages their workflow rhythm, and uses \
(or avoids) visual/multimodal elements.

Be specific: cite concrete patterns you observed (e.g., "always creates README \
files", "uses grep before reading", "flat directory structure"). Avoid vague \
or generic statements.

Respond in plain text (not JSON). Write 300-600 words.
"""


def _load_attribute_rubrics() -> dict[str, dict[str, str]]:
    """Load per-attribute rubrics from judge_scoring.py."""
    try:
        from bench.evaluation.judge_scoring import ATTRIBUTE_RUBRICS
        return ATTRIBUTE_RUBRICS
    except ImportError:
        # Fallback: add bench/ to path
        _b = str(ROOT / "bench")
        if _b not in sys.path:
            sys.path.insert(0, _b)
        from evaluation.judge_scoring import ATTRIBUTE_RUBRICS
        return ATTRIBUTE_RUBRICS


_ATTRIBUTE_RUBRICS = None  # lazy loaded


def _get_rubrics() -> dict[str, dict[str, str]]:
    global _ATTRIBUTE_RUBRICS
    if _ATTRIBUTE_RUBRICS is None:
        _ATTRIBUTE_RUBRICS = _load_attribute_rubrics()
    return _ATTRIBUTE_RUBRICS


def _judge_one_attribute(
    attr: str, gt_value: str, response: str
) -> tuple[str, int, str]:
    """Judge a single attribute from a free-form response. Returns (attr, score, justification)."""
    rubrics = _get_rubrics()
    rubric_info = rubrics.get(attr, {})
    prompt = _FREEFORM_PER_ATTR_PROMPT.format(
        attr_name=attr,
        attr_description=rubric_info.get("description", attr),
        value_space=rubric_info.get("value_space", "See rubric"),
        rubric=rubric_info.get("rubric", "5=exact, 4=close, 3=partial, 2=wrong, 1=missing"),
        vagueness_penalty=rubric_info.get("vagueness_penalty", "Generic → max 3"),
        gt_value=gt_value,
        response=response,
    )
    try:
        resp = call_llm(prompt, max_tokens=256)
        parsed = _parse_json_response(resp)
        score = parsed.get("score", 0)
        if not isinstance(score, (int, float)) or score < 1 or score > 5:
            score = 0
        return attr, int(score), parsed.get("justification", "")
    except Exception as e:
        return attr, 0, f"error: {e}"


def _run_freeform_one(
    profile_id: str, method: str, cache_dir: Path
) -> dict | None:
    """Run free-form profile reconstruction + per-attribute judge.

    Steps: load pkl → build context → ask open-ended question →
           16 parallel per-attribute judge calls → aggregate scores.
    Results are cached in cache_dir to avoid re-running.
    """
    from concurrent.futures import ThreadPoolExecutor

    # Check for cached result
    cache_file = cache_dir / f"{profile_id}_{method}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        if "mean_score" in data:
            return data

    # Skip baselines that have no memory
    if method in ("random", "no_context"):
        return None

    # Load pkl cache
    state = load_cache(profile_id, method)
    if state is None:
        return None

    # Build context from ALL trajectories (use all task IDs)
    extractor = EXTRACTORS.get(method)
    if not extractor:
        return None
    all_task_ids = set(TASK_IDS)

    use_multimodal = (method == "visrag" and _llm_multimodal_fn is not None)

    try:
        context = extractor(state, all_task_ids)
    except Exception as e:
        print(f"  [{profile_id}][{method}] context extraction error: {e}")
        return None

    if not context or len(context.strip()) < 50:
        return None

    # Step 1: Ask open-ended question
    try:
        if use_multimodal:
            pairs = [(profile_id, all_task_ids)]
            mm_parts, _ = _build_visrag_multimodal_parts(pairs, method)
            mm_parts = [
                "You are analyzing behavioral trajectories from a computer user's "
                "file-system activity logs.\n\nMemory context:\n"
            ] + mm_parts + [f"\n\n{_FREEFORM_QUESTION}"]
            response = call_llm_multimodal(mm_parts, max_tokens=4096)
        else:
            ask_prompt = (
                "You are analyzing behavioral trajectories from a computer user's "
                "file-system activity logs.\n\n"
                f"Memory context:\n{context}\n\n{_FREEFORM_QUESTION}"
            )
            response = call_llm(ask_prompt, max_tokens=4096)
    except Exception as e:
        print(f"  [{profile_id}][{method}] freeform response error: {e}")
        return None

    # Step 2: Per-attribute judge (16 parallel calls)
    gt = _load_ground_truth(profile_id)
    if not gt:
        return None

    per_attr_scores = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_judge_one_attribute, attr, gt.get(attr, "N/A"), response): attr
            for attr in PROFILE_ATTRIBUTES
        }
        for fut in futures:
            attr, score, justification = fut.result()
            per_attr_scores[attr] = {"score": score, "justification": justification}

    valid = [v["score"] for v in per_attr_scores.values() if v["score"] > 0]
    mean_score = round(sum(valid) / len(valid), 3) if valid else 0.0

    result = {
        "profile_id": profile_id,
        "method": method,
        "mean_score": mean_score,
        "per_attribute": per_attr_scores,
        "n_valid": len(valid),
        "response_preview": response[:500],
    }
    # Cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


# ── Question helpers ─────────────────────────────────────────

def get_profiles_and_tasks(q: dict) -> list[tuple[str, set[str]]]:
    """Extract (profile, {task_ids}) pairs needed for a question."""
    results = defaultdict(set)

    # S1: direct input_trajectories
    for t in q.get("input_trajectories", []):
        parts = t.rsplit("_T-", 1)
        if len(parts) == 2:
            results[parts[0]].add(f"T-{parts[1]}")

    # S2/S3: mixing_spec / stitch_spec
    spec = q.get("mixing_spec", q.get("stitch_spec", {}))
    for v in spec.values():
        if isinstance(v, dict):
            pid = v.get("profile_id", "")
            if "dir_name" in v:
                p = v["dir_name"].rsplit("_T-", 1)
                if len(p) == 2:
                    results[pid or p[0]].add(f"T-{p[1]}")
            for t in v.get("trajectories", []):
                p = t.rsplit("_T-", 1)
                if len(p) == 2:
                    results[pid or p[0]].add(f"T-{p[1]}")

    return list(results.items())


def is_runnable(q: dict, cached_profiles: set[str]) -> bool:
    """Check if all profiles needed for this question have caches."""
    pairs = get_profiles_and_tasks(q)
    return all(p in cached_profiles for p, _ in pairs)


def build_context(method: str, q: dict) -> str:
    """Build memory context for a question from cached method data."""
    pairs = get_profiles_and_tasks(q)
    extractor = EXTRACTORS[method]
    parts = []
    for i, (profile, task_ids) in enumerate(pairs):
        state = load_cache(profile, method)
        if state is None:
            parts.append(f"(no cache for profile)")
            continue
        ctx = extractor(state, task_ids)
        if len(pairs) > 1:
            # Anonymize profile names to prevent data leakage
            parts.append(f"--- User {i+1} ---\n{ctx}")
        else:
            parts.append(ctx)
    return "\n\n".join(parts)


def format_question(q: dict) -> str:
    lines = [q["question"], ""]
    choices = q.get("choices") or {}
    if not choices:
        return q["question"] + "\n(No choices available)"
    for label, text in sorted(choices.items()):
        lines.append(f"  {label}) {text}")
    lines.append("\nAnswer with ONLY the letter (e.g., A, B, C, D, E, or F).")
    return "\n".join(lines)


# ── Evaluate one question ────────────────────────────────────

def evaluate_one(method: str, q: dict, dry: bool = False) -> dict:
    """Evaluate a single question with a given method."""
    t0 = time.time()

    if not q.get("choices"):
        return {
            "question_id": q.get("question_id", ""),
            "method": method,
            "track": q.get("track", ""),
            "sub_track": q.get("sub_track", ""),
            "sub_class": q.get("sub_class", ""),
            "channel": q.get("channel", ""),
            "predicted": "?",
            "correct": q.get("correct", "?"),
            "match": False,
            "skipped": True,
            "reason": "no_choices",
            "latency_ms": 0,
        }, ""

    if method == "random":
        predicted = random.choice(list(q["choices"].keys()))
        return {
            "question_id": q.get("question_id", ""),
            "method": method,
            "track": q.get("track", ""),
            "sub_track": q.get("sub_track", ""),
            "sub_class": q.get("sub_class", ""),
            "channel": q.get("channel", ""),
            "predicted": predicted,
            "correct": q["correct"],
            "match": predicted == q["correct"],
            "context_length": 0,
            "latency_ms": 0,
        }, ""

    question_text = format_question(q)

    # ── VisRAG multimodal path: pass rendered images to Gemini VLM ──
    use_multimodal = (method == "visrag" and _llm_multimodal_fn is not None)

    if method == "no_context":
        context = ""
        prompt = (
            "You are answering a behavioral analysis question about a computer user's "
            "work habits. You have NO information about this user. "
            "Pick the most likely answer.\n\n"
            f"{question_text}"
        )
    elif use_multimodal:
        pairs = get_profiles_and_tasks(q)
        mm_parts, context = _build_visrag_multimodal_parts(pairs, method)
        # Prepend system instruction, append question
        mm_parts = [
            "You are answering a behavioral analysis question about a computer user's "
            "work habits based on their file-system activity logs.\n\n"
            "Memory context:\n"
        ] + mm_parts + [f"\n\n{question_text}"]
        prompt = f"(multimodal: {len([p for p in mm_parts if isinstance(p, bytes)])} images)"
    else:
        context = build_context(method, q)
        prompt = (
            "You are answering a behavioral analysis question about a computer user's "
            "work habits based on their file-system activity logs.\n\n"
            f"Memory context:\n{context}\n\n{question_text}"
        )

    if dry:
        return {
            "question_id": q.get("question_id", ""),
            "method": method,
            "track": q.get("track", ""),
            "sub_track": q.get("sub_track", ""),
            "sub_class": q.get("sub_class", ""),
            "channel": q.get("channel", ""),
            "predicted": "?",
            "correct": q["correct"],
            "match": False,
            "context_length": len(context),
            "prompt_length": len(prompt),
            "latency_ms": 0,
        }, context

    try:
        if use_multimodal:
            response = call_llm_multimodal(mm_parts, max_tokens=16)
        else:
            response = call_llm(prompt, max_tokens=16)
        predicted = "?"
        for ch in response.strip().upper():
            if ch in q["choices"]:
                predicted = ch
                break
        raw_response = response.strip()
    except Exception as e:
        print(f"  LLM error ({method}, {q.get('question_id','')}): {e}")
        predicted = "?"
        raw_response = f"ERROR: {e}"

    return {
        "question_id": q.get("question_id", ""),
        "method": method,
        "track": q.get("track", ""),
        "sub_track": q.get("sub_track", ""),
        "sub_class": q.get("sub_class", ""),
        "channel": q.get("channel", ""),
        "predicted": predicted,
        "correct": q["correct"],
        "match": predicted == q["correct"],
        "raw_response": raw_response,
        "context_length": len(context),
        "prompt_length": len(prompt) if isinstance(prompt, str) else 0,
        "latency_ms": int((time.time() - t0) * 1000),
    }, context


# ── Save helpers ─────────────────────────────────────────────
_save_lock = Lock()


def save_method_result(method: str, results: list[dict], run_meta: dict, out_dir: Path = None):
    """Save per-method intermediate results."""
    out = out_dir or RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    by_sub_track = defaultdict(list)
    by_sub_class = defaultdict(list)
    for r in results:
        by_sub_track[r.get("sub_track", "?")].append(r)
        sc = r.get("sub_class", "") or r.get("sub_type", "?")
        by_sub_class[sc].append(r)

    summary = {
        "method": method,
        "model": run_meta["model"],
        "timestamp": run_meta["timestamp"],
        "total_questions": len(results),
        "overall_accuracy": sum(r["match"] for r in results) / max(len(results), 1),
        "by_sub_track": {
            st: {
                "count": len(rs),
                "accuracy": sum(r["match"] for r in rs) / max(len(rs), 1),
            }
            for st, rs in by_sub_track.items()
        },
        "by_sub_class": {
            sc: {
                "count": len(rs),
                "accuracy": sum(r["match"] for r in rs) / max(len(rs), 1),
            }
            for sc, rs in by_sub_class.items()
        },
        "questions": results,
    }

    suffix = _PS_RESULT_SUFFIX
    fname = f"{method}{suffix}.json" if suffix else f"{method}.json"
    path = out / fname
    with _save_lock:
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  [saved] {path.name} ({len(results)} questions, acc={summary['overall_accuracy']:.1%})")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run QA evaluation")
    parser.add_argument("--methods", nargs="+", default=METHODS + ["no_context", "random"])
    parser.add_argument("--settings", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--parallel", type=int, default=3, help="concurrent LLM calls")
    parser.add_argument("--dry", action="store_true", help="preview without LLM calls")
    parser.add_argument("--max-per-subtype", type=int, default=0, help="cap questions per sub-type (0=all)")
    parser.add_argument("--resume", action="store_true", help="skip questions already in results/")
    parser.add_argument("--api", choices=["azure", "gemini"], default="azure", help="LLM API to use")
    parser.add_argument("--questions-dir", type=str, default=None,
                        help="Custom questions directory (default: filegramQA/questions/). "
                             "Use 'v2' as shorthand for questions_v2/")
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Ingest cache directory (REQUIRED). Use model slug like 'gpt4.1', "
                             "'gemini_2.5_pro' (maps to bench/ingest_cache_{slug}/). "
                             "Or a full/relative path.")
    parser.add_argument("--mode", choices=["qa", "openended", "all"], default="all",
                        help="What to run: 'qa' (MCQ only), 'openended' (free-form ProfRec only), "
                             "'all' (both). Default: all.")
    # Parameter sensitivity experiment overrides
    parser.add_argument("--display-chars", type=int, default=None,
                        help="Override RETRIEVER_DISPLAY_CHARS (default: use tuning.py)")
    parser.add_argument("--preview-chars", type=int, default=200,
                        help="Per-task content preview char limit (default: 200)")
    parser.add_argument("--files-per-task", type=int, default=3,
                        help="Max created files shown per task (default: 3)")
    parser.add_argument("--edit-diffs", action="store_true",
                        help="Include edit chain diffs in per-task context")
    parser.add_argument("--no-content-previews", action="store_true",
                        help="Exclude content previews from per-task context")
    parser.add_argument("--result-suffix", type=str, default="",
                        help="Suffix appended to result filenames for experiment tracking")
    args = parser.parse_args()

    # Apply parameter sensitivity overrides
    global _PS_DISPLAY_CHARS, _PS_PREVIEW_CHARS, _PS_FILES_PER_TASK
    global _PS_INCLUDE_EDIT_DIFFS, _PS_INCLUDE_CONTENT_PREVIEWS, _PS_RESULT_SUFFIX
    _PS_PREVIEW_CHARS = args.preview_chars
    _PS_FILES_PER_TASK = args.files_per_task
    _PS_INCLUDE_EDIT_DIFFS = args.edit_diffs
    _PS_INCLUDE_CONTENT_PREVIEWS = not args.no_content_previews
    _PS_RESULT_SUFFIX = args.result_suffix

    if args.display_chars is not None:
        _PS_DISPLAY_CHARS = args.display_chars
        # Monkey-patch retriever module to use overridden display chars
        import filegramos.retriever as _ret_mod
        _ret_mod.RETRIEVER_DISPLAY_CHARS = args.display_chars
        print(f"[param-sensitivity] RETRIEVER_DISPLAY_CHARS = {args.display_chars}")

    init_llm_client(args.api)

    # Resolve cache directory (required — independent of --api)
    global CACHE_DIR
    p = Path(args.cache_dir)
    if p.is_absolute() or p.exists():
        CACHE_DIR = p
    else:
        # Treat as model slug: "gpt4.1" -> bench/ingest_cache_gpt4.1/
        CACHE_DIR = ROOT / "bench" / f"ingest_cache_{args.cache_dir}"

    # Resolve questions directory
    if args.questions_dir in ("v2", "v3"):
        q_dir = Path(__file__).resolve().parent / f"questions_{args.questions_dir}"
    elif args.questions_dir:
        q_dir = Path(args.questions_dir)
    else:
        q_dir = QUESTIONS_DIR

    # Discover cached profiles
    cached_profiles = {
        d for d in os.listdir(CACHE_DIR) if (CACHE_DIR / d).is_dir() and d != ".DS_Store"
    }
    print(f"Cached profiles: {sorted(cached_profiles)} ({len(cached_profiles)})")
    print(f"Questions dir: {q_dir}")

    # Use separate results dir for v2 questions
    results_dir = RESULTS_DIR
    if args.questions_dir in ("v2", "v3") or (args.questions_dir and ("v2" in str(args.questions_dir) or "v3" in str(args.questions_dir))):
        ver = "v3" if (args.questions_dir == "v3" or (args.questions_dir and "v3" in str(args.questions_dir))) else "v2"
        results_dir = Path(__file__).resolve().parent / f"results_{ver}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load questions — support both v1 and v2 naming conventions
    setting_files_v1 = {
        1: "setting1_basic.json",
        2: "setting2_cross_task.json",
        3: "setting3_profile_shift.json",
        4: "setting4_multimodal.json",
    }
    setting_files_v2 = {
        1: "setting_basic.json",
        2: "setting_cross_task.json",
        3: "setting_profile_shift.json",
        4: "setting_multimodal.json",
    }
    all_questions = []
    for s in args.settings:
        # Try v2 naming first, then v1
        qf = q_dir / setting_files_v2.get(s, "")
        if not qf.exists():
            qf = q_dir / setting_files_v1.get(s, "")
        if not qf.exists():
            print(f"Setting {s}: no question file found in {q_dir}")
            continue
        qs = json.loads(qf.read_text())
        runnable = [q for q in qs if is_runnable(q, cached_profiles)]
        print(f"Setting {s}: {len(runnable)}/{len(qs)} runnable")

        if args.max_per_subtype > 0:
            by_sub = defaultdict(list)
            for q in runnable:
                by_sub[q.get("sub_class", q.get("sub_type", ""))].append(q)
            runnable = []
            rng = random.Random(42)
            for st, sq in by_sub.items():
                runnable.extend(rng.sample(sq, min(args.max_per_subtype, len(sq))))

        all_questions.extend(runnable)

    run_qa = args.mode in ("qa", "all")
    run_openended = args.mode in ("openended", "all")

    methods = [m for m in args.methods if m in ("random", "no_context") or m in METHODS]

    if run_qa and not all_questions:
        print("No runnable MCQ questions.")
        if not run_openended:
            print("Exiting.")
            return

    run_meta = {
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "cache_dir": str(CACHE_DIR),
        "cached_profiles": sorted(cached_profiles),
        "settings": args.settings,
        "methods": methods,
        "total_questions": len(all_questions) if run_qa else 0,
        "parallel": args.parallel,
        "mode": args.mode,
    }

    all_results = {}

    # ── MCQ evaluation ────────────────────────────────────────
    if run_qa and all_questions:
        total_calls = len(all_questions) * len([m for m in methods if m != "random"])
        print(f"\nMCQ: {len(all_questions)} questions × {len(methods)} methods = {total_calls} LLM calls")
        if args.dry:
            print("(DRY RUN — no LLM calls)")

        # Resume: load existing results
        _prior = {}  # method -> {question_id: result_dict}
        if args.resume:
            for method in methods:
                suffix = _PS_RESULT_SUFFIX
                fname = f"{method}{suffix}.json" if suffix else f"{method}.json"
                path = results_dir / fname
                if path.exists():
                    prev = json.loads(path.read_text())
                    done = {r["question_id"]: r for r in prev.get("questions", [])}
                    _prior[method] = done
            total_skipped = sum(len(v) for v in _prior.values())
            if total_skipped:
                print(f"Resuming: {total_skipped} question-method pairs already done, will skip")

        # Run evaluation: all methods in parallel, each with its own tqdm bar
        _results_lock = Lock()
        _method_data = {}  # method -> {"results": [], "correct": 0, "pbar": tqdm}

        # Create all progress bars (one per method, stacked)
        for mi, method in enumerate(methods):
            prior = _prior.get(method, {})
            pbar = tqdm(total=len(all_questions),
                        desc=f"{method:<20s}",
                        initial=len(prior),
                        position=mi,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] acc={postfix}",
                        leave=True)
            prior_results = list(prior.values())
            prior_correct = sum(r["match"] for r in prior_results)
            if prior_results:
                pbar.set_postfix_str(f"{prior_correct}/{len(prior_results)}={prior_correct/len(prior_results):.1%}")
            else:
                pbar.set_postfix_str("--")
            _method_data[method] = {
                "results": prior_results,
                "correct": prior_correct,
                "contexts": {},  # question_id -> context string
                "pbar": pbar,
                "done_ids": set(prior.keys()),
            }

        def run_one(method: str, q: dict) -> dict | None:
            qid = q.get("question_id", "")
            md = _method_data[method]
            if qid in md["done_ids"]:
                return None  # already done
            r, ctx = evaluate_one(method, q, dry=args.dry)
            with _results_lock:
                md["results"].append(r)
                md["correct"] += r["match"]
                md["contexts"][qid] = ctx
                n = len(md["results"])
                md["pbar"].set_postfix_str(f"{md['correct']}/{n}={md['correct']/n:.1%}")
                md["pbar"].update(1)
                # Checkpoint every 50 questions
                if n % 50 == 0:
                    save_method_result(method, list(md["results"]), run_meta, results_dir)
            return r

        # Submit all (method × question) pairs in round-robin order
        n_workers = max(len(methods), args.parallel)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = []
            for qi in range(len(all_questions)):
                for method in methods:
                    futures.append(pool.submit(run_one, method, all_questions[qi]))
            for fut in as_completed(futures):
                fut.result()  # propagate exceptions

        # Close bars and collect results
        print()  # newline after tqdm bars
        contexts_dir = results_dir / "contexts"
        contexts_dir.mkdir(parents=True, exist_ok=True)
        for method in methods:
            md = _method_data[method]
            md["pbar"].close()
            save_method_result(method, md["results"], run_meta, results_dir)
            all_results[method] = md["results"]
            # Save full contexts for qualitative analysis
            if md["contexts"]:
                ctx_path = contexts_dir / f"{method}.json"
                ctx_path.write_text(json.dumps(md["contexts"], indent=2, ensure_ascii=False))

    # ── Free-form evaluation (from pkl caches, at eval time) ──
    freeform_scores = defaultdict(list)  # method -> list of {score, ...}
    if run_openended and not args.dry:
        # Build (profile, method) pairs from cached profiles × methods
        ff_pairs = []
        for pid in sorted(cached_profiles):
            for method in methods:
                if method in ("random", "no_context"):
                    continue
                ff_pairs.append((pid, method))

        if ff_pairs:
            freeform_cache_dir = results_dir / "freeform_cache"
            freeform_cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nFree-form evaluation: {len(ff_pairs)} (profile, method) pairs")

            ff_pbar = tqdm(total=len(ff_pairs), desc="Free-form eval    ", ncols=90)

            def _run_ff(pair):
                pid, m = pair
                result = _run_freeform_one(pid, m, freeform_cache_dir)
                ff_pbar.update(1)
                return pid, m, result

            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                for pid, m, result in pool.map(_run_ff, ff_pairs):
                    if result:
                        freeform_scores[m].append(result)

            ff_pbar.close()
            print(f"Free-form eval done: {sum(len(v) for v in freeform_scores.values())} results")

    # ── Build unified results ─────────────────────────────────
    # method -> (track, sub_track) -> list of {match: 0|1 or float}
    # method -> channel -> list of results
    by_subtype = defaultdict(lambda: defaultdict(list))
    by_channel = defaultdict(lambda: defaultdict(list))

    for method in methods:
        for r in all_results.get(method, []):
            # Use new taxonomy fields if available, fallback to TRACK_MAP/CHANNEL_MAP
            track = r.get("track", "")
            sub_track = r.get("sub_track", "")
            channel = r.get("channel", "")
            if track and sub_track:
                by_subtype[method][(track, sub_track)].append(r)
            else:
                # Fallback for old cached results without new fields
                old_sc = r.get("sub_class", "") or r.get("sub_type", "")
                mapping = TRACK_MAP.get(old_sc)
                if mapping:
                    by_subtype[method][mapping].append(r)
                channel = CHANNEL_MAP.get(old_sc, "")
            if channel:
                by_channel[method][channel].append(r)

        # Add free-form as T1.Profile Reconstruction
        ff_list = freeform_scores.get(method, [])
        if ff_list:
            # Normalize Likert 1-5 → 0-1 for "match" compatibility: (mean-1)/4
            ff_pseudo = [{"match": (x["mean_score"] - 1) / 4} for x in ff_list]
            by_subtype[method][("T1", "Profile Reconstruction")] = ff_pseudo
            by_channel[method]["semantic"].extend(ff_pseudo)

    # ── Precompute per-method free-form aggregates ──────────────
    ff_agg = {}  # method -> {mean, n, per_attr}
    for method, ff_list in freeform_scores.items():
        if ff_list:
            mean = sum(x["mean_score"] for x in ff_list) / len(ff_list)
            # Aggregate per-attribute across profiles
            per_attr_means = {}
            for attr in PROFILE_ATTRIBUTES:
                attr_scores = [
                    x["per_attribute"][attr]["score"]
                    for x in ff_list
                    if attr in x.get("per_attribute", {}) and x["per_attribute"][attr]["score"] > 0
                ]
                if attr_scores:
                    per_attr_means[attr] = round(sum(attr_scores) / len(attr_scores), 2)
            ff_agg[method] = {"mean": round(mean, 3), "n": len(ff_list), "per_attr": per_attr_means}

    # ── Precompute sub_class → sub_track mapping ───────────────
    SUB_CLASS_TO_SUBTRACK = defaultdict(list)
    for sc, (track, st) in TRACK_MAP.items():
        SUB_CLASS_TO_SUBTRACK[(track, st)].append(sc)

    # ── Build per-method, per-sub_class results (for component table) ──
    by_sub_class = defaultdict(lambda: defaultdict(list))  # method -> sub_class -> [results]
    for method in methods:
        for r in all_results.get(method, []):
            sc = r.get("sub_class", "") or r.get("sub_type", "")
            if sc:
                by_sub_class[method][sc].append(r)

    # ── Main table ────────────────────────────────────────────
    # Columns: 7 sub-type + 3 channel + 1 overall = 11 data cols
    # ProfRec shows Likert mean (1-5 scale)
    # T4:VisGrd is a placeholder (no data yet)
    print(f"\n{'='*140}")
    print(f"RESULTS SUMMARY  (model={MODEL})")
    print(f"{'='*140}")

    cw = 9   # normal column width

    # Row 1: Track group headers
    row1 = f"{'':22}"
    row1 += f"{'T1:Understanding':^{cw * 2}}"
    row1 += f"{'T2:Reasoning':^{cw * 2}}"
    row1 += f"{'T3:Detection':^{cw * 2}}"
    row1 += f"{'T4':^{cw}}"
    row1 += f"   {'Channel':^{cw * 3}}"
    print(row1)

    # Row 2: Sub-type / channel headers
    row2 = f"{'Method':<22}"
    row2 += f"{'AttrRec':>{cw}}"
    row2 += f"{'ProfRec':>{cw}}"
    row2 += f"{'BehavInf':>{cw}}"
    row2 += f"{'TraceDis':>{cw}}"
    row2 += f"{'AnomDet':>{cw}}"
    row2 += f"{'ShiftAna':>{cw}}"
    row2 += f"{'VisGrd':>{cw}}"
    row2 += " | "
    for ch in CHANNEL_ORDER:
        row2 += f"{CHANNEL_SHORT[ch]:>{cw}}"
    row2 += " | "
    row2 += f"{'Avg':>{cw}}"
    print(row2)
    print("-" * len(row2))

    summary_data = {}
    for method in methods:
        row = f"{method:<22}"
        method_scores = {}
        all_flat = []

        # 7 sub-type columns (T1-T3 from MCQ/freeform, T4 placeholder)
        for track, st in SUBTYPE_ORDER:
            if st == "Profile Reconstruction":
                # ProfRec: show Likert mean (1-5 scale)
                agg = ff_agg.get(method)
                if agg:
                    row += f"{agg['mean']:>{cw - 3}.2f}/5"
                    method_scores[f"{track}_{st}"] = agg["mean"]
                    # Add pseudo-results for overall calculation
                    ff_pseudo = [{"match": (x["mean_score"] - 1) / 4} for x in freeform_scores[method]]
                    all_flat.extend(ff_pseudo)
                else:
                    row += f"{'—':>{cw}}"
            elif st == "visual_grounding":
                # T4: placeholder — no data yet
                row += f"{'—':>{cw}}"
            else:
                rs = by_subtype[method].get((track, st), [])
                if rs:
                    acc = sum(r["match"] for r in rs) / len(rs) * 100
                    row += f"{acc:>{cw - 2}.1f}% "
                    method_scores[f"{track}_{st}"] = round(acc, 1)
                    all_flat.extend(rs)
                else:
                    row += f"{'—':>{cw}}"

        row += " | "

        # 3 channel columns
        for ch in CHANNEL_ORDER:
            rs = by_channel[method].get(ch, [])
            # Add free-form to semantic channel
            if ch == "semantic" and method in freeform_scores:
                rs = list(rs) + [{"match": (x["mean_score"] - 1) / 4} for x in freeform_scores[method]]
            if rs:
                acc = sum(r["match"] for r in rs) / len(rs) * 100
                row += f"{acc:>{cw - 2}.1f}% "
                method_scores[ch] = round(acc, 1)
            else:
                row += f"{'—':>{cw}}"

        row += " | "

        # Overall
        if all_flat:
            overall = sum(r["match"] for r in all_flat) / len(all_flat) * 100
            row += f"{overall:>{cw - 2}.1f}% "
            method_scores["overall"] = round(overall, 1)
        else:
            row += f"{'—':>{cw}}"

        summary_data[method] = method_scores
        print(row)

    # ── Component breakdown (all in one block) ────────────────
    print(f"\n{'='*100}")
    print(f"COMPONENT BREAKDOWN")
    print(f"{'='*100}")

    for track, sub_track in SUBTYPE_ORDER:
        if sub_track == "Profile Reconstruction":
            # Show per-attribute Likert breakdown
            if not ff_agg:
                continue
            print(f"\n  T1.ProfRec — Per-Attribute Likert (1-5)")
            # Show procedural attrs then semantic attrs
            proc_attrs = [a for a in PROFILE_ATTRIBUTES if CHANNEL_MAP.get("dimension_classification") == "procedural"
                          or a in ("working_style","thoroughness","error_handling","reading_strategy",
                                   "directory_style","edit_strategy","version_strategy","output_detail")]
            sem_attrs = [a for a in PROFILE_ATTRIBUTES if a not in proc_attrs]
            for label, attrs in [("Procedural", proc_attrs), ("Semantic", sem_attrs)]:
                short_attrs = [a[:8] for a in attrs]
                print(f"\n    {label} attributes:")
                header = f"    {'Method':<22}" + "".join(f"{s:>10}" for s in short_attrs) + f"{'Mean':>8}"
                print(header)
                print(f"    {'-' * (len(header) - 4)}")
                for method in methods:
                    agg = ff_agg.get(method)
                    if not agg:
                        continue
                    row = f"    {method:<22}"
                    vals = []
                    for attr in attrs:
                        v = agg["per_attr"].get(attr)
                        if v is not None:
                            row += f"{v:>10.2f}"
                            vals.append(v)
                        else:
                            row += f"{'—':>10}"
                    if vals:
                        row += f"{sum(vals)/len(vals):>8.2f}"
                    print(row)
            # Overall summary
            print(f"\n    {'Method':<22} {'Mean(/5)':>9} {'N':>5}")
            print(f"    {'-'*38}")
            for method in methods:
                agg = ff_agg.get(method)
                if agg:
                    print(f"    {method:<22} {agg['mean']:>7.3f} {agg['n']:>5}")
            continue

        sub_classes = SUB_CLASS_TO_SUBTRACK.get((track, sub_track), [])
        if not sub_classes:
            continue
        print(f"\n  {track}.{SUBTYPE_SHORT.get(sub_track, sub_track)}:")
        st_header = f"  {'Method':<22}" + "".join(
            f"{SUB_CLASS_SHORT.get(sc, sc):>10}" for sc in sub_classes
        )
        print(st_header)
        print(f"  {'-' * (len(st_header) - 2)}")
        for method in methods:
            row = f"  {method:<22}"
            for sc in sub_classes:
                rs = by_sub_class[method].get(sc, [])
                if rs:
                    acc = sum(r["match"] for r in rs) / len(rs) * 100
                    row += f"{acc:>8.1f}% "
                else:
                    row += f"{'—':>10}"
            print(row)

    # ── Save summary ──────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    final = {
        "metadata": run_meta,
        "summary": summary_data,
        "by_sub_track": {
            method: {
                f"{t}_{SUBTYPE_SHORT.get(st, st)}": {
                    "count": len(rs),
                    "accuracy": round(sum(r["match"] for r in rs) / max(len(rs), 1) * 100, 1),
                }
                for (t, st), rs in by_subtype[method].items()
            }
            for method in methods
        },
        "freeform_detail": {
            method: [
                {"profile_id": x["profile_id"], "mean_score": x["mean_score"],
                 "n_valid": x["n_valid"]}
                for x in scores
            ]
            for method, scores in freeform_scores.items()
        },
        "total_llm_calls": _llm_calls,
    }
    suffix = _PS_RESULT_SUFFIX
    summary_fname = f"summary{suffix}.json" if suffix else "summary.json"
    (results_dir / summary_fname).write_text(
        json.dumps(final, indent=2, ensure_ascii=False)
    )
    print(f"\nResults saved to {results_dir}/{summary_fname}")
    print(f"Total LLM calls: {_llm_calls}")


if __name__ == "__main__":
    main()
