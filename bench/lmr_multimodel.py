#!/usr/bin/env python3
"""Multi-Model L/M/R Classification with Inter-Annotator Agreement.

Uses multiple LLMs (GPT-4.1-mini, Gemini 2.5 Flash, Claude Haiku 4.5)
as independent classifiers to:
1. Classify profiles on 6 L/M/R dimensions from behavioral summaries
2. Measure inter-model agreement (Cohen's κ, Fleiss' κ)
3. Compare with rule-based baseline from lmr_validation.py

Usage:
    python bench/lmr_multimodel.py --models azure gemini anthropic --parallel 3
    python bench/lmr_multimodel.py --dry           # print one prompt, no API calls
    python bench/lmr_multimodel.py --resume         # skip completed
    python bench/lmr_multimodel.py --profiles p1,p3 # subset
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

_BENCH = Path(__file__).resolve().parent
_ROOT = _BENCH.parent
sys.path.insert(0, str(_BENCH))

load_dotenv(_ROOT / ".env")

SIGNAL_DIR = _ROOT / "signal"
MATRIX_PATH = _ROOT / "profiles" / "profile_matrix.json"
RESULTS_PATH = _BENCH / "lmr_multimodel_results.json"

# ── Ground truth ──────────────────────────────────────────────
with open(MATRIX_PATH) as f:
    _matrix = json.load(f)
GT = {pid: info["dimensions"] for pid, info in _matrix["profiles"].items()}
DIM_LABELS = _matrix["dimensions"]
DIMS = ["A", "B", "C", "D", "E", "F"]
LMR = ["L", "M", "R"]

# ── Dimension definitions for prompt ─────────────────────────
DIMENSION_DEFINITIONS = """\
# L/M/R Classification Rules — 6 Dimensions with Exact Observable Indicators

For each dimension, use the OBSERVABLE INDICATORS to classify. Match the user's statistics against the thresholds below. Choose the tier (L/M/R) that best matches the majority of indicators.

---

## Dimension A: Consumption Pattern (信息消费模式)
How the user explores, locates, and reads files.

### A:L (sequential)
- total_searches <= 1 (almost never uses grep/glob search)
- search_ratio ≈ 0
- revisit_ratio >= 0.10 (revisits files to cross-check)
- browse_ratio low (< 0.10)
- Reads most of each file (large read windows, not small snippets)

### A:M (targeted)
- total_searches >= 3
- search_ratio >= 0.15 (uses search relative to reads)
- Partial/targeted reads (not full files)
- revisit_ratio low (finds info via search, no need to revisit)

### A:R (breadth-first)
- total_browses >= 3
- browse_ratio >= 0.15 (heavy directory browsing)
- Small read windows (skims first 10-20 lines per file)
- revisit_ratio < 0.05 (rarely re-reads)
- total_searches <= 1 (browses directories, does not keyword-search)

---

## Dimension B: Production Style (生产风格)
How the user produces content — format, detail, structure, file count.

### B:L (comprehensive)
- avg_output_length >= 3000 chars per created file
- files_created >= 3 per task (multiple output files)
- total_output_chars >= 8000 per task
- Deep heading structure (#### present, heading_max_depth >= 4)
- Creates auxiliary files (README, index, appendix, glossary)

### B:M (balanced)
- avg_output_length in [1000, 3000] chars
- files_created in [1, 3] per task
- total_output_chars in [3000, 8000] per task
- heading_max_depth 2-3 (### but not ####)

### B:R (minimal)
- avg_output_length <= 1000 chars
- files_created == 1 per task
- total_output_chars <= 3000 per task
- heading_max_depth <= 2 (at most ##)
- Zero auxiliary files

---

## Dimension C: Organization Preference (组织偏好)
How the user manages directories, naming, versioning, cleanup.

### C:L (deeply_nested)
- dirs_created >= 3 per task
- max_dir_depth >= 3
- files_moved >= 2 (actively organizes files into subdirs)
- backup_copies >= 1 (creates backups before editing)
- total_deletes == 0 (never deletes old files)
- Long descriptive filenames (mean filename length >= 25 chars)

### C:M (adaptive)
- dirs_created in [1, 2] per task
- max_dir_depth in [1, 2]
- files_moved in [0, 2]
- Mixed naming patterns, practical filenames
- Occasional deletes

### C:R (flat)
- dirs_created == 0 per task
- max_dir_depth == 0
- files_moved == 0
- total_deletes >= 1 (cleans up temp files)
- total_overwrites >= 1 (overwrites rather than versioning)
- Short abbreviated filenames (mean filename length <= 12 chars)

---

## Dimension D: Iteration Strategy (迭代策略)
How the user modifies and refines existing work.

### D:L (incremental)
- total_edits >= 5 per modified file (many small edits)
- avg_lines_changed <= 10 (each edit is small)
- small_edit_ratio >= 0.5 (majority of edits are small)
- backup_copies >= 1 (backs up before editing)
- total_overwrites == 0 (never overwrites whole file)

### D:M (balanced)
- total_edits in [2, 4] per modified file
- avg_lines_changed in [10, 30]
- May occasionally overwrite

### D:R (rewrite)
- total_edits <= 1 per modified file (or zero — writes once, done)
- avg_lines_changed >= 30 OR file rewritten via overwrite
- backup_copies == 0 (no backups)
- total_overwrites >= 1
- Possible delete-then-recreate pattern

---

## Dimension E: Curation (信息管护)
Workspace artifact management: selective vs preservative vs pragmatic.

### E:L (selective)
- total_deletes >= 2 (actively prunes files)
- delete_to_create_ratio >= 0.15 (deletes a meaningful fraction of created files)
- Backs up before pruning (file_copy with is_backup=true)

### E:M (pragmatic)
- total_deletes in [0, 1] (occasional cleanup)
- delete_to_create_ratio < 0.15 (rarely deletes relative to creates)
- Occasional cleanup but no systematic curation

### E:R (preservative)
- total_deletes == 0 (never deletes anything)
- No backup copies; workspace only grows
- All intermediate and temporary files persist at session end

---

## Dimension F: Cross-Modal Behavior (跨模态行为)
Whether user creates/uses visual materials (images, charts, tables) alongside text.

### F:L (visual-heavy)
- image_files_created >= 1 (creates png/jpg/svg/csv/mermaid files)
- structured_files_created >= 1
- markdown_table_rows >= 5
- Figure references in text ("Figure 1", "see chart", "see diagram")
- Media/figures/assets directory created

### F:M (balanced)
- image_files_created == 0
- has_tables > 0 (markdown table syntax present in output)
- markdown_table_rows >= 1
- No figure references to external image files

### F:R (text-only)
- image_files_created == 0
- structured_files_created == 0
- markdown_table_rows == 0
- has_tables == 0
- No table syntax, no figure references, pure prose/bullet lists
"""

CLASSIFICATION_INSTRUCTION = """\
Based on the behavioral data above, classify this user on each of the 6 dimensions (A through F).

IMPORTANT: Use the OBSERVABLE INDICATORS and THRESHOLDS from the dimension definitions above. For each dimension:
1. Look at the CROSS-TRAJECTORY MEAN values for the relevant metrics
2. Compare against the thresholds specified for L, M, and R
3. Choose the tier whose indicators best match the data

For each dimension, assign exactly one of: L (Left), M (Middle), or R (Right).

Respond ONLY in valid JSON with this exact structure:
{
  "A": {"classification": "L|M|R", "justification": "cite specific numbers: metric_name=value vs threshold"},
  "B": {"classification": "L|M|R", "justification": "cite specific numbers: metric_name=value vs threshold"},
  "C": {"classification": "L|M|R", "justification": "cite specific numbers: metric_name=value vs threshold"},
  "D": {"classification": "L|M|R", "justification": "cite specific numbers: metric_name=value vs threshold"},
  "E": {"classification": "L|M|R", "justification": "cite specific numbers: metric_name=value vs threshold"},
  "F": {"classification": "L|M|R", "justification": "cite specific numbers: metric_name=value vs threshold"}
}
"""


# ── LLM clients ──────────────────────────────────────────────
_llm_lock = Lock()
_llm_calls = 0


def _make_azure_client(model_override: str | None = None):
    from openai import AzureOpenAI

    # Support both full-URL endpoint and base endpoint
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    # Strip deployment path from endpoint if embedded (e.g., .../deployments/gpt-4.1-mini/...)
    # AzureOpenAI needs just the base URL: https://<resource>.openai.azure.com/
    base = re.sub(r"/openai/deployments/.*", "", endpoint).rstrip("/")
    if not base:
        base = endpoint.rstrip("/")

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        azure_endpoint=base,
    )
    model = model_override or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

    def call(system: str, user: str, max_tokens: int = 2048) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    return call, model


def _make_gemini_client(model_override: str | None = None):
    from google import genai
    from google.genai import types

    gclient = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model = model_override or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    is_pro = "pro" in model.lower()

    def call(system: str, user: str, max_tokens: int = 2048) -> str:
        prompt = f"{system}\n\n{user}"
        if is_pro:
            # Pro models use thinking by default; give explicit budget so
            # thinking doesn't consume the entire output allowance.
            config = types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=max(max_tokens, 16384),
                thinking_config=types.ThinkingConfig(thinking_budget=4096),
            )
        else:
            config = types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        resp = gclient.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return (resp.text or "").strip()

    return call, model


def _make_anthropic_client(model_override: str | None = None):
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = model_override or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    def call(system: str, user: str, max_tokens: int = 2048) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0,
        )
        return resp.content[0].text.strip()

    return call, model


# Maps --models key -> (factory, default_model_name)
CLIENT_FACTORIES = {
    "azure": _make_azure_client,
    "gemini": _make_gemini_client,
    "anthropic": _make_anthropic_client,
}


def call_with_retry(fn, system: str, user: str, retries: int = 12) -> str:
    global _llm_calls
    with _llm_lock:
        _llm_calls += 1
    for attempt in range(retries):
        try:
            return fn(system, user)
        except Exception as e:
            err = str(e)
            if any(k in err for k in ("429", "RateLimit", "RESOURCE_EXHAUSTED", "rate_limit", "overloaded")):
                wait = min(2**attempt * 5 + random.uniform(0, 3), 120)
                print(f"    Rate limited (attempt {attempt + 1}), waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Rate limited after {retries} retries")


# ── Feature extraction (reuse from lmr_validation.py) ────────


def load_profile_summary(profile_id: str, mode: str = "rich") -> str | None:
    """Load all trajectories for a profile and build an LLM-readable summary.

    Modes:
        "stats" — aggregated statistics only (~2.5K tokens, fast but lossy)
        "rich"  — per-trajectory features + semantic (filenames, content previews)
                   + aggregated behavioral patterns (~15-20K tokens, recommended)
    """
    from filegramos.aggregation import FeatureAggregator
    from filegramos.feature_extraction import FeatureExtractor

    traj_dirs = sorted(SIGNAL_DIR.glob(f"{profile_id}_T-*"))
    if not traj_dirs:
        return None

    all_features = []
    all_semantic = []
    traj_names = []
    for td in traj_dirs:
        events_path = td / "events.json"
        if not events_path.exists():
            continue
        with open(events_path) as f:
            events = json.load(f)
        fe = FeatureExtractor(events)
        features = fe.extract_all()
        all_features.append(features)
        all_semantic.append(fe.extract_semantic_channel())
        traj_names.append(td.name)

    if not all_features:
        return None

    if mode == "stats":
        agg = FeatureAggregator(all_features)
        return agg.to_summary_text()

    # ── "rich" mode: per-trajectory detail + patterns ─────
    parts = [f"# Behavioral Data ({len(all_features)} trajectories)\n"]

    # Per-trajectory structured features + semantic info
    for i, (feat, sem, tname) in enumerate(zip(all_features, all_semantic, traj_names)):
        lines = [f"## {tname}"]

        # Key procedural stats (compact, one line per attribute)
        rs = feat.get("reading_strategy", {})
        lines.append(
            f"  Reading: {rs.get('total_reads', 0)} reads, "
            f"{rs.get('total_searches', 0)} searches, "
            f"{rs.get('total_browses', 0)} browses, "
            f"revisit_ratio={rs.get('revisit_ratio', 0)}, "
            f"switch_rate={rs.get('context_switch_rate', 0)}"
        )

        od = feat.get("output_detail", {})
        lines.append(
            f"  Output: {od.get('files_created', 0)} files, "
            f"total={od.get('total_output_chars', 0)} chars, "
            f"avg={od.get('avg_output_length', 0):.0f} chars/file"
        )

        ds = feat.get("directory_style", {})
        lines.append(
            f"  Dirs: {ds.get('dirs_created', 0)} created, "
            f"max_depth={ds.get('max_dir_depth', 0)}, "
            f"files_moved={ds.get('files_moved', 0)}"
        )

        es = feat.get("edit_strategy", {})
        lines.append(
            f"  Edits: {es.get('total_edits', 0)} total, "
            f"avg_lines={es.get('avg_lines_changed', 0)}, "
            f"small_ratio={es.get('small_edit_ratio', 0)}"
        )

        vs = feat.get("version_strategy", {})
        lines.append(
            f"  Version: {vs.get('backup_copies', 0)} backups, "
            f"{vs.get('total_deletes', 0)} deletes, "
            f"{vs.get('total_overwrites', 0)} overwrites"
        )

        cm = feat.get("cross_modal_behavior", {})
        lines.append(
            f"  Cross-modal: {cm.get('image_files_created', 0)} images, "
            f"{cm.get('structured_files_created', 0)} structured, "
            f"table_rows={cm.get('markdown_table_rows', 0)}"
        )

        tn = feat.get("tone", {})
        lines.append(
            f"  Structure: headings={tn.get('heading_count', 0)} "
            f"(max_depth={tn.get('heading_max_depth', 0)}), "
            f"lists={tn.get('list_item_count', 0)}, "
            f"prose_ratio={tn.get('prose_to_structure_ratio', 0)}"
        )

        # Semantic: created filenames and dir structure
        filenames = sem.get("created_filenames", [])
        if filenames:
            lines.append(f"  Created files: {', '.join(filenames)}")

        dir_diff = sem.get("dir_structure_diff", [])
        if dir_diff:
            lines.append(f"  New dirs/files: {', '.join(dir_diff[:15])}")

        # Content previews (first 150 chars of each created file)
        for cf in sem.get("created_files", [])[:3]:
            preview = (cf.get("preview", "") or "")[:150].replace("\n", " ").strip()
            if preview:
                lines.append(f"  Content preview ({cf.get('path', '?')}): {preview}...")

        parts.append("\n".join(lines))

    # Aggregated behavioral patterns (from FeatureAggregator)
    agg = FeatureAggregator(all_features)
    patterns = agg._detect_behavioral_patterns(agg.aggregate_all())
    if patterns:
        parts.append("\n## Detected Cross-Trajectory Patterns")
        for p in patterns:
            parts.append(f"  - {p}")

    return "\n\n".join(parts)


# ── Rule-based baseline (reuse from lmr_validation.py) ───────


def _load_rule_baseline():
    """Load rule-based classifications from lmr_validation.py functions."""
    from lmr_validation import aggregate_stats, classify_from_stats, load_trajectory_features

    results = {}
    for pid in sorted(GT.keys()):
        features_list = load_trajectory_features(pid)
        agg = aggregate_stats(features_list)
        pred = classify_from_stats(agg)
        results[pid] = pred
    return results


# ── Prompt building ──────────────────────────────────────────


def build_prompt(summary: str) -> tuple[str, str]:
    """Build (system, user) prompt pair for classification."""
    system = (
        "You are a behavioral analyst. Your task is to classify a user on 6 dimensions "
        "by matching their behavioral statistics against predefined thresholds.\n\n"
        "CRITICAL: Each dimension has EXACT observable indicators with numeric thresholds. "
        "You MUST compare the user's actual numbers against these thresholds to classify. "
        "Do NOT rely on subjective interpretation — use the numbers.\n\n" + DIMENSION_DEFINITIONS
    )
    user = (
        "Here is the aggregated behavioral summary for a user across multiple tasks:\n\n"
        + summary
        + "\n\n"
        + CLASSIFICATION_INSTRUCTION
    )
    return system, user


# ── JSON parsing ─────────────────────────────────────────────


def parse_classification(text: str) -> dict[str, dict] | None:
    """Parse JSON classification from LLM response.

    Falls back to regex extraction if JSON is truncated/malformed.
    """
    text = text.strip()
    # Strip markdown fences
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
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Fallback: regex extraction for truncated JSON
    # Look for patterns like "A": {"classification": "L", ...} or "A": "L"
    result = {}
    for dim in DIMS:
        # Pattern 1: "A": {"classification": "L", ...}
        m = re.search(rf'"{dim}"\s*:\s*\{{\s*"classification"\s*:\s*"([LMR])"', text)
        if m:
            result[dim] = {"classification": m.group(1)}
            # Try to also grab justification
            j = re.search(rf'"{dim}"\s*:\s*\{{[^}}]*"justification"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
            if j:
                result[dim]["justification"] = j.group(1)
            continue
        # Pattern 2: "A": "L"
        m = re.search(rf'"{dim}"\s*:\s*"([LMR])"', text)
        if m:
            result[dim] = {"classification": m.group(1)}

    return result if result else None


def extract_predictions(parsed: dict) -> dict[str, str]:
    """Extract dimension -> L/M/R predictions from parsed JSON."""
    preds = {}
    for dim in DIMS:
        entry = parsed.get(dim, {})
        if isinstance(entry, dict):
            cls = entry.get("classification", "").upper().strip()
        elif isinstance(entry, str):
            cls = entry.upper().strip()
        else:
            cls = ""
        if cls in LMR:
            preds[dim] = cls
        else:
            preds[dim] = "?"
    return preds


# ── Statistical analysis ─────────────────────────────────────


def confusion_matrix_3x3(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute 3x3 confusion matrix for L/M/R classification."""
    matrix = {t: {p: 0 for p in LMR} for t in LMR}
    for t, p in zip(y_true, y_pred):
        if t in LMR and p in LMR:
            matrix[t][p] += 1
    return matrix


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def cohens_kappa(y1: list[str], y2: list[str], labels: list[str] | None = None) -> float:
    """Cohen's kappa for two raters."""
    if labels is None:
        labels = LMR
    n = len(y1)
    if n == 0:
        return 0.0

    # Observed agreement
    po = sum(a == b for a, b in zip(y1, y2)) / n

    # Expected agreement
    pe = 0.0
    for label in labels:
        c1 = sum(1 for x in y1 if x == label) / n
        c2 = sum(1 for x in y2 if x == label) / n
        pe += c1 * c2

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings: list[list[str]], labels: list[str] | None = None) -> float:
    """Fleiss' kappa for 3+ raters.

    Args:
        ratings: list of lists, each inner list has one rating per rater for one item.
                 E.g., ratings[i] = ["L", "M", "L"] for item i rated by 3 raters.
        labels: category labels (default: L/M/R).
    """
    if labels is None:
        labels = LMR
    N = len(ratings)  # number of items
    if N == 0:
        return 0.0
    n = len(ratings[0])  # number of raters
    if n <= 1:
        return 0.0
    k = len(labels)

    # Build count matrix: N x k
    counts = []
    for item_ratings in ratings:
        row = {label: 0 for label in labels}
        for r in item_ratings:
            if r in row:
                row[r] += 1
        counts.append(row)

    # P_i for each item
    P_i = []
    for row in counts:
        total = sum(row[label] ** 2 for label in labels)
        P_i.append((total - n) / (n * (n - 1)) if n > 1 else 0)

    P_bar = sum(P_i) / N

    # p_j for each category
    p_j = {}
    for label in labels:
        total = sum(row[label] for row in counts)
        p_j[label] = total / (N * n)

    P_e_bar = sum(p**2 for p in p_j.values())

    if P_e_bar == 1.0:
        return 1.0
    return (P_bar - P_e_bar) / (1 - P_e_bar)


# ── Classify one profile with one model ──────────────────────


def classify_profile(
    profile_id: str,
    summary: str,
    call_fn,
    model_name: str,
) -> dict:
    """Call LLM to classify a profile's L/M/R dimensions."""
    system, user = build_prompt(summary)
    t0 = time.time()
    try:
        response = call_with_retry(call_fn, system, user)
        parsed = parse_classification(response)
        if parsed is None:
            return {
                "profile": profile_id,
                "model": model_name,
                "predictions": {d: "?" for d in DIMS},
                "justifications": {},
                "error": "parse_failed",
                "raw_response": response[:500],
                "latency_ms": int((time.time() - t0) * 1000),
            }
        preds = extract_predictions(parsed)
        justifications = {}
        for dim in DIMS:
            entry = parsed.get(dim, {})
            if isinstance(entry, dict):
                justifications[dim] = entry.get("justification", "")
        return {
            "profile": profile_id,
            "model": model_name,
            "predictions": preds,
            "justifications": justifications,
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        return {
            "profile": profile_id,
            "model": model_name,
            "predictions": {d: "?" for d in DIMS},
            "justifications": {},
            "error": str(e),
            "latency_ms": int((time.time() - t0) * 1000),
        }


# ── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Multi-model L/M/R classification with agreement analysis")
    parser.add_argument(
        "--models", nargs="+", default=["azure", "gemini", "anthropic"], choices=list(CLIENT_FACTORIES.keys())
    )
    parser.add_argument(
        "--model-override",
        nargs="+",
        default=None,
        help="Override model names per --models entry. "
        "E.g. --models azure gemini --model-override gpt-4.1 gemini-2.5-pro",
    )
    parser.add_argument("--profiles", type=str, default=None, help="Comma-separated profile IDs (default: all)")
    parser.add_argument("--parallel", type=int, default=3, help="Concurrent LLM calls")
    parser.add_argument(
        "--context-mode",
        choices=["stats", "rich"],
        default="rich",
        help="'stats'=aggregated statistics only (~2.5K tok), "
        "'rich'=per-trajectory features+semantic (~15K tok, default)",
    )
    parser.add_argument("--dry", action="store_true", help="Print one prompt, no API calls")
    parser.add_argument("--resume", action="store_true", help="Skip completed profiles")
    args = parser.parse_args()

    # Select profiles
    if args.profiles:
        profile_ids = []
        for p in args.profiles.split(","):
            p = p.strip()
            # Allow short form (p1) or full form (p1_methodical)
            # Short form must match "pN_" to avoid p1 matching p10-p19
            matches = [pid for pid in GT if pid == p or pid.startswith(p + "_")]
            profile_ids.extend(matches)
        profile_ids = sorted(set(profile_ids))
    else:
        profile_ids = sorted(GT.keys())

    if not profile_ids:
        print("No matching profiles found.")
        return

    print(f"Profiles: {len(profile_ids)}")
    print(f"Models: {args.models}")
    print(f"Total API calls: {len(profile_ids) * len(args.models)}")

    # Load existing results for resume
    prior_results = {}  # (profile, model_key) -> result dict
    if args.resume and RESULTS_PATH.exists():
        prior = json.loads(RESULTS_PATH.read_text())
        for r in prior.get("model_results", []):
            key = (r["profile"], r.get("model_key", ""))
            prior_results[key] = r
        print(f"Resume: {len(prior_results)} results loaded")

    # Step 1: Load all profile summaries
    print("\nLoading behavioral summaries...")
    summaries = {}
    for pid in profile_ids:
        s = load_profile_summary(pid, mode=args.context_mode)
        if s:
            summaries[pid] = s
            print(f"  {pid}: {len(s)} chars")
        else:
            print(f"  {pid}: NO DATA (skipping)")

    if not summaries:
        print("No profile summaries loaded. Check signal/ directory.")
        return

    # Dry run: print one prompt and exit
    if args.dry:
        first_pid = next(iter(summaries))
        system, user = build_prompt(summaries[first_pid])
        print(f"\n{'=' * 80}")
        print(f"DRY RUN — Prompt for {first_pid}")
        print(f"{'=' * 80}")
        print(f"\n[SYSTEM]\n{system[:500]}...")
        print(f"\n[USER]\n{user}")
        print(f"\n{'=' * 80}")
        print(f"Total prompt length: {len(system) + len(user)} chars")
        return

    # Step 2: Initialize LLM clients
    overrides = args.model_override or []
    clients = {}  # model_key -> (call_fn, model_name)
    for i, mk in enumerate(args.models):
        override = overrides[i] if i < len(overrides) else None
        try:
            call_fn, model_name = CLIENT_FACTORIES[mk](model_override=override)
            clients[mk] = (call_fn, model_name)
            print(f"  {mk}: {model_name}")
        except Exception as e:
            print(f"  {mk}: FAILED to init ({e})")

    if not clients:
        print("No LLM clients initialized. Check API keys.")
        return

    # Step 3: Run classifications
    print(f"\nClassifying {len(summaries)} profiles × {len(clients)} models...")
    all_model_results = []
    results_lock = Lock()

    def run_one(pid: str, model_key: str) -> dict:
        key = (pid, model_key)
        if key in prior_results:
            return prior_results[key]
        call_fn, model_name = clients[model_key]
        result = classify_profile(pid, summaries[pid], call_fn, model_name)
        result["model_key"] = model_key
        return result

    tasks = [(pid, mk) for pid in summaries for mk in clients]

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, pid, mk): (pid, mk) for pid, mk in tasks}
        done = 0
        for fut in as_completed(futures):
            pid, mk = futures[fut]
            try:
                result = fut.result()
                with results_lock:
                    all_model_results.append(result)
                    done += 1
                err = result.get("error", "")
                preds = result.get("predictions", {})
                pred_str = "".join(preds.get(d, "?") for d in DIMS)
                status = f"err={err}" if err else pred_str
                print(f"  [{done}/{len(tasks)}] {pid} × {mk}: {status}")
            except Exception as e:
                print(f"  [{done}/{len(tasks)}] {pid} × {mk}: EXCEPTION {e}")
                done += 1

    # Step 4: Load rule-based baseline
    print("\nLoading rule-based baseline...")
    try:
        rule_preds = _load_rule_baseline()
    except Exception as e:
        print(f"  Failed to load rule baseline: {e}")
        rule_preds = {}

    # ── Organize results ─────────────────────────────────────
    # model_key -> profile -> predictions dict
    model_predictions = defaultdict(dict)  # model_key -> {profile: {dim: L/M/R}}
    for r in all_model_results:
        mk = r.get("model_key", r.get("model", ""))
        model_predictions[mk][r["profile"]] = r.get("predictions", {})

    if rule_preds:
        model_predictions["rule_based"] = rule_preds

    all_model_keys = list(model_predictions.keys())

    # Common profiles (those with data from all models)
    common_profiles = (
        sorted(set.intersection(*[set(model_predictions[mk].keys()) for mk in all_model_keys]))
        if all_model_keys
        else []
    )

    # ── Per-model accuracy ───────────────────────────────────
    print(f"\n{'=' * 90}")
    print("PER-MODEL ACCURACY vs Ground Truth")
    print(f"{'=' * 90}")

    # Header
    header = f"{'Model':<22}"
    for d in DIMS:
        header += f"  {d:>5}"
    header += f"  {'Overall':>8}"
    print(header)
    print("-" * len(header))

    per_model_accuracy = {}
    for mk in all_model_keys:
        preds = model_predictions[mk]
        dim_correct = {d: 0 for d in DIMS}
        dim_total = {d: 0 for d in DIMS}
        for pid in common_profiles:
            gt = GT[pid]
            pred = preds.get(pid, {})
            for d in DIMS:
                if pred.get(d) in LMR:
                    dim_total[d] += 1
                    if pred[d] == gt[d]:
                        dim_correct[d] += 1

        row = f"{mk:<22}"
        total_c, total_t = 0, 0
        dim_accs = {}
        for d in DIMS:
            if dim_total[d] > 0:
                acc = dim_correct[d] / dim_total[d] * 100
                row += f"  {acc:>4.0f}%"
                dim_accs[d] = round(acc, 1)
            else:
                row += f"  {'—':>5}"
                dim_accs[d] = None
            total_c += dim_correct[d]
            total_t += dim_total[d]

        overall = total_c / total_t * 100 if total_t > 0 else 0
        row += f"  {overall:>6.1f}%"
        print(row)
        per_model_accuracy[mk] = {"per_dim": dim_accs, "overall": round(overall, 1)}

    # ── Confusion matrices ───────────────────────────────────
    print(f"\n{'=' * 90}")
    print("CONFUSION MATRICES (per model, per dimension)")
    print(f"{'=' * 90}")

    per_model_confusion = {}
    for mk in all_model_keys:
        preds = model_predictions[mk]
        per_model_confusion[mk] = {}
        for d in DIMS:
            y_true = [GT[pid][d] for pid in common_profiles if preds.get(pid, {}).get(d) in LMR]
            y_pred = [preds[pid][d] for pid in common_profiles if preds.get(pid, {}).get(d) in LMR]
            cm = confusion_matrix_3x3(y_true, y_pred)
            per_model_confusion[mk][d] = cm

        # Print compact confusion matrices
        print(f"\n  {mk}:")
        for d in DIMS:
            cm = per_model_confusion[mk][d]
            total = sum(sum(cm[t].values()) for t in LMR)
            if total == 0:
                continue
            print(f"    Dim {d}:  pred→  L   M   R")
            for t in LMR:
                vals = [cm[t][p] for p in LMR]
                marks = []
                for i, p in enumerate(LMR):
                    v = cm[t][p]
                    marks.append(f"[{v}]" if t == p else f" {v} ")
                print(f"      gt {t}:     {''.join(f'{m:>4}' for m in marks)}")

    # ── Cohen's κ (pairwise) ─────────────────────────────────
    print(f"\n{'=' * 90}")
    print("COHEN'S KAPPA (pairwise inter-model agreement)")
    print(f"{'=' * 90}")

    llm_keys = [mk for mk in all_model_keys if mk != "rule_based"]
    pairwise_kappas = {}

    if len(llm_keys) >= 2:
        # Overall kappa
        print("\n  Overall (all dimensions pooled):")
        header = f"  {'':>22}"
        for mk2 in all_model_keys:
            header += f"  {mk2[:10]:>10}"
        print(header)

        for mk1 in all_model_keys:
            row = f"  {mk1:<22}"
            for mk2 in all_model_keys:
                if mk1 == mk2:
                    row += f"  {'—':>10}"
                else:
                    # Pool all dimensions
                    y1, y2 = [], []
                    for pid in common_profiles:
                        p1 = model_predictions[mk1].get(pid, {})
                        p2 = model_predictions[mk2].get(pid, {})
                        for d in DIMS:
                            if p1.get(d) in LMR and p2.get(d) in LMR:
                                y1.append(p1[d])
                                y2.append(p2[d])
                    if y1:
                        kappa = cohens_kappa(y1, y2)
                        row += f"  {kappa:>10.3f}"
                        pairwise_kappas[(mk1, mk2)] = round(kappa, 3)
                    else:
                        row += f"  {'—':>10}"
            print(row)

        # Per-dimension kappa
        print("\n  Per-dimension:")
        for d in DIMS:
            print(f"\n    Dim {d}:")
            header = f"    {'':>22}"
            for mk2 in all_model_keys:
                header += f"  {mk2[:10]:>10}"
            print(header)

            for mk1 in all_model_keys:
                row = f"    {mk1:<22}"
                for mk2 in all_model_keys:
                    if mk1 == mk2:
                        row += f"  {'—':>10}"
                    else:
                        y1, y2 = [], []
                        for pid in common_profiles:
                            p1 = model_predictions[mk1].get(pid, {})
                            p2 = model_predictions[mk2].get(pid, {})
                            if p1.get(d) in LMR and p2.get(d) in LMR:
                                y1.append(p1[d])
                                y2.append(p2[d])
                        if y1:
                            kappa = cohens_kappa(y1, y2)
                            row += f"  {kappa:>10.3f}"
                        else:
                            row += f"  {'—':>10}"
                print(row)

    # ── Fleiss' κ (all LLM models) ───────────────────────────
    print(f"\n{'=' * 90}")
    print(f"FLEISS' KAPPA (all {len(llm_keys)} LLM models)")
    print(f"{'=' * 90}")

    fleiss_results = {}
    if len(llm_keys) >= 2:
        # Overall
        all_ratings = []
        for pid in common_profiles:
            for d in DIMS:
                item_ratings = []
                for mk in llm_keys:
                    pred = model_predictions[mk].get(pid, {}).get(d, "?")
                    if pred in LMR:
                        item_ratings.append(pred)
                if len(item_ratings) == len(llm_keys):
                    all_ratings.append(item_ratings)

        if all_ratings:
            fk = fleiss_kappa(all_ratings)
            print(f"  Overall: κ = {fk:.3f}  (N={len(all_ratings)} items, {len(llm_keys)} raters)")
            fleiss_results["overall"] = round(fk, 3)

        # Per dimension
        for d in DIMS:
            dim_ratings = []
            for pid in common_profiles:
                item_ratings = []
                for mk in llm_keys:
                    pred = model_predictions[mk].get(pid, {}).get(d, "?")
                    if pred in LMR:
                        item_ratings.append(pred)
                if len(item_ratings) == len(llm_keys):
                    dim_ratings.append(item_ratings)
            if dim_ratings:
                fk = fleiss_kappa(dim_ratings)
                print(f"  Dim {d}: κ = {fk:.3f}  (N={len(dim_ratings)})")
                fleiss_results[d] = round(fk, 3)

    # ── Majority vote accuracy ───────────────────────────────
    print(f"\n{'=' * 90}")
    print("MAJORITY VOTE (LLM ensemble)")
    print(f"{'=' * 90}")

    if len(llm_keys) >= 2:
        majority_correct = {d: 0 for d in DIMS}
        majority_total = {d: 0 for d in DIMS}

        for pid in common_profiles:
            gt = GT[pid]
            for d in DIMS:
                votes = []
                for mk in llm_keys:
                    pred = model_predictions[mk].get(pid, {}).get(d, "?")
                    if pred in LMR:
                        votes.append(pred)
                if votes:
                    majority = Counter(votes).most_common(1)[0][0]
                    majority_total[d] += 1
                    if majority == gt[d]:
                        majority_correct[d] += 1

        header = f"{'':>22}"
        for d in DIMS:
            header += f"  {d:>5}"
        header += f"  {'Overall':>8}"
        print(header)
        print("-" * len(header))

        row = f"{'Majority Vote':<22}"
        tc, tt = 0, 0
        for d in DIMS:
            if majority_total[d] > 0:
                acc = majority_correct[d] / majority_total[d] * 100
                row += f"  {acc:>4.0f}%"
            else:
                row += f"  {'—':>5}"
            tc += majority_correct[d]
            tt += majority_total[d]
        overall = tc / tt * 100 if tt > 0 else 0
        row += f"  {overall:>6.1f}%"
        print(row)

    # ── Per-profile detail ───────────────────────────────────
    print(f"\n{'=' * 90}")
    print("PER-PROFILE DETAIL")
    print(f"{'=' * 90}")

    for pid in common_profiles:
        gt = GT[pid]
        gt_str = "".join(gt[d] for d in DIMS)
        print(f"\n  {pid} (GT: {gt_str}):")
        for mk in all_model_keys:
            preds = model_predictions[mk].get(pid, {})
            pred_str = ""
            for d in DIMS:
                p = preds.get(d, "?")
                if p == gt[d]:
                    pred_str += p
                else:
                    pred_str += f"[{p}]"
            matches = sum(1 for d in DIMS if preds.get(d) == gt[d])
            print(f"    {mk:<22} {pred_str}  ({matches}/6)")

    # ── Save JSON results ────────────────────────────────────
    output = {
        "timestamp": datetime.now().isoformat(),
        "profiles": profile_ids,
        "models": {mk: clients[mk][1] if mk in clients else "rule_based" for mk in all_model_keys},
        "n_profiles": len(common_profiles),
        "n_models": len(all_model_keys),
        "per_model_accuracy": per_model_accuracy,
        "fleiss_kappa": fleiss_results,
        "pairwise_cohens_kappa": {f"{k[0]}_vs_{k[1]}": v for k, v in pairwise_kappas.items()},
        "ground_truth": {pid: GT[pid] for pid in common_profiles},
        "model_results": all_model_results,
        "per_model_confusion": {mk: {d: per_model_confusion[mk][d] for d in DIMS} for mk in per_model_confusion},
        "total_llm_calls": _llm_calls,
    }

    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Total LLM calls: {_llm_calls}")


if __name__ == "__main__":
    main()
