"""LLM-as-Judge per-attribute scoring implementation.

Given a ground-truth profile and an inferred profile from a baseline,
uses an LLM judge to score each attribute 1-5 with justification.

Supports two modes:
1. Legacy: single LLM call scores all attributes (build_judge_prompt / parse_judge_response)
2. Per-attribute: one LLM call per attribute with detailed rubrics (judge_all_attributes)
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for user profiling systems. Your task is to \
compare an inferred user profile against the ground truth profile and \
score each attribute on accuracy.

IMPORTANT: The inferred values may use different vocabulary than ground truth \
labels. You MUST evaluate based on SEMANTIC meaning, not exact string match. \
For example, if ground truth is "sequential_deep" and inferred is "thorough \
sequential reading with revisits", that is semantically identical (score 5).

Scoring rubric (1-5):
5 = Semantically identical — captures the same behavioral concept even if worded differently
4 = Close match — right direction, captures the core behavior with minor deviations
3 = Partially correct — some aspects match but key behavioral elements are wrong or missing
2 = Mostly wrong — only superficial overlap, misidentifies the core behavioral pattern
1 = Completely wrong — contradicts ground truth behavioral pattern

## Attribute Value Definitions

Use these definitions to judge semantic equivalence. The inferred value does NOT \
need to use these exact labels — it just needs to describe the SAME behavior.

### reading_strategy
- sequential_deep: Reads files one-by-one in logical order, full file content, revisits to cross-check, does NOT use search/grep
- breadth_first: Scans broadly with directory browsing, reads only first 10-20 lines, high coverage low depth, rarely revisits
- targeted_search: Searches first (grep/glob), reads only matched files, reads targeted sections only

### working_style
- methodical: Systematic, step-by-step approach, follows a plan, thorough in each phase
- exploratory: Tries multiple approaches, experiments, flexible strategy
- pragmatic: Efficient, direct path to solution, skips unnecessary steps

### thoroughness
- exhaustive: Reads everything available, covers all angles, leaves nothing unexamined
- balanced: Reasonable coverage without over-examining
- minimal: Reads only what's strictly needed, skips optional material

### error_handling
- defensive: Creates backups, validates before modifying, cautious approach
- balanced: Moderate caution, some checks but not excessive
- optimistic: Proceeds without backups, trusts first attempt, fixes only if problems arise

### directory_style
- nested_by_topic: Creates 3+ level directory hierarchies organized by topic/category
- flat: All files in root directory, no subdirectory creation
- by_type: Organizes by file type rather than topic

### edit_strategy
- incremental_small: Many small edits (<10 lines each), refines gradually, multiple passes per file
- bulk_rewrite: Few edits but large scope, or overwrites entire files, single-pass creation

### version_strategy
- keep_history: Creates backups/copies before modifying, never deletes old versions
- archive_old: Moves old versions to archive directory
- overwrite: Directly overwrites existing files, no backup behavior

### output_detail
- detailed: Comprehensive output (3000+ chars/file), multi-level headings, thorough coverage
- balanced: Moderate detail (1000-3000 chars), adequate but not exhaustive
- concise: Minimal output (<1000 chars), brief and to-the-point

### tone
- professional: Formal language, structured headings, academic/business style
- friendly: Warm, accessible, conversational but competent
- casual: Informal, abbreviated, minimal structure

### output_structure
- hierarchical: Multi-level headings (##/###/####), clear sections, table of contents style
- flat: Single level or no headings, linear flow
- freeform: Mixed structure, prose-heavy, no consistent pattern

### documentation
- comprehensive: Creates README, index files, explains everything thoroughly
- moderate: Some documentation where needed
- minimal: No auxiliary documentation files

### naming
- date_prefix_descriptive: Date-prefixed (YYYY-MM-DD) with long descriptive names
- long_descriptive: Long, descriptive filenames with multiple words/separators
- short_abbrev: Short abbreviated filenames, cryptic but compact

### cross_modal_behavior
- tables_and_references: Creates data files (CSV/JSON), markdown tables, figure references
- minimal_tables: Occasional markdown tables, no standalone data files
- text_only: Pure text output, no tables, no data files, no images

### language
- Chinese: Content primarily in Chinese
- English: Content primarily in English

### name / role
- Free text: Judge based on whether the inferred name/role matches or is close to ground truth"""

JUDGE_USER_PROMPT = """\
Compare the following inferred user profile against the ground truth \
and score each attribute.

### Ground Truth Profile
{ground_truth}

### Inferred Profile (by {method_name})
{inferred_profile}

### Attributes to Score
{attributes_list}

For each attribute, provide:
1. The ground truth value
2. The inferred value
3. A score (1-5) following the rubric
4. A brief justification

Respond in JSON format:
{{
  "scores": {{
    "<attribute_name>": {{
      "ground_truth": "<value>",
      "inferred": "<value>",
      "score": <1-5>,
      "justification": "<reasoning>"
    }}
  }},
  "overall_mean": <average score>
}}"""


# ---------------------------------------------------------------------------
# Per-attribute rubrics — detailed scoring criteria for each of 16 attributes
# ---------------------------------------------------------------------------
# Core design principle: if the inferred value is vague/generic/describes a
# "normal user" without specific behavioral evidence, cap at 3.  Scores of
# 4-5 REQUIRE concrete, distinguishing behavioral descriptions.
# ---------------------------------------------------------------------------

ATTRIBUTE_RUBRICS: dict[str, dict[str, str]] = {
    "name": {
        "description": "The user's name",
        "value_space": "Free text (e.g. Chen Wei, Sam Taylor, Maria Santos)",
        "rubric": (
            "5 = Exact match (including transliteration / pinyin variants, e.g. 'Chen Wei' = '陈伟')\n"
            "4 = Partial match — correct family name OR given name but not both\n"
            "3 = Wrong name but correct cultural/linguistic background (e.g. inferred a Chinese name when GT is Chinese)\n"
            "2 = Completely wrong name, wrong cultural background\n"
            "1 = Not inferred / left blank / said 'unknown'"
        ),
        "vagueness_penalty": "'unknown', 'a Chinese user', 'the user', or no name at all → 1",
    },
    "role": {
        "description": "The user's professional role",
        "value_space": "Free text (e.g. Research Analyst, Operations Manager, Event Planner)",
        "rubric": (
            "5 = Exact role match or very close synonym (e.g. 'Research Analyst' = 'Research Specialist')\n"
            "4 = Same professional domain, slightly different scope (e.g. 'Data Analyst' for GT 'Research Analyst')\n"
            "3 = Related field but wrong specific role (e.g. 'Professor' for GT 'Research Analyst')\n"
            "2 = Completely different domain (e.g. 'Software Engineer' for GT 'Event Planner')\n"
            "1 = Not inferred / generic (e.g. 'professional', 'office worker')"
        ),
        "vagueness_penalty": "'professional', 'knowledge worker', 'office worker' → max 1",
    },
    "language": {
        "description": "Primary language of user's content output",
        "value_space": "Chinese | English",
        "rubric": (
            "5 = Correct language identified with confidence\n"
            "4 = Correct language but hedged (e.g. 'probably Chinese')\n"
            "3 = Mentioned the correct language among multiple guesses\n"
            "2 = Wrong language\n"
            "1 = Not inferred"
        ),
        "vagueness_penalty": "'multilingual' or 'various languages' without specifying primary → max 3",
    },
    "tone": {
        "description": "Communication tone/style in produced content",
        "value_space": "professional | friendly | casual",
        "rubric": (
            "5 = Exact match with behavioral evidence (e.g. 'professional — uses formal headings and structured sections')\n"
            "4 = Correct tone label without strong evidence\n"
            "3 = Adjacent tone (e.g. 'friendly' when GT is 'professional') OR correct but GENERIC phrasing\n"
            "2 = Wrong tone direction (e.g. 'casual' when GT is 'professional')\n"
            "1 = Not inferred"
        ),
        "vagueness_penalty": "'normal tone', 'appropriate style' → max 2",
    },
    "output_detail": {
        "description": "Level of detail/verbosity in produced content",
        "value_space": "detailed | balanced | concise",
        "rubric": (
            "5 = Correct level + quantitative evidence (e.g. 'detailed — creates 200+ line files with appendices')\n"
            "4 = Correct level with qualitative evidence (e.g. 'detailed — very thorough output')\n"
            "3 = Correct direction but GENERIC (e.g. 'writes a reasonable amount') OR adjacent level\n"
            "2 = Wrong direction (e.g. 'concise' when GT is 'detailed')\n"
            "1 = Not inferred"
        ),
        "vagueness_penalty": "'moderate detail', 'appropriate level of detail' → max 3",
    },
    "working_style": {
        "description": "How the user approaches tasks — overall work pattern",
        "value_space": "methodical | pragmatic | exploratory",
        "rubric": (
            "5 = Correct style + specific behavioral evidence (e.g. 'methodical — reads all files before writing, follows plan-execute-review phases')\n"
            "4 = Correct style + some evidence (e.g. 'methodical — systematic approach')\n"
            "3 = Right direction but GENERIC (e.g. 'organized worker', 'efficient') — these descriptions could apply to anyone\n"
            "2 = Wrong style (e.g. 'exploratory' when GT is 'methodical')\n"
            "1 = Contradicts GT pattern (e.g. 'chaotic, no structure' when GT is 'methodical')"
        ),
        "vagueness_penalty": "'organized', 'efficient', 'productive' without specifying methodical/pragmatic/exploratory → max 3",
    },
    "thoroughness": {
        "description": "How exhaustively the user consumes information before acting",
        "value_space": "exhaustive | balanced | minimal",
        "rubric": (
            "5 = Correct level + evidence from reading behavior (e.g. 'exhaustive — reads every file fully, revisits to cross-check')\n"
            "4 = Correct level + partial evidence (e.g. 'exhaustive — very thorough reader')\n"
            "3 = Generic description (e.g. 'reads carefully', 'thorough approach') — cap here even if compatible with GT\n"
            "2 = Wrong level (e.g. 'minimal' when GT is 'exhaustive')\n"
            "1 = Opposite of GT"
        ),
        "vagueness_penalty": "'thorough', 'careful', 'reads files' → max 3 — must specify exhaustive vs minimal vs balanced",
    },
    "error_handling": {
        "description": "How the user handles errors and risk",
        "value_space": "defensive | balanced | optimistic",
        "rubric": (
            "5 = Correct strategy + specific evidence (e.g. 'defensive — creates backups before editing, validates changes')\n"
            "4 = Correct strategy + partial evidence\n"
            "3 = Generic (e.g. 'handles errors appropriately') — cap here even if direction is right\n"
            "2 = Wrong strategy (e.g. 'optimistic' when GT is 'defensive')\n"
            "1 = Opposite / not inferred"
        ),
        "vagueness_penalty": "'handles errors well', 'deals with problems' → max 3",
    },
    "reading_strategy": {
        "description": "Information acquisition strategy when exploring files",
        "value_space": "sequential_deep | targeted_search | breadth_first",
        "rubric": (
            "5 = Correct strategy + specific behavioral evidence (e.g. 'sequential_deep — reads files one-by-one in order, full content, revisits for cross-checking')\n"
            "4 = Correct strategy + partial evidence (e.g. 'sequential — reads files thoroughly')\n"
            "3 = Right general direction but GENERIC (e.g. 'reads files carefully', 'thorough reader') — MUST name the specific strategy\n"
            "2 = Wrong strategy (e.g. 'targeted_search' when GT is 'sequential_deep')\n"
            "1 = Opposite strategy / not inferred"
        ),
        "vagueness_penalty": "'reads files carefully', 'thorough approach', 'explores files' → max 3. Must explicitly indicate sequential/search/browse pattern.",
    },
    "directory_style": {
        "description": "How the user organizes directory structures",
        "value_space": "deeply_nested | flat | adaptive (by_type)",
        "rubric": (
            "5 = Correct style + evidence (e.g. 'deeply_nested — creates 3+ level hierarchies, organizes by topic with README at each level')\n"
            "4 = Correct style + partial evidence (e.g. 'creates organized directory structures')\n"
            "3 = Generic (e.g. 'organizes files', 'creates folders') — could describe anyone\n"
            "2 = Wrong style (e.g. 'flat' when GT is 'deeply_nested')\n"
            "1 = Opposite / not inferred"
        ),
        "vagueness_penalty": "'organizes files into folders' → max 3. Must specify depth/nesting pattern.",
    },
    "edit_strategy": {
        "description": "How the user modifies existing content",
        "value_space": "incremental_small | bulk_rewrite | balanced",
        "rubric": (
            "5 = Correct strategy + evidence (e.g. 'incremental_small — many small edits of <10 lines, multiple passes, reviews after each edit')\n"
            "4 = Correct strategy + partial evidence\n"
            "3 = Generic (e.g. 'edits files', 'makes changes') — cap here\n"
            "2 = Wrong strategy (e.g. 'bulk_rewrite' when GT is 'incremental_small')\n"
            "1 = Opposite / not inferred"
        ),
        "vagueness_penalty": "'edits files as needed' → max 3",
    },
    "version_strategy": {
        "description": "How the user handles file versions and backups",
        "value_space": "keep_history | overwrite | archive_old",
        "rubric": (
            "5 = Correct strategy + evidence (e.g. 'keep_history — creates backup copies before modifying, never deletes old versions')\n"
            "4 = Correct strategy + partial evidence\n"
            "3 = Generic (e.g. 'manages file versions') — cap here\n"
            "2 = Wrong strategy (e.g. 'overwrite' when GT is 'keep_history')\n"
            "1 = Opposite / not inferred"
        ),
        "vagueness_penalty": "'manages versions appropriately' → max 3",
    },
    "output_structure": {
        "description": "Structural organization of produced content",
        "value_space": "hierarchical | flat_list | narrative",
        "rubric": (
            "5 = Correct structure + evidence (e.g. 'hierarchical — multi-level headings ##/###/####, clear sections, table-of-contents style')\n"
            "4 = Correct structure + partial evidence\n"
            "3 = Generic (e.g. 'well-structured output') — cap here\n"
            "2 = Wrong structure (e.g. 'flat_list' when GT is 'hierarchical')\n"
            "1 = Opposite / not inferred"
        ),
        "vagueness_penalty": "'well-organized', 'structured output' → max 3",
    },
    "documentation": {
        "description": "Whether and how much the user creates auxiliary documentation",
        "value_space": "comprehensive | moderate | minimal",
        "rubric": (
            "5 = Correct level + evidence (e.g. 'comprehensive — creates README files, index documents, explains everything thoroughly')\n"
            "4 = Correct level + partial evidence\n"
            "3 = Generic (e.g. 'documents work') — cap here\n"
            "2 = Wrong level (e.g. 'minimal' when GT is 'comprehensive')\n"
            "1 = Opposite / not inferred"
        ),
        "vagueness_penalty": "'documents things' → max 3",
    },
    "naming": {
        "description": "File naming convention used",
        "value_space": "long_descriptive | short_abbrev | date_prefix | mixed",
        "rubric": (
            "5 = Correct convention + example evidence (e.g. 'date_prefix — uses YYYY-MM-DD prefix like 2024-01-report.md')\n"
            "4 = Correct convention without examples\n"
            "3 = Generic (e.g. 'uses clear file names') — cap here\n"
            "2 = Wrong convention (e.g. 'short_abbrev' when GT is 'long_descriptive')\n"
            "1 = Not inferred"
        ),
        "vagueness_penalty": "'descriptive names', 'clear naming' → max 3. Must identify the specific pattern.",
    },
    "cross_modal_behavior": {
        "description": "Whether and how the user creates/uses visual and tabular materials",
        "value_space": "visual_heavy | tables_and_references | balanced | text_only",
        "rubric": (
            "5 = Correct behavior + evidence (e.g. 'text_only — no tables, no images, no data files, pure prose and bullets')\n"
            "4 = Correct behavior + partial evidence\n"
            "3 = Generic (e.g. 'uses various formats') — cap here\n"
            "2 = Wrong behavior (e.g. 'visual_heavy' when GT is 'text_only')\n"
            "1 = Opposite / not inferred"
        ),
        "vagueness_penalty": "'uses appropriate formats' → max 3",
    },
}

# ---------------------------------------------------------------------------
# Per-attribute prompt templates
# ---------------------------------------------------------------------------

PER_ATTRIBUTE_SYSTEM_PROMPT = """\
You are a strict evaluator for user profiling systems. You will score ONE \
specific attribute of an inferred user profile against ground truth.

## Attribute: {attr_name}
{attr_description}

## Possible values
{value_space}

## Scoring rubric (1-5)
{rubric}

## VAGUENESS PENALTY
{vagueness_penalty}

## CRITICAL RULE
If the inferred value is vague, generic, or could describe ANY user without \
specific behavioral evidence, the MAXIMUM score is 3 — even if the vague \
description happens to be compatible with the ground truth. Scores of 4-5 \
REQUIRE concrete, distinguishing behavioral descriptions that differentiate \
this specific user from others.

Respond in JSON:
{{"ground_truth": "<GT value>", "inferred": "<inferred value>", "score": <1-5>, "justification": "<1-2 sentences>"}}"""

PER_ATTRIBUTE_USER_PROMPT = """\
## Ground Truth
Attribute: {attr_name}
Value: {gt_value}

## Inferred Value (by {method_name})
{inferred_value}

Score this single attribute according to the rubric. Respond ONLY in JSON."""


class JudgeScorer:
    """Orchestrates LLM-as-Judge evaluation across methods and profiles."""

    def __init__(self, profiles_dir: Path, output_dir: Path):
        """
        Args:
            profiles_dir: Path to profiles/ with ground truth YAMLs
            output_dir: Path to bench/scores/ for results
        """
        self.profiles_dir = profiles_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_judge_prompt(
        self,
        ground_truth: dict[str, Any],
        inferred_profile: dict[str, Any],
        method_name: str,
        attributes: list[str],
    ) -> dict[str, str]:
        """Build the judge prompt for a single evaluation.

        Returns:
            Dict with "system" and "user" prompt strings.
        """
        gt_text = json.dumps(ground_truth, indent=2, ensure_ascii=False)
        inferred_text = json.dumps(inferred_profile, indent=2, ensure_ascii=False)
        attr_list = "\n".join(f"- {a}" for a in attributes)

        return {
            "system": JUDGE_SYSTEM_PROMPT,
            "user": JUDGE_USER_PROMPT.format(
                ground_truth=gt_text,
                method_name=method_name,
                inferred_profile=inferred_text,
                attributes_list=attr_list,
            ),
        }

    def parse_judge_response(self, response_text: str) -> dict[str, Any]:
        """Parse the judge's JSON response into structured scores."""
        try:
            # Try to extract JSON from response
            text = response_text.strip()
            if text.startswith("```"):
                # Remove markdown code fences
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse judge response", "raw": response_text}

    # ------------------------------------------------------------------
    # Per-attribute judge methods (new)
    # ------------------------------------------------------------------

    def build_single_attribute_prompt(
        self,
        ground_truth: dict[str, Any],
        inferred_profile: dict[str, Any],
        method_name: str,
        attr: str,
    ) -> dict[str, str]:
        """Build a judge prompt for a single attribute.

        Returns:
            Dict with "system" and "user" prompt strings.
        """
        rubric_info = ATTRIBUTE_RUBRICS.get(attr, {})
        system = PER_ATTRIBUTE_SYSTEM_PROMPT.format(
            attr_name=attr,
            attr_description=rubric_info.get("description", attr),
            value_space=rubric_info.get("value_space", "See rubric"),
            rubric=rubric_info.get("rubric", "5=exact match, 4=close, 3=partial, 2=mostly wrong, 1=wrong"),
            vagueness_penalty=rubric_info.get("vagueness_penalty", "Generic/vague answers → max 3"),
        )
        gt_value = ground_truth.get(attr, "N/A")
        inferred_value = inferred_profile.get(attr, "N/A")
        user = PER_ATTRIBUTE_USER_PROMPT.format(
            attr_name=attr,
            gt_value=gt_value,
            method_name=method_name,
            inferred_value=inferred_value,
        )
        return {"system": system, "user": user}

    def parse_single_attribute_response(self, response_text: str, attr: str) -> dict[str, Any]:
        """Parse a single-attribute judge response.

        Returns:
            Dict with ground_truth, inferred, score, justification.
            Falls back to score=0 on parse error.
        """
        try:
            text = response_text.strip()
            # Strip markdown code fences
            if "```" in text:
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            # Find JSON object
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start : end + 1]
            parsed = json.loads(text)
            # Validate score is 1-5
            score = parsed.get("score", 0)
            if not isinstance(score, (int, float)) or score < 1 or score > 5:
                score = 0
            return {
                "ground_truth": parsed.get("ground_truth", ""),
                "inferred": parsed.get("inferred", ""),
                "score": score,
                "justification": parsed.get("justification", ""),
            }
        except (json.JSONDecodeError, ValueError):
            return {
                "ground_truth": "",
                "inferred": "",
                "score": 0,
                "justification": f"Parse error for {attr}",
            }

    def judge_all_attributes(
        self,
        ground_truth: dict[str, Any],
        inferred_profile: dict[str, Any],
        method_name: str,
        attributes: list[str],
        llm_fn: Callable[..., str],
        verbose: bool = False,
        max_workers: int = 8,
    ) -> dict[str, Any]:
        """Score each attribute with a separate LLM judge call (parallel).

        Args:
            ground_truth: GT profile dict {attr: value}
            inferred_profile: Inferred profile dict {attr: value}
            method_name: Name of the baseline method
            attributes: List of attribute names to score
            llm_fn: Callable(prompt, system_prompt, temperature, max_tokens) → str
            verbose: Print per-attribute progress
            max_workers: Max concurrent judge calls (default 8)

        Returns:
            Dict matching legacy format:
            {"scores": {attr: {ground_truth, inferred, score, justification}},
             "overall_mean": float}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _judge_one(attr: str) -> tuple[str, dict[str, Any], float]:
            prompt = self.build_single_attribute_prompt(ground_truth, inferred_profile, method_name, attr)
            t0 = time.time()
            try:
                response = llm_fn(
                    prompt=prompt["user"],
                    system_prompt=prompt["system"],
                    temperature=0.1,
                    max_tokens=512,
                )
                parsed = self.parse_single_attribute_response(response, attr)
            except Exception as e:
                parsed = {
                    "ground_truth": ground_truth.get(attr, ""),
                    "inferred": inferred_profile.get(attr, ""),
                    "score": 0,
                    "justification": f"LLM error: {e}",
                }
            return attr, parsed, time.time() - t0

        scores: dict[str, dict[str, Any]] = {}
        t_start = time.time()
        done_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_judge_one, attr): attr for attr in attributes}
            for future in as_completed(futures):
                attr, parsed, dt = future.result()
                scores[attr] = parsed
                done_count += 1
                if verbose:
                    s = parsed.get("score", 0)
                    print(f"      [{done_count}/{len(attributes)}] {attr}: {s}/5 ({dt:.1f}s)")

        total_time = time.time() - t_start

        # Compute overall mean
        valid_scores = [
            d["score"] for d in scores.values() if isinstance(d.get("score"), (int, float)) and d["score"] > 0
        ]
        overall_mean = round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else 0.0

        if verbose:
            print(f"      Judge total: {total_time:.1f}s, mean={overall_mean:.2f}/5")

        return {"scores": scores, "overall_mean": overall_mean}

    def aggregate_scores(self, all_scores: list[dict[str, Any]], method_name: str) -> dict[str, Any]:
        """Aggregate scores across profiles for a single method.

        Args:
            all_scores: List of parsed judge responses (one per profile)
            method_name: Name of the method being evaluated

        Returns:
            Aggregated results with per-attribute and overall statistics.
        """
        attr_scores: dict[str, list[float]] = {}

        for score_result in all_scores:
            scores = score_result.get("scores", {})
            for attr, detail in scores.items():
                if attr not in attr_scores:
                    attr_scores[attr] = []
                s = detail.get("score")
                if isinstance(s, (int, float)):
                    attr_scores[attr].append(float(s))

        summary = {
            "method": method_name,
            "n_profiles": len(all_scores),
            "per_attribute": {},
            "overall_mean": 0.0,
        }

        all_means = []
        for attr, scores in sorted(attr_scores.items()):
            mean_score = statistics.mean(scores) if scores else 0
            summary["per_attribute"][attr] = {
                "mean": round(mean_score, 3),
                "median": round(statistics.median(scores), 3) if scores else 0,
                "std": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0,
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
                "n": len(scores),
            }
            all_means.append(mean_score)

        summary["overall_mean"] = round(statistics.mean(all_means), 3) if all_means else 0
        return summary

    def save_results(
        self,
        method_name: str,
        per_profile_scores: dict[str, dict[str, Any]],
        aggregated: dict[str, Any],
    ) -> Path:
        """Save scoring results to disk."""
        result = {
            "method": method_name,
            "per_profile": per_profile_scores,
            "aggregated": aggregated,
        }

        output_file = self.output_dir / f"{method_name}_scores.json"
        output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_file

    def generate_comparison_table(self, all_method_results: list[dict[str, Any]]) -> str:
        """Generate a markdown comparison table across all methods.

        Returns:
            Markdown table string.
        """
        if not all_method_results:
            return "No results to compare."

        # Get all attributes from first result
        first = all_method_results[0]
        attributes = sorted(first.get("per_attribute", {}).keys())

        # Header
        header = "| Method | " + " | ".join(attributes) + " | Overall |"
        separator = "|--------|" + "|".join("-" * (len(a) + 2) for a in attributes) + "|---------|"

        rows = []
        for result in sorted(all_method_results, key=lambda r: -r.get("overall_mean", 0)):
            method = result.get("method", "?")
            per_attr = result.get("per_attribute", {})
            cells = []
            for attr in attributes:
                mean = per_attr.get(attr, {}).get("mean", 0)
                cells.append(f"{mean:.2f}")
            overall = result.get("overall_mean", 0)
            rows.append(f"| {method} | " + " | ".join(cells) + f" | {overall:.2f} |")

        return "\n".join([header, separator] + rows)
