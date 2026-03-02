#!/usr/bin/env python3
"""Automated parameter tuning experiments for FileGramOS.

Runs multiple ablation experiments by modifying tuning.py, clearing caches,
rebuilding, and running QA eval.

Usage:
    python bench/run_tuning.py                  # run all experiments
    python bench/run_tuning.py --exp A B        # run specific experiments
    python bench/run_tuning.py --list           # list experiment configs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
_ROOT = _BENCH.parent

# ── Experiment configurations ──
# Each config is a dict of parameter_name -> value
# matching the variable names in tuning.py

BASELINE = {
    "PREVIEW_FILE_CHARS": 500,
    "PREVIEW_DIFF_CHARS": 300,
    "LLM_ENCODE_MAX_FILES": 4,
    "LLM_ENCODE_FILE_CHARS": 600,
    "LLM_ENCODE_MAX_EDITS": 3,
    "LLM_ENCODE_EDIT_CHARS": 400,
    "SEMANTIC_BUDGET": 8000,
    "RETRIEVER_DISPLAY_CHARS": 300,
}

EXPERIMENTS = {
    "A": {
        "name": "original_baseline",
        "desc": "Revert to original params (reproduce 52.5/55.5 baseline)",
        "params": {**BASELINE},
    },
    "B": {
        "name": "preview_750",
        "desc": "Only increase preview (750/400), everything else original",
        "params": {**BASELINE, "PREVIEW_FILE_CHARS": 750, "PREVIEW_DIFF_CHARS": 400},
    },
    "C": {
        "name": "llm_encode_up",
        "desc": "Only increase LLM encoding (5x800, 3x500)",
        "params": {**BASELINE, "LLM_ENCODE_MAX_FILES": 5, "LLM_ENCODE_FILE_CHARS": 800, "LLM_ENCODE_EDIT_CHARS": 500},
    },
    "D": {
        "name": "budget_12k",
        "desc": "Only increase sampler budget (12000)",
        "params": {**BASELINE, "SEMANTIC_BUDGET": 12000},
    },
    "E": {
        "name": "display_500",
        "desc": "Only increase retriever display (500) — affects formal only",
        "params": {**BASELINE, "RETRIEVER_DISPLAY_CHARS": 500},
    },
    "F": {
        "name": "display_800",
        "desc": "Increase retriever display to 800",
        "params": {**BASELINE, "RETRIEVER_DISPLAY_CHARS": 800},
    },
    "G": {
        "name": "display_1000",
        "desc": "Increase retriever display to 1000",
        "params": {**BASELINE, "RETRIEVER_DISPLAY_CHARS": 1000},
    },
    "H": {
        "name": "display_500_preview_750",
        "desc": "Display 500 + preview 750 (retriever sees longer content)",
        "params": {**BASELINE, "PREVIEW_FILE_CHARS": 750, "PREVIEW_DIFF_CHARS": 400, "RETRIEVER_DISPLAY_CHARS": 500},
    },
}

TUNING_PY = _BENCH / "filegramos" / "tuning.py"
CACHE_DIR = _BENCH / "ingest_cache_gemini_2.5_flash"
RESULTS_FILE = _BENCH / "tuning_results.json"


def write_tuning_config(params: dict[str, int]):
    """Overwrite tuning.py with the given parameter values."""
    content = '"""Centralized tunable parameters for ablation experiments.\n\n'
    content += "All downstream files import from here so experiments only need to\n"
    content += 'modify this single file.\n"""\n\n'
    content += "# ── Feature Extraction (feature_extraction.py) ──\n"
    content += f"PREVIEW_FILE_CHARS = {params['PREVIEW_FILE_CHARS']}\n"
    content += f"PREVIEW_DIFF_CHARS = {params['PREVIEW_DIFF_CHARS']}\n\n"
    content += "# ── Encoder LLM encoding (encoder.py) ──\n"
    content += f"LLM_ENCODE_MAX_FILES = {params['LLM_ENCODE_MAX_FILES']}\n"
    content += f"LLM_ENCODE_FILE_CHARS = {params['LLM_ENCODE_FILE_CHARS']}\n"
    content += f"LLM_ENCODE_MAX_EDITS = {params['LLM_ENCODE_MAX_EDITS']}\n"
    content += f"LLM_ENCODE_EDIT_CHARS = {params['LLM_ENCODE_EDIT_CHARS']}\n\n"
    content += "# ── Sampler / Consolidator (sampler.py, consolidator.py) ──\n"
    content += f"SEMANTIC_BUDGET = {params['SEMANTIC_BUDGET']}\n\n"
    content += "# ── Retriever display (retriever.py) ──\n"
    content += f"RETRIEVER_DISPLAY_CHARS = {params['RETRIEVER_DISPLAY_CHARS']}\n"

    TUNING_PY.write_text(content, encoding="utf-8")
    print(f"  tuning.py updated: {params}")


def clear_filegramos_caches():
    """Delete all filegramos*.pkl caches to force rebuild."""
    count = 0
    for pkl in CACHE_DIR.rglob("filegramos*.pkl"):
        pkl.unlink()
        count += 1
    print(f"  Deleted {count} filegramos*.pkl cache files")
    return count


def run_ablation() -> dict:
    """Run rebuild_caches + run_qa via run_ablation.py and parse results."""
    print("  Running ablation (rebuild caches + QA eval)...")
    t0 = time.time()

    # Import and run directly to capture output
    ret = os.system(f"cd {_ROOT} && python bench/run_ablation.py 2>&1 | tee /tmp/tuning_run.log")
    elapsed = time.time() - t0
    print(f"  Ablation completed in {elapsed:.0f}s (exit code {ret})")

    # Parse results from the QA eval output
    results = parse_qa_results("/tmp/tuning_run.log")
    return results


def parse_qa_results(log_path: str) -> dict:
    """Parse QA eval results from log output.

    The results table format is:
    filegramos_simple        49.0%     —   48.1%    78.3%    48.7%    39.6%     — |    58.8%    66.7%    42.8%  |    52.6%
    filegramos               45.6%     —   46.4%    85.7%    56.4%    38.9%     — |    60.8%    61.7%    45.0%  |    54.3%
    """
    results = {}
    try:
        with open(log_path, encoding="utf-8") as f:
            text = f.read()

        for line in text.split("\n"):
            line = line.strip()
            # Match lines starting with method name followed by percentages
            for method in ("filegramos_simple", "filegramos"):
                if not line.startswith(method):
                    continue
                # Avoid partial match: "filegramos" should not match "filegramos_simple"
                if method == "filegramos" and line.startswith("filegramos_simple"):
                    continue
                # Extract all percentage values from the line
                pcts = re.findall(r"(\d+\.\d+)%", line)
                if len(pcts) >= 6:
                    # Format: AttrRec BehavInf TraceDis AnomDet ShiftAna ... Proc Sem Epi Avg
                    # The last value is the overall average
                    results[method] = {
                        "AttrRec": float(pcts[0]),
                        "BehavInf": float(pcts[1]),
                        "TraceDis": float(pcts[2]),
                        "AnomDet": float(pcts[3]),
                        "ShiftAna": float(pcts[4]),
                        "Proc": float(pcts[5]) if len(pcts) > 5 else 0,
                        "Sem": float(pcts[6]) if len(pcts) > 6 else 0,
                        "Epi": float(pcts[7]) if len(pcts) > 7 else 0,
                        "Overall": float(pcts[-1]),  # last pct is always Avg
                    }

    except Exception as e:
        print(f"  Warning: could not parse results: {e}")

    return results


def run_experiment(exp_id: str, exp_config: dict) -> dict:
    """Run a single experiment."""
    print(f"\n{'=' * 70}")
    print(f"Experiment {exp_id}: {exp_config['name']}")
    print(f"  {exp_config['desc']}")
    print(f"{'=' * 70}")

    # 1. Write tuning config
    write_tuning_config(exp_config["params"])

    # 2. Clear caches
    clear_filegramos_caches()

    # 3. Run ablation
    results = run_ablation()

    # 4. Record
    record = {
        "exp_id": exp_id,
        "name": exp_config["name"],
        "desc": exp_config["desc"],
        "params": exp_config["params"],
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f"\n  Results for {exp_id} ({exp_config['name']}):")
    for method, scores in results.items():
        overall = scores.get("Overall", "?")
        print(f"    {method}: {overall}%")

    return record


def main():
    parser = argparse.ArgumentParser(description="Automated parameter tuning")
    parser.add_argument("--exp", nargs="+", help="Experiment IDs to run (e.g., A B C)")
    parser.add_argument("--list", action="store_true", help="List experiments and exit")
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        print(f"{'ID':<4} {'Name':<25} {'Description'}")
        print("-" * 70)
        for eid, cfg in EXPERIMENTS.items():
            diff = {k: v for k, v in cfg["params"].items() if v != BASELINE.get(k)}
            diff_str = ", ".join(f"{k}={v}" for k, v in diff.items()) or "(all original)"
            print(f"{eid:<4} {cfg['name']:<25} {diff_str}")
        return

    exp_ids = args.exp if args.exp else list(EXPERIMENTS.keys())

    # Validate
    for eid in exp_ids:
        if eid not in EXPERIMENTS:
            print(f"Unknown experiment: {eid}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            return

    # Load existing results
    all_results = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            all_results = json.load(f)

    print(f"Running {len(exp_ids)} experiments: {', '.join(exp_ids)}")
    print(f"Cache dir: {CACHE_DIR}")

    for eid in exp_ids:
        record = run_experiment(eid, EXPERIMENTS[eid])
        all_results.append(record)

        # Save after each experiment (in case of interruption)
        RESULTS_FILE.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Exp':<4} {'Name':<25} {'simple':<10} {'formal':<10} {'Δ simple':<10} {'Δ formal':<10}")
    print("-" * 70)

    baseline_simple = 52.5
    baseline_formal = 55.5

    for r in all_results[-len(exp_ids) :]:
        eid = r["exp_id"]
        name = r["name"]
        s = r["results"].get("filegramos_simple", {}).get("Overall", "?")
        f = r["results"].get("filegramos", {}).get("Overall", "?")
        ds = f"{s - baseline_simple:+.1f}" if isinstance(s, (int, float)) else "?"
        df = f"{f - baseline_formal:+.1f}" if isinstance(f, (int, float)) else "?"
        print(f"{eid:<4} {name:<25} {s:<10} {f:<10} {ds:<10} {df:<10}")

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
