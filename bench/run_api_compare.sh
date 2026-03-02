#!/usr/bin/env bash
# ===========================================================================
# Quick A/B comparison: GPT-4.1 vs Gemini 2.5 Pro
#
# Runs a small subset (3 profiles × 4 methods) with both APIs,
# then prints side-by-side results.
#
# Usage:
#   ./bench/run_api_compare.sh
#   ./bench/run_api_compare.sh --profiles p1_methodical,p3_efficient_executor
#   ./bench/run_api_compare.sh --methods filegramos_simple,filegramos
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."  # project root

# Defaults — small scale
PROFILES="${PROFILES:-p1_methodical,p3_efficient_executor,p5_balanced_organizer}"
METHODS="${METHODS:-filegramos_simple,filegramos,eager_summarization,full_context}"
PARALLEL=2

# Parse overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profiles) PROFILES="$2"; shift 2 ;;
        --methods)  METHODS="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "==========================================="
echo " API Comparison: GPT-4.1 vs Gemini 2.5 Pro"
echo " Profiles: $PROFILES"
echo " Methods:  $METHODS"
echo "==========================================="

# ── Run with Azure (GPT-4.1) ──────────────────────────────
echo ""
echo ">>> Phase 1: Azure / GPT-4.1"
echo "-------------------------------------------"
python3 bench/test_baselines.py \
    --api azure \
    --methods "$METHODS" \
    --profiles "$PROFILES" \
    --parallel "$PARALLEL" \
    --no-cache

# ── Run with Gemini ────────────────────────────────────────
echo ""
echo ">>> Phase 2: Gemini 2.5 Pro"
echo "-------------------------------------------"
python3 bench/test_baselines.py \
    --api gemini \
    --methods "$METHODS" \
    --profiles "$PROFILES" \
    --parallel "$PARALLEL" \
    --no-cache

# ── Compare results ────────────────────────────────────────
echo ""
echo "==========================================="
echo " Comparison Summary"
echo "==========================================="
python3 -c "
import json
from pathlib import Path

import os
_azure_model = os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'gpt-4.1')
_gemini_model = os.environ.get('GEMINI_MODEL_BASELINE', 'gemini-2.5-pro')
_azure_slug = _azure_model.replace('-', '') if _azure_model.startswith('gpt') else _azure_model.replace('-', '_')
_gemini_slug = _gemini_model.replace('-', '') if _gemini_model.startswith('gpt') else _gemini_model.replace('-', '_')
results_azure = Path(f'bench/test_results_{_azure_slug}')
results_gemini = Path(f'bench/test_results_{_gemini_slug}')

profiles = '$PROFILES'.split(',')
methods = '$METHODS'.split(',')

print()
print(f'{'Method':<25s} {'GPT-4.1':>10s} {'Gemini':>10s} {'Delta':>10s}')
print('-' * 55)

for method in methods:
    azure_scores = []
    gemini_scores = []
    for pid in profiles:
        for rdir, scores_list in [(results_azure, azure_scores), (results_gemini, gemini_scores)]:
            rfile = rdir / pid / 'results.json'
            if not rfile.exists():
                continue
            data = json.loads(rfile.read_text())
            mdata = data.get(method, {})
            judge = mdata.get('judge_scores', {})
            for attr, info in judge.items():
                s = info.get('score')
                if isinstance(s, (int, float)):
                    scores_list.append(s)

    avg_a = sum(azure_scores) / len(azure_scores) if azure_scores else 0
    avg_g = sum(gemini_scores) / len(gemini_scores) if gemini_scores else 0
    delta = avg_g - avg_a
    sign = '+' if delta > 0 else ''
    print(f'{method:<25s} {avg_a:>10.3f} {avg_g:>10.3f} {sign}{delta:>9.3f}')

print()
print(f'Azure results ({_azure_model}):  {results_azure}/')
print(f'Gemini results ({_gemini_model}): {results_gemini}/')
"

echo ""
echo "Done!"
