#!/usr/bin/env bash
# ===========================================================================
# FileGramBench Evaluation Runner
#
# Usage:
#   ./bench/run_eval.sh                          # full run (all methods, all profiles)
#   ./bench/run_eval.sh --methods filegramos     # only new formal adapter
#   ./bench/run_eval.sh --methods filegramos,filegramos_simple  # A/B comparison
#   ./bench/run_eval.sh --status                 # check completion status (default: gpt4.1)
#   ./bench/run_eval.sh --status --cache-dir gemini_2.5_pro
#   ./bench/run_eval.sh --phase 1                # ingest+infer only (no judge)
#   ./bench/run_eval.sh --qa --cache-dir gpt4.1 --api gemini
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."  # project root

# ── Parse flags ──────────────────────────────────────────────
RUN_STATUS=false
RUN_QA=false
CACHE_DIR_SLUG=""
BASELINE_ARGS=()
QA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --status)    RUN_STATUS=true; shift ;;
        --qa)        RUN_QA=true; shift ;;
        --api)       QA_ARGS+=(--api "$2"); shift 2 ;;
        --cache-dir) CACHE_DIR_SLUG="$2"; shift 2 ;;
        *)           BASELINE_ARGS+=("$1"); shift ;;
    esac
done

# ── Resolve model slug for directory paths ──────────────────
if [[ -z "$CACHE_DIR_SLUG" ]]; then
    # Default: derive from AZURE_OPENAI_DEPLOYMENT env var
    _model="${AZURE_OPENAI_DEPLOYMENT:-gpt-4.1}"
    if [[ "$_model" == gpt* ]]; then
        CACHE_DIR_SLUG="${_model//-/}"
    else
        CACHE_DIR_SLUG="${_model//-/_}"
    fi
fi

# ── Status check ─────────────────────────────────────────────
if $RUN_STATUS; then
    echo "==========================================="
    echo " FileGramBench Completion Status"
    echo " Cache: ingest_cache_${CACHE_DIR_SLUG}"
    echo "==========================================="
    python3 -c "
import json, sys
from pathlib import Path

slug = '${CACHE_DIR_SLUG}'
results_dir = Path(f'bench/test_results_{slug}')
cache_dir = Path(f'bench/ingest_cache_{slug}')

if not results_dir.exists() and not cache_dir.exists():
    print(f'\n  No data found for slug \"{slug}\".')
    print(f'  Expected: {results_dir}/ and {cache_dir}/')
    print(f'  Run test_baselines.py first, or use --cache-dir <slug>.')
    sys.exit(0)

profiles = sorted([d.name for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('p')]) if results_dir.exists() else []
# Also include profiles that have cache but no results yet
if cache_dir.exists():
    cached_only = sorted([d.name for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith('p') and d.name not in profiles])
    profiles = sorted(set(profiles) | set(cached_only))

methods = [
    'naive_rag', 'eager_summarization', 'mem0', 'zep', 'memos',
    'memu', 'evermemos', 'filegramos_simple', 'filegramos', 'full_context',
]

print(f'\n  Profiles: {len(profiles)}')
print(f'  Methods:  {len(methods)}')
print(f'  Total:    {len(profiles) * len(methods)} jobs\n')

# Header
hdr = f\"{'Profile':<25s}\"
for m in methods:
    short = m[:8]
    hdr += f'{short:>9s}'
print(hdr)
print('-' * (25 + 9 * len(methods)))

method_done = {m: 0 for m in methods}
method_cached = {m: 0 for m in methods}

for pid in profiles:
    row = f'{pid:<25s}'
    for m in methods:
        inferred = (results_dir / pid / f'{m}_inferred.json').exists()
        cached = (cache_dir / pid / f'{m}.pkl').exists()
        judged = False
        try:
            r = json.loads((results_dir / pid / 'results.json').read_text())
            judged = m in r and 'judge_scores' in r[m]
        except Exception:
            pass

        if judged:
            symbol = '\033[92mJ\033[0m'  # green J = judged
            method_done[m] += 1
        elif inferred:
            symbol = '\033[93mI\033[0m'  # yellow I = inferred
            method_done[m] += 1
        elif cached:
            symbol = '\033[96mC\033[0m'  # cyan C = cached
            method_cached[m] += 1
        else:
            symbol = '\033[91m.\033[0m'  # red . = missing
        row += f'{symbol:>18s}'  # 18 because ANSI codes add invisible chars
    print(row)

print()
print('Legend: \033[92mJ\033[0m=judged  \033[93mI\033[0m=inferred  \033[96mC\033[0m=cached  \033[91m.\033[0m=missing')
print()

total_done = sum(method_done.values())
total_jobs = len(profiles) * len(methods)
print(f'Overall: {total_done}/{total_jobs} inferred/judged, {sum(method_cached.values())} cached-only')
print()
for m in methods:
    print(f'  {m:<25s} {method_done[m]:>3d}/{len(profiles)} done')
"
    exit 0
fi

# ── Run baseline evaluation ──────────────────────────────────
echo "==========================================="
echo " FileGramBench Evaluation"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "==========================================="

python3 bench/test_baselines.py "${BASELINE_ARGS[@]}"

# ── Optionally run QA eval ────────────────────────────────────
if $RUN_QA; then
    echo ""
    echo "==========================================="
    echo " FileGramQA MCQ Evaluation"
    echo "==========================================="

    # Inherit --methods from baseline args if present
    QA_METHODS=""
    for i in "${!BASELINE_ARGS[@]}"; do
        if [[ "${BASELINE_ARGS[$i]}" == "--methods" ]]; then
            QA_METHODS="${BASELINE_ARGS[$((i+1))]}"
        fi
    done

    QA_CMD=(python3 -m filegramQA.run_qa_eval --cache-dir "$CACHE_DIR_SLUG" --parallel 4)
    if [[ -n "$QA_METHODS" ]]; then
        QA_CMD+=(--methods $QA_METHODS)
    fi
    QA_CMD+=("${QA_ARGS[@]}")

    echo "Running: ${QA_CMD[*]}"
    "${QA_CMD[@]}"
fi

echo ""
echo "Done! Run './bench/run_eval.sh --status' to check completion."
