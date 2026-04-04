#!/bin/bash
# Run a single filegram trajectory and collect results
# Usage: ./run_trajectory.sh <profile_id> <task_id> [prompt] [suffix]
# Example: ./run_trajectory.sh p15_thorough_surveyor T-04
# Example: ./run_trajectory.sh p1_methodical T-01 "" _multimodal
# If prompt is omitted or empty, auto-reads from tasks/tXX.json (picks prompt_en or prompt_zh by profile)
# NOTE: Safe for parallel execution — session matching uses profile_id in events.json

set -e

PROFILE="$1"
TASK_ID="$2"
PROMPT="$3"
SUFFIX="${4:-$TRAJECTORY_SUFFIX}"

# Derive task number from task_id (T-01 -> 01)
TASK_NUM=$(echo "$TASK_ID" | sed 's/T-//')

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Auto-read prompt from tasks JSON if not provided
if [ -z "$PROMPT" ]; then
    TASK_JSON="$PROJECT_ROOT/tasks/t${TASK_NUM}.json"
    if [ ! -f "$TASK_JSON" ]; then
        echo "ERROR: No prompt provided and task file not found: $TASK_JSON"
        exit 1
    fi
    # English profiles
    EN_PROFILES="p3_efficient_executor p6_quick_curator p8_minimal_editor p10_silent_auditor p11_meticulous_planner p14_concise_organizer p15_thorough_surveyor p16_phased_minimalist p18_decisive_scanner p20_visual_auditor"
    if echo "$EN_PROFILES" | grep -qw "$PROFILE"; then
        PROMPT=$(python3 -c "import json; d=json.load(open('$TASK_JSON')); print(d.get('prompt_en') or d.get('prompt_zh',''))")
    else
        PROMPT=$(python3 -c "import json; d=json.load(open('$TASK_JSON')); print(d.get('prompt_zh') or d.get('prompt_en',''))")
    fi
    echo "[$(date +%H:%M:%S)] Auto-loaded prompt from $TASK_JSON"
fi

SANDBOX_DIR="$PROJECT_ROOT/sandbox/${PROFILE}_${TASK_ID}${SUFFIX}"
SIGNAL_BASE="${SIGNAL_BASE:-$PROJECT_ROOT/signal}"
SIGNAL_DIR="$SIGNAL_BASE/${PROFILE}_${TASK_ID}${SUFFIX}"
WORKSPACE_SRC="$PROJECT_ROOT/workspace/t${TASK_NUM}_workspace"
SESSIONS_DIR="$PROJECT_ROOT/data/behavior/sessions"
LOG_DIR="${LOG_BASE:-$PROJECT_ROOT/logs}"
LOG_FILE="$LOG_DIR/${PROFILE}_${TASK_ID}${SUFFIX}.log"

mkdir -p "$LOG_DIR"

# Skip if trajectory already exists.
# - Complete (has session_end): skip
# - Incomplete (no session_end, e.g. LLM error): skip with warning
#   To re-run incomplete trajectories, delete their signal dirs first.
if [ -f "$SIGNAL_DIR/events.json" ]; then
    if python3 -c "
import json, sys
events = json.load(open('$SIGNAL_DIR/events.json'))
sys.exit(0 if any(e.get('event_type') == 'session_end' for e in events) else 1)
" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] SKIP: ${PROFILE} × ${TASK_ID} (completed)"
        exit 0
    else
        echo "[$(date +%H:%M:%S)] SKIP: ${PROFILE} × ${TASK_ID} (incomplete — delete signal dir to retry)"
        exit 0
    fi
fi

# Tee all output to both stdout and log file
exec > >(tee "$LOG_FILE") 2>&1

MODEL="${ANTHROPIC_MODEL:-$(python3 -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('ANTHROPIC_MODEL','claude-haiku-4-5-20251001'))")}"
echo "[$(date +%H:%M:%S)] Starting: ${PROFILE} × ${TASK_ID} | Model: ${MODEL}"

# Record timestamp before run (for narrowing session search)
BEFORE_TS=$(date +%s)

# Create sandbox
rm -rf "$SANDBOX_DIR"
mkdir -p "$SANDBOX_DIR"

# Copy workspace files (if any exist beyond .gitkeep)
if [ -d "$WORKSPACE_SRC" ]; then
    FILE_COUNT=$(find "$WORKSPACE_SRC" -maxdepth 1 -not -name '.gitkeep' -not -name '.' -not -name '..' | wc -l)
    if [ "$FILE_COUNT" -gt 0 ]; then
        cp -r "$WORKSPACE_SRC"/* "$SANDBOX_DIR/" 2>/dev/null || true
        # Copy hidden .annotation directory (not matched by *)
        cp -r "$WORKSPACE_SRC"/.annotation "$SANDBOX_DIR/" 2>/dev/null || true
        # Remove .gitkeep if copied
        rm -f "$SANDBOX_DIR/.gitkeep"
    fi
fi

echo "[$(date +%H:%M:%S)] Sandbox ready: $(ls "$SANDBOX_DIR" | wc -l) files"

# Run filegram
cd "$PROJECT_ROOT"
filegramengine -1 --autonomous -d "$SANDBOX_DIR" -p "$PROFILE" "$PROMPT" 2>&1 || {
    echo "[$(date +%H:%M:%S)] ERROR: filegramengine failed for ${PROFILE} × ${TASK_ID}"
    exit 1
}

echo "[$(date +%H:%M:%S)] Filegram completed"

# Find the new session by matching BOTH profile_id AND target_directory (parallel-safe)
# target_directory in session_start event contains the unique sandbox path per trajectory
EXPECTED_SANDBOX="$SANDBOX_DIR"
NEW_SESSION=""
for sess_dir in "$SESSIONS_DIR"/*/; do
    [ -d "$sess_dir" ] || continue
    events_file="$sess_dir/events.json"
    [ -f "$events_file" ] || continue
    # Only check sessions created after BEFORE_TS (by file modification time)
    file_ts=$(stat -f %m "$events_file" 2>/dev/null || stat -c %Y "$events_file" 2>/dev/null || echo 0)
    [ "$file_ts" -ge "$BEFORE_TS" ] || continue
    # Match profile_id AND target_directory (sandbox path is unique per trajectory)
    if python3 -c "
import json, sys
events = json.load(open('$events_file'))
if not events:
    sys.exit(1)
first = events[0]
if first.get('profile_id') == '$PROFILE' and first.get('target_directory', '').rstrip('/') == '$EXPECTED_SANDBOX'.rstrip('/'):
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
        NEW_SESSION=$(basename "$sess_dir")
        break
    fi
done

if [ -z "$NEW_SESSION" ]; then
    echo "[$(date +%H:%M:%S)] ERROR: No new session found for profile $PROFILE"
    exit 1
fi

echo "[$(date +%H:%M:%S)] New session: $NEW_SESSION"

# Collect trajectory
mkdir -p "$SIGNAL_DIR"
cp -r "$SESSIONS_DIR/$NEW_SESSION"/* "$SIGNAL_DIR/"

# Validate
EVENT_COUNT=$(python3 -c "import json; f=open('$SIGNAL_DIR/events.json'); print(len(json.load(f)))")
echo "[$(date +%H:%M:%S)] DONE: ${PROFILE} × ${TASK_ID} → $EVENT_COUNT events"

# Cleanup sandbox
rm -rf "$SANDBOX_DIR"
