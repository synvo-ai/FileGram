# FileGram

Behavioral data generation engine for **FileGram** — a multimodal memory framework that grounds memory in file evolution and access patterns.

FileGram uses profiled code agents to simulate realistic human work behavior. The core loop: **Task + Profile + Environment -> Agent execution -> Bottom-up file-level signals -> Memory modeling**.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Vision](#vision)
- [Architecture](#architecture)
- [Memory Signal Mapping](#memory-signal-mapping)
- [Profile System](#profile-system)
- [Data Output Format](#data-output-format)
- [Experiment Runner](#experiment-runner)
- [Project Structure](#project-structure)
- [Development Setup](#development-setup)
- [License](#license)

---

## Quick Start

### Install

```bash
uv pip install -e .
```

### Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Run

```bash
# One-shot mode with a profile
filegram -d playground/task1_alex "Create a config validator"

# Interactive mode
filegram -d /path/to/project -i

# Switch profile at runtime
/profile alex
```

---

## Vision

These signals feed three memory types in the downstream FileGramOS pipeline:

| Memory Type | What It Models | Signal Source |
|-------------|---------------|---------------|
| **Procedural** | How you do things — tool preferences, exploration strategies, workflow patterns | File operation sequences, tool call ordering, search patterns |
| **Semantic** | What changed and why — file content evolution, changelogs | File write/edit diffs, before/after hashes, diff summaries |
| **Episodic** | What happened over time — long-range behavioral consistency | Cross-session patterns, iteration timing, decision rhythm |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         INPUT                                │
│                                                              │
│   Task Prompt            Profile             Environment     │
│   "Create a validator"   alex.yaml           playground/     │
│                          (persona,            task1_alex/    │
│                           work habits,                       │
│                           coding style)                      │
└──────────┬───────────────────┬──────────────────┬───────────┘
           │                   │                  │
           ▼                   ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 FileGram Agent Runtime                       │
│                                                              │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────────────┐  │
│  │ LLM Loop │──│ Tools     │──│ BehaviorCollector        │  │
│  │ (GPT-4.1,│  │ read/     │  │ • Real-time event capture│  │
│  │  Claude,  │  │ write/    │  │ • File hash tracking     │  │
│  │  etc.)    │  │ edit/     │  │ • Revisit interval calc  │  │
│  │          │  │ bash/     │  │ • Context switch detect  │  │
│  │          │  │ grep/     │  │ • Session statistics     │  │
│  │          │  │ glob      │  │                          │  │
│  └──────────┘  └───────────┘  └──────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT                                     │
│                                                              │
│  data/behavior/sessions/{session_id}/                        │
│  ├── events.json        # Structured behavioral signals      │
│  ├── summary.json       # Aggregated session statistics      │
│  ├── summary.md         # Markdown conversation log          │
│  └── media/             # Externalized file content          │
│      ├── 0001_write.md  #   Write content snapshots          │
│      ├── 0002_old.md    #   Edit before-state                │
│      └── 0003_new.md    #   Edit after-state                 │
│                                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              FileGramOS Memory Pipeline                       │
│                                                              │
│  signals → procedural memory (file operation patterns)       │
│          → semantic memory   (file changelogs & evolution)   │
│          → episodic memory   (cross-session behavior)        │
│          → personalized user model                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Memory Signal Mapping

### Procedural Memory Signals

| Event Type | Key Fields | What It Reveals |
|-----------|-----------|-----------------|
| `file_read` | `file_path`, `view_count`, `revisit_interval_ms`, `view_range` | Exploration strategy — breadth-first vs depth-first, which files get re-read |
| `file_search` | `search_type`, `query`, `files_matched`, `files_opened_after` | Information seeking behavior — what patterns are searched, grep vs glob preference |
| `tool_call` | `tool_name`, `sequence_position`, `execution_time_ms`, `retry_count` | Tool preference and workflow — which tools used in what order, error recovery |
| `context_switch` | `from_file`, `to_file`, `trigger`, `switch_count` | Navigation patterns — how the agent moves between files |
| `iteration_end` | `tools_called`, `duration_ms`, `has_tool_error` | Work rhythm — tool density per iteration, error tolerance |

### Semantic Memory Signals

| Event Type | Key Fields | What It Reveals |
|-----------|-----------|-----------------|
| `file_write` | `file_path`, `operation`, `content_length`, `before_hash`, `after_hash` | Content creation — new files vs overwrites, file size patterns |
| `file_edit` | `edit_tool`, `lines_added/deleted/modified`, `diff_summary`, `before_hash`, `after_hash` | Content evolution — incremental refinement vs large rewrites, edit tool preference |

### Episodic Memory Signals

| Event Type | Key Fields | What It Reveals |
|-----------|-----------|-----------------|
| `llm_response` | `response_time_ms`, `input_tokens`, `output_tokens`, `has_reasoning`, `stop_reason` | Decision-making rhythm — reasoning density, token efficiency |
| `iteration_start/end` | `iteration_number`, `duration_ms` | Session pacing — how many iterations to reach a solution |
| `compaction_triggered` | `reason`, `messages_before/after`, `tokens_saved` | Context management — when the agent hits limits |
| `session_start/end` | session-level timing | Overall session structure |

---

## Profile System

Profiles live in `filegram/profile/profiles/*.yaml` and define agent personas that produce differentiated behavioral data.

### Current Profiles

| Profile | Persona | Key Behavioral Traits |
|---------|---------|----------------------|
| **alex** | The Meticulous Craftsman (Chinese, 28) | Detail-oriented, thorough docs, Chinese comments, defensive error handling |
| **luna** | The Creative Explorer (Japanese, 25) | Enthusiastic, aggressive refactoring, tries new patterns, balanced verbosity |
| **sam** | The Pragmatic Problem Solver (American, 32) | Ship fast, minimal comments, concise, 80/20 focus |

### Profile Fields

```yaml
basic:
  name: Alex
  age: 28
  role: Senior Software Engineer
  nationality: Chinese
  language: Chinese

personality:
  traits: [detail-oriented, patient, methodical]
  tone: professional
  humor_level: low
  emoji_usage: minimal
  verbosity: detailed

work_habits:
  coding_style: clean
  comment_preference: detailed
  testing_approach: thorough
  refactoring_tendency: moderate
  error_handling: defensive
  documentation: comprehensive
  preferences: [...]
  avoidances: [...]

greeting: |
  First-person introduction in character

system_prompt_addition: |
  Identity reinforcement injected into system prompt
```

### Data Quality Goal

Running the **same task** with **different profiles** should produce measurably differentiated behavioral data:
- Alex reads more files before writing, adds more comments, writes longer docs
- Luna tries creative patterns, refactors aggressively, uses friendly tone
- Sam writes minimal code fast, skips docs, uses pragmatic tool choices

---

## Data Output Format

### events.json

JSON array of events. Each event has common metadata + type-specific data fields:

```json
{
  "event_id": "uuid",
  "event_type": "file_read",
  "timestamp": 1770542208476.703,
  "session_id": "uuid",
  "profile_id": "alex",
  "message_id": "uuid",
  "model_provider": "azure_openai",
  "model_name": "azure_openai/gpt-4.1",
  "file_path": "src/validator.ts",
  "file_type": "ts",
  "directory_depth": 1,
  "view_count": 2,
  "view_range": [1, 50],
  "content_length": 1234,
  "revisit_interval_ms": 45000
}
```

### Active Event Types (11 implemented)

| Event Type | Key Data Fields |
|-----------|-----------------|
| `file_read` | file_path, view_count, revisit_interval_ms, view_range, content_length |
| `file_write` | file_path, operation (create/overwrite), content_length, before/after_hash |
| `file_edit` | file_path, edit_tool, lines_added/deleted/modified, diff_summary, before/after_hash |
| `file_search` | search_type (grep/glob), query, files_matched, files_opened_after |
| `tool_call` | tool_name, tool_parameters, execution_time_ms, success, error_type, retry_count, sequence_position |
| `iteration_start` | iteration_number |
| `iteration_end` | iteration_number, duration_ms, tools_called, has_tool_error |
| `llm_response` | response_time_ms, input/output_tokens, has_reasoning, stop_reason |
| `context_switch` | from_file, to_file, trigger, switch_count |
| `compaction_triggered` | reason, messages_before/after, tokens_saved |
| `session_start/end` | (metadata only) |

---

## Experiment Runner

Batch experiment runner for generating behavioral data across task x profile combinations.

```bash
# Run all profiles x all tasks
python experiments/run.py

# Filter by profile or task
python experiments/run.py --profile alex
python experiments/run.py --task task1

# Dry run (show combinations without executing)
python experiments/run.py --dry-run
```

Configuration is in `experiments/config.json`. Each run:
1. Sets up an isolated workspace directory with git init
2. Executes the agent with the specified profile and task
3. Captures behavioral signals to `data/experiments/`
4. Enforces a configurable timeout (default 300s)

---

## Project Structure

```
filegram/
├── agent/          # Agent loop, orchestration, BehaviorCollector integration
├── behavior/       # Behavioral signal collection
│   ├── events.py   #   Event type definitions (EventType enum, dataclasses)
│   ├── collector.py #  BehaviorCollector (real-time recording, session stats)
│   └── exporter.py #   BehaviorExporter (JSON output, media externalization)
├── tools/          # Tool implementations (read, write, edit, grep, glob, bash, etc.)
├── profile/        # Profile system
│   ├── loader.py   #   ProfileLoader (YAML parsing, system prompt injection)
│   └── profiles/   #   alex.yaml, luna.yaml, sam.yaml
├── skill/          # Skill system (SKILL.md loader/parser)
├── llm/            # LLM provider integrations (Azure OpenAI, Anthropic, OpenAI)
├── auth/           # Authentication system
├── storage/        # Persistent JSON storage
├── session/        # Session management and revert
├── snapshot/       # File state tracking and rollback
├── compaction/     # Context window compression
├── permission/     # Access control
├── instruction/    # AGENTS.md loader
├── context/        # Token counting
├── console/        # Console UI
├── prompts/        # Prompt templates (provider-specific)
├── models/         # Data models (messages, tools)
├── mcp/            # MCP server support
├── bus/            # Event bus
├── file/           # File utilities
├── utils/          # Utilities
├── config.py       # Configuration and env vars
└── main.py         # Entry point

experiments/        # Batch experiment runner
├── config.json     #   Task x profile configuration
└── run.py          #   Parallel experiment orchestration

data/
└── behavior/
    └── sessions/
        └── {session_id}/
            ├── events.json     # Behavioral signal log
            ├── summary.json    # Session statistics
            ├── summary.md      # Markdown conversation log
            └── media/          # Externalized file content

playground/         # Workspace directories for agent tasks
```

---

## Development Setup

### Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

### Install

```bash
uv pip install -e .
```

### Linting & Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [detect-secrets](https://github.com/Yelp/detect-secrets) for preventing API key leaks. Both are automated via [pre-commit](https://pre-commit.com/).

#### One-time setup

```bash
# Install dev tools
uv pip install pre-commit detect-secrets

# Generate secrets baseline (marks existing non-sensitive patterns)
detect-secrets scan > .secrets.baseline

# Install git hooks (runs checks automatically on every commit)
pre-commit install
```

#### What happens on `git commit`

After setup, every `git commit` automatically runs:

1. **ruff check** — Lint errors, import sorting, naming conventions (auto-fixes where possible)
2. **ruff format** — Code formatting (Black-compatible)
3. **detect-secrets** — Blocks commits containing hardcoded API keys, tokens, or passwords

If any check fails, the commit is blocked. Fix the issues and commit again.

#### Manual usage

```bash
# Lint (with auto-fix)
ruff check . --fix

# Format
ruff format .

# Check for secrets
detect-secrets scan
```

#### Configuration

- Ruff config: `pyproject.toml` under `[tool.ruff]`
- Pre-commit hooks: `.pre-commit-config.yaml`
- Secrets baseline: `.secrets.baseline`

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and fill in your API keys.

| Variable | Description |
|----------|-------------|
| `SYNVOCODE_LLM_PROVIDER` | LLM provider: `azure_openai`, `anthropic`, or `openai` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | Azure deployment name |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ANTHROPIC_MODEL` | Anthropic model (default: `claude-sonnet-4-20250514`) |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | OpenAI model (default: `gpt-4o`) |
| `EXA_API_KEY` | Exa API key (for web/code search) |

See `.env.example` for the full list of configuration options.

---

## Key Conventions

- Tools inherit from `BaseTool` in `tools/base.py`; all `execute()` methods are async
- Tools access BehaviorCollector via `ToolContext` for event recording
- Profiles are YAML files in `profile/profiles/`
- Skills are `SKILL.md` files with YAML frontmatter
- Prompt templates are `.txt` files in `prompts/`
- Use dataclasses for models, not Pydantic (except config)

---

## License

MIT
