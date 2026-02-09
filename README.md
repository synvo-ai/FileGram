# FileGram

**File System Memory as Behavioral Engrams for Personalized Agentic Reasoning**

This repository contains the **FileGram** behavioral data generation pipeline and the **FileGramOS** bottom-up memory framework. FileGram treats file-level behavioral traces as *engrams* — persistent memory traces grounded in file evolution and access patterns — rather than explicit user statements.

> **Paper**: *FileGram: File System Memory as Behavioral Engrams for Personalized Agentic Reasoning* (ECCV 2026 submission)

---

## Overview

The FileGram project is a three-part framework for file-system behavioral memory:

| Component | What It Does | Status |
|-----------|-------------|--------|
| **FileGram** | Persona-driven behavioral data generation pipeline | This repo |
| **FileGramBench** | Multimodal file-system memory benchmark | In progress |
| **FileGramOS** | Bottom-up memory framework (procedural / semantic / episodic) | In progress |

### Core Idea

Current memory systems for LLM agents are *interaction-driven*: they store and retrieve explicit dialogue records. But in OS-level settings, the most valuable personal signals — work habits, exploration strategies, project structure preferences — are never stated explicitly. They are distributed across **behavioral traces** in the file system.

FileGram takes a **bottom-up** approach:

```
File-level behavioral signals  -->  Structured memory channels  -->  Personalized reasoning
(file reads, writes, edits,        (procedural, semantic,           (query-time composition
 searches, tool sequences)          episodic)                        and interpretation)
```

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

## Architecture

### Data Generation Pipeline (FileGram)

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
└─────────────────────────────────────────────────────────────┘
```

### Memory Pipeline (FileGramOS)

FileGramOS decomposes file-level signals into three complementary memory channels, deferring semantic abstraction to query time:

```
Raw event stream
       │
       ├──→ Procedural Memory   (how the user works)
       │     • File access graph: exploration strategy, revisit patterns
       │     • Tool transition model: P(tool_k+1 | tool_k), error recovery
       │     • Information-seeking profile: search strategies, query complexity
       │
       ├──→ Semantic Memory     (what changed and why)
       │     • File changelog: ordered mutation history with content fingerprints
       │     • Causal edit chains: cross-file read→write dependencies
       │
       └──→ Episodic Memory     (behavioral consistency over time)
              • Work rhythm signature: iterations, tool density, error tolerance
              • Decision-making profile: reasoning density, token efficiency
              • Behavioral stability metrics: cross-session fingerprint similarity
```

**Deferred semantic abstraction**: Memory channels store structured representations, not natural language summaries. Interpretation occurs at query time — preserving temporal fidelity, enabling query adaptivity, and maintaining conflict awareness when user behavior evolves.

---

## Behavioral Event Schema

The behavioral collector captures **11 event types** organized into five categories. Each event is recorded as a structured object with common metadata (unique ID, millisecond timestamp, session ID, profile ID, model provider) and type-specific fields.

### File Access (Procedural)

| Event Type | Key Fields | What It Reveals |
|-----------|-----------|-----------------|
| `file_read` | `file_path`, `view_count`, `revisit_interval_ms`, `view_range` | Exploration strategy — breadth-first vs depth-first, which files get re-read |
| `file_search` | `search_type`, `query`, `files_matched`, `files_opened_after` | Information seeking — search strategy preference, query-to-action patterns |

### File Mutation (Semantic)

| Event Type | Key Fields | What It Reveals |
|-----------|-----------|-----------------|
| `file_write` | `file_path`, `operation`, `content_length`, `before_hash`, `after_hash` | Content creation — new files vs overwrites, file size patterns |
| `file_edit` | `edit_tool`, `lines_added/deleted/modified`, `diff_summary`, `before_hash`, `after_hash` | Content evolution — incremental refinement vs large rewrites |

### Workflow Structure (Procedural)

| Event Type | Key Fields | What It Reveals |
|-----------|-----------|-----------------|
| `tool_call` | `tool_name`, `sequence_position`, `execution_time_ms`, `retry_count` | Tool preference and workflow ordering, error recovery patterns |
| `context_switch` | `from_file`, `to_file`, `trigger`, `switch_count` | Navigation patterns between files |
| `iteration_start/end` | `iteration_number`, `duration_ms`, `tools_called`, `has_tool_error` | Work rhythm — tool density per iteration |

### Cognitive Indicators (Episodic)

| Event Type | Key Fields | What It Reveals |
|-----------|-----------|-----------------|
| `llm_response` | `response_time_ms`, `input/output_tokens`, `has_reasoning`, `stop_reason` | Decision-making rhythm, reasoning density |
| `compaction_triggered` | `reason`, `messages_before/after`, `tokens_saved` | Context management behavior |

### Session Boundaries (Episodic)

| Event Type | Key Fields | What It Reveals |
|-----------|-----------|-----------------|
| `session_start/end` | session-level timing | Overall session structure and temporal segmentation |

### Event Example

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

---

## Profile System

Profiles define agent personas that produce **differentiated behavioral data** on identical tasks. They live in `filegram/profile/profiles/*.yaml`.

### Profiles

| Profile | Persona | Behavioral Axis |
|---------|---------|----------------|
| **alex** | The Meticulous Craftsman (Chinese, 28) | Thoroughness — extensive reading before writing, comprehensive docs, defensive error handling |
| **luna** | The Creative Explorer (Japanese, 25) | Creativity — aggressive refactoring, novel pattern exploration, balanced verbosity |
| **sam** | The Pragmatic Problem Solver (American, 32) | Pragmatism — minimal exploration, fast write cycles, concise output, 80/20 focus |

### Profile Schema

```yaml
basic:
  name: Alex
  age: 28
  role: Senior Software Engineer
  nationality: Chinese
  language: Chinese

personality:
  traits: [detail-oriented, patient, methodical]
  tone: professional          # professional | friendly | casual
  humor_level: low            # low | moderate | high
  emoji_usage: minimal        # none | minimal | moderate | heavy
  verbosity: detailed         # concise | balanced | detailed

work_habits:
  coding_style: clean         # clean | pragmatic | creative
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

### Profile Differentiation Validation

Profiles are validated by measuring cross-profile divergence on behavioral statistics:
- **Read-to-write ratio**: How much exploration before action
- **Mean revisit interval**: Re-reading frequency
- **Tool transition entropy**: Workflow variability
- **Edit granularity distribution**: Incremental vs bulk changes

Profiles that fail to produce statistically distinguishable traces are revised before benchmark construction.

---

## FileGramBench

FileGramBench evaluates whether memory systems can extract, maintain, and reason over behavioral patterns embedded in file-level signals. Tasks are organized along two axes:

### Memory Type Axis

| Category | What It Tests | Example |
|----------|-------------|---------|
| **Procedural** | Workflow pattern reconstruction | Predict next-file access, infer tool preferences, recognize exploration strategies |
| **Semantic** | Content evolution tracking | Summarize a file's edit history, detect version conflicts, infer edit intent from diffs |
| **Episodic** | Cross-session behavioral consistency | Attribute session to profile, predict work rhythm, detect behavioral drift |

### Reasoning Capability Axis

| Capability | Description |
|-----------|-------------|
| **Temporal reasoning** | Ordering events and reasoning over time |
| **Cross-file inference** | Maintaining coherence across related files |
| **Conflict resolution** | Handling contradictory information across versions/sessions |
| **Selective context usage** | Leveraging relevant personal context while ignoring noise |
| **Profile attribution** | Identifying behavioral signatures |

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
```

---

## Setup

### Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

### Install

```bash
uv pip install -e .
```

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

See `.env.example` for the full list.

### Linting & Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [detect-secrets](https://github.com/Yelp/detect-secrets) for preventing API key leaks. Both are automated via [pre-commit](https://pre-commit.com/).

```bash
# One-time setup
uv pip install pre-commit detect-secrets
detect-secrets scan > .secrets.baseline
pre-commit install

# Manual usage
ruff check . --fix    # Lint
ruff format .         # Format
detect-secrets scan   # Check for secrets
```

---

## License

MIT
