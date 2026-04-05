# FileGram

[![arXiv](https://img.shields.io/badge/arXiv-FileGram-b31b1b.svg)](#)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Choiszt/FileGram)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://filegram.choiszt.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Grounding Agent Personalization in File-System Behavioral Traces**

FileGram is a comprehensive framework that grounds agent memory and personalization in file-system behavioral traces. It comprises three core components:

- **FileGramEngine** — A scalable, persona-driven data engine that simulates realistic file-system workflows to generate fine-grained, multimodal behavioral traces.
- **FileGramBench** — A diagnostic benchmark with 4,600+ QA pairs across four evaluation tracks: profile reconstruction, trace disentanglement, anomaly detection, and multimodal grounding.
- **FileGramOS** — A bottom-up memory architecture that builds user profiles directly from atomic file-level signals through procedural, semantic, and episodic channels.

<p align="center">
  <img src="assets/teaser.png" width="100%" />
</p>

---

## Quick Start

### Install

```bash
uv sync
```

### Configure

```bash
cp .env.example .env
# Fill in your API keys (Anthropic / Gemini / Cohere)
```

### Run a Trajectory

```bash
# Single trajectory with a profile
filegramengine -1 --autonomous -d /path/to/workspace -p p1_methodical "Analyze and organize the files"

# List available profiles
filegramengine --list-profiles
```

### Run Batch Generation (640 trajectories)

```bash
python scripts/run_all_200.py
```

### Run Evaluation

```bash
# Step 1: Build ingest caches for all baselines
python bench/test_baselines.py --ingest-only

# Step 2: Run QA evaluation
python -m filegramQA.run_qa_eval --cache-dir gemini_2.5_flash --api gemini --mode qa --settings 1 2 3 4 --parallel 20
```

---

## Project Structure

```
FileGram/
├── filegramengine/        # Core package (FileGramEngine)
│   ├── agent/             #   Agent loop and orchestration
│   ├── behavior/          #   Behavioral signal collection (11 event types)
│   ├── llm/               #   LLM providers (Anthropic, Gemini, Azure OpenAI)
│   ├── tools/             #   File operation tools (read, write, edit, grep, bash, etc.)
│   ├── profile/           #   Profile loader + 20 persona YAMLs
│   ├── prompts/           #   System and tool prompt templates
│   └── ...                #   session, storage, snapshot, compaction, etc.
│
├── bench/                 # FileGramBench + FileGramOS
│   ├── baselines/         #   12 baseline adapters + FileGramOS
│   ├── filegramos/        #   FileGramOS core (encoder, consolidator, retriever)
│   ├── evaluation/        #   LLM-as-Judge scoring + MCQ generator
│   └── run_*.py           #   Evaluation runners
│
├── filegramQA/            # QA generation and evaluation
│   ├── generators/        #   Question generators (4 settings)
│   ├── questions/         #   Generated question bank (4,600+)
│   └── run_qa_eval.py     #   QA evaluation runner
│
├── profiles/              # 20 user profile definitions (YAML)
├── tasks/                 # 32 task definitions (JSON)
├── scripts/               # Utility scripts
│   ├── run_all_200.py     #   Generate 640 trajectories (20 profiles × 32 tasks)
│   ├── run_trajectory.sh  #   Run a single trajectory
│   └── convert_multimodal.py  # Convert text outputs to PDF/DOCX/images
│
├── web/                   # Interactive dashboard (local visualization)
├── pyproject.toml
├── .env.example
└── uv.lock
```

---

## Data

### 20 User Profiles

Each profile is defined by 6 behavioral dimensions with L/M/R tiers:

| Dimension | L (Left) | M (Middle) | R (Right) |
|-----------|----------|------------|-----------|
| A. Consumption | Sequential deep reader | Targeted searcher | Breadth-first scanner |
| B. Production | Comprehensive | Balanced | Minimal |
| C. Organization | Deeply nested | Adaptive | Flat |
| D. Iteration | Incremental | Balanced | Rewrite |
| E. Curation | Selective | Pragmatic | Preservative |
| F. Cross-Modal | Visual-heavy | Mixed | Text-only |

All 20 profiles: `p1_methodical`, `p2_thorough_reviser`, `p3_efficient_executor`, `p4_structured_analyst`, `p5_balanced_organizer`, `p6_quick_curator`, `p7_visual_reader`, `p8_minimal_editor`, `p9_visual_organizer`, `p10_silent_auditor`, `p11_meticulous_planner`, `p12_prolific_scanner`, `p13_visual_architect`, `p14_concise_organizer`, `p15_thorough_surveyor`, `p16_phased_minimalist`, `p17_creative_archivist`, `p18_decisive_scanner`, `p19_agile_pragmatist`, `p20_visual_auditor`.

### 32 Tasks

Tasks span five categories: understand, create, organize, synthesize, iterate, and maintain — producing ~10K multimodal output files across 640 trajectories.

### Behavioral Events

The behavioral collector captures **11 event types**: `session_start/end`, `iteration_start/end`, `file_read`, `file_write`, `file_edit`, `file_search`, `file_browse`, `tool_call`, `llm_response`, `context_switch`, `compaction_triggered`, and file organization events (`file_rename`, `file_move`, `file_copy`, `file_delete`, `dir_create`).

---

## Evaluation

### FileGramBench (4 Tracks)

| Track | Sub-tasks | Questions |
|-------|-----------|-----------|
| T1: Understanding | Attribute Recognition, Behavioral Fingerprint, Profile Reconstruction | 886 |
| T2: Reasoning | Behavioral Inference, Trace Disentanglement | 1,694 |
| T3: Detection | Anomaly Detection, Shift Analysis | 1,103 |
| T4: Multimodal | File Grounding, Visual Grounding | 650 |

<p align="center">
  <img src="assets/qa_example.png" width="100%" />
</p>

### Baselines (12 methods)

Context methods: Full Context, Naive RAG, VisRAG.
Text memory: Eager Summarization, Mem0, Zep, MemOS, EverMemOS, SimpleMem.
Multimodal memory: MMA, MemU.

### Results

| Method | Avg |
|--------|-----|
| **FileGramOS** | **59.6** |
| VisRAG | 51.9 |
| EverMemOS | 49.9 |
| Eager Summ. | 49.5 |
| Full Context | 48.0 |
| MMA | 44.7 |
| MemU | 44.4 |
| Zep | 40.2 |
| Naive RAG | 40.5 |
| MemOS | 36.2 |
| Mem0 | 33.2 |
| SimpleMem | 32.9 |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FILEGRAMENGINE_LLM_PROVIDER` | LLM provider: `anthropic`, `google`, or `azure_openai` |
| `ANTHROPIC_API_KEY` | Anthropic API key (for trajectory generation) |
| `GEMINI_API_KEY` | Google Gemini API key (for evaluation) |
| `COHERE_API_KEY` | Cohere API key (for embedding in baselines) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key (optional) |

See `.env.example` for the full configuration template.

---

## Citation

Coming soon.

## License

MIT
