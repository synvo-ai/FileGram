# FileGramBench Evaluation Framework

## Overview

Evaluates 12 baseline memory systems and **FileGramOS** on profile reconstruction from file-system behavioral trajectories (640 trajectories, 20 profiles × 32 tasks).

## Structure

```
bench/
├── baselines/              # 12 baselines + FileGramOS adapter
│   ├── base.py             # BaseAdapter interface
│   ├── registry.py         # Adapter registration
│   ├── filegramos_adapter.py   # FileGramOS (ours)
│   ├── full_context.py     # Full context baseline
│   ├── naive_rag.py        # Naive RAG
│   ├── eager_summarization.py
│   ├── mem0_adapter.py
│   ├── zep_adapter.py
│   ├── memos_adapter.py
│   ├── memu_adapter.py
│   ├── evermemos_adapter.py
│   ├── simplemem_adapter.py
│   ├── mma_adapter.py      # Multimodal Memory Agent
│   └── visrag_adapter.py
├── filegramos/             # FileGramOS core modules
│   ├── encoder.py          # Stage 1: Engram encoding
│   ├── consolidator.py     # Stage 2: Cross-engram consolidation
│   ├── retriever.py        # Stage 3: Query-adaptive retrieval
│   ├── engram.py           # Engram + MemoryStore data structures
│   ├── schema.py           # Event normalization schema
│   └── ...
├── evaluation/
│   ├── judge_scoring.py    # LLM-as-Judge scoring
│   └── mcq_generator.py    # MCQ generation from trajectories
├── run_eval.py             # Main evaluation runner
├── test_baselines.py       # Run all baselines
├── run_ablation.py         # Ablation experiments
├── run_curve_quick.py      # Learning curve evaluation
└── run_tuning.py           # Parameter tuning
```

## Usage

```bash
# Run full evaluation
python -m bench.run_eval

# Run all baselines on all profiles
python bench/test_baselines.py

# Learning curve (N=1,3,5,10,15,20,30 trajectories)
python bench/run_curve_quick.py
```

## Adapter Interface

All methods implement `BaseAdapter`:

```python
class BaseAdapter(ABC):
    def ingest(self, trajectories: list[dict]) -> None: ...
    def infer_profile(self, profile_attributes: list[str]) -> dict: ...
    def reset(self) -> None: ...
```
