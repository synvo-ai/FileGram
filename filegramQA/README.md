# FileGramQA — Benchmark Evaluation

## Overview

FileGramBench provides 4,600+ QA pairs across 4 evaluation tracks. Questions are available on HuggingFace:

> **Dataset**: [Choiszt/FileGram](https://huggingface.co/datasets/Choiszt/FileGram)

## Data

`run_qa_eval.py` reads the original question JSON from `filegramQA/questions/` and trajectories from `signal/`. Use the JSON files rather than the parquet files under `data/`: the parquet drops `mixing_spec`, which Trace Disentanglement questions need to locate their trajectories.

```bash
uv run hf download Choiszt/FileGram --repo-type dataset --include "questions/*" --local-dir filegramQA
uv run hf download Choiszt/FileGram --repo-type dataset --include "signal/*" --local-dir .
```

## Run Evaluation

Requires ingest caches in `bench/ingest_cache_*/` (run `bench/test_baselines.py --ingest-only` first).

```bash
# MCQ evaluation with Gemini
python -m filegramQA.run_qa_eval --cache-dir gemini_2.5_flash --api gemini --mode qa

# Open-ended profile reconstruction
python -m filegramQA.run_qa_eval --cache-dir gemini_2.5_flash --mode openended

# All modes, parallel
python -m filegramQA.run_qa_eval --cache-dir gemini_2.5_flash --mode all --parallel 20

# Resume from checkpoint
python -m filegramQA.run_qa_eval --cache-dir gemini_2.5_flash --resume
```

Results are saved to `filegramQA/results/`.
