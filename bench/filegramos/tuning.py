"""Centralized tunable parameters for ablation experiments.

All downstream files import from here so experiments only need to
modify this single file.
"""

# ── Feature Extraction (feature_extraction.py) ──
PREVIEW_FILE_CHARS = 500
PREVIEW_DIFF_CHARS = 300

# ── Encoder LLM encoding (encoder.py) ──
LLM_ENCODE_MAX_FILES = 4
LLM_ENCODE_FILE_CHARS = 600
LLM_ENCODE_MAX_EDITS = 3
LLM_ENCODE_EDIT_CHARS = 400

# ── Sampler / Consolidator (sampler.py, consolidator.py) ──
SEMANTIC_BUDGET = 8000

# ── Retriever display (retriever.py) ──
RETRIEVER_DISPLAY_CHARS = 800
