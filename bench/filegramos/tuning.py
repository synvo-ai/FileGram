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

# ── Episode segmentation (episode.py) ──
EPISODE_MIN_EVENTS = 3
EPISODE_MAX_PER_TRAJECTORY = 5
EPISODE_SUMMARY_CHARS = 0  # 0 = no truncation

# ── Content chunking (consolidator.py) ──
CONTENT_CHUNK_SIZE = 800  # chars per chunk
CONTENT_CHUNK_MAX = 50  # max chunks stored across all trajectories

# ── Embedding + clustering (embedder.py, consolidator.py) ──
EPISODE_CLUSTER_THRESHOLD = 0.6  # cosine similarity threshold for average-linkage

# ── Sequence analysis (sequence.py) ──
PHASE_WINDOW_SIZE = 5

# ── Behavioral clustering (consolidator.py) ──
BEHAVIORAL_CLUSTER_MAX = 3
