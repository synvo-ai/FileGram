"""FileGramOS: dimension-aware profile extraction.

Two versions:
    - Simple (ablation baseline): flat extraction + one-shot synthesis
    - Formal: engram-based memory with fingerprints, deviation detection,
      stratified sampling, and query-adaptive retrieval

Core differentiator from all baselines:
    Step 1: Deterministic feature extraction (no LLM) per attribute
    Step 2: Cross-trajectory aggregation (10 trajectories -> statistics)
    Step 3: LLM synthesis (structured features -> inferred profile)
"""

from . import schema
from .aggregation import FeatureAggregator
from .consolidator import EngramConsolidator
from .encoder import EngramEncoder
from .engram import ContentSample, CrossRef, EditChainSample, Engram, MemoryStore, SemanticUnit
from .feature_extraction import FeatureExtractor
from .fingerprint import compute_deviations, compute_fingerprint, normalize_fingerprints
from .normalizer import EventNormalizer
from .parsers import ParserRegistry
from .retriever import QueryAdaptiveRetriever
from .sampler import StratifiedSampler
from .schema import ConsumerEventType, NormalizedEvent

__all__ = [
    "schema",
    "ConsumerEventType",
    "NormalizedEvent",
    "EventNormalizer",
    "FeatureExtractor",
    "FeatureAggregator",
    "ContentSample",
    "EditChainSample",
    "CrossRef",
    "SemanticUnit",
    "Engram",
    "MemoryStore",
    "compute_fingerprint",
    "normalize_fingerprints",
    "compute_deviations",
    "EngramEncoder",
    "EngramConsolidator",
    "QueryAdaptiveRetriever",
    "StratifiedSampler",
    "ParserRegistry",
]
