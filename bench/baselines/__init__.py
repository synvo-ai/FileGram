"""Baseline adapters for FileGramBench evaluation.

Each adapter transforms file-system behavioral trajectories (events.json)
into a format consumable by a specific memory system, then uses the system
to infer a user profile from the stored memories.

All dialogue-memory systems are expected to perform poorly on file-system
trajectories — this is the core motivation for FileGramOS.
"""

from .base import BaseAdapter
from .registry import ADAPTER_REGISTRY, get_adapter

__all__ = ["BaseAdapter", "ADAPTER_REGISTRY", "get_adapter"]
