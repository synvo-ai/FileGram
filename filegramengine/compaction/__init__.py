"""Compaction (summarization) module with intelligent pruning."""

from .compactor import (
    PRUNE_MINIMUM,
    PRUNE_PROTECT,
    AutoCompactor,
    CompactionResult,
    Compactor,
    PruneResult,
    SmartPruner,
)

__all__ = [
    "Compactor",
    "CompactionResult",
    "AutoCompactor",
    "SmartPruner",
    "PruneResult",
    "PRUNE_MINIMUM",
    "PRUNE_PROTECT",
]
