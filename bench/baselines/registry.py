"""Adapter registry for baseline memory systems."""

from __future__ import annotations

from typing import Any

from .base import BaseAdapter

ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {}


def register_adapter(name: str):
    """Decorator to register an adapter class."""

    def decorator(cls: type[BaseAdapter]):
        ADAPTER_REGISTRY[name] = cls
        return cls

    return decorator


def get_adapter(name: str, config: dict[str, Any] | None = None, llm_fn=None) -> BaseAdapter:
    """Instantiate a registered adapter by name."""
    if name not in ADAPTER_REGISTRY:
        available = ", ".join(sorted(ADAPTER_REGISTRY.keys()))
        raise ValueError(f"Unknown adapter '{name}'. Available: {available}")
    return ADAPTER_REGISTRY[name](config, llm_fn=llm_fn)
