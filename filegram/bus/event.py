"""Event definitions for the bus system."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class EventDefinition(Generic[T]):
    """Definition of an event type."""

    type: str
    properties_class: type[T]


class BusEvent:
    """Event definition factory and registry."""

    _registry: dict[str, EventDefinition] = {}

    @classmethod
    def define(cls, event_type: str, properties_class: type[T]) -> EventDefinition[T]:
        """
        Define a new event type.

        Args:
            event_type: Unique string identifier for the event
            properties_class: Pydantic model for event properties

        Returns:
            Event definition
        """
        definition = EventDefinition(type=event_type, properties_class=properties_class)
        cls._registry[event_type] = definition
        return definition

    @classmethod
    def get(cls, event_type: str) -> EventDefinition | None:
        """Get an event definition by type."""
        return cls._registry.get(event_type)

    @classmethod
    def all(cls) -> dict[str, EventDefinition]:
        """Get all registered event definitions."""
        return cls._registry.copy()
