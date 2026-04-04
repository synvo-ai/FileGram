"""Event bus for pub/sub messaging."""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from .event import EventDefinition

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class Event(Generic[T]):
    """An event with type and properties."""

    type: str
    properties: T


class Bus:
    """Event bus for pub/sub messaging within a session."""

    def __init__(self):
        self._subscriptions: dict[str, list[Callable]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def publish(
        self,
        definition: EventDefinition[T],
        properties: T,
    ) -> None:
        """
        Publish an event to all subscribers.

        Args:
            definition: Event definition
            properties: Event properties (must match definition's properties_class)
        """
        event = Event(type=definition.type, properties=properties)

        logger.debug(f"Publishing event: {definition.type}")

        # Get subscribers for specific type and wildcard
        async with self._lock:
            specific_subs = list(self._subscriptions.get(definition.type, []))
            wildcard_subs = list(self._subscriptions.get("*", []))

        all_subs = specific_subs + wildcard_subs

        # Call all subscribers
        pending = []
        for callback in all_subs:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    pending.append(result)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}")

        # Wait for async subscribers
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def subscribe(
        self,
        definition: EventDefinition[T],
        callback: Callable[[Event[T]], Any],
    ) -> Callable[[], None]:
        """
        Subscribe to an event type.

        Args:
            definition: Event definition to subscribe to
            callback: Function to call when event is published

        Returns:
            Unsubscribe function
        """
        return self._raw_subscribe(definition.type, callback)

    def subscribe_all(
        self,
        callback: Callable[[Event[Any]], Any],
    ) -> Callable[[], None]:
        """
        Subscribe to all events.

        Args:
            callback: Function to call for any event

        Returns:
            Unsubscribe function
        """
        return self._raw_subscribe("*", callback)

    def once(
        self,
        definition: EventDefinition[T],
        callback: Callable[[Event[T]], bool | None],
    ) -> Callable[[], None]:
        """
        Subscribe to an event and unsubscribe after callback returns True.

        Args:
            definition: Event definition to subscribe to
            callback: Function that returns True when done

        Returns:
            Unsubscribe function
        """

        def wrapper(event: Event[T]):
            result = callback(event)
            if result:
                unsub()

        unsub = self.subscribe(definition, wrapper)
        return unsub

    def _raw_subscribe(
        self,
        event_type: str,
        callback: Callable,
    ) -> Callable[[], None]:
        """
        Internal subscribe method.

        Args:
            event_type: Event type string or "*" for all
            callback: Callback function

        Returns:
            Unsubscribe function
        """
        logger.debug(f"Subscribing to: {event_type}")
        self._subscriptions[event_type].append(callback)

        def unsubscribe():
            logger.debug(f"Unsubscribing from: {event_type}")
            try:
                self._subscriptions[event_type].remove(callback)
            except ValueError:
                pass

        return unsubscribe


# Global bus instance for cross-module events
_global_bus: Bus | None = None


def get_global_bus() -> Bus:
    """Get or create the global event bus."""
    global _global_bus
    if _global_bus is None:
        _global_bus = Bus()
    return _global_bus


def reset_global_bus() -> None:
    """Reset the global event bus (mainly for testing)."""
    global _global_bus
    _global_bus = None
