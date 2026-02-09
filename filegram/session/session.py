"""Session management for conversations."""

import logging
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from ..bus import Bus, BusEvent
from ..storage import NotFoundError, Storage

logger = logging.getLogger(__name__)


# Session events
class SessionCreatedProperties(BaseModel):
    """Properties for session created event."""

    session_id: str
    title: str


class SessionUpdatedProperties(BaseModel):
    """Properties for session updated event."""

    session_id: str


class SessionDeletedProperties(BaseModel):
    """Properties for session deleted event."""

    session_id: str


class SessionEvent:
    """Session event definitions."""

    Created = BusEvent.define("session.created", SessionCreatedProperties)
    Updated = BusEvent.define("session.updated", SessionUpdatedProperties)
    Deleted = BusEvent.define("session.deleted", SessionDeletedProperties)


class SessionInfo(BaseModel):
    """Session information."""

    id: str
    slug: str
    project_id: str
    directory: str
    parent_id: str | None = None
    title: str
    version: str = "1.0.0"
    time: dict[str, int] = {}
    metadata: dict[str, Any] = {}

    class Config:
        extra = "allow"


def generate_slug() -> str:
    """Generate a human-readable slug."""
    import random

    adjectives = [
        "happy",
        "bright",
        "swift",
        "calm",
        "bold",
        "wise",
        "keen",
        "warm",
        "cool",
        "soft",
    ]
    nouns = [
        "river",
        "mountain",
        "forest",
        "ocean",
        "meadow",
        "valley",
        "stream",
        "cloud",
        "breeze",
        "dawn",
    ]
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{uuid.uuid4().hex[:6]}"


def generate_session_id() -> str:
    """Generate a unique session ID."""
    # Use timestamp-based descending ID for newest-first ordering
    timestamp = int(datetime.now().timestamp() * 1000)
    # Invert to get descending order
    inverted = 9999999999999 - timestamp
    return f"session_{inverted}_{uuid.uuid4().hex[:8]}"


class Session:
    """Session management namespace."""

    @staticmethod
    async def create(
        project_id: str,
        directory: str,
        title: str | None = None,
        parent_id: str | None = None,
        bus: Bus | None = None,
    ) -> SessionInfo:
        """Create a new session.

        Args:
            project_id: ID of the project
            directory: Working directory
            title: Optional title (auto-generated if not provided)
            parent_id: Optional parent session ID
            bus: Optional event bus

        Returns:
            Created session info
        """
        now = int(datetime.now().timestamp() * 1000)

        session = SessionInfo(
            id=generate_session_id(),
            slug=generate_slug(),
            project_id=project_id,
            directory=directory,
            parent_id=parent_id,
            title=title or f"New session - {datetime.now().isoformat()}",
            time={"created": now, "updated": now},
        )

        await Storage.write(["session", project_id, session.id], session.model_dump())

        logger.info(f"Created session: {session.id}")

        if bus:
            await bus.publish(
                SessionEvent.Created,
                SessionCreatedProperties(session_id=session.id, title=session.title),
            )

        return session

    @staticmethod
    async def get(project_id: str, session_id: str) -> SessionInfo | None:
        """Get a session by ID.

        Args:
            project_id: Project ID
            session_id: Session ID

        Returns:
            Session info or None if not found
        """
        try:
            data = await Storage.read(["session", project_id, session_id])
            return SessionInfo(**data)
        except NotFoundError:
            return None

    @staticmethod
    async def update(
        project_id: str,
        session_id: str,
        updater: Callable[[dict], None],
        bus: Bus | None = None,
        touch: bool = True,
    ) -> SessionInfo:
        """Update a session.

        Args:
            project_id: Project ID
            session_id: Session ID
            updater: Function to modify session data
            bus: Optional event bus
            touch: Whether to update the updated timestamp

        Returns:
            Updated session info
        """

        def _update(data: dict) -> None:
            updater(data)
            if touch:
                data["time"]["updated"] = int(datetime.now().timestamp() * 1000)

        data = await Storage.update(["session", project_id, session_id], _update)

        if bus:
            await bus.publish(
                SessionEvent.Updated,
                SessionUpdatedProperties(session_id=session_id),
            )

        return SessionInfo(**data)

    @staticmethod
    async def touch(project_id: str, session_id: str) -> None:
        """Update session's updated timestamp.

        Args:
            project_id: Project ID
            session_id: Session ID
        """
        await Session.update(project_id, session_id, lambda _: None, touch=True)

    @staticmethod
    async def remove(project_id: str, session_id: str, bus: Bus | None = None) -> None:
        """Remove a session and all its data.

        Args:
            project_id: Project ID
            session_id: Session ID
            bus: Optional event bus
        """
        # Remove messages
        message_keys = await Storage.list(["message", session_id])
        for key in message_keys:
            # Remove parts for each message
            message_id = key[-1]
            part_keys = await Storage.list(["part", message_id])
            for part_key in part_keys:
                await Storage.remove(part_key)
            await Storage.remove(key)

        # Remove session
        await Storage.remove(["session", project_id, session_id])

        logger.info(f"Removed session: {session_id}")

        if bus:
            await bus.publish(
                SessionEvent.Deleted,
                SessionDeletedProperties(session_id=session_id),
            )

    @staticmethod
    async def list_sessions(project_id: str) -> AsyncIterator[SessionInfo]:
        """List all sessions for a project.

        Args:
            project_id: Project ID

        Yields:
            Session info objects
        """
        keys = await Storage.list(["session", project_id])
        for key in keys:
            try:
                data = await Storage.read(key)
                yield SessionInfo(**data)
            except NotFoundError:
                continue

    @staticmethod
    async def fork(
        project_id: str,
        session_id: str,
        message_id: str | None = None,
        bus: Bus | None = None,
    ) -> SessionInfo:
        """Fork a session, optionally up to a specific message.

        Args:
            project_id: Project ID
            session_id: Session ID to fork
            message_id: Optional message ID to fork up to
            bus: Optional event bus

        Returns:
            New forked session
        """

        # Get original session
        original = await Session.get(project_id, session_id)
        if not original:
            raise NotFoundError(f"Session not found: {session_id}")

        # Create new session
        new_session = await Session.create(
            project_id=project_id,
            directory=original.directory,
            title=f"Fork of {original.title}",
            bus=bus,
        )

        # Copy messages up to message_id
        message_keys = await Storage.list(["message", session_id])
        id_map: dict[str, str] = {}

        for key in message_keys:
            try:
                msg_data = await Storage.read(key)
                old_id = msg_data["id"]

                if message_id and old_id >= message_id:
                    break

                # Generate new ID
                new_id = f"msg_{int(datetime.now().timestamp() * 1000)}_{uuid.uuid4().hex[:8]}"
                id_map[old_id] = new_id

                # Update references
                msg_data["id"] = new_id
                msg_data["session_id"] = new_session.id
                if msg_data.get("parent_id") and msg_data["parent_id"] in id_map:
                    msg_data["parent_id"] = id_map[msg_data["parent_id"]]

                await Storage.write(["message", new_session.id, new_id], msg_data)

                # Copy parts
                part_keys = await Storage.list(["part", old_id])
                for part_key in part_keys:
                    part_data = await Storage.read(part_key)
                    part_data["message_id"] = new_id
                    part_data["session_id"] = new_session.id
                    new_part_id = f"part_{int(datetime.now().timestamp() * 1000)}_{uuid.uuid4().hex[:8]}"
                    part_data["id"] = new_part_id
                    await Storage.write(["part", new_id, new_part_id], part_data)

            except NotFoundError:
                continue

        return new_session

    @staticmethod
    def is_default_title(title: str) -> bool:
        """Check if a title is a default auto-generated title."""
        return title.startswith("New session - ") or title.startswith("Child session - ")
