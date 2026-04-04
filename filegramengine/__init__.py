"""FileGram - A Python code agent powered by Azure OpenAI."""

__version__ = "0.1.0"

from .agent import AgentInfo, AgentLoop, AgentMode, get_agent_registry
from .bus import Bus, BusEvent
from .compaction import AutoCompactor, Compactor
from .config import AzureOpenAIConfig, Config
from .console import ConsoleApp, Display
from .context import ContextManager, TokenCounter

# New modules (OpenCode parity)
from .file import FileIgnore, Ripgrep
from .instruction import InstructionLoader
from .permission import Permission, PermissionAction, PermissionError, PermissionRule
from .session import Session, SessionEvent, SessionInfo
from .storage import NotFoundError, Storage

__all__ = [
    "Config",
    "AzureOpenAIConfig",
    "AgentLoop",
    "AgentInfo",
    "AgentMode",
    "get_agent_registry",
    "Permission",
    "PermissionAction",
    "PermissionRule",
    "PermissionError",
    "InstructionLoader",
    "TokenCounter",
    "ContextManager",
    "Compactor",
    "AutoCompactor",
    # New modules (OpenCode parity)
    "FileIgnore",
    "Ripgrep",
    "Bus",
    "BusEvent",
    "Storage",
    "NotFoundError",
    "Session",
    "SessionInfo",
    "SessionEvent",
    # Console module
    "ConsoleApp",
    "Display",
]
