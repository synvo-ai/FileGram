"""Permission control for tool execution."""

import fnmatch
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PermissionAction(Enum):
    """Permission action types."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class PermissionRule:
    """A single permission rule."""

    permission: str  # Tool name or pattern (e.g., "bash", "edit", "*")
    pattern: str  # File/argument pattern (e.g., "*.py", "/tmp/*", "*")
    action: PermissionAction

    def matches(self, tool: str, target: str) -> bool:
        """Check if this rule matches the given tool and target."""
        tool_match = fnmatch.fnmatch(tool, self.permission)
        pattern_match = fnmatch.fnmatch(target, self.pattern)
        return tool_match and pattern_match


class PermissionError(Exception):
    """Base class for permission errors."""

    pass


class PermissionDeniedError(PermissionError):
    """Raised when permission is denied by rule."""

    def __init__(self, tool: str, target: str, rule: PermissionRule):
        self.tool = tool
        self.target = target
        self.rule = rule
        super().__init__(
            f"Permission denied for tool '{tool}' on '{target}' by rule: "
            f"{rule.permission}:{rule.pattern} -> {rule.action.value}"
        )


class PermissionRejectedError(PermissionError):
    """Raised when user rejects permission."""

    def __init__(self, tool: str, target: str, message: str | None = None):
        self.tool = tool
        self.target = target
        self.message = message
        msg = f"User rejected permission for tool '{tool}' on '{target}'"
        if message:
            msg += f": {message}"
        super().__init__(msg)


@dataclass
class Permission:
    """Permission manager for controlling tool access."""

    rules: list[PermissionRule] = field(default_factory=list)
    session_approvals: list[PermissionRule] = field(default_factory=list)
    ask_callback: Callable[[str, str, dict[str, Any]], PermissionAction] | None = None

    # Edit tools that share the same permission
    EDIT_TOOLS = {"edit", "write", "patch", "multiedit"}

    # Default rules
    DEFAULT_RULES = [
        # Allow most tools by default
        PermissionRule("*", "*", PermissionAction.ALLOW),
        # Allow skill tool by default
        PermissionRule("skill", "*", PermissionAction.ALLOW),
        # But ask for external directories
        PermissionRule("read", "*.env", PermissionAction.ASK),
        PermissionRule("read", "*.env.*", PermissionAction.ASK),
        PermissionRule("read", "*.env.example", PermissionAction.ALLOW),
        # Dangerous bash commands should ask
        PermissionRule("bash", "rm -rf *", PermissionAction.ASK),
        PermissionRule("bash", "sudo *", PermissionAction.ASK),
    ]

    @classmethod
    def create_default(cls, ask_callback: Callable | None = None) -> "Permission":
        """Create a Permission instance with default rules."""
        return cls(
            rules=list(cls.DEFAULT_RULES),
            ask_callback=ask_callback,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any], ask_callback: Callable | None = None) -> "Permission":
        """
        Create Permission from config dict.

        Config format:
        {
            "permission": {
                "*": "allow",
                "bash": {
                    "*": "allow",
                    "rm -rf *": "ask"
                },
                "read": {
                    "*.env": "ask"
                }
            }
        }
        """
        rules = []
        permission_config = config.get("permission", {})

        for tool, value in permission_config.items():
            if isinstance(value, str):
                action = PermissionAction(value)
                rules.append(PermissionRule(tool, "*", action))
            elif isinstance(value, dict):
                for pattern, action_str in value.items():
                    # Expand home directory
                    if pattern.startswith("~/"):
                        pattern = os.path.expanduser(pattern)
                    action = PermissionAction(action_str)
                    rules.append(PermissionRule(tool, pattern, action))

        # Merge with defaults (user rules take precedence)
        all_rules = list(cls.DEFAULT_RULES) + rules

        return cls(rules=all_rules, ask_callback=ask_callback)

    def add_rule(self, rule: PermissionRule) -> None:
        """Add a permission rule."""
        self.rules.append(rule)

    def add_session_approval(self, tool: str, pattern: str) -> None:
        """Add a session-level approval (from user 'always allow')."""
        self.session_approvals.append(PermissionRule(tool, pattern, PermissionAction.ALLOW))

    def evaluate(self, tool: str, target: str) -> PermissionAction:
        """
        Evaluate permission for a tool call.

        Returns the action based on rules (last matching rule wins).
        """
        # Normalize tool name for edit tools
        permission_name = "edit" if tool in self.EDIT_TOOLS else tool

        # Check session approvals first
        for rule in reversed(self.session_approvals):
            if rule.matches(permission_name, target):
                return rule.action

        # Check rules (last match wins)
        matched_action = PermissionAction.ASK  # Default to ask
        for rule in self.rules:
            if rule.matches(permission_name, target):
                matched_action = rule.action

        return matched_action

    def check(self, tool: str, target: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Check permission for a tool call.

        Raises PermissionDeniedError or PermissionRejectedError if not allowed.
        """
        action = self.evaluate(tool, target)

        if action == PermissionAction.ALLOW:
            return

        if action == PermissionAction.DENY:
            # Find the denying rule for error message
            permission_name = "edit" if tool in self.EDIT_TOOLS else tool
            for rule in reversed(self.rules):
                if rule.matches(permission_name, target) and rule.action == PermissionAction.DENY:
                    raise PermissionDeniedError(tool, target, rule)
            raise PermissionDeniedError(tool, target, PermissionRule(tool, target, PermissionAction.DENY))

        if action == PermissionAction.ASK:
            if self.ask_callback is None:
                # No callback, default to allow
                return

            user_action = self.ask_callback(tool, target, metadata or {})

            if user_action == PermissionAction.ALLOW:
                return
            elif user_action == PermissionAction.DENY:
                raise PermissionRejectedError(tool, target)
            # If ASK returned again, treat as reject
            raise PermissionRejectedError(tool, target, "User did not approve")

    def is_tool_disabled(self, tool: str) -> bool:
        """Check if a tool is completely disabled (deny for all patterns)."""
        permission_name = "edit" if tool in self.EDIT_TOOLS else tool

        for rule in reversed(self.rules):
            if rule.permission == permission_name and rule.pattern == "*":
                return rule.action == PermissionAction.DENY
            if rule.permission == "*" and rule.pattern == "*":
                return rule.action == PermissionAction.DENY

        return False

    def get_disabled_tools(self, available_tools: list[str]) -> set[str]:
        """Get set of tools that are disabled."""
        return {tool for tool in available_tools if self.is_tool_disabled(tool)}
