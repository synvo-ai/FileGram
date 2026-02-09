"""Configuration management for CodeAgent."""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv


class LLMProvider(Enum):
    """Available LLM providers."""

    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration."""

    api_key: str
    endpoint: str
    deployment: str
    api_version: str


@dataclass
class AnthropicConfig:
    """Anthropic Claude configuration."""

    api_key: str
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192
    temperature: float = 1.0
    enable_thinking: bool = False
    thinking_budget_tokens: int = 10000
    oauth_token: str | None = None  # OAuth Bearer token (alternative to api_key)


@dataclass
class OpenAIConfig:
    """OpenAI configuration."""

    api_key: str
    model: str = "gpt-4o"
    max_tokens: int = 8192
    temperature: float = 1.0


@dataclass
class LLMConfig:
    """LLM configuration with multi-provider support."""

    provider: LLMProvider = LLMProvider.AZURE_OPENAI
    azure_openai: AzureOpenAIConfig | None = None
    anthropic: AnthropicConfig | None = None
    openai: OpenAIConfig | None = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load LLM configuration from environment variables.

        After loading from env vars, checks stored credentials (via auth login)
        for any providers not yet configured. Environment variables take priority.
        """
        # Determine provider from environment
        provider_str = os.environ.get("SYNVOCODE_LLM_PROVIDER", "azure_openai").lower()
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            provider = LLMProvider.AZURE_OPENAI

        config = cls(provider=provider)

        # Load Azure OpenAI config if available
        azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if azure_key:
            config.azure_openai = AzureOpenAIConfig(
                api_key=azure_key,
                endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "https://haku-chat.openai.azure.com"),
                deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            )

        # Load Anthropic config if available
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            config.anthropic = AnthropicConfig(
                api_key=anthropic_key,
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                max_tokens=int(os.environ.get("ANTHROPIC_MAX_TOKENS", "8192")),
                temperature=float(os.environ.get("ANTHROPIC_TEMPERATURE", "1.0")),
                enable_thinking=os.environ.get("ANTHROPIC_ENABLE_THINKING", "").lower() in ("1", "true", "yes"),
            )

        # Load OpenAI config if available
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            config.openai = OpenAIConfig(
                api_key=openai_key,
                model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
                max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "8192")),
                temperature=float(os.environ.get("OPENAI_TEMPERATURE", "1.0")),
            )

        # Fill in any providers not yet configured from stored credentials
        config._load_stored_credentials()

        # If the user didn't explicitly set a provider via env var,
        # auto-select based on what's actually configured
        if not os.environ.get("SYNVOCODE_LLM_PROVIDER"):
            active = config._get_provider_config(config.provider)
            if active is None:
                # Current default provider has no credentials — pick the first available
                for candidate, cfg in [
                    (LLMProvider.ANTHROPIC, config.anthropic),
                    (LLMProvider.OPENAI, config.openai),
                    (LLMProvider.AZURE_OPENAI, config.azure_openai),
                ]:
                    if cfg is not None:
                        config.provider = candidate
                        break

        return config

    def _get_provider_config(self, provider: LLMProvider):
        """Get the config object for a given provider, or None."""
        if provider == LLMProvider.ANTHROPIC:
            return self.anthropic
        elif provider == LLMProvider.OPENAI:
            return self.openai
        elif provider == LLMProvider.AZURE_OPENAI:
            return self.azure_openai
        return None

    def _load_stored_credentials(self) -> None:
        """Load stored credentials for providers not configured via env vars."""
        try:
            from .auth.auth import Auth

            stored = Auth.all_sync()
        except Exception:
            return

        if not stored:
            return

        # Only fill providers that aren't already set from env vars
        if self.anthropic is None and "anthropic" in stored:
            cred = stored["anthropic"]
            cred_type = cred.get("type", "api")
            if cred_type == "oauth" and cred.get("access_token"):
                # OAuth credential — refresh if needed, then use Bearer token
                access_token = self._maybe_refresh_oauth(cred, stored)
                # Use a placeholder api_key (required field); actual auth via oauth_token
                self.anthropic = AnthropicConfig(
                    api_key="oauth-placeholder",
                    oauth_token=access_token,
                )
            elif cred.get("key"):
                self.anthropic = AnthropicConfig(api_key=cred["key"])

        if self.openai is None and "openai" in stored:
            cred = stored["openai"]
            if cred.get("key"):
                self.openai = OpenAIConfig(api_key=cred["key"])

        if self.azure_openai is None and "azure_openai" in stored:
            cred = stored["azure_openai"]
            if cred.get("key"):
                self.azure_openai = AzureOpenAIConfig(
                    api_key=cred["key"],
                    endpoint=cred.get("endpoint", "https://haku-chat.openai.azure.com"),
                    deployment=cred.get("deployment", "gpt-4.1-mini"),
                    api_version="2025-01-01-preview",
                )

    @staticmethod
    def _maybe_refresh_oauth(cred: dict, all_stored: dict) -> str:
        """Check if OAuth token needs refresh and refresh it synchronously.

        Args:
            cred: The Anthropic credential dict with access_token, refresh_token, expires_at
            all_stored: Full stored credentials dict (to write back updated tokens)

        Returns:
            Valid access token
        """
        import time

        expires_at = cred.get("expires_at", 0)
        # Refresh if token expires within 5 minutes
        if time.time() < expires_at - 300:
            return cred["access_token"]

        refresh_token = cred.get("refresh_token")
        if not refresh_token:
            return cred["access_token"]

        try:
            import httpx

            resp = httpx.post(
                "https://console.anthropic.com/v1/oauth/token",
                json={
                    "grant_type": "refresh_token",
                    "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
                    "refresh_token": refresh_token,
                },
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                tokens = resp.json()
                cred["access_token"] = tokens["access_token"]
                if tokens.get("refresh_token"):
                    cred["refresh_token"] = tokens["refresh_token"]
                cred["expires_at"] = int(time.time()) + tokens.get("expires_in", 3600)
                # Write back updated credentials
                all_stored["anthropic"] = cred
                try:
                    import json as _json

                    from .storage.storage import Storage

                    path = Storage._key_to_path(["auth", "credentials"])
                    with open(path, "w", encoding="utf-8") as f:
                        _json.dump(all_stored, f, indent=2)
                except Exception:
                    pass
                return cred["access_token"]
        except Exception:
            pass

        return cred["access_token"]

    def get_active_config(self) -> AzureOpenAIConfig | AnthropicConfig | OpenAIConfig:
        """Get the active provider configuration."""
        if self.provider == LLMProvider.ANTHROPIC:
            if self.anthropic is None:
                raise ValueError("Anthropic configuration not available")
            return self.anthropic
        elif self.provider == LLMProvider.OPENAI:
            if self.openai is None:
                raise ValueError("OpenAI configuration not available")
            return self.openai
        else:  # Default to Azure OpenAI
            if self.azure_openai is None:
                raise ValueError("Azure OpenAI configuration not available")
            return self.azure_openai


@dataclass
class SkillConfig:
    """Skill system configuration."""

    enabled: bool = True
    custom_paths: list[str] = field(default_factory=list)
    disable_claude_skills: bool = False

    @classmethod
    def from_env(cls) -> "SkillConfig":
        """Load skill configuration from environment variables."""
        enabled = os.environ.get("SYNVOCODE_DISABLE_SKILLS", "").lower() not in (
            "1",
            "true",
            "yes",
        )

        custom_paths_str = os.environ.get("SYNVOCODE_SKILL_PATHS", "")
        custom_paths = [p.strip() for p in custom_paths_str.split(":") if p.strip()]

        disable_claude = os.environ.get("SYNVOCODE_DISABLE_CLAUDE_SKILLS", "").lower() in ("1", "true", "yes")

        return cls(
            enabled=enabled,
            custom_paths=custom_paths,
            disable_claude_skills=disable_claude,
        )


@dataclass
class MCPConfig:
    """MCP (Model Context Protocol) configuration."""

    enabled: bool = True
    config_path: str | None = None
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "MCPConfig":
        """Load MCP configuration from environment variables."""
        enabled = os.environ.get("SYNVOCODE_DISABLE_MCP", "").lower() not in (
            "1",
            "true",
            "yes",
        )

        config_path = os.environ.get("SYNVOCODE_MCP_CONFIG")
        timeout = float(os.environ.get("SYNVOCODE_MCP_TIMEOUT", "30.0"))

        return cls(
            enabled=enabled,
            config_path=config_path,
            timeout=timeout,
        )


@dataclass
class Config:
    """CodeAgent configuration."""

    llm: LLMConfig
    target_directory: Path
    skill: SkillConfig = field(default_factory=SkillConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    max_output_chars: int = 30000
    default_timeout: int = 120000  # 2 minutes in ms
    sandbox_enabled: bool = True  # If True, restrict file access to target_directory

    # Legacy property for backwards compatibility
    @property
    def azure_openai(self) -> AzureOpenAIConfig:
        """Get Azure OpenAI config (legacy property)."""
        if self.llm.azure_openai is None:
            raise ValueError("Azure OpenAI configuration not available")
        return self.llm.azure_openai

    def apply_model_override(self, model_spec: str) -> None:
        """Override model from CLI or slash command.

        Format: provider/model (e.g., anthropic/claude-sonnet-4-20250514)
        """
        parts = model_spec.split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid model format: {model_spec}. "
                "Use provider/model format (e.g., anthropic/claude-sonnet-4-20250514)"
            )

        provider_str, model_name = parts

        provider_map = {
            "azure_openai": LLMProvider.AZURE_OPENAI,
            "azure": LLMProvider.AZURE_OPENAI,
            "anthropic": LLMProvider.ANTHROPIC,
            "openai": LLMProvider.OPENAI,
        }

        provider = provider_map.get(provider_str.lower())
        if provider is None:
            raise ValueError(f"Unknown provider: {provider_str}. Available: {', '.join(provider_map.keys())}")

        self.llm.provider = provider

        if provider == LLMProvider.ANTHROPIC:
            if self.llm.anthropic is None:
                raise ValueError("Anthropic API key not configured. Set ANTHROPIC_API_KEY.")
            self.llm.anthropic.model = model_name
        elif provider == LLMProvider.OPENAI:
            if self.llm.openai is None:
                raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY.")
            self.llm.openai.model = model_name
        elif provider == LLMProvider.AZURE_OPENAI:
            if self.llm.azure_openai is None:
                raise ValueError("Azure OpenAI not configured. Set AZURE_OPENAI_API_KEY.")
            self.llm.azure_openai.deployment = model_name

    def get_model_display(self) -> str:
        """Get display string for current model (provider/model)."""
        provider = self.llm.provider.value
        active = self.llm.get_active_config()
        if hasattr(active, "model"):
            model = active.model
        elif hasattr(active, "deployment"):
            model = active.deployment
        else:
            model = "unknown"
        return f"{provider}/{model}"

    @classmethod
    def from_env(
        cls,
        target_directory: str | None = None,
        sandbox_enabled: bool = True,
    ) -> "Config":
        """Load configuration from environment variables.

        Args:
            target_directory: Target directory for file operations
            sandbox_enabled: If True, restrict file access to target_directory
        """
        load_dotenv()

        llm_config = LLMConfig.from_env()

        # Validate that at least one provider is configured
        if llm_config.azure_openai is None and llm_config.anthropic is None and llm_config.openai is None:
            raise ValueError(
                "At least one LLM provider must be configured. "
                "Set AZURE_OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY"
            )

        target_dir = Path(target_directory) if target_directory else Path.cwd()
        if not target_dir.exists():
            raise ValueError(f"Target directory does not exist: {target_dir}")

        skill_config = SkillConfig.from_env()
        mcp_config = MCPConfig.from_env()

        return cls(
            llm=llm_config,
            target_directory=target_dir.resolve(),
            skill=skill_config,
            mcp=mcp_config,
            sandbox_enabled=sandbox_enabled,
        )


# Environment variable reference:
# ================================
# LLM Provider Selection:
#   SYNVOCODE_LLM_PROVIDER = "azure_openai" | "anthropic" | "openai"
#
# Azure OpenAI:
#   AZURE_OPENAI_API_KEY (required for Azure)
#   AZURE_OPENAI_ENDPOINT (default: https://haku-chat.openai.azure.com)
#   AZURE_OPENAI_DEPLOYMENT (default: gpt-4.1-mini)
#   AZURE_OPENAI_API_VERSION (default: 2025-01-01-preview)
#
# Anthropic Claude:
#   ANTHROPIC_API_KEY (required for Anthropic)
#   ANTHROPIC_MODEL (default: claude-sonnet-4-20250514)
#   ANTHROPIC_MAX_TOKENS (default: 8192)
#   ANTHROPIC_TEMPERATURE (default: 1.0)
#   ANTHROPIC_ENABLE_THINKING (default: false)
#
# OpenAI:
#   OPENAI_API_KEY (required for OpenAI)
#   OPENAI_MODEL (default: gpt-4o)
#   OPENAI_MAX_TOKENS (default: 8192)
#   OPENAI_TEMPERATURE (default: 1.0)
#
# Skill System:
#   SYNVOCODE_DISABLE_SKILLS (default: false)
#   SYNVOCODE_SKILL_PATHS (colon-separated paths)
#   SYNVOCODE_DISABLE_CLAUDE_SKILLS (default: false)
#
# MCP (Model Context Protocol):
#   SYNVOCODE_DISABLE_MCP (default: false)
#   SYNVOCODE_MCP_CONFIG (path to mcp.json config file)
#   SYNVOCODE_MCP_TIMEOUT (default: 30.0 seconds)
