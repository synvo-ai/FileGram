"""Core authentication management using Storage backend."""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
from typing import Any
from urllib.parse import urlencode

from ..storage.storage import NotFoundError, Storage

# Anthropic OAuth constants (from opencode-anthropic-auth@0.0.13)
OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
OAUTH_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
OAUTH_SCOPE = "org:create_api_key user:profile user:inference"
# "max" mode → claude.ai (for OAuth inference with Pro/Max plan)
# "console" mode → console.anthropic.com (for creating API keys)
OAUTH_AUTHORIZE_URL_MAX = "https://claude.ai/oauth/authorize"
OAUTH_AUTHORIZE_URL_CONSOLE = "https://console.anthropic.com/oauth/authorize"
OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
OAUTH_CREATE_API_KEY_URL = "https://api.anthropic.com/api/oauth/claude_cli/create_api_key"  # pragma: allowlist secret


STORAGE_KEY = ["auth", "credentials"]

PROVIDERS = {
    "anthropic": {
        "label": "Anthropic (Claude)",
        "env_key": "ANTHROPIC_API_KEY",
        "fields": ["key"],
    },
    "openai": {
        "label": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "fields": ["key"],
    },
    "azure_openai": {
        "label": "Azure OpenAI",
        "env_key": "AZURE_OPENAI_API_KEY",
        "fields": ["key", "endpoint", "deployment"],
    },
}


def _mask_key(key: str) -> str:
    """Mask an API key for display, showing first 4 and last 4 chars."""
    if len(key) <= 12:
        return key[:4] + "..." + key[-2:]
    return key[:4] + "..." + key[-4:]


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge pair.

    Returns:
        Tuple of (code_verifier, code_challenge)
    """
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def build_authorize_url(code_challenge: str, verifier: str, mode: str = "max") -> str:
    """Build the Anthropic OAuth authorization URL.

    Args:
        code_challenge: PKCE code challenge
        verifier: PKCE code verifier (used as the state parameter)
        mode: "max" for Claude Pro/Max OAuth login (claude.ai),
              "console" for API key creation (console.anthropic.com)

    Returns:
        Full authorization URL to open in browser
    """
    base_url = OAUTH_AUTHORIZE_URL_MAX if mode == "max" else OAUTH_AUTHORIZE_URL_CONSOLE
    params = {
        "code": "true",
        "client_id": OAUTH_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": OAUTH_REDIRECT_URI,
        "scope": OAUTH_SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    return f"{base_url}?{urlencode(params)}"


async def exchange_code(code: str, verifier: str) -> dict[str, Any]:
    """Exchange authorization code for OAuth tokens.

    The pasted code is in the format 'authorization_code#state'.
    Both parts must be sent to the token endpoint.

    Args:
        code: Raw code string from user (format: 'code#state')
        verifier: PKCE code_verifier

    Returns:
        Token response dict with access_token, refresh_token, expires_in, etc.

    Raises:
        RuntimeError: If token exchange fails
    """
    import httpx

    # Split code#state
    parts = code.split("#", 1)
    auth_code = parts[0]
    state = parts[1] if len(parts) > 1 else ""

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            OAUTH_TOKEN_URL,
            json={
                "code": auth_code,
                "state": state,
                "grant_type": "authorization_code",
                "client_id": OAUTH_CLIENT_ID,
                "redirect_uri": OAUTH_REDIRECT_URI,
                "code_verifier": verifier,
            },
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Token exchange failed ({resp.status_code}): {resp.text}")
        return resp.json()


async def refresh_oauth_token(refresh_token: str) -> dict[str, Any]:
    """Refresh an expired OAuth access token.

    Args:
        refresh_token: The refresh token

    Returns:
        New token response dict

    Raises:
        RuntimeError: If refresh fails
    """
    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            OAUTH_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": OAUTH_CLIENT_ID,
            },
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Token refresh failed ({resp.status_code}): {resp.text}")
        return resp.json()


async def create_api_key(access_token: str) -> str:
    """Create an API key using an OAuth access token.

    Args:
        access_token: OAuth access token

    Returns:
        The created API key string (sk-ant-...)

    Raises:
        RuntimeError: If API key creation fails
    """
    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            OAUTH_CREATE_API_KEY_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"API key creation failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        api_key = data.get("raw_key") or data.get("api_key") or data.get("key")
        if not api_key:
            raise RuntimeError(f"No API key in response: {data}")
        return api_key


class Auth:
    """Credential management backed by Storage.

    Credentials are stored at ~/.synvocode/storage/auth/credentials.json.
    Environment variables always take priority over stored credentials.
    """

    @staticmethod
    async def _read_all() -> dict[str, Any]:
        """Read the credentials file, returning empty dict if not found."""
        try:
            data = await Storage.read(STORAGE_KEY)
            if isinstance(data, dict):
                return data
            return {}
        except NotFoundError:
            return {}

    @staticmethod
    async def _write_all(data: dict[str, Any]) -> None:
        """Write credentials and set restrictive file permissions."""
        await Storage.write(STORAGE_KEY, data)
        # Set 0o600 (owner read/write only)
        path = Storage._key_to_path(STORAGE_KEY)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass  # Best effort on platforms that don't support chmod

    @staticmethod
    async def get(provider: str) -> dict[str, Any] | None:
        """Get stored credential for a provider.

        Args:
            provider: Provider name (anthropic, openai, azure_openai)

        Returns:
            Credential dict or None if not stored
        """
        data = await Auth._read_all()
        return data.get(provider)

    @staticmethod
    async def set(provider: str, credential: dict[str, Any]) -> None:
        """Store a credential for a provider.

        Args:
            provider: Provider name
            credential: Dict with at least {"type": "api", "key": "..."}
        """
        data = await Auth._read_all()
        data[provider] = credential
        await Auth._write_all(data)

    @staticmethod
    async def remove(provider: str) -> bool:
        """Remove a stored credential.

        Returns:
            True if the credential existed and was removed
        """
        data = await Auth._read_all()
        if provider in data:
            del data[provider]
            await Auth._write_all(data)
            return True
        return False

    @staticmethod
    async def all() -> dict[str, Any]:
        """Get all stored credentials."""
        return await Auth._read_all()

    @staticmethod
    def all_sync() -> dict[str, Any]:
        """Synchronously read all stored credentials (for config loading).

        Falls back to direct file read to avoid async context issues.
        """
        import json

        path = Storage._key_to_path(STORAGE_KEY)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    @staticmethod
    async def status() -> list[dict[str, str]]:
        """Get status of all providers (env + stored).

        Returns:
            List of dicts with provider info and credential source
        """
        stored = await Auth._read_all()
        result = []

        for name, info in PROVIDERS.items():
            entry = {
                "provider": name,
                "label": info["label"],
                "source": "none",
                "masked_key": "",
            }

            # Check environment variable first (takes priority)
            env_val = os.environ.get(info["env_key"])
            if env_val:
                entry["source"] = "env"
                entry["masked_key"] = _mask_key(env_val)
            elif name in stored:
                cred = stored[name]
                cred_type = cred.get("type", "api")
                if cred_type == "oauth" and cred.get("access_token"):
                    entry["source"] = "oauth"
                    entry["masked_key"] = _mask_key(cred["access_token"])
                elif cred.get("key"):
                    entry["source"] = "stored"
                    entry["masked_key"] = _mask_key(cred["key"])

            result.append(entry)

        return result


async def validate_credential(provider: str, key: str, **kwargs: Any) -> tuple[bool, str]:
    """Validate an API key by making a minimal API call.

    Args:
        provider: Provider name
        key: API key to validate
        **kwargs: Additional provider-specific params (endpoint, deployment)

    Returns:
        Tuple of (success, message)
    """
    try:
        if provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=key)
            client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True, "Valid Anthropic API key"

        elif provider == "openai":
            import openai

            client = openai.OpenAI(api_key=key)
            client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True, "Valid OpenAI API key"

        elif provider == "azure_openai":
            import openai

            endpoint = kwargs.get("endpoint", "")
            deployment = kwargs.get("deployment", "")
            if not endpoint or not deployment:
                return False, "Azure OpenAI requires endpoint and deployment"
            client = openai.AzureOpenAI(
                api_key=key,
                azure_endpoint=endpoint,
                api_version="2025-01-01-preview",
            )
            client.chat.completions.create(
                model=deployment,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True, "Valid Azure OpenAI credentials"

        else:
            return False, f"Unknown provider: {provider}"

    except Exception as e:
        error_msg = str(e)
        # Truncate long error messages
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."
        return False, f"Validation failed: {error_msg}"
