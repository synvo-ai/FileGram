"""Interactive authentication commands for CLI and console."""

from __future__ import annotations

import time
import webbrowser

from rich.console import Console
from rich.table import Table

from .auth import (
    PROVIDERS,
    Auth,
    build_authorize_url,
    create_api_key,
    exchange_code,
    generate_pkce,
    validate_credential,
)


async def _anthropic_oauth_login(console: Console) -> bool:
    """OAuth browser login for Anthropic. Stores OAuth tokens."""
    verifier, challenge = generate_pkce()
    url = build_authorize_url(challenge, verifier)

    console.print("\n  [bold]Opening browser for Anthropic login...[/bold]")
    console.print("  [dim]If the browser doesn't open, visit:[/dim]")
    console.print(f"  [link={url}]{url}[/link]\n")

    webbrowser.open(url)

    try:
        auth_code = console.input("  Paste the authorization code here: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/dim]")
        return False

    if not auth_code:
        console.print("[red]Authorization code cannot be empty.[/red]")
        return False

    console.print("  [dim]Exchanging code for tokens...[/dim]", end=" ")
    try:
        tokens = await exchange_code(auth_code, verifier)
    except RuntimeError as e:
        console.print("[red]Failed[/red]")
        console.print(f"  [red]{e}[/red]")
        return False

    console.print("[green]OK[/green]")

    credential = {
        "type": "oauth",
        "access_token": tokens["access_token"],
        "refresh_token": tokens.get("refresh_token", ""),
        "expires_at": int(time.time()) + tokens.get("expires_in", 3600),
    }

    await Auth.set("anthropic", credential)
    from ..storage.storage import Storage

    path = Storage._key_to_path(["auth", "credentials"])
    console.print(f"\n  [green]OAuth credentials saved to {path}[/green]")
    return True


async def _anthropic_oauth_create_key(console: Console) -> bool:
    """OAuth flow that creates an API key via Anthropic's OAuth endpoint."""
    verifier, challenge = generate_pkce()
    url = build_authorize_url(challenge, verifier, mode="console")

    console.print("\n  [bold]Opening browser for Anthropic login...[/bold]")
    console.print("  [dim]If the browser doesn't open, visit:[/dim]")
    console.print(f"  [link={url}]{url}[/link]\n")

    webbrowser.open(url)

    try:
        auth_code = console.input("  Paste the authorization code here: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/dim]")
        return False

    if not auth_code:
        console.print("[red]Authorization code cannot be empty.[/red]")
        return False

    console.print("  [dim]Exchanging code for tokens...[/dim]", end=" ")
    try:
        tokens = await exchange_code(auth_code, verifier)
    except RuntimeError as e:
        console.print("[red]Failed[/red]")
        console.print(f"  [red]{e}[/red]")
        return False
    console.print("[green]OK[/green]")

    console.print("  [dim]Creating API key...[/dim]", end=" ")
    try:
        api_key = await create_api_key(tokens["access_token"])
    except RuntimeError as e:
        console.print("[red]Failed[/red]")
        console.print(f"  [red]{e}[/red]")
        return False
    console.print("[green]OK[/green]")

    credential = {"type": "api", "key": api_key}
    await Auth.set("anthropic", credential)
    from ..storage.storage import Storage

    path = Storage._key_to_path(["auth", "credentials"])
    console.print(f"\n  [green]API key created and saved to {path}[/green]")
    return True


async def auth_login_interactive(console: Console) -> bool:
    """Interactive login flow: select provider, enter key, validate, save.

    Args:
        console: Rich console for output

    Returns:
        True if credentials were saved successfully
    """
    console.print("\n[bold]Select provider:[/bold]")
    provider_list = list(PROVIDERS.items())
    for i, (name, info) in enumerate(provider_list, 1):
        console.print(f"  {i}. {info['label']}")

    # Get provider selection
    try:
        choice = console.input("\n  Select [1-3]: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/dim]")
        return False

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(provider_list):
            raise ValueError
    except ValueError:
        console.print("[red]Invalid selection.[/red]")
        return False

    provider_name, provider_info = provider_list[idx]

    # Anthropic: browser-only login methods
    if provider_name == "anthropic":
        console.print("\n[bold]Select login method:[/bold]")
        console.print("  1. Claude Pro/Max (OAuth login)")
        console.print("  2. Create API Key via browser")

        try:
            method = console.input("\n  Select [1-2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Cancelled.[/dim]")
            return False

        if method == "1":
            return await _anthropic_oauth_login(console)
        elif method == "2":
            return await _anthropic_oauth_create_key(console)
        else:
            console.print("[red]Invalid selection.[/red]")
            return False

    # Non-Anthropic providers: existing manual key flow
    try:
        api_key = console.input(f"\n  {provider_info['label']} API Key: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/dim]")
        return False

    if not api_key:
        console.print("[red]API key cannot be empty.[/red]")
        return False

    # Get extra fields for Azure
    extra_kwargs = {}
    credential = {"type": "api", "key": api_key}

    if provider_name == "azure_openai":
        try:
            endpoint = console.input("  Azure Endpoint (e.g., https://myapp.openai.azure.com): ").strip()
            deployment = console.input("  Azure Deployment (e.g., gpt-4.1-mini): ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Cancelled.[/dim]")
            return False

        if not endpoint or not deployment:
            console.print("[red]Endpoint and deployment are required for Azure OpenAI.[/red]")
            return False

        extra_kwargs["endpoint"] = endpoint
        extra_kwargs["deployment"] = deployment
        credential["endpoint"] = endpoint
        credential["deployment"] = deployment

    # Validate the credential
    console.print("\n  [dim]Validating...[/dim]", end=" ")
    success, message = await validate_credential(provider_name, api_key, **extra_kwargs)

    if success:
        console.print("[green]OK[/green]")
    else:
        console.print("[red]Failed[/red]")
        console.print(f"  [red]{message}[/red]")
        # Ask if they want to save anyway
        try:
            save_anyway = console.input("\n  Save anyway? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Cancelled.[/dim]")
            return False
        if save_anyway != "y":
            console.print("[dim]Not saved.[/dim]")
            return False

    # Save credential
    await Auth.set(provider_name, credential)
    from ..storage.storage import Storage

    path = Storage._key_to_path(["auth", "credentials"])
    console.print(f"\n  [green]Credentials saved to {path}[/green]")
    return True


async def auth_logout_interactive(console: Console) -> bool:
    """Interactive logout flow: select provider, remove credential.

    Args:
        console: Rich console for output

    Returns:
        True if credentials were removed
    """
    stored = await Auth.all()
    if not stored:
        console.print("[dim]No stored credentials found.[/dim]")
        return False

    console.print("\n[bold]Stored credentials:[/bold]")
    stored_list = []
    for i, (name, _cred) in enumerate(stored.items(), 1):
        label = PROVIDERS.get(name, {}).get("label", name)
        console.print(f"  {i}. {label}")
        stored_list.append(name)

    try:
        choice = console.input(f"\n  Select to remove [1-{len(stored_list)}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/dim]")
        return False

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(stored_list):
            raise ValueError
    except ValueError:
        console.print("[red]Invalid selection.[/red]")
        return False

    provider_name = stored_list[idx]
    removed = await Auth.remove(provider_name)
    if removed:
        label = PROVIDERS.get(provider_name, {}).get("label", provider_name)
        console.print(f"  [green]Removed credentials for {label}.[/green]")
        return True
    else:
        console.print("[dim]No credentials to remove.[/dim]")
        return False


async def auth_status(console: Console) -> None:
    """Display authentication status for all providers.

    Args:
        console: Rich console for output
    """
    statuses = await Auth.status()

    table = Table(title="Authentication Status", show_header=True, header_style="bold")
    table.add_column("Provider", style="cyan")
    table.add_column("Source", style="white")
    table.add_column("Key", style="dim")

    for entry in statuses:
        source = entry["source"]
        if source == "env":
            source_display = "[green]env var[/green]"
        elif source == "oauth":
            source_display = "[magenta]oauth[/magenta]"
        elif source == "stored":
            source_display = "[blue]stored[/blue]"
        else:
            source_display = "[dim]not set[/dim]"

        table.add_row(
            entry["label"],
            source_display,
            entry["masked_key"] or "-",
        )

    console.print()
    console.print(table)
    console.print("\n[dim]Priority: environment variables > stored credentials[/dim]")
