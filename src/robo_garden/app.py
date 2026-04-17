"""robo-garden-app entry point.

Probes for a running Isaac Sim server.  If found, starts the Studio backend
(which bridges the Textual TUI chat to the Isaac Sim viewport) in a daemon
thread and then opens the Textual TUI.  If Isaac Sim is not running, opens the
TUI alone so the user can still use the chat interface and press `s` to launch
Isaac Sim later.

Usage:
    uv run robo-garden-app
"""

from __future__ import annotations

import logging
import socket
import sys
import threading
import urllib.parse
from typing import Optional

log = logging.getLogger(__name__)


def _probe_isaac(url: str, timeout: float = 1.5) -> bool:
    """Return True if an Isaac Sim WebSocket server is listening at *url*."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8765
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _start_studio_backend(isaac_url: str) -> Optional[object]:
    """Connect a Studio instance and run it in a daemon thread.

    Returns the Studio on success, None on failure.
    """
    try:
        from robo_garden.studio import Studio
        studio = Studio(isaac_url=isaac_url)
        if not studio.connect():
            log.warning("robo-garden-app: Studio.connect() failed")
            return None
        t = threading.Thread(target=studio.run_forever, daemon=True, name="studio-backend")
        t.start()
        return studio
    except Exception as exc:
        log.warning(f"robo-garden-app: could not start Studio backend — {exc}")
        return None


def main() -> None:
    """Entry point for `robo-garden-app`."""
    from rich.console import Console
    console = Console()

    isaac_url = "ws://localhost:8765"
    studio = None

    console.print("[bold cyan]Robo Garden[/bold cyan]  —  initializing…")

    if _probe_isaac(isaac_url):
        console.print(f"[green]✓[/green] Isaac Sim found at [bold]{isaac_url}[/bold]")
        studio = _start_studio_backend(isaac_url)
        if studio:
            console.print("[green]✓[/green] Studio backend connected")
        else:
            console.print(
                "[yellow]⚠[/yellow] Isaac Sim reachable but Studio backend failed to connect.\n"
                "  TUI will open without live viewport sync."
            )
    else:
        console.print(
            f"[yellow]⚠[/yellow] Isaac Sim not found at [bold]{isaac_url}[/bold].\n"
            "  Press [bold]s[/bold] in the TUI to launch it."
        )

    from robo_garden.tui.app import RoboGardenApp
    app = RoboGardenApp(studio_connected=studio is not None)
    try:
        app.run()
    finally:
        if studio is not None:
            try:
                studio.close()
            except Exception:
                pass
