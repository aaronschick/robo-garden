"""Main Textual application with tab-based navigation across all 5 spaces.

Also exposes a keyboard shortcut ``s`` / action ``open_studio`` which launches
the Isaac Sim Design Studio (by spawning the server script and running
``robo-garden --mode studio`` in the foreground).  The Textual TUI continues
to act as the headless-friendly launcher / status hub.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane

from robo_garden.tui.screens.chat import ChatScreen
from robo_garden.tui.screens.building import BuildingScreen
from robo_garden.tui.screens.environments import EnvironmentsScreen
from robo_garden.tui.screens.training import TrainingScreen
from robo_garden.tui.screens.rewards import RewardsScreen


class RoboGardenApp(App):
    """Robo Garden: Claude-Powered Robot Studio."""

    TITLE = "Robo Garden"
    CSS = """
    Screen { background: $surface; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 0; height: 1fr; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "open_studio", "Open Studio"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Chat", id="tab-chat"):
                yield ChatScreen()
            with TabPane("Building", id="tab-building"):
                yield BuildingScreen()
            with TabPane("Environments", id="tab-environments"):
                yield EnvironmentsScreen()
            with TabPane("Training", id="tab-training"):
                yield TrainingScreen()
            with TabPane("Rewards", id="tab-rewards"):
                yield RewardsScreen()
        yield Footer()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_open_studio(self) -> None:
        """Launch the Isaac Sim Design Studio in an external terminal.

        Spawns two processes detached from the TUI so the user can continue
        using the Textual interface while the Studio runs:
          1. Isaac Sim server (isaac_server/launch.ps1)
          2. robo-garden --mode studio (connects to it)

        On non-Windows or when PowerShell is unavailable, prints instructions.
        """
        from rich.console import Console
        console = Console()

        project_root = Path(__file__).parent.parent.parent.parent
        launcher = project_root / "isaac_server" / "launch.ps1"

        if sys.platform == "win32" and launcher.exists():
            try:
                new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                subprocess.Popen(
                    ["powershell", "-NoExit", "-File", str(launcher)],
                    creationflags=new_console,
                )
                subprocess.Popen(
                    [
                        "powershell", "-NoExit", "-Command",
                        "Start-Sleep 8; uv run robo-garden --mode studio",
                    ],
                    creationflags=new_console,
                )
                console.print(
                    "[green]Launched Isaac Sim + Studio in new terminals.[/]"
                )
                return
            except Exception as exc:
                console.print(f"[red]Failed to spawn Studio:[/] {exc}")

        console.print(
            "[yellow]Open Studio manually:[/]\n"
            "  1. Start Isaac Sim:   ./isaac_server/launch.ps1\n"
            "  2. In another shell:  uv run robo-garden --mode studio"
        )
