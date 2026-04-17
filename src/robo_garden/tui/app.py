"""Main Textual application — mode-taxonomy tabs mirroring the Isaac Sim extension.

Tab layout:
  Home | Design (Chat / Robot / Envs) | Simulate | Train (Progress / Rewards)
  | Skills | Compose | Deploy

The `s` keybinding (and the robo-garden-app launcher) still spawns Isaac Sim +
the Studio backend in external windows when running standalone.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane

from robo_garden.tui.screens.home import HomeScreen
from robo_garden.tui.screens.chat import ChatScreen
from robo_garden.tui.screens.building import BuildingScreen
from robo_garden.tui.screens.environments import EnvironmentsScreen
from robo_garden.tui.screens.training import TrainingScreen
from robo_garden.tui.screens.rewards import RewardsScreen
from robo_garden.tui.screens.stub import StubScreen


class RoboGardenApp(App):
    """Robo Garden: Claude-Powered Robot Studio."""

    TITLE = "Robo Garden"
    CSS = """
    Screen { background: $surface; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 0; height: 1fr; }
    #home-welcome { padding: 1 2; }
    #home-log { height: auto; padding: 0 2; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "open_studio", "Open Studio"),
    ]

    def __init__(self, studio_connected: bool = False) -> None:
        super().__init__()
        self.studio_connected = studio_connected

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(id="modes"):
            # ── Home ───────────────────────────────────────────────────
            with TabPane("Home", id="tab-home"):
                yield HomeScreen()

            # ── Design (Chat + Building + Environments) ────────────────
            with TabPane("Design", id="tab-design"):
                with TabbedContent():
                    with TabPane("Chat", id="tab-design-chat"):
                        yield ChatScreen()
                    with TabPane("Robot", id="tab-design-robot"):
                        yield BuildingScreen()
                    with TabPane("Environments", id="tab-design-envs"):
                        yield EnvironmentsScreen()

            # ── Simulate (Phase 4) ─────────────────────────────────────
            with TabPane("Simulate", id="tab-simulate"):
                yield StubScreen(
                    "Simulate",
                    "Load any robot and drive it with a gamepad.",
                    "Phase 4",
                )

            # ── Train (Progress + Rewards) ─────────────────────────────
            with TabPane("Train", id="tab-train"):
                with TabbedContent():
                    with TabPane("Progress", id="tab-train-progress"):
                        yield TrainingScreen()
                    with TabPane("Rewards", id="tab-train-rewards"):
                        yield RewardsScreen()

            # ── Skills (Phase 3) ───────────────────────────────────────
            with TabPane("Skills", id="tab-skills"):
                yield StubScreen(
                    "Skills Library",
                    "Browse named behaviors per robot and play them back.",
                    "Phase 3",
                )

            # ── Compose (Phase 6) ──────────────────────────────────────
            with TabPane("Compose", id="tab-compose"):
                yield StubScreen(
                    "Policy Composer",
                    "Combine skills into named policy versions.",
                    "Phase 6",
                )

            # ── Deploy (Phase 8) ───────────────────────────────────────
            with TabPane("Deploy", id="tab-deploy"):
                yield StubScreen(
                    "Deploy / Export",
                    "Export checkpoints for real-world deployment.",
                    "Phase 8",
                )

        yield Footer()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_open_studio(self) -> None:
        """Launch the Isaac Sim Design Studio in an external terminal.

        Spawns two processes detached from the TUI so the user can continue
        using the Textual interface while the Studio runs.
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
