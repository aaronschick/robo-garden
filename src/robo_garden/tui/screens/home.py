"""Home screen for the Textual TUI — welcome, quick-start, recent activity."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import RichLog, Static


_WELCOME = """\
[bold cyan]Robo Garden[/bold cyan]  —  Claude-powered robot design and training studio

[dim]─────────────────────────────────────────────────────────[/dim]

[bold]Modes[/bold]
  [cyan]Design[/cyan]    —  describe robots in chat; Claude generates MJCF
  [cyan]Simulate[/cyan]  —  drive robots with a controller  [dim](Phase 4)[/dim]
  [cyan]Train[/cyan]     —  run and monitor RL training jobs
  [cyan]Skills[/cyan]    —  browse and replay trained behaviors  [dim](Phase 3)[/dim]
  [cyan]Compose[/cyan]   —  combine skills into policy versions  [dim](Phase 6)[/dim]
  [cyan]Deploy[/cyan]    —  export checkpoints for deployment  [dim](Phase 8)[/dim]

[bold]Quick start[/bold]
  1. Press [bold]s[/bold] or click [bold]Open Studio[/bold] to launch Isaac Sim + backend.
  2. Switch to the [cyan]Design[/cyan] tab and describe a robot in the chat.
  3. Claude generates MJCF, validates physics, simulates, and iterates.

[bold]Keyboard shortcuts[/bold]
  [bold]q[/bold]  quit        [bold]s[/bold]  open Studio        [bold]ctrl+l[/bold]  clear chat log
"""


class HomeScreen(Widget):
    """Textual TUI home screen: welcome message and quick-start guide."""

    def compose(self) -> ComposeResult:
        yield Static(_WELCOME, markup=True, id="home-welcome")
        yield RichLog(id="home-log", wrap=True, markup=True, highlight=False)

    def on_mount(self) -> None:
        log = self.query_one("#home-log", RichLog)
        self._refresh_recent(log)

    def _refresh_recent(self, log: RichLog) -> None:
        """Show recent training runs if any exist."""
        try:
            from robo_garden.training.history import load_recent
            runs = load_recent(limit=5)
            if not runs:
                return
            log.write("[bold]Recent training runs[/bold]")
            for r in runs:
                status = "[green]✓[/green]" if r.get("success") else "[red]✗[/red]"
                robot = r.get("robot_name", "?")
                best = r.get("best_reward")
                best_txt = f"best={best:+.3f}" if best is not None else ""
                steps = r.get("total_timesteps")
                steps_txt = f"{int(steps):,} steps" if steps else ""
                log.write(f"  {status} [cyan]{robot}[/cyan]  {best_txt}  {steps_txt}")
        except Exception:
            pass
