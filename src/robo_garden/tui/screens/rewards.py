"""Rewards screen: shows Eureka reward iteration history."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import RichLog


class RewardsScreen(Widget):
    """Displays the Eureka reward-design iteration table."""

    BINDINGS = [
        Binding("ctrl+l", "clear_log", "Clear"),
    ]

    def compose(self) -> ComposeResult:
        yield RichLog(id="log", wrap=True, markup=True, highlight=False)

    def on_mount(self) -> None:
        log = self.query_one("#log", RichLog)
        log.write("[dim]No reward iterations yet. Use the Chat tab to design rewards.[/dim]")

    def add_iteration(
        self,
        iteration: int,
        mean_reward: float,
        max_reward: float,
        description: str,
    ) -> None:
        """Append a formatted reward-iteration row to the log."""
        log = self.query_one("#log", RichLog)
        log.write(
            f"[bold]#{iteration:>3}[/bold]  "
            f"mean={mean_reward:>8.3f}  "
            f"max={max_reward:>8.3f}  "
            f"[dim]{description}[/dim]"
        )

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()
