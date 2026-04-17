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

    def __init__(self) -> None:
        super().__init__()
        # Tracks line index per iteration so update_best_reward can append a
        # follow-up line rather than trying to mutate the immutable RichLog.
        self._iteration_descriptions: dict[int, str] = {}

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
        self._iteration_descriptions[iteration] = description
        reward_str = (
            f"mean={mean_reward:>8.3f}  max={max_reward:>8.3f}"
            if mean_reward != 0.0 or max_reward != 0.0
            else "[dim]training pending…[/dim]"
        )
        log.write(
            f"[bold]#{iteration:>3}[/bold]  "
            f"{reward_str}  "
            f"[dim]{description}[/dim]"
        )

    def update_best_reward(self, iteration: int, best_reward: float) -> None:
        """Append a training-result line for an existing iteration."""
        log = self.query_one("#log", RichLog)
        desc = self._iteration_descriptions.get(iteration, "")
        log.write(
            f"[bold]#{iteration:>3}[/bold]  "
            f"[green]best={best_reward:>8.3f}[/green]  "
            f"[dim]{desc} — training complete[/dim]"
        )

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()
