"""Training screen: shows live training progress."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import RichLog


class TrainingScreen(Widget):
    """Displays training step metrics as they arrive from the RL engine."""

    BINDINGS = [
        Binding("ctrl+l", "clear_log", "Clear"),
    ]

    def compose(self) -> ComposeResult:
        yield RichLog(id="log", wrap=True, markup=True, highlight=False)

    def on_mount(self) -> None:
        log = self.query_one("#log", RichLog)
        log.write("[dim]No training run active. Use the Chat tab to start training.[/dim]")

    def log_update(self, step: int, metrics: dict) -> None:
        """Append a single training step line to the log."""
        log = self.query_one("#log", RichLog)
        mean_reward = metrics.get("eval/episode_reward", 0)
        log.write(f"step={step:,}  mean_reward={mean_reward:.3f}")

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()
