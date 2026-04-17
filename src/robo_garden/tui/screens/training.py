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
        mean_reward = float(metrics.get("eval/episode_reward", 0))
        best_reward = metrics.get("best_reward")
        elapsed = metrics.get("elapsed_s")
        total = metrics.get("total_timesteps")
        tps = metrics.get("timesteps_per_second")
        backend = metrics.get("_backend", "")

        # JIT compilation status line
        if step == 0 and backend and "compiling" in backend.lower():
            log.write(f"[yellow]⟳ {backend}[/yellow]")
            return

        parts: list[str] = []

        if total and total > 0:
            pct = min(100.0, 100.0 * step / total)
            parts.append(f"[dim]{pct:5.1f}%[/dim]  step={step:,}")
        else:
            parts.append(f"step={step:,}")

        parts.append(f"reward=[cyan]{mean_reward:+.3f}[/cyan]")

        if best_reward is not None:
            parts.append(f"best=[green]{best_reward:+.3f}[/green]")

        if elapsed is not None:
            m, s = divmod(int(elapsed), 60)
            parts.append(f"elapsed={m}m{s:02d}s")

        if tps is not None and tps > 0:
            parts.append(f"[dim]{tps:,.0f} t/s[/dim]")

        if backend:
            parts.append(f"[dim][{backend}][/dim]")

        log.write("  ".join(parts))

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()
