"""Training progress visualization using Rich live display."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()


def print_training_progress(
    timestep: int,
    total_timesteps: int,
    mean_reward: float,
    episode_length: float,
    fps: float,
    extra_metrics: dict | None = None,
):
    """Print a training progress update to the terminal."""
    pct = timestep / total_timesteps * 100
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()

    table.add_row("Progress", f"{pct:.1f}% ({timestep:,}/{total_timesteps:,})")
    table.add_row("Mean Reward", f"{mean_reward:.3f}")
    table.add_row("Episode Length", f"{episode_length:.0f}")
    table.add_row("FPS", f"{fps:.0f}")

    if extra_metrics:
        for k, v in extra_metrics.items():
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))

    console.print(table)
